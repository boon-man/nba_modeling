import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, Trials, STATUS_OK
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm


def sanitize_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize pandas dtypes for packages (notably XGBoost) that choke on
    pandas StringDtype-backed categoricals (e.g., categories dtype 'string[python]').

    - Converts pandas StringDtype columns -> object
    - Converts object columns to category (XGBoost enable_categorical support)
    - Forces category columns to be object-backed categories
    - Leaves numeric columns untouched

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe to sanitize.

    Returns
    -------
    pd.DataFrame
        DataFrame with sanitized dtypes ready for XGBoost.
    """
    out = X.copy()

    # Convert pandas StringDtype columns to plain object
    string_cols = out.select_dtypes(include=["string"]).columns
    for c in string_cols:
        out[c] = out[c].astype(object)

    # Convert object columns to category (for XGBoost categorical support)
    object_cols = out.select_dtypes(include=["object"]).columns
    for c in object_cols:
        out[c] = out[c].astype("category")

    # Force categoricals to have object categories (avoid 'string[python]' categories)
    cat_cols = out.select_dtypes(include=["category"]).columns
    for c in cat_cols:
        out[c] = out[c].astype(object).astype("category")

    return out


def align_categorical_dtypes(
    X: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """
    Cast the categorical columns of ``X`` to match the CategoricalDtype of the same column
    in ``reference`` (typically the training feature matrix).

    Any value present in ``X`` but not in the reference's category set is mapped to NaN,
    which XGBoost treats as missing. This prevents XGBoost 3.x from erroring on categories
    that appear at prediction time but were never seen during training.

    Parameters
    ----------
    X : pd.DataFrame
        Frame to align (e.g., the prediction feature matrix).
    reference : pd.DataFrame
        Frame whose categorical dtypes define the valid category sets (e.g., X_train).

    Returns
    -------
    pd.DataFrame
        Copy of ``X`` with categorical columns re-cast to the reference category sets.

    Notes
    -----
    - Only columns that are categorical in ``reference`` and present in ``X`` are touched;
      all other columns (numeric, etc.) are left unchanged.
    """
    out = X.copy()

    # Re-cast each categorical column to the training category set (unseen -> NaN)
    reference_cat_cols = reference.select_dtypes(include=["category"]).columns
    for c in reference_cat_cols:
        if c in out.columns:
            out[c] = out[c].astype(reference[c].dtype)

    return out


def split_data_nba(
    df: pd.DataFrame,
    pred_year: int,
    target_col: str = "fantasy_points_future",
    drop_cols: List[str] = None,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 62820,
):
    """
    Build modeling matrix and split into train/val/test.

    Notes
    -----
    - Data from pred_year and beyond is excluded (no target available).
    - Rows with target_col <= 0 are excluded (often injury/retirement/etc.).
    - Test set is held out and should never be used in tuning / early stopping.
    - Validation set is used for tuning + early stopping.
    """
    df = df.copy()

    if drop_cols is None:
        drop_cols = ["player_name_clean", "player_id", "season"]

    # Remove prediction season rows (no real target available)
    df = df.loc[df["year"] < pred_year].copy()

    # Remove cases where the "future" fantasy points are 0 (often injury/retirement/etc.)
    df = df[df[target_col] > 0].copy()

    y = df[target_col]

    cols_to_drop = set(drop_cols + [target_col])
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    # Sanitize dtypes once on the full matrix (not per-split) so every categorical column
    # carries its complete category set. Row-slicing preserves the CategoricalDtype, so the
    # train/val/test splits share identical categories and survive the pd.concat that the
    # downstream model functions perform (under pandas 3, concatenating categoricals with
    # mismatched category sets silently collapses them to a 'str' dtype XGBoost rejects).
    X = sanitize_dtypes(df[feature_cols])

    # Train/Test split (test is pure holdout)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train/Val split (val used for early stopping + tuning)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def scale_numeric_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on numeric columns of X_train and apply the transformation to both X_train and X_test.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix. Must contain only columns to be used as features.
    X_test : pd.DataFrame
        Testing feature matrix. Must contain the same columns as X_train.

    Returns
    -------
    X_train_scaled : pd.DataFrame
        Scaled training feature matrix with numeric columns standardized (mean=0, std=1).
    X_test_scaled : pd.DataFrame
        Scaled testing feature matrix with numeric columns standardized using the scaler fit on X_train.
    scaler : StandardScaler
        The fitted StandardScaler object, which can be used to transform new data in the same way.

    Notes
    -----
    - Only columns of numeric dtype are scaled; non-numeric columns are left unchanged.
    - The scaler is fit only on X_train to avoid data leakage.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    numeric_cols = X_train.select_dtypes(include=["number"]).columns

    scaler = StandardScaler()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train_scaled, X_test_scaled, scaler


def create_baseline_nba(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Tuple[XGBRegressor, np.ndarray]:
    """
    Train a baseline XGBoost regression model to predict future fantasy points and evaluate its performance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (predictors).
    X_val : pd.DataFrame
        Validation feature matrix (predictors).
    X_test : pd.DataFrame
        Testing feature matrix (predictors).
    y_train : pd.Series
        Training target vector (future fantasy points).
    y_val : pd.Series
        Validation target vector (future fantasy points).
    y_test : pd.Series
        Testing target vector (future fantasy points).

    Returns
    -------
    model : XGBRegressor
        The trained XGBoost regression model.
    y_pred : np.ndarray
        Predicted fantasy points for the test set.

    Notes
    -----
    - Combines train and validation sets for training (no early stopping for baseline).
    - Prints RMSE, MAE, and R^2 metrics for the test set.
    - Uses fixed hyperparameters for the baseline model.
    - Assumes categorical features are handled appropriately in X_train/X_val/X_test.
    """
    # Combine train and validation sets for baseline model
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1234,
        n_jobs=-1,
        enable_categorical=True,
    )

    model.fit(X_train_full, y_train_full)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    print(f"[Baseline] RMSE: {rmse:.3f}")
    print(f"[Baseline] MAE:  {mae:.3f}")
    print(f"[Baseline] R^2:  {r2:.3f}")
    print(f"[Baseline] Spearman: {spearman_corr:.3f}")

    return model, y_pred


def attach_model_results(
    X_test: pd.DataFrame,
    y_pred: np.ndarray,
    source_df: pd.DataFrame,
    result_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Attach model predictions back onto the test set and calculate performance metrics.

    Parameters:
        X_test (pd.DataFrame): Feature matrix from train/test split.
        y_pred (np.ndarray): Model predictions.
        source_df (pd.DataFrame): Original combined DataFrame (to pull player/season info).
        result_cols (List[str]): Columns to include in final results. If None, uses defaults.

    Returns:
        pd.DataFrame: Results DataFrame with predictions, actuals, and differences.
    """
    if result_cols is None:
        result_cols = [
            "player_name_clean",
            "season",
            "year",
            "age",
            "fantasy_points",
            "prediction_diff",
            "fantasy_points_future",
            "predicted_fantasy_points",
        ]

    results = X_test.copy()
    results["predicted_fantasy_points"] = y_pred

    # Join player/season info from source DataFrame using index alignment
    results = results.join(
        source_df[["player_name_clean", "season", "fantasy_points_future"]],
        how="left",
    )

    # Calculate prediction difference
    results["prediction_diff"] = round(
        results["predicted_fantasy_points"] - results["fantasy_points_future"], 2
    )

    # Return only specified columns
    return results[result_cols].sort_values(
        by="predicted_fantasy_points", ascending=False
    )


# Function to tune XGBoost hyperparameters
def tune_xgb_nba(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    space: dict,
    metric: str = "asymmetric",
    evals: int = 75,
    random_state: int = 62820,
) -> Tuple[dict, int]:
    """
    Performs hyperparameter optimization for an XGBoost regressor using Hyperopt.

    This function tunes XGBoost model hyperparameters by minimizing a specified loss metric
    (RMSE, MAE, or an asymmetric loss) on the validation set. The search is performed using
    the Hyperopt library and the Tree of Parzen Estimators (TPE) algorithm. The function
    returns the best set of hyperparameters found and the best iteration count.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training predictors.
    X_val : pd.DataFrame
        Validation predictors.
    y_train : pd.Series
        Training target.
    y_val : pd.Series
        Validation target.
    space : dict
        Hyperparameter search space for Hyperopt.
    metric : str, default "asymmetric"
        Metric to optimize ("rmse", "mae", or "asymmetric").
    evals : int, default 75
        Number of Hyperopt evaluations.
    random_state : int, default 62820
        Random seed for reproducibility.

    Returns
    -------
    best_params : dict
        Dictionary of the best hyperparameters found.
    best_iteration : int
        Number of boosting rounds from the best trial (for use in final model).
    """

    def objective(params):
        model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=float(params["learning_rate"]),
            # leaf-based tree growth
            grow_policy="lossguide",
            max_leaves=int(params["max_leaves"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            gamma=float(params["gamma"]),
            enable_categorical=True,
            n_estimators=3000,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
            early_stopping_rounds=100,
        )

        model.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
        )  # noqa: F821
        y_pred = model.predict(X_val)

        # --- compute all metrics for visibility ---
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        spearman_corr, _ = spearmanr(y_val, y_pred)

        # --- choose the one to optimize ---
        if metric == "rmse":
            loss = rmse
        elif metric == "mae":
            loss = mae
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Hyperopt will store this in trials; handy for later analysis
        return {
            "loss": loss,
            "status": STATUS_OK,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman": float(spearman_corr),
            "best_iteration": getattr(model, "best_iteration", None),
        }

    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=evals, trials=trials
    )

    best_params = {
        "learning_rate": float(best["learning_rate"]),
        "max_leaves": int(best["max_leaves"]),
        "grow_policy": "lossguide",
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": float(best["min_child_weight"]),
        "reg_lambda": float(best["reg_lambda"]),
        "reg_alpha": float(best["reg_alpha"]),
        "gamma": float(best["gamma"]),
    }

    # print the best trial's metrics
    best_trial = trials.best_trial["result"]
    best_iteration = best_trial.get("best_iteration", 100)
    print("Best Parameters:", best_params)
    print(
        f"[Best trial @ val] optimized={metric} "
        f"| RMSE={best_trial.get('rmse', float('nan')):.3f} "
        f"| MAE={best_trial.get('mae', float('nan')):.3f} "
        f"| R^2={best_trial.get('r2', float('nan')):.3f} "
        f"| Spearman={best_trial.get('spearman', float('nan')):.3f} "
        f"| best_iteration={best_iteration}"
    )

    return best_params, best_iteration


# Function to create final model after tuning
def create_model_nba(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    final_params: dict,
    n_estimators: int,
    n_estimators_mult: float = 1.15,
    random_state: int = 62820,
) -> Tuple[XGBRegressor, np.ndarray]:
    """
    Fit a final XGBoost regression model using provided hyperparameters on combined train+val data,
    then evaluate performance on a held-out test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix (combined with train for final fit).
    X_test : pd.DataFrame
        Test feature matrix (for final evaluation).
    y_train : pd.Series
        Training target vector.
    y_val : pd.Series
        Validation target vector (combined with train for final fit).
    y_test : pd.Series
        Test target vector.
    final_params : dict
        Dictionary of tuned XGBoost hyperparameters.
    n_estimators : int
        Best iteration count from tuning (used to set n_estimators for final model).
    n_estimators_mult : float, default 1.15
        Multiplier applied to n_estimators to account for larger training set.
    random_state : int, default 62820
        Random seed for reproducibility.

    Returns
    -------
    model : XGBRegressor
        The trained XGBoost regression model.
    final_pred : np.ndarray
        Predicted values for the test set.

    Notes
    -----
    - Combines train and validation sets for final model fitting.
    - Uses n_estimators * n_estimators_mult (default 1.15) to set boosting rounds.
    - Prints RMSE, MAE, and R^2 metrics for the test set.
    - Assumes categorical features are handled appropriately in X_train/X_val/X_test.
    """
    # Combine train and validation sets for final model
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    # Calculate final n_estimators with multiplier
    final_n_estimators = int(n_estimators * n_estimators_mult)

    model = XGBRegressor(
        objective="reg:squarederror",
        **final_params,
        enable_categorical=True,
        n_estimators=final_n_estimators,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train_full, y_train_full)

    # --- Test metrics ---
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_spearman, _ = spearmanr(y_test, test_pred)

    print(
        f"[Test] RMSE: {test_rmse:.3f} | MAE: {test_mae:.3f} | R^2: {test_r2:.3f} | Spearman: {test_spearman:.3f}"
    )

    return model, test_pred


def build_prediction_frame(
    df: pd.DataFrame,
    pred_year: int,
    feature_cols: List[str],
    drop_cols: List[str] = None,
    train_features: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the prediction dataframe and feature matrix for making future predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing all seasons and features.
    pred_year : int
        The season/year for which predictions are to be made.
    feature_cols : List[str]
        List of feature column names to use for prediction.
    drop_cols : List[str], optional
        Columns to drop from the prediction dataframe (default: ["player_name_clean", "player_id", "season"]).
    train_features : pd.DataFrame, optional
        Training feature matrix (e.g., X_train from split_data_nba). When provided, X_pred's
        categorical columns are cast to the training category sets so that categories unseen
        during training map to NaN. Required for XGBoost 3.x, which errors on unseen categories.

    Returns
    -------
    df_pred : pd.DataFrame
        Subset of df for pred_year, used to attach predictions back to players.
    X_pred : pd.DataFrame
        Feature matrix for prediction, containing only columns in feature_cols.

    Notes
    -----
    - Raises ValueError if any feature in feature_cols is missing from df_pred.
    - drop_cols is not used for filtering features, but can be used for downstream processing if needed.
    - When train_features is omitted, X_pred keeps its own categorical encodings, which can
      trip XGBoost 3.x if the prediction season introduces categories unseen in training.
    """
    df = df.copy()

    if drop_cols is None:
        drop_cols = [
            "player_name_clean",
            "player_id",
            "season",
        ]

    # Subset to the season you want to predict
    df_pred = df.loc[df["year"] == pred_year].copy()

    # Ensure all required feature columns exist
    missing = [c for c in feature_cols if c not in df_pred.columns]
    if missing:
        raise ValueError(f"Missing feature columns in prediction frame: {missing}")

    X_pred = df_pred[feature_cols]

    # Align categoricals to the training category sets so pred-season-only categories
    # (never seen in training) become NaN rather than tripping XGBoost 3.x.
    if train_features is not None:
        X_pred = align_categorical_dtypes(X_pred, train_features)

    return df_pred, X_pred


def index_100(x: np.ndarray) -> np.ndarray:
    """
    Standardize a vector to mean 100 / sd 15 (IQ / wRC+-style). Returns a flat 100 if the
    input has no spread. Uses sample sd (ddof=1) to match R's sd().
    """
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x, ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.full(x.shape, 100.0)
    return 100.0 + 15.0 * (x - np.nanmean(x)) / s


def _lowess_predict(y: np.ndarray, level: np.ndarray, frac: float = 0.75) -> np.ndarray:
    """
    Smooth ``y`` against ``level`` and return the fitted value at every ``level`` (the
    Python analog of R's ``predict(loess(y ~ level), newdata=...)``).

    Fits LOWESS on the finite (y, level) pairs, then interpolates the smoothed curve onto
    all levels (values outside the fitted range are clamped to the nearest endpoint). Falls
    back to a linear fit for tiny pools (<10 usable points), matching the R ``lm`` fallback.
    """
    y = np.asarray(y, dtype=float)
    level = np.asarray(level, dtype=float)
    mask = np.isfinite(y) & np.isfinite(level)

    if mask.sum() >= 10:
        smoothed = lowess(y[mask], level[mask], frac=frac, return_sorted=True)
        xs, ys = smoothed[:, 0], smoothed[:, 1]
        # np.interp needs strictly increasing x; average any tied LOWESS x-values
        uniq_x, inverse = np.unique(xs, return_inverse=True)
        if uniq_x.size != xs.size:
            ys = np.array([ys[inverse == i].mean() for i in range(uniq_x.size)])
            xs = uniq_x
        return np.interp(level, xs, ys)

    # Linear fallback for thin pools
    if mask.sum() >= 2:
        coef = np.polyfit(level[mask], y[mask], 1)
        return np.polyval(coef, level)

    return np.full(level.shape, np.nanmean(y[mask]) if mask.any() else 0.0)


def level_adjust(x: np.ndarray, level: np.ndarray) -> np.ndarray:
    """
    Level-adjust a signal into a studentized residual against the projection level, so that
    BOTH its average AND its spread are equalized across levels. A mean-only detrend flattens
    the average but a higher-variance level still over-populates the tail; dividing by the
    local spread fixes that, giving every projection level an equal shot at scoring high.

    center = residual vs a smooth of value-vs-level; scale = smooth of |residual|-vs-level
    (~ the conditional sd), floored to avoid blow-ups on thin/flat regions.
    """
    x = np.asarray(x, dtype=float)
    level = np.asarray(level, dtype=float)

    resid = x - _lowess_predict(x, level)  # center: above/below typical for the level
    local_scale = _lowess_predict(np.abs(resid), level)  # scale: conditional |residual|

    floor_scale = 0.05 * np.nanmedian(np.abs(resid))
    if not np.isfinite(floor_scale) or floor_scale <= 0:
        floor_scale = 1.0
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > floor_scale), local_scale, floor_scale
    )
    return resid / local_scale


def generate_prediction_intervals(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    X_pred: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    base_params: dict,
    model_objective: str = "reg:squarederror",
    metric: str = "rmse",
    n_bootstrap: int = 30,
    n_noise_draws: int = 50,
    n_resid_bins: int = 10,
    min_bin: int = 30,
    random_state: int = 62820,
    group_col: str = "player_id",
    id_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Simulate floor/ceiling prediction intervals via a player-level cluster bootstrap.

    Reuses the already-fit full-data ``model`` (its tuned hyperparameters via ``base_params``
    and its tree count), so no re-tuning happens. Each of ``n_bootstrap`` iterations draws
    players WITH replacement and replicates every drawn player's rows by its draw count (a
    true cluster bootstrap), fits at the model's FIXED tree count (no early stopping), and
    predicts on the prediction players — the spread of these refit predictions is the
    epistemic (model) uncertainty. Out-of-bag (fitted, residual) pairs from every iteration
    accumulate into ONE global pool, binned by fitted value so the aleatoric noise a player
    receives is sized to his own projection level (heteroscedastic). The predictive sample
    per player is::

        point_pred + epistemic deviation (refit spread, re-centered on point_pred) + binned OOB noise

    giving ``n_bootstrap * n_noise_draws`` Monte Carlo draws each. Bands are centered on the
    SAME full-data ``model`` prediction carried onto the draft sheet, not the bootstrap mean.

    Parameters
    ----------
    model : XGBRegressor
        The fitted full-data model (from create_model_nba). Supplies the band center
        (model.predict(X_pred)) and the fixed tree count for the bootstrap refits.
    X_train, X_val : pd.DataFrame
        Training / validation feature matrices (from split_data_nba).
    y_train, y_val : pd.Series
        Training / validation targets.
    X_pred : pd.DataFrame
        Prediction feature matrix (from build_prediction_frame).
    source_df : pd.DataFrame
        Combined DataFrame supplying the player ID columns for grouping / output.
    base_params : dict
        Tuned XGBoost hyperparameters (from tune_xgb_nba) used for the refits.
    model_objective : str, default "reg:squarederror"
        XGBoost objective for the refits.
    metric : str, default "rmse"
        XGBoost eval_metric for the refits (no early stopping is used).
    n_bootstrap : int, default 30
        Number of cluster-bootstrap refits (epistemic spread). Kept modest so the tails
        retain a little run-to-run variety.
    n_noise_draws : int, default 50
        Aleatoric noise draws layered on each refit -> n_bootstrap * n_noise_draws samples.
    n_resid_bins : int, default 10
        Number of fitted-value quantile bins for the heteroscedastic residual pool.
    min_bin : int, default 30
        If a fitted-value bin holds fewer residuals than this, fall back to the global pool.
    random_state : int, default 62820
        Seed; a fresh value yields a genuinely fresh simulation.
    group_col : str, default "player_id"
        Player-level grouping column recovered from source_df.
    id_cols : List[str], optional
        ID columns to prepend to the output. Default: ["player_id", "player_name_clean"].

    Returns
    -------
    pd.DataFrame
        One row per prediction player with:
        - pred_mean, pred_p05, pred_p10, pred_p50, pred_p90, pred_p95
        - pred_width   : pred_p95 - pred_p05
        - pred_upside  : pred_p95 - pred_mean (ceiling distance above the mean)
        - pred_downside: pred_mean - pred_p05 (floor distance below the mean)
        - ceiling_index: level-neutral, high = unusually high CEILING for the projection level
        - floor_index  : level-neutral, high = unusually high / safe FLOOR for the level
        - upside_index : single sortable OR-score; high = a strong ceiling-or-floor outlier

    Notes
    -----
    - ceiling_index / floor_index / upside_index are standardized across ALL prediction
      players (mean 100 / sd 15). upside_index is the magnitude of the positive parts of the
      two indices, so it rewards spiking on EITHER axis (and both) without penalizing an
      ordinary axis; because it is zero-inflated + right-skewed its median sits below 100.
    - Assumes X_train, X_val, and X_pred share categorical encodings (sanitized/aligned
      upstream) and that model was fit on the same feature set.
    """
    if id_cols is None:
        id_cols = ["player_id", "player_name_clean"]

    # --- Combine train and validation sets (the bootstrap universe) ---
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)

    # --- Recover the player grouping column from source_df via index alignment ---
    if group_col not in source_df.columns:
        raise ValueError(
            f"Player-level bootstrap requires '{group_col}' in source_df."
        )
    group_ids = source_df.loc[X_combined.index, group_col].to_numpy()

    X_tr = X_combined.reset_index(drop=True)
    y_tr = y_combined.reset_index(drop=True)
    n_pred = len(X_pred)

    # Params for the refits (same tuned hyperparameters as the point model)
    params = {
        "objective": model_objective,
        "eval_metric": metric,
        "tree_method": "hist",
        "enable_categorical": True,
    }
    params.update(base_params or {})

    # Reuse the point model's tree count so refits mirror the full-data fit (no early stopping)
    fixed_nrounds = model.get_booster().num_boosted_rounds()

    # Center every band on the SAME full-data prediction carried onto the draft sheet
    point_pred = model.predict(X_pred)

    # Map each player to its row positions once (for fast row replication per iteration)
    unique_players = pd.unique(group_ids)
    player_to_rows = {p: np.where(group_ids == p)[0] for p in unique_players}

    # --- Pass 1: cluster-bootstrap refits at a FIXED tree count (no early stopping) ---
    # base_all[b] holds each refit's prediction-player predictions (epistemic spread); every
    # iteration's OOB (fitted, residual) pairs accumulate into one global pool for the
    # heteroscedastic aleatoric noise below.
    base_all = np.full((n_bootstrap, n_pred), np.nan)
    pool_fitted: List[np.ndarray] = []
    pool_resid: List[np.ndarray] = []

    for b in tqdm(range(n_bootstrap), desc="Bootstrapping prediction intervals"):
        rng_b = np.random.default_rng(random_state + b)

        # True player-level bootstrap: sample players WITH replacement, then replicate each
        # drawn player's rows by its draw count (duplicates are the point of the bootstrap).
        boot_players = rng_b.choice(
            unique_players, size=len(unique_players), replace=True
        )
        draw_counts = pd.Series(boot_players).value_counts()
        in_bag_idx = np.concatenate(
            [np.tile(player_to_rows[p], count) for p, count in draw_counts.items()]
        )

        # Out-of-bag players (never drawn) feed the honest global residual pool
        drawn = set(boot_players.tolist())
        oob_players = [p for p in unique_players if p not in drawn]
        oob_idx = (
            np.concatenate([player_to_rows[p] for p in oob_players])
            if oob_players
            else np.array([], dtype=int)
        )

        # Fit at the fixed tree count — no eval_set, so OOB rows stay unused for round
        # selection and remain fully honest for the residual pool.
        booster = XGBRegressor(
            n_estimators=fixed_nrounds,
            random_state=random_state + b,
            n_jobs=-1,
            **params,
        )
        booster.fit(X_tr.iloc[in_bag_idx], y_tr.iloc[in_bag_idx], verbose=False)

        base_all[b] = booster.predict(X_pred)

        if oob_idx.size > 0:
            oob_preds = booster.predict(X_tr.iloc[oob_idx])
            pool_fitted.append(oob_preds)
            pool_resid.append(y_tr.iloc[oob_idx].to_numpy() - oob_preds)

    # --- Global, fitted-value-binned residual pool (heteroscedastic aleatoric noise) ---
    # Bin pooled OOB residuals by fitted value so a player draws noise sized to his own
    # projection level; residuals are centered WITHIN each bin so noise is mean-zero per level.
    g_fitted = np.concatenate(pool_fitted)
    g_resid = np.concatenate(pool_resid)

    bin_breaks = np.unique(np.quantile(g_fitted, np.linspace(0, 1, n_resid_bins + 1)))
    n_bins_eff = len(bin_breaks) - 1
    interior_edges = bin_breaks[1:-1]

    def _assign_bins(values: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(values, interior_edges), 0, n_bins_eff - 1)

    g_bin = _assign_bins(g_fitted)
    bin_residuals = [
        (g_resid[g_bin == k] - g_resid[g_bin == k].mean())
        if np.any(g_bin == k)
        else np.array([])
        for k in range(n_bins_eff)
    ]
    global_centered = g_resid - g_resid.mean()  # fallback for thin/empty bins
    pred_bin = _assign_bins(point_pred)

    # --- Pass 2: assemble the predictive sample per player ---
    # final = point_pred + epistemic deviation (refit spread re-centered on point_pred) + binned noise
    ensemble_mean = base_all.mean(axis=0)
    n_samples = n_bootstrap * n_noise_draws
    pred_mat = np.empty((n_samples, n_pred))

    rng_noise = np.random.default_rng(random_state)  # reproducible noise for a given seed
    for j in range(n_pred):
        epi_dev = np.repeat(base_all[:, j] - ensemble_mean[j], n_noise_draws)
        pool_j = bin_residuals[pred_bin[j]]
        if pool_j.size < min_bin:
            pool_j = global_centered
        noise_j = (
            rng_noise.choice(pool_j, size=n_samples, replace=True)
            if pool_j.size
            else 0.0
        )
        pred_mat[:, j] = point_pred[j] + epi_dev + noise_j

    # --- Aggregate into per-player percentile bands (centered on point_pred) ---
    pred_mean = pred_mat.mean(axis=0)
    p05, p10, p50, p90, p95 = (
        np.percentile(pred_mat, q, axis=0) for q in (5, 10, 50, 90, 95)
    )

    # Scale-free band edges relative to the projection (intermediates, dropped after indexing)
    ceiling_room = np.where(pred_mean > 0, p95 / pred_mean, np.nan)
    floor_share = np.where(pred_mean > 0, p05 / pred_mean, np.nan)

    # Two level-neutral signals: level_adjust studentizes each ratio against the projection
    # level (equalizing average AND spread), then index_100 standardizes to mean 100 / sd 15.
    ceiling_index = index_100(level_adjust(ceiling_room, pred_mean))
    floor_index = index_100(level_adjust(floor_share, pred_mean))

    # Single sortable OR-score: rewarded for spiking on EITHER axis, never penalized for an
    # ordinary (below-100) axis; re-standardized to 100 / 15.
    upside_index = index_100(
        np.sqrt(
            np.maximum(ceiling_index - 100, 0) ** 2
            + np.maximum(floor_index - 100, 0) ** 2
        )
    )

    out = pd.DataFrame(
        {
            "pred_mean": pred_mean,
            "pred_p05": p05,
            "pred_p10": p10,
            "pred_p50": p50,
            "pred_p90": p90,
            "pred_p95": p95,
            "pred_width": p95 - p05,
            "pred_upside": p95 - pred_mean,
            "pred_downside": pred_mean - p05,
            "ceiling_index": ceiling_index,
            "floor_index": floor_index,
            "upside_index": upside_index,
        },
        index=X_pred.index,
    )

    # Prepend id columns from source_df for the prediction rows
    id_present = [c for c in id_cols if c in source_df.columns]
    if id_present:
        ids = source_df.loc[X_pred.index, id_present].reset_index(drop=True)
        out = pd.concat([ids, out.reset_index(drop=True)], axis=1)
        out.index = X_pred.index

    return out
