import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, Trials, STATUS_OK
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
    X = df[feature_cols]

    # Train/Test split (test is pure holdout)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train/Val split (val used for early stopping + tuning)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )

    # Sanitize dtypes for XGBoost compatibility
    X_train = sanitize_dtypes(X_train)
    X_val = sanitize_dtypes(X_val)
    X_test = sanitize_dtypes(X_test)

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

    return df_pred, X_pred


def generate_prediction_intervals(
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
    random_state: int = 62820,
    group_col: str = "player_id",
    id_cols: List[str] = None,
    n_estimators: int = 5000,
    early_stopping_rounds: int = 50,
) -> pd.DataFrame:
    """
    Estimate prediction intervals via bootstrap-resampled XGBoost models.

    Default strategy:
      - Combines train and validation sets internally.
      - Bootstrap sample at the *player* level (group_col).
      - Use out-of-bag (OOB) *players* each iteration for early stopping.
      - Aggregate predictions across bootstraps to produce percentile intervals.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (from split_data_nba).
    X_val : pd.DataFrame
        Validation feature matrix (from split_data_nba).
    y_train : pd.Series
        Training target vector.
    y_val : pd.Series
        Validation target vector.
    X_pred : pd.DataFrame
        Prediction feature matrix (from build_prediction_frame).
    source_df : pd.DataFrame
        Original combined DataFrame containing player ID columns for grouping.
    base_params : dict
        Tuned XGBoost hyperparameters (from tune_xgb_nba).
    model_objective : str, default "reg:squarederror"
        XGBoost objective function.
    metric : str, default "rmse"
        Evaluation metric for early stopping.
    n_bootstrap : int, default 30
        Number of bootstrap iterations.
    random_state : int, default 62820
        Random seed for reproducibility.
    group_col : str, default "player_id"
        Column to use for player-level grouping in bootstrap resampling.
    id_cols : List[str], optional
        ID columns to include in output. Default: ["player_id", "player_name_clean"]
    n_estimators : int, default 5000
        Maximum number of boosting rounds per bootstrap model.
    early_stopping_rounds : int, default 50
        Early stopping patience for OOB evaluation.

    Returns
    -------
    pd.DataFrame
        DataFrame with prediction intervals and upside/downside metrics:
        - pred_mean: Mean prediction across bootstraps
        - pred_p10: 10th percentile prediction
        - pred_p50: Median prediction
        - pred_p90: 90th percentile prediction
        - pred_width_80: Width of 80% prediction interval (p90 - p10)
        - pred_upside: Upside potential (p90 - mean)
        - pred_downside: Downside risk (mean - p10)
        - implied_upside: Ratio of upside to downside

    Notes
    -----
    - Combines train and validation sets internally for bootstrap resampling.
    - Recovers player ID columns from source_df using index alignment.
    - If OOB set is too small in an iteration, falls back to training without early stopping.
    - Assumes X_train, X_val, and X_pred have been sanitized for XGBoost compatibility.
    """
    if id_cols is None:
        id_cols = ["player_id", "player_name_clean"]

    # --- Combine train and validation sets ---
    X_combined = pd.concat([X_train, X_val], axis=0)
    y_combined = pd.concat([y_train, y_val], axis=0)

    # --- Recover ID columns from source_df using index alignment ---
    for col in id_cols:
        if col in source_df.columns:
            X_combined[col] = source_df.loc[X_combined.index, col].values

    # --- Validate group_col is present ---
    if group_col not in X_combined.columns:
        raise ValueError(
            f"OOB-by-player requires '{group_col}' to be present in source_df. "
            f"Ensure source_df contains the column '{group_col}'."
        )

    def _drop_ids(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

    # --- Prepare matrices ---
    X_tr = _drop_ids(X_combined).reset_index(drop=True)
    X_p = _drop_ids(X_pred).reset_index(drop=True)
    y_tr_full = y_combined.reset_index(drop=True)

    # --- Params ---
    params = {
        "objective": model_objective,
        "eval_metric": metric,
        "tree_method": "hist",
        "enable_categorical": True,
    }
    params.update(base_params or {})

    # --- Out of Bag (OOB) grouping (player-level) ---
    group_ids = X_combined[group_col].reset_index(drop=True).values
    unique_players = pd.unique(group_ids)

    # Guardrails
    min_oob_rows = 200  # keep early stopping stable

    preds_list: List[np.ndarray] = []

    for b in tqdm(range(n_bootstrap), desc="Bootstrapping prediction intervals"):
        rng_b = np.random.default_rng(random_state + b)

        # Sample players with replacement, then include all rows for sampled players
        # TODO: Fix the bootstrapping methodology, conforming to a set makes bootstrapping redundant
        boot_players = rng_b.choice(
            unique_players, size=len(unique_players), replace=True
        )
        boot_set = set(boot_players)

        in_bag_mask = np.isin(group_ids, list(boot_set))
        idx_boot = np.where(in_bag_mask)[0]
        idx_oob = np.where(~in_bag_mask)[0]

        X_fit = X_tr.iloc[idx_boot]
        y_fit = y_tr_full.iloc[idx_boot]

        model = XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state + b,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=-1,
            **params,
        )

        # Use OOB players for early stopping when we have enough rows
        use_oob = idx_oob.size >= min_oob_rows
        if use_oob:
            X_oob = X_tr.iloc[idx_oob]
            y_oob = y_tr_full.iloc[idx_oob]
            model.fit(X_fit, y_fit, eval_set=[(X_oob, y_oob)], verbose=False)
        else:
            # Fallback: fit without eval_set (early stopping won't activate)
            model.fit(X_fit, y_fit, verbose=False)

        base_preds = model.predict(X_p)

        # --- Add residual noise from OOB predictions to generate player prediction intervals ---
        if use_oob:
            oob_preds = model.predict(X_oob)
            residuals = y_oob.values - oob_preds
            residuals = residuals - residuals.mean()  # de-bias
            noise = rng_b.choice(residuals, size=len(base_preds), replace=True)
            preds = base_preds + noise
        else:
            preds = base_preds

        preds_list.append(preds)

    pred_mat = np.vstack(preds_list)  # (n_bootstrap, n_rows_pred)

    out = pd.DataFrame(
        {
            "pred_mean": pred_mat.mean(axis=0),
            "pred_p10": np.percentile(pred_mat, 10, axis=0),
            "pred_p50": np.percentile(pred_mat, 50, axis=0),
            "pred_p90": np.percentile(pred_mat, 90, axis=0),
        },
        index=X_pred.index,
    )

    downside_floor = 0.01 * out["pred_mean"].abs().clip(lower=1.0)  # 1% of mean

    out["pred_width_80"] = out["pred_p90"] - out["pred_p10"]
    out["pred_upside"] = out["pred_p90"] - out["pred_mean"]
    out["pred_downside"] = out["pred_mean"] - out["pred_p10"]
    out["implied_upside"] = out["pred_upside"] / (out["pred_downside"] + downside_floor)

    # Prepend id columns from source_df for prediction rows
    id_present = [c for c in id_cols if c in source_df.columns]
    if id_present:
        ids = source_df.loc[X_pred.index, id_present].reset_index(drop=True)
        out = pd.concat([ids, out.reset_index(drop=True)], axis=1)
        out.index = X_pred.index

    return out
