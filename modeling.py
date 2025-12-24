import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, Trials, STATUS_OK


def split_data_nba(
    df: pd.DataFrame,
    pred_year: int,
    target_col: str = "fantasy_points_future",
    drop_cols: List[str] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
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
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Train a baseline XGBoost regression model to predict future fantasy points and evaluate its performance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (predictors).
    X_test : pd.DataFrame
        Testing feature matrix (predictors).
    y_train : pd.Series
        Training target vector (future fantasy points).
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
    - Prints RMSE, MAE, and R^2 metrics for the test set.
    - Uses fixed hyperparameters for the baseline model.
    - Assumes categorical features are handled appropriately in X_train/X_test.
    """
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

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[Baseline] RMSE: {rmse:.3f}")
    print(f"[Baseline] MAE:  {mae:.3f}")
    print(f"[Baseline] R^2:  {r2:.3f}")

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
            "fantasy_points",
            "fantasy_points_future",
            "predicted_fantasy_points",
            "prediction_diff",
        ]

    results = X_test.copy()
    results["predicted_fantasy_points"] = y_pred

    # Join player/season info from source DataFrame using index alignment
    results = results.join(
        source_df[["player_name_clean", "season", "fantasy_points_future"]], how="left"
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
    metric: str = "rmse",
    alpha: float = 1.5,
    evals: int = 60,
    random_state: int = 62820,
    max_depth_choices: list = None,
):
    """
    Tune XGBoost hyperparameters using Hyperopt with early stopping on a validation set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_train : pd.Series
        Training target vector.
    y_val : pd.Series
        Validation target vector.
    space : dict
        Hyperopt search space for XGBoost hyperparameters.
    metric : str, default "rmse"
        Optimization metric: "rmse", "mae", or "asymmetric".
    alpha : float, default 1.5
        Asymmetry parameter for custom loss (used if metric="asymmetric").
    evals : int, default 60
        Number of Hyperopt trials (max_evals).
    random_state : int, default 62820
        Random seed for reproducibility.

    Returns
    -------
    best_params : dict
        Dictionary of best hyperparameters found by Hyperopt.

    Notes
    -----
    - Uses early stopping on the validation set for each trial.
    - Supports RMSE, MAE, or a custom asymmetric loss for optimization.
    - Prints the best hyperparameters found.
    - Assumes categorical features are handled appropriately in X_train/X_val.
    """

    def asymmetric_loss(y_true, y_pred, alpha=1.5):
        residuals = y_true - y_pred
        loss = np.where(residuals > 0, alpha * (residuals**2), residuals**2)
        return np.mean(loss)

    def objective(params):
        model = XGBRegressor(
            objective="reg:squarederror",
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            colsample_bynode=float(params["colsample_bynode"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            gamma=float(params["gamma"]),
            n_estimators=2000,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="rmse",
            early_stopping_rounds=100,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        if metric == "rmse":
            loss = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        elif metric == "mae":
            loss = float(mean_absolute_error(y_val, y_pred))
        elif metric == "asymmetric":
            loss = float(asymmetric_loss(y_val, y_pred, alpha=alpha))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()
    best_params = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=evals, trials=trials
    )

    best_params = {
        "learning_rate": float(best_params["learning_rate"]),
        "max_depth": int(max_depth_choices[best_params["max_depth"]]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "colsample_bynode": float(best_params["colsample_bynode"]),
        "min_child_weight": float(best_params["min_child_weight"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "gamma": float(best_params["gamma"]),
    }

    print("Best Parameters:", best_params)
    return best_params


# Function to create final model after tuning
def create_model_nba(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    final_params: dict,
    random_state: int = 62820,
):
    """
    Fit a final XGBoost regression model using provided hyperparameters and early stopping on a validation set,
    then evaluate performance on a held-out test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix (for early stopping).
    X_test : pd.DataFrame
        Test feature matrix (for final evaluation).
    y_train : pd.Series
        Training target vector.
    y_val : pd.Series
        Validation target vector.
    y_test : pd.Series
        Test target vector.
    final_params : dict
        Dictionary of tuned XGBoost hyperparameters.
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
    - Prints RMSE, MAE, and R^2 metrics for the test set.
    - Uses early stopping on the validation set.
    - Assumes categorical features are handled appropriately in X_train/X_val/X_test.
    - Prints the best iteration if available.
    """

    model = XGBRegressor(
        objective="reg:squarederror",
        **final_params,
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Helpful diagnostic: where early stopping landed
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"Best iteration: {model.best_iteration}")

    final_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    mae = mean_absolute_error(y_test, final_pred)
    r2 = r2_score(y_test, final_pred)

    print(f"[Final tuned] RMSE: {rmse:.3f}")
    print(f"[Final tuned] MAE:  {mae:.3f}")
    print(f"[Final tuned] R^2:  {r2:.3f}")

    return model, final_pred


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
