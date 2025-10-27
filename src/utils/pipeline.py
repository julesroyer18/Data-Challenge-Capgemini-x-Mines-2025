from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from typing import List, Dict, Union, Tuple
import xgboost as xgb
from xgboost import XGBRegressor
from mapie.regression import MapieRegressor, MapieQuantileRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import os
import time
import json
from sklearn.model_selection import cross_val_score
import json
import os
from pathlib import Path  # Added for Path object
from sklearn.model_selection import (
    RandomizedSearchCV,
)  # Make sure RandomizedSearchCV is imported
import random
from sklearn import set_config

set_config(display="diagram")


# data loading
def prepare_data_no_multiindex(
    dataset_train: pd.DataFrame,
    group_col: str = "station_code",
    datetime_col: str = "ObsDate",
) -> Tuple[pd.DataFrame, pd.Series]:
    # 1. BEST PRACTICE: Start with an explicit copy to guarantee immutability
    dataset_train = dataset_train.copy()

    # 2. Sort by datetime THEN group to ensure correct temporal order
    if datetime_col not in dataset_train.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in data.")
    dataset_train[datetime_col] = pd.to_datetime(dataset_train[datetime_col])

    # dataset_train = dataset_train.set_index([datetime_col, group_col]).sort_index()
    dataset_train = dataset_train.sort_values(by=[datetime_col, group_col]).reset_index(
        drop=True
    )

    return dataset_train


def prepare_data_multiindex(
    dataset_train: pd.DataFrame,
    datetime_col: str = "ObsDate",
    group_col: str = "station_code",
    cols_to_drop: List[str] = ["index"],
) -> Tuple[pd.DataFrame, pd.Series]:
    """pass datetime_col and group_col to set multiindex"""
    # 1. BEST PRACTICE: Start with an explicit copy to guarantee immutability
    dataset_train = dataset_train.copy()

    # Drop any specified columns if they exist
    print(f"Dropping columns: {cols_to_drop}")
    dataset_train = dataset_train.drop(columns=cols_to_drop, errors="ignore")

    # 2. Set index datetime, group to ensure correct temporal order
    if datetime_col not in dataset_train.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in data.")
    dataset_train[datetime_col] = pd.to_datetime(dataset_train[datetime_col])

    dataset_train = dataset_train.set_index([datetime_col, group_col]).sort_index()

    return dataset_train


def split_features_target(
    data,
    target_column: str = "water_flow_week1",
    cols_to_drop: List[str] = [
        "station_code",
        "water_flow_week2",
        "water_flow_week3",
        "water_flow_week4",
    ],
) -> Tuple[pd.DataFrame, pd.Series]:
    # 3. Split into X and y (Scikit-learn convention: X is matrix, y is vector)
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    # Separate columns to drop from X vs. those not in data
    cols_to_drop_present = [col for col in cols_to_drop if col in data.columns]
    target_and_drops = cols_to_drop_present + [target_column]

    X = data.drop(columns=target_and_drops)
    y = data[target_column]

    # Drop rows where target is NaN (e.g., future weeks in training data)
    print(f"Dropping {y.isna().sum()} rows with NaN target values.")
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    return X, y


# --- UTILITY TRANSFORMER (FOR DROPPING FEATURES) ---
def drop_feature(X: pd.DataFrame, features: Union[str, List[str]]):
    """
    Drops specified features (columns) from a DataFrame.
    Returns a DataFrame for pipeline compatibility.
    """
    Xc = X.copy()
    if isinstance(features, str):
        features = [features]
    # Ensure all drops are performed while ignoring errors
    Xc = Xc.drop(columns=features, errors="ignore")
    return Xc


# --- 1. Custom Feature Engineering (Time Features) ---
def create_datetime_features_no_multiindex(
    X: pd.DataFrame, datetime_col: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Extracts time-based features from the index and adds them to the DataFrame.
    Returns a DataFrame.
    """
    X_copy = X.copy()
    date_series = X_copy[datetime_col].dt
    dt_features = pd.DataFrame(
        {
            # "dayofyear": X_copy.index.dayofyear,
            # "month": X_copy.index.month,
            # "year": X_copy.index.year,
            # "dayofweek": X_copy.index.dayofweek,
            "sin_month": np.sin(2 * np.pi * date_series.month / 12),
            "cos_month": np.cos(2 * np.pi * date_series.month / 12),
            "sin_dayofyear": np.sin(2 * np.pi * date_series.dayofyear / 365.25),
            "cos_dayofyear": np.cos(2 * np.pi * date_series.dayofyear / 365.25),
            # "is_winter": X_copy[datetime_col].dt.month.isin([12, 1, 2]).astype(int),
            # "is_spring": X_copy[datetime_col].dt.month.isin([3, 4, 5]).astype(int),
            # "is_summer": X_copy[datetime_col].dt.month.isin([6, 7, 8]).astype(int),
            # "is_fall": X_copy[datetime_col].dt.month.isin([9, 10, 11]).astype(int),
        },
        index=X_copy.index,  # Preserve index
    )
    # Concatenate new features with the original X_copy
    X_processed = pd.concat([X_copy, dt_features], axis=1)

    # Debug print
    if verbose:
        print("DATETIME FEATURES ADDED:")
        print(X_processed.head())

    # Return DataFrame (CRITICAL for next sequential steps)
    return X_processed


# --- 1. Custom Feature Engineering (Time Features) ---
def create_datetime_features_multiindex(
    X: pd.DataFrame, datetime_index_level: str, verbose: bool = False
) -> pd.DataFrame:
    """
    assume index is (datetime, group)
    Extracts time-based features from the index and adds them to the DataFrame.
    Returns a DataFrame.
    """
    X_copy = X.copy()
    datetime_series = X_copy.index.get_level_values(datetime_index_level)
    dt_features = pd.DataFrame(
        {
            "sin_month": np.sin(2 * np.pi * datetime_series.month / 12),
            "cos_month": np.cos(2 * np.pi * datetime_series.month / 12),
            "sin_dayofyear": np.sin(2 * np.pi * datetime_series.dayofyear / 365.25),
            "cos_dayofyear": np.cos(2 * np.pi * datetime_series.dayofyear / 365.25),
        },
        index=X_copy.index,  # Preserve index
    )
    # Concatenate new features with the original X_copy
    X_processed = pd.concat([X_copy, dt_features], axis=1)

    # Debug print
    if verbose:
        print("DATETIME FEATURES ADDED:")
        print(X_processed.head())

    # Return DataFrame (CRITICAL for next sequential steps)
    return X_processed


class LagFeatureGeneratorNoMultiIndex(BaseEstimator, TransformerMixin):
    """
    Generates lag features for specified numerical columns, grouped by an identifier.

    Best Practice: Does NOT drop rows or impute NaNs, relying on subsequent pipeline
    steps (SimpleImputer) to handle the missing values created by the shift operation.
    """

    def __init__(
        self,
        lags: Union[int, List[int]] = [1, 2],
        cols_to_lag: List[str] = None,
        group_col: str = "station_code",
    ):
        self.lags = lags
        self.cols_to_lag = cols_to_lag
        self.group_col = group_col
        # NOTE: Impute method removed, SimpleImputer handles it later.

    def fit(self, X, y=None):
        if self.cols_to_lag is None:
            raise ValueError("cols_to_lag must be specified")
        return self

    def transform(
        self, X: pd.DataFrame, group_col: str, datetime_col: str
    ) -> pd.DataFrame:
        X_ = X.copy().sort_values(by=[datetime_col, group_col]).reset_index(drop=True)

        if not isinstance(X_.index, pd.DatetimeIndex):
            print(
                "Warning: Index is not DatetimeIndex. Assuming data is chronologically sorted."
            )

        if self.group_col not in X_.columns:
            # If group_col is not found, assume all data is a single time series
            print(
                f"Warning: Group column '{self.group_col}' not found. Lagging across entire dataset."
            )

        for col in self.cols_to_lag:
            if col not in X_.columns:
                print(f"Warning: Column '{col}' to lag not found. Skipping.")
                continue

            for lag in self.lags:
                lag_name = f"{col}_lag_{lag}"

                # Check if group_col exists before grouping
                if self.group_col in X_.columns:
                    # Best practice: Group by station before shifting to prevent leakage
                    X_[lag_name] = X_.groupby(self.group_col, group_keys=False)[
                        col
                    ].shift(lag)
                else:
                    raise ValueError(
                        f"Group column '{self.group_col}' not found in DataFrame."
                    )

        # Return the DataFrame with new columns (containing NaNs at the start of each group)
        return X_


class LagFeatureGeneratorMultiIndex(BaseEstimator, TransformerMixin):
    """
    Generates lag features for specified numerical columns.
    Assumes the input DataFrame X has a MultiIndex (datetime_level, group_level).

    Best Practice: Does NOT drop rows or impute NaNs, relying on subsequent pipeline
    steps (e.g., SimpleImputer) to handle the missing values created by the shift operation.
    """

    def __init__(
        self,
        lags: Union[int, List[int]] = [1, 2],
        cols_to_lag: List[str] = None,
        # The name of the index level for the group identifier (e.g., 'station_code')
        group_level_name: str = "station_code",
    ):
        # Ensure lags is a list
        self.lags = [lags] if isinstance(lags, int) else lags
        self.cols_to_lag = cols_to_lag
        self.group_level_name = group_level_name

    def fit(self, X: pd.DataFrame, y=None):
        if self.cols_to_lag is None:
            raise ValueError("cols_to_lag must be specified.")

        # Check if the MultiIndex has the required group level
        if not isinstance(X.index, pd.MultiIndex):
            raise TypeError("Input DataFrame X must have a MultiIndex.")

        if self.group_level_name not in X.index.names:
            raise ValueError(
                f"Group level '{self.group_level_name}' not found in the DataFrame's MultiIndex names: {X.index.names}."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generates lag features by shifting data grouped by the group_level_name
        in the MultiIndex.
        """

        # NOTE: We assume the MultiIndex is already sorted correctly:
        # (datetime_level, group_level_name)

        X_ = X.copy()

        # Check for required index type and level name
        if not isinstance(X_.index, pd.MultiIndex):
            raise TypeError("Input DataFrame X must have a MultiIndex.")
        if self.group_level_name not in X_.index.names:
            # This check is technically also in fit, but good to have in transform
            raise ValueError(
                f"Group level '{self.group_level_name}' not found in the DataFrame's MultiIndex names: {X_.index.names}."
            )

        for col in self.cols_to_lag:
            if col not in X_.columns:
                print(f"Warning: Column '{col}' to lag not found. Skipping.")
                continue

            for lag in self.lags:
                lag_name = f"{col}_lag_{lag}"

                # Key Change: Use groupby on the index level, which keeps the MultiIndex
                # The shift operation is correctly applied within each group defined
                # by the group_level_name.
                X_[lag_name] = X_.groupby(
                    level=self.group_level_name, group_keys=False
                )[col].shift(lag)

        # Return the DataFrame with new columns. The MultiIndex is preserved.
        return X_


# --- 2. FINAL PREPROCESSOR (ColumnTransformer) ---
# ✅ FIX 1: Replaced with data-agnostic version using make_column_selector
def create_final_preprocessor() -> ColumnTransformer:
    """
    Creates the final ColumnTransformer, placed at the end of feature engineering.
    It infers dtypes at fit-time.
    """
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preproc = ColumnTransformer(
        [
            ("num", num_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipe, make_column_selector(dtype_include=object)),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # ✅ FIX 5: Ensure pandas output for compatibility
    try:
        preproc.set_output(transform="pandas")
    except Exception:
        pass  # Fails silently on older sklearn
    return preproc


# --- 1b. Custom Feature Engineering (Non-Linear Features) ---
# ✅ FIX 6: Made robust to missing columns
def create_non_linear_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Creates non-linear interaction and threshold features.
    It safely skips interactions if required columns are missing.
    Returns a DataFrame.
    """
    X_df = X.copy()

    # --- Feature 1: Precipitation and Antecedent Soil Moisture Interaction (Lagged) ---
    if "precipitations" in X_df.columns and "soil_moisture_lag_1" in X_df.columns:
        X_df["rain_soil_interaction"] = (
            X_df["precipitations"] * X_df["soil_moisture_lag_1"]
        )

    # --- Feature 2: Temperature and Altitude Interaction ---
    if "temperatures" in X_df.columns and "altitude" in X_df.columns:
        X_df["temp_altitude_interaction"] = X_df["temperatures"] * X_df["altitude"]

    # --- Feature 3: Thawing Cycle Binary Flag ---
    if "temperatures" in X_df.columns and "temperatures_lag_1" in X_df.columns:
        X_df["is_thawing_binary"] = np.where(
            (X_df["temperatures"] > 0) & (X_df["temperatures_lag_1"] <= 0), 1, 0
        )

    # --- 2. MultiScale Soil and Weather Interactions ---
    # granularity = "region"  # other "zone", "sub_sector"
    for granularity in ["region", "zone", "subsector", "sector"]:
        clay_col = f"clay_0-5cm_mean_index__{granularity}"
        sand_col = f"sand_0-5cm_mean_index__{granularity}"
        cfvo_col = f"cfvo_0-5cm_mean_index__{granularity}"
        bdod_col = f"bdod_0-5cm_mean_index__{granularity}"

        if "precipitations" in X_df.columns:
            if clay_col in X_df.columns:
                X_df[f"precip_x_clay__index_{granularity}"] = (
                    X_df["precipitations"] * X_df[clay_col]
                )
            if sand_col in X_df.columns:
                X_df[f"precip_x_sand__index_{granularity}"] = (
                    X_df["precipitations"] * X_df[sand_col]
                )

        if "evaporation" in X_df.columns:
            if cfvo_col in X_df.columns:
                X_df[f"evap_x_coarse_frag__index_{granularity}"] = (
                    X_df["evaporation"] * X_df[cfvo_col]
                )
            if bdod_col in X_df.columns:
                X_df[f"evap_x_bulk_density__index_{granularity}"] = (
                    X_df["evaporation"] * X_df[bdod_col]
                )

    # --- 3. Geographical Interactions ---
    if "temperatures" in X_df.columns and "altitude" in X_df.columns:
        X_df["temp_x_altitude"] = X_df["temperatures"] * X_df["altitude"]
    if "precipitations" in X_df.columns and "catchment" in X_df.columns:
        X_df["precip_x_catchment"] = X_df["precipitations"] * X_df["catchment"]

    return X_df


# Define log and inverse-log functions
def log_transform(y):
    return np.log1p(y)  # log(1 + y) to handle zeros


def inverse_log_transform(y):
    return np.expm1(y)  # exp(y) - 1 to reverse log(1 + y)


# --- 3. Model Creation (XGBoost wrapped in MAPIE) ---
def create_xgboost_model() -> MapieQuantileRegressor:
    """Creates an XGBoost model wrapped in MAPIE for prediction intervals."""
    xgb_model = XGBRegressor(
        random_state=42,
        n_estimators=100,
        n_jobs=1,  # ✅ FIX 11: Set to 1 to avoid nested parallelism issues
        objective="reg:squarederror",
        tree_method="hist",
    )
    # MAPIE is used for uncertainty quantification/conformal prediction
    # mapie_model = MapieRegressor(estimator=xgb_model)

    # mapie_model = MapiePointWrapper(mapie_model)  # ✅ FIX 3: Wrap for CV compatibility
    return xgb_model


def create_quantile_xgboost_model() -> MapieQuantileRegressor:
    """Creates an XGBoost model wrapped in MAPIE Quantile Regressor for quantile predictions."""
    xgb_model = XGBRegressor(
        random_state=42,
        n_estimators=100,
        n_jobs=1,  # ✅ FIX 11: Set to 1 to avoid nested parallelism issues
        objective="reg:squarederror",
        tree_method="hist",
    )
    # MAPIE Quantile Regressor for quantile predictions
    mapie_quantile_model = MapieQuantileRegressor(estimator=xgb_model)

    mapie_quantile_model = MapiePointWrapper(
        mapie_quantile_model
    )  # ✅ FIX 3: Wrap for CV compatibility
    return mapie_quantile_model


# ✅ FIX 3: Add MapiePointWrapper to fix CV scoring
class MapiePointWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps MapieRegressor to ensure .predict() returns a 1D array
    of point predictions, compatible with scikit-learn scorers.
    """

    def __init__(self, mapie):
        self.mapie = mapie

    def fit(self, X, y):
        self.mapie.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        out = self.mapie.predict(X, **kwargs)
        # Return only point predictions if tuple (y_pred, intervals)
        return out[0] if isinstance(out, tuple) else out

    def __getattr__(self, name):
        # Delegate other attributes (e.g., predict_interval) if needed
        return getattr(self.mapie, name)


# --- 4. FULL PIPELINE CREATOR (SEQUENTIAL) ---
# ✅ FIX 1: Removed `X` argument as it's no longer needed
from functools import partial


class DebugDataSaver(BaseEstimator, TransformerMixin):
    """
    Saves the final processed features (entire X) to a CSV file on the first
    call during the pipeline fit, then operates as a pass-through.
    """

    def __init__(
        self,
        output_dir: str,
        file_name: str = "final_features.csv",
    ):
        self.output_dir = output_dir
        self.file_name = file_name
        self._has_logged = False  # Internal flag to ensure single logging

    def fit(self, X, y=None):
        if not self._has_logged:
            self._save_data(X)
            self._has_logged = True
        return self

    def transform(self, X):
        # Pass-through with negligible overhead
        return X

    def _save_data(self, X: Union[pd.DataFrame, np.ndarray], num_samples: int = 5):
        """Internal method to save X as a CSV file."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.file_name)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            print(f"\n✨ [DEBUG SUCCESS]: Processed features saved to: {output_path}")
        else:
            # Convert NumPy array to DataFrame with generic column names
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

            print(
                f"\n[DEBUG WARNING]: Input was a NumPy array. "
                f"Saved as CSV with generic column names to: {output_path}"
            )

        X_df.sample(num_samples).to_csv(output_path)

        # save columns names
        cols_output_path = os.path.join(self.output_dir, "final_feature_names.json")
        with open(cols_output_path, "w") as f:
            json.dump(X_df.columns.tolist(), f, indent=4)


def create_ts_pipeline(model, run_dir):
    """
    Creates the full sequential pipeline with feature engineering, final preprocessing,
    and the regressor model.
    """
    # 2. Define the sequential pipeline flow
    base_pipeline = Pipeline(
        steps=[
            # Step 1: Create Time Features (DF -> DF)
            (
                "datetime_features",
                FunctionTransformer(
                    partial(
                        create_datetime_features_multiindex,
                        datetime_index_level="ObsDate",
                    ),
                    validate=False,
                ),
            ),
            # Step 3: Create Non-Linear Features (DF -> DF) - Uses outputs of Step 1 & 2
            (
                "nonlinear_features",
                FunctionTransformer(create_non_linear_features, validate=False),
            ),
            # Step 4: Drop Unnecessary Features (DF -> DF)
            (
                "feature_dropping",
                FunctionTransformer(
                    drop_feature, kw_args={"features": ["station_code"]}
                ),
            ),
            # Step 4: Final Preprocessing (DF -> DF or NumPy array for model)
            # This call is now correct as create_final_preprocessor takes no args
            ("final_preprocessor", create_final_preprocessor()),
            # Debug Step: Save final feature names
            (
                "debugger",
                DebugDataSaver(output_dir=run_dir),
            ),
            # Step 5: Regressor Model
            # ✅ FIX 3: Wrap model in MapiePointWrapper for CV
            # ("model", MapiePointWrapper(model)),
            ("model", model),
        ]
    )

    # ✅ FIX 5: Set pandas output on pipeline
    try:
        base_pipeline.set_output(transform="pandas")
    except Exception:
        pass  # Fails silently on older sklearn

    # Wrap the pipeline with log transform for the target
    print("Creating TransformedTargetRegressor with log transformation.")
    ts_pipeline = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=log_transform,
        inverse_func=inverse_log_transform,
    )
    return ts_pipeline


def get_pipeline_params(pipeline):
    params = pipeline.get_params()
    # Filter out internal scikit-learn parameters (e.g., those starting with steps__)
    return {
        k: v for k, v in params.items() if not k.startswith("steps__") and "__" in k
    }


def save_experiment_params(experiment_params, output_dir="experiment"):
    # Create the experiment folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save to a JSON file
    output_path = os.path.join(output_dir, "experiment_params.json")
    with open(output_path, "w") as f:
        json.dump(experiment_params, f, indent=4)
    print(f"\nExperiment results saved to {output_path}")


# Helper to predict and calculate RMSE, handling potential MAPIE tuple output
def evaluate_rmse(model, X_val, y_val):
    y_pred = model.predict(X_val)

    # Handle MAPIE's potential tuple output (prediction, intervals)
    # Note: Our wrapper handles this for CV, but final model might be unwrapped
    # or predict() might be called directly. This check is still good.
    y_pred_point = y_pred[0] if isinstance(y_pred, tuple) else y_pred

    # Ensure y_val and y_pred_point align
    # Convert y_pred_point to a pandas Series with X_val's index
    y_pred_series = pd.Series(y_pred_point, index=X_val.index)

    common_index = y_val.index.intersection(y_pred_series.index)

    if len(common_index) < len(y_val):
        print(
            f"Warning: Evaluating on {len(common_index)} / {len(y_val)} samples due to index mismatch after prediction."
        )
    if len(common_index) == 0:
        print(
            "Error: No common indices between y_val and predictions. Cannot calculate RMSE."
        )
        return np.nan

    rmse = mean_squared_error(
        y_val.loc[common_index],
        y_pred_series.loc[common_index],
        squared=False,
    )
    return rmse, len(common_index)


def save_date_station_code(
    X: pd.DataFrame, output_path: Union[str, Path] = "experiment"
):
    # check index levels names
    if X.index.names != ["ObsDate", "station_code"]:
        raise ValueError(
            f"Expected index levels ['ObsDate', 'station_code'], but got {X.index.names}"
        )  # Extract index levels
    df_index = X.index.to_frame(index=False)[["ObsDate", "station_code"]]
    df_index.to_csv(output_path, index=False)
    print(f"Saved date and station_code to {output_path}")


def run_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val_st: pd.DataFrame,
    y_val_st: pd.Series,
    X_val_t: pd.DataFrame,
    y_val_t: pd.Series,
    pipeline: Pipeline,  # This is the full pipeline, likely including TransformedTargetRegressor
    n_splits: int = 5,
    n_iter: int = 25,  # Number of iterations for RandomizedSearchCV
    run_dir: Union[str, Path] = "experiment",
    extra_infos: Dict = {},
):
    """
    Performs RandomizedSearchCV HP tuning, final training, and evaluation with logging.
    Follows scikit-learn best practices.
    """
    print(f"\n--- Running Experiment: {run_dir} ---")
    run_dir = Path(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    print("--- Starting Hyperparameter Tuning (RandomizedSearchCV) ---")

    # 1. Define Time Series Split (CV)
    tscv = TimeSeriesSplit(
        n_splits=n_splits, gap=1
    )  # Gap prevents adjacent fold leakage

    # ✅ FIX 9: Use built-in neg_root_mean_squared_error
    scoring = "neg_root_mean_squared_error"

    param_dist = {
        "regressor__model__estimator__n_estimators": [50],
        "regressor__model__estimator__learning_rate": [0.05],
        "regressor__model__estimator__max_depth": [3, 4, 5],
        "regressor__model__estimator__subsample": [
            0.7,
        ],
        "regressor__model__estimator__colsample_bytree": [0.7],
        "regressor__model__estimator__reg_lambda": [
            0.5,
            1,
        ],  # L2 regularization
        "regressor__model__estimator__min_child_weight": [1, 3],
    }

    search_cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=tscv,
        n_jobs=-1,  # n_jobs on Search CV is OK
        random_state=42,
        verbose=2,
    )

    # 3. Perform Hyperparameter Search on training data
    start_time = time.time()

    # ✅ FIX 2: We no longer need to trim X/y. NaNs from lagging
    # will be created and then imputed by the pipeline's SimpleImputer.
    try:
        search_cv.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during RandomizedSearchCV fit: {e}")
        print(
            "Please check parameter grid prefixes against pipeline.get_params().keys()"
        )
        print("Available params snippet:")
        print([k for k in pipeline.get_params().keys() if "estimator" in k])
        return None, None

    cv_time = time.time() - start_time
    print(f"Randomized Search CV complete in {cv_time:.2f} seconds.")

    # Print statements are correct since score is negative
    print(f"Best CV RMSE: {-search_cv.best_score_:.4f}")
    print("Best Hyperparameters found:")
    print(json.dumps(search_cv.best_params_, indent=2))

    best_pipeline = search_cv.best_estimator_
    cv_scores = search_cv.cv_results_["mean_test_score"]  # These are negative RMSE
    avg_cv_rmse = -np.mean(cv_scores)

    print(
        f"\nCV Evaluation Summary. Avg RMSE: {avg_cv_rmse:.4f} across {n_splits} folds."
    )
    print(f"CV RMSE scores for each fold: {-cv_scores}")  # Negate to show positive RMSE

    # 4. Final Training
    print("\nRefitting the best pipeline on the complete training set (X_train)...")
    start_time = time.time()
    # RandomizedSearchCV with refit=True (default) already retrains
    # the best_estimator_ on the *full* dataset (X_train, y_train).
    final_pipeline = best_pipeline
    training_time = time.time() - start_time
    print(f"Final model fitting complete (time: {training_time:.2f} seconds).")

    # 5. Save the Final Trained Pipeline
    model_path = run_dir / "trained_pipeline_best.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"Final trained pipeline saved to {model_path}")

    # 6. Evaluation on Hold-Out Validation Sets
    print("\n--- Evaluating Final Model on Hold-Out Sets ---")
    rmse_st, n_st = evaluate_rmse(final_pipeline, X_val_st, y_val_st)
    rmse_t, n_t = evaluate_rmse(final_pipeline, X_val_t, y_val_t)

    print("\n--- Validation Results ---")
    print(
        f"1. Spatio-Temporal RMSE (New Space/Time): {rmse_st:.4f} (on {n_st} samples)"
    )
    print(f"2. Temporal RMSE (Known Space/New Time): {rmse_t:.4f} (on {n_t} samples)")

    experiment_params = {
        "best_hyperparameters": search_cv.best_params_,
        "cv_n_splits": n_splits,
        "cv_avg_rmse": avg_cv_rmse,
        "val_spatio_temporal_rmse": rmse_st,
        "val_temporal_rmse": rmse_t,
        "final_training_time_s": training_time,
        "cv_tuning_time_s": cv_time,
        "input_feature_count": X_train.shape[1],
    }
    all_params = {**experiment_params, **extra_infos}

    # Save parameters
    output_path = run_dir / "experiment_params.json"
    with open(output_path, "w") as f:
        json.dump(all_params, f, indent=4, default=str)
    print(f"\nExperiment results saved to {output_path}")

    return final_pipeline, experiment_params


import os
import joblib
import json
from pathlib import Path
from typing import Union, Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from mapie.metrics import regression_coverage_score, regression_mean_width_score


# Note: The original function signature had arguments (n_splits, n_iter)
# that are no longer used. I've removed them for clarity.
def run_experiment_no_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val_st: pd.DataFrame,
    y_val_st: pd.Series,
    X_val_t: pd.DataFrame,
    y_val_t: pd.Series,
    pipeline: Pipeline,  # This is the full pipeline, likely including TransformedTargetRegressor
    run_dir: Union[str, Path] = "experiment",
    extra_infos: Dict = {},
):
    """
    Performs a single model training and evaluation with logging.
    No hyperparameter tuning is performed.
    """
    print(f"\n--- Running Experiment: {run_dir} ---")
    run_dir = Path(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    print("--- Starting Model Training ---")

    # 1. Fit the pipeline
    # This will train the base estimator(s) and compute conformity scores
    # using the CV method defined inside your MapieRegressor (e.g., cv=5)
    pipeline.fit(X_train, y_train)
    print("--- Model Training Complete ---")

    # 2. Save the fitted pipeline
    model_path = run_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Fitted pipeline saved to {model_path}")

    # 3. Evaluate on the "st" validation set
    print(f"--- Evaluating on Validation Set (st) ---")

    # Define the desired coverage level
    # For 5% and 95% intervals, you want 90% coverage, so alpha = 0.10
    ALPHA = 0.10

    # Get point predictions and prediction intervals
    y_pred_st, y_pis_st = pipeline.predict(X_val_st, alpha=ALPHA)

    # y_pis_st shape is (n_samples, 1, 2) for one alpha
    # [:, 0, 0] is the lower bound (5% quantile)
    # [:, 0, 1] is the upper bound (95% quantile)
    y_lower_st = y_pis_st[:, 0, 0]
    y_upper_st = y_pis_st[:, 0, 1]

    # Calculate metrics
    rmse_st = root_mean_squared_error(y_val_st, y_pred_st)
    coverage_st = regression_coverage_score(y_val_st, y_lower_st, y_upper_st)
    width_st = regression_mean_width_score(y_lower_st, y_upper_st)

    print(f"Validation (st) RMSE: {rmse_st:.4f}")
    print(f"Validation (st) Target Coverage: {1-ALPHA:.2f}")
    print(f"Validation (st) Effective Coverage: {coverage_st:.4f}")
    print(f"Validation (st) Mean Interval Width: {width_st:.4f}")

    # 4. Evaluate on the "t" validation set (assuming this is a test set)
    print(f"\n--- Evaluating on Validation Set (t) ---")
    y_pred_t, y_pis_t = pipeline.predict(X_val_t, alpha=ALPHA)
    y_lower_t = y_pis_t[:, 0, 0]
    y_upper_t = y_pis_t[:, 0, 1]

    rmse_t = root_mean_squared_error(y_val_t, y_pred_t)
    coverage_t = regression_coverage_score(y_val_t, y_lower_t, y_upper_t)
    width_t = regression_mean_width_score(y_lower_t, y_upper_t)

    print(f"Validation (t) RMSE: {rmse_t:.4f}")
    print(f"Validation (t) Target Coverage: {1-ALPHA:.2f}")
    print(f"Validation (t) Effective Coverage: {coverage_t:.4f}")
    print(f"Validation (t) Mean Interval Width: {width_t:.4f}")

    # 5. Log results
    results = {
        "validation_st": {
            "rmse": rmse_st,
            "target_coverage": 1 - ALPHA,
            "effective_coverage": coverage_st,
            "mean_interval_width": width_st,
        },
        "validation_t": {
            "rmse": rmse_t,
            "target_coverage": 1 - ALPHA,
            "effective_coverage": coverage_t,
            "mean_interval_width": width_t,
        },
        "extra_infos": extra_infos,
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")
    print(f"--- Experiment {run_dir} Finished ---")

    return results


import os
import joblib
import json
from pathlib import Path
from typing import Union, Dict
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error
from mapie.metrics import regression_coverage_score, regression_mean_width_score

# Make sure to import these
from sklearn.compose import TransformedTargetRegressor
from mapie.regression import MapieRegressor


def run_experiment_no_cv2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val_st: pd.DataFrame,
    y_val_st: pd.Series,
    X_val_t: pd.DataFrame,
    y_val_t: pd.Series,
    pipeline: Pipeline,  # Your full pipeline, e.g., Pipeline([('features', ...), ('regressor', TTR(...))])
    run_dir: Union[str, Path] = "experiment",
    extra_infos: Dict = {},
):
    """
    Performs a single model training and evaluation with logging.
    Handles the incompatibility between TransformedTargetRegressor and MapieRegressor.
    """
    print(f"\n--- Running Experiment: {run_dir} ---")
    run_dir = Path(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    print("--- Starting Model Training ---")

    # 1. Fit the pipeline (this is unchanged)
    pipeline.fit(X_train, y_train)
    print("--- Model Training Complete ---")

    # 2. Save the fitted pipeline
    model_path = run_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Fitted pipeline saved to {model_path}")

    # 3. Define Alpha
    # For 5% and 95% intervals, you want 90% coverage, so alpha = 0.10
    ALPHA = 0.10

    # --- Helper function to get predictions and intervals ---
    # This is needed to handle the TTR/Mapie incompatibility
    def get_preds_and_intervals(X_data, fitted_pipeline):
        # 3a. Get point predictions (this works fine)
        # TTR's predict() calls Mapie's predict(alpha=None), which returns one array,
        # which TTR then inverse-transforms.
        print("Getting point predictions...")
        y_pred = fitted_pipeline.predict(X_data)

        # 3b. Manually get intervals
        print("Manually extracting intervals...")

        # This assumes your TTR is the last step, named 'regressor'
        # If your TTR is named differently, change 'regressor'
        ttr_step = fitted_pipeline.named_steps["regressor"]
        if not isinstance(ttr_step, TransformedTargetRegressor):
            raise ValueError(
                "The last step of the pipeline is not a TransformedTargetRegressor. Adjust logic."
            )

        # Get the fitted MapieRegressor and the target-transformer
        mapie_model = ttr_step.regressor_
        transformer = ttr_step.transformer_

        # Get all preceding steps to transform X
        # This creates a temporary pipeline of all steps *except* the TTR
        if len(fitted_pipeline.steps) > 1:
            preprocessor = Pipeline(fitted_pipeline.steps[:-1])
            X_data_transformed = preprocessor.transform(X_data)
        else:
            # No preprocessing steps
            X_data_transformed = X_data

        # 3c. Predict on *transformed* data with Mapie
        # This gives predictions and intervals *in the transformed space*
        y_pred_trans, y_pis_trans = mapie_model.predict(X_data_transformed, alpha=ALPHA)

        # 3d. Inverse-transform the intervals
        # y_pis_trans shape is (n_samples, 1, 2)
        lower_bound_trans = y_pis_trans[:, 0, 0].reshape(-1, 1)
        upper_bound_trans = y_pis_trans[:, 0, 1].reshape(-1, 1)

        y_lower = transformer.inverse_transform(lower_bound_trans).ravel()
        y_upper = transformer.inverse_transform(upper_bound_trans).ravel()

        return y_pred, y_lower, y_upper

    # 4. Evaluate on the "st" validation set
    print(f"--- Evaluating on Validation Set (st) ---")
    y_pred_st, y_lower_st, y_upper_st = get_preds_and_intervals(X_val_st, pipeline)

    # Calculate metrics
    rmse_st = root_mean_squared_error(y_val_st, y_pred_st)
    coverage_st = regression_coverage_score(y_val_st, y_lower_st, y_upper_st)
    width_st = regression_mean_width_score(y_lower_st, y_upper_st)

    print(f"Validation (st) RMSE: {rmse_st:.4f}")
    print(f"Validation (st) Target Coverage: {1-ALPHA:.2f}")
    print(f"Validation (st) Effective Coverage: {coverage_st:.4f}")
    print(f"Validation (st) Mean Interval Width: {width_st:.4f}")

    # 5. Evaluate on the "t" validation set
    print(f"\n--- Evaluating on Validation Set (t) ---")
    y_pred_t, y_lower_t, y_upper_t = get_preds_and_intervals(X_val_t, pipeline)

    rmse_t = root_mean_squared_error(y_val_t, y_pred_t)
    coverage_t = regression_coverage_score(y_val_t, y_lower_t, y_upper_t)
    width_t = regression_mean_width_score(y_lower_t, y_upper_t)

    print(f"Validation (t) RMSE: {rmse_t:.4f}")
    print(f"Validation (t) Target Coverage: {1-ALPHA:.2f}")
    print(f"Validation (t) Effective Coverage: {coverage_t:.4f}")
    print(f"Validation (t) Mean Interval Width: {width_t:.4f}")

    # 6. Log results
    results = {
        "validation_st": {
            "rmse": rmse_st,
            "target_coverage": 1 - ALPHA,
            "effective_coverage": coverage_st,
            "mean_interval_width": width_st,
        },
        "validation_t": {
            "rmse": rmse_t,
            "target_coverage": 1 - ALPHA,
            "effective_coverage": coverage_t,
            "mean_interval_width": width_t,
        },
        "extra_infos": extra_infos,
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")
    print(f"--- Experiment {run_dir} Finished ---")

    return results


def split_dataset_custom_multiindex(
    ds: pd.DataFrame,
    p: float = 0.75,
    time: str = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, spatio-temporal test, and temporal test sets.
    Requires a MultiIndex with levels named ("station_code", "ObsDate").

    Args:
        ds (pd.DataFrame): Input dataset with a MultiIndex: ("station_code", "ObsDate").
                           'ObsDate' must be datetime-like.
        p (float): Proportion of stations to use for training (default: 0.75).
        time (str): Timestamp for temporal split (e.g., "2003-01-01").
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: (train, spatio_temporal_test, temporal_test)
    """
    # ⚠️ Assumes ds has a MultiIndex with levels ("station_code", "ObsDate")
    if not isinstance(ds.index, pd.MultiIndex):
        raise TypeError("Data must have a MultiIndex.")
    if list(ds.index.names) != ["ObsDate", "station_code"]:
        raise ValueError("MultiIndex levels must be named ('ObsDate', 'station_code').")

    # Get unique station codes from the first level of the MultiIndex
    station_codes = ds.index.get_level_values("station_code").unique().tolist()

    random.seed(seed)
    random.shuffle(station_codes)

    # Split stations into train and test sets (Spatial Split)
    split_idx = int(len(station_codes) * p)
    train_stations = set(station_codes[:split_idx])
    test_stations = set(station_codes[split_idx:])

    # Convert the time string to a pandas datetime object
    split_time = pd.to_datetime(time)

    # Filtering is done using the MultiIndex levels

    # Train set: Data for train_stations AND ObsDate < split_time
    train = ds.loc[
        (ds.index.get_level_values("station_code").isin(train_stations))
        & (ds.index.get_level_values("ObsDate") < split_time)
    ]

    # Spatio-temporal test: Data for test_stations AND ObsDate >= split_time
    spatio_temporal_test = ds.loc[
        (ds.index.get_level_values("station_code").isin(test_stations))
        & (ds.index.get_level_values("ObsDate") >= split_time)
    ]

    # Temporal test: Data for train_stations AND ObsDate >= split_time
    temporal_test = ds.loc[
        (ds.index.get_level_values("station_code").isin(train_stations))
        & (ds.index.get_level_values("ObsDate") >= split_time)
    ]

    # Print diagnostics
    print(
        f"Train stations: {len(train_stations)} | Test stations: {len(test_stations)}"
    )
    print(f"Train data shape: {train.shape}")
    print(f"Spatio-temporal test shape: {spatio_temporal_test.shape}")
    print(f"Temporal test shape: {temporal_test.shape}")

    return train, spatio_temporal_test, temporal_test


def split_dataset_custom(
    ds: pd.DataFrame,
    p: float = 0.75,
    time: str = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- Validation ---
    required_cols = {"station_code", "ObsDate"}
    if not required_cols.issubset(ds.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Ensure ObsDate is datetime
    ds = ds.copy()
    ds["ObsDate"] = pd.to_datetime(ds["ObsDate"])

    # --- Prepare station splits (spatial split) ---
    station_codes = ds["station_code"].unique().tolist()
    random.seed(seed)
    random.shuffle(station_codes)

    split_idx = int(len(station_codes) * p)
    train_stations = set(station_codes[:split_idx])
    test_stations = set(station_codes[split_idx:])

    # --- Temporal split ---
    if time is None:
        raise ValueError("You must provide a 'time' argument (e.g., '2003-01-01').")
    split_time = pd.to_datetime(time)

    # --- Create subsets ---
    train = ds[(ds["station_code"].isin(train_stations)) & (ds["ObsDate"] < split_time)]

    spatio_temporal_test = ds[
        (ds["station_code"].isin(test_stations)) & (ds["ObsDate"] >= split_time)
    ]

    temporal_test = ds[
        (ds["station_code"].isin(train_stations)) & (ds["ObsDate"] >= split_time)
    ]

    # --- Diagnostics ---
    print("📊 Dataset split summary")
    print(
        f"  • Train stations: {len(train_stations)} | Test stations: {len(test_stations)}"
    )
    print(f"  • Train data shape: {train.shape}")
    print(f"  • Spatio-temporal test shape: {spatio_temporal_test.shape}")
    print(f"  • Temporal test shape: {temporal_test.shape}")

    return train, spatio_temporal_test, temporal_test
