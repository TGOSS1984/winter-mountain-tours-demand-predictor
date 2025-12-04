# src/data_loaders.py

from pathlib import Path
from functools import lru_cache
from typing import Any  # NEW

import pandas as pd
import joblib

# NEW: imports to help patch XGBoost models inside sklearn Pipelines
from xgboost import XGBModel  # NEW
from sklearn.pipeline import Pipeline  # NEW


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_PROCESSED = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"


# NEW: helper to patch XGBoost models so version-mismatch attrs exist
def _patch_xgboost(obj: Any) -> Any:
    """
    Patch XGBoost estimators (optionally inside a sklearn Pipeline) so that
    attributes expected by some XGBoost versions ('gpu_id', 'predictor',
    'use_label_encoder') are always present.

    This avoids AttributeError when calling get_params()/predict()
    on models that were trained under a slightly different XGBoost version.
    """
    # Case 1: plain XGBModel
    if isinstance(obj, XGBModel):
        for attr in ["gpu_id", "predictor", "use_label_encoder"]:
            if not hasattr(obj, attr):
                setattr(obj, attr, None)
        return obj

    # Case 2: sklearn Pipeline containing an XGBModel
    if isinstance(obj, Pipeline):
        for name, step in obj.steps:
            if isinstance(step, XGBModel):
                for attr in ["gpu_id", "predictor", "use_label_encoder"]:
                    if not hasattr(step, attr):
                        setattr(step, attr, None)
        return obj

    # Anything else: return unchanged
    return obj


@lru_cache(maxsize=1)
def load_weekly_regression_data() -> pd.DataFrame:
    """
    Load the weekly region-level dataset used for regression modelling.
    This is useful for the EDA page and for checking forecast context.
    """
    path = DATA_PROCESSED / "weekly_bookings_regression.csv"
    return pd.read_csv(path, parse_dates=["week_start"])


@lru_cache(maxsize=1)
def load_bookings_classification_data() -> pd.DataFrame:
    """
    Load the booking-level dataset used for cancellation classification.
    This is useful for summary stats and example bookings in the dashboard.
    """
    path = DATA_PROCESSED / "bookings_for_classification.csv"
    return pd.read_csv(
        path,
        parse_dates=["tour_date", "booking_date", "week_start"],
    )


@lru_cache(maxsize=1)
def load_regression_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test splits for regression. Useful if we want to display
    metrics or actual vs predicted plots in the dashboard without
    re-running notebooks.
    """
    train_path = DATA_PROCESSED / "train_regression.csv"
    test_path = DATA_PROCESSED / "test_regression.csv"
    train = pd.read_csv(train_path, parse_dates=["week_start"])
    test = pd.read_csv(test_path, parse_dates=["week_start"])
    return train, test


@lru_cache(maxsize=1)
def load_classification_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test splits for classification.
    """
    train_path = DATA_PROCESSED / "train_classification.csv"
    test_path = DATA_PROCESSED / "test_classification.csv"
    train = pd.read_csv(
        train_path,
        parse_dates=["tour_date", "booking_date", "week_start"],
    )
    test = pd.read_csv(
        test_path,
        parse_dates=["tour_date", "booking_date", "week_start"],
    )
    return train, test


@lru_cache(maxsize=1)
def load_bookings_model():
    """
    Load the trained regression model for weekly bookings.

    Model file is produced in 03_model_regression.ipynb.
    """
    model_path = MODELS_DIR / "v1_bookings_model.pkl"
    model = joblib.load(model_path)
    # PATCH: ensure XGBoost inside this pipeline has expected attributes
    model = _patch_xgboost(model)
    return model


@lru_cache(maxsize=1)
def load_cancellation_model():
    """
    Load the trained classification model for cancellation risk.

    Model file is produced in 04_model_classification.ipynb.
    """
    model_path = MODELS_DIR / "v1_cancel_model.pkl"
    model = joblib.load(model_path)
    # PATCH: ensure XGBoost inside this pipeline has expected attributes
    model = _patch_xgboost(model)
    return model


def load_model_metrics(filename: str) -> dict:
    """
    Load a JSON metrics summary created by the modelling notebooks.
    For example:
    - v1_bookings_model_metrics.json
    - v1_cancel_model_metrics.json
    """
    import json

    path = MODELS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
