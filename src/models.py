# src/models.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBModel

from . import data_loaders
from . import features

from pathlib import Path
from functools import lru_cache

from src.image_features import extract_weather_features_from_bytes

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
WEATHER_MODEL_PATH = MODELS_DIR / "weather_severity_model.pkl"


def _ensure_xgb_gpu_id(model):
    """
    Heroku CPU builds of xgboost still expect a `gpu_id` attribute on the
    underlying XGBModel in some code paths. Patch it defensively.
    """
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
    else:
        est = model

    if isinstance(est, XGBModel) and not hasattr(est, "gpu_id"):
        est.gpu_id = -1

    return model

@lru_cache(maxsize=1)
def load_weather_severity_model():
   
    if not WEATHER_MODEL_PATH.exists():
        # Fail gracefully instead of crashing the whole app
        return None

    model = joblib.load(WEATHER_MODEL_PATH)

    # Optional defensive check: make sure has predict()
    if not hasattr(model, "predict"):
        raise TypeError(
            f"Loaded weather model from {WEATHER_MODEL_PATH} "
            "does not implement predict()."
        )

    return model

def predict_weather_severity_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Predict weather severity (e.g. 'mild', 'moderate', 'severe') from raw image bytes.

    """
    model = load_weather_severity_model()
    if model is None:
        raise RuntimeError(
            "Weather severity model not found. Please ensure "
            "models/weather_severity_model.pkl exists."
        )

    # 1) Extract features from image
    feats = extract_weather_features_from_bytes(image_bytes)

    # 2) Build single-row DataFrame
    df = pd.DataFrame([feats])

    # 3) Arrange columns in the order the model was trained on (prevents silent bugs)
    if hasattr(model, "feature_names_in_"):
        df = df.filter(model.feature_names_in_)

    # 4) Predict label
    label = model.predict(df)[0]

    # 5) Predict probabilities (if supported)
    proba_dict: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            proba_dict = {
                str(cls): float(p) for cls, p in zip(classes, proba)
            }

    result: Dict[str, Any] = {
        "label": str(label),
        "proba": proba_dict,
        "features": feats,
    }

    return result



def forecast_weekly_bookings(
    region: str,
    week_start: pd.Timestamp,
    scenario: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Use the trained regression model to forecast weekly bookings for a given
    region and week_start.

    Parameters
    ----------
    region : str
        Region name (must exist in the training data, e.g. "lake_district").
    week_start : pd.Timestamp
        Monday of the week to forecast. In this prototype we expect it to
        match a week present in the processed dataset (or a simple "future"
        copy of a known week).
    scenario : dict, optional
        Optional "what-if" overrides, e.g. {
            "is_bank_holiday_week": 1,
            "weather_severity_bin": "severe",
        }

    Returns
    -------
    dict with keys:
        - "prediction": float
        - "inputs": dict of final feature values used
    """
    model = data_loaders.load_bookings_model()
    model = _ensure_xgb_gpu_id(model)

    weekly_df = data_loaders.load_weekly_regression_data()

    row = features.build_regression_feature_row(
        weekly_df=weekly_df,
        region=region,
        week_start=week_start,
        scenario=scenario or {},
    )

    # Model expects a DataFrame with one row
    X = row[features.REGRESSION_FEATURE_COLUMNS]
    y_pred = model.predict(X)[0]

    result = {
        "prediction": float(y_pred),
        "inputs": row.to_dict(orient="records")[0],
    }
    return result


def predict_cancellation_risk(
    booking_features: Dict[str, Any], threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Use the trained classification model to predict the cancellation risk
    for a single booking defined by feature values.

    Parameters
    ----------
    booking_features : dict
        Dictionary with keys matching the classification feature columns
        defined in features.CLASSIFICATION_FEATURE_COLUMNS, e.g.
        {
            "region": "lake_district",
            "route_difficulty": "moderate",
            "weather_severity_bin": "severe",
            "party_size": 3,
            "lead_time_days": 21,
            "year": 2025,
            "week_number": 5,
            "month": 2,
            "is_bank_holiday_week": 0,
            "is_peak_winter": 1,
        }
    threshold : float
        Probability threshold above which a booking is flagged as high risk.

    Returns
    -------
    dict with keys:
        - "probability": float
        - "is_high_risk": bool
        - "threshold": float
    """
    model = data_loaders.load_cancellation_model()
    model = _ensure_xgb_gpu_id(model)

    # Wrap single dict into a DataFrame with one row
    X = pd.DataFrame([booking_features])[features.CLASSIFICATION_FEATURE_COLUMNS]

    proba = model.predict_proba(X)[:, 1][0]
    is_high_risk = bool(proba >= threshold)

    return {
        "probability": float(proba),
        "is_high_risk": is_high_risk,
        "threshold": float(threshold),
    }

