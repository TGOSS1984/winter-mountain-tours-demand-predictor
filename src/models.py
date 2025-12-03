# src/models.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from . import data_loaders
from . import features


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

    # Wrap single dict into a DataFrame with one row
    X = pd.DataFrame([booking_features])[features.CLASSIFICATION_FEATURE_COLUMNS]

    proba = model.predict_proba(X)[:, 1][0]
    is_high_risk = bool(proba >= threshold)

    return {
        "probability": float(proba),
        "is_high_risk": is_high_risk,
        "threshold": float(threshold),
    }
