# src/features.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd


# Columns expected by the regression model (must match notebook training)
REGRESSION_CATEGORICAL = ["region", "weather_severity_bin"]
REGRESSION_NUMERIC = [
    "mean_temp_c",
    "precip_mm",
    "snowfall_flag",
    "wind_speed_kph",
    "visibility_km",
    "year",
    "week_number",
    "month",
    "is_bank_holiday_week",
    "is_peak_winter",
    "lag_1w_bookings",
    "lag_4w_mean",
    "lag_52w_bookings",
]

REGRESSION_FEATURE_COLUMNS = REGRESSION_CATEGORICAL + REGRESSION_NUMERIC

# Columns expected by the classification model (must match notebook training)
CLASSIFICATION_CATEGORICAL = [
    "region",
    "route_difficulty",
    "weather_severity_bin",
]

CLASSIFICATION_NUMERIC = [
    "party_size",
    "lead_time_days",
    "year",
    "week_number",
    "month",
    "is_bank_holiday_week",
    "is_peak_winter",
]

CLASSIFICATION_FEATURE_COLUMNS = CLASSIFICATION_CATEGORICAL + CLASSIFICATION_NUMERIC


@dataclass
class ForecastScenario:
    """
    Optional overrides for regression forecasting in the UI.

    For example, the user can toggle a holiday flag or change
    the weather severity bin.
    """
    is_bank_holiday_week: int | None = None
    is_peak_winter: int | None = None
    weather_severity_bin: str | None = None


def build_regression_feature_row(
    weekly_df: pd.DataFrame,
    region: str,
    week_start: pd.Timestamp,
    scenario: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with all regression features for the given
    region and week_start.

    In this prototype, we assume `week_start` exists in the processed
    dataset (or is a cloned row from an existing week). We then apply any
    scenario overrides.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Full weekly regression dataset from `load_weekly_regression_data`.
    region : str
        Region name to filter on.
    week_start : pd.Timestamp
        Monday of the week.
    scenario : dict, optional
        Optional overrides for feature values, e.g.
        {"is_bank_holiday_week": 1, "weather_severity_bin": "severe"}

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame including all REGRESSION_FEATURE_COLUMNS and
        any additional columns the model might ignore.
    """
    scenario = scenario or {}

    mask = (weekly_df["region"] == region) & (weekly_df["week_start"] == week_start)
    if not mask.any():
        # Fallback: if the exact week_start is not found, use the closest
        # previous week for that region.
        region_df = weekly_df[weekly_df["region"] == region].sort_values("week_start")
        closest_idx = (region_df["week_start"] - week_start).abs().idxmin()
        row = region_df.loc[[closest_idx]].copy()
    else:
        row = weekly_df.loc[mask].copy()

    # Apply overrides from scenario dict
    for key, value in scenario.items():
        if key in row.columns:
            row.loc[:, key] = value

    # Ensure all expected columns are present
    missing_cols = [c for c in REGRESSION_FEATURE_COLUMNS if c not in row.columns]
    if missing_cols:
        raise ValueError(f"Missing expected regression feature columns: {missing_cols}")

    return row


def build_classification_feature_dict(
    *,
    region: str,
    route_difficulty: str,
    weather_severity_bin: str,
    party_size: int,
    lead_time_days: int,
    tour_date: pd.Timestamp,
    calendar_row: pd.Series,
    is_bank_holiday_week: int | None = None,
    is_peak_winter: int | None = None,
) -> Dict[str, Any]:
    """
    Build a dictionary of features for the cancellation model, combining
    booking-level inputs with calendar information.

    Parameters
    ----------
    region : str
    route_difficulty : str
    weather_severity_bin : str
    party_size : int
    lead_time_days : int
    tour_date : Timestamp
    calendar_row : pd.Series
        Row with `year`, `week_number`, `month`, `is_bank_holiday_week`,
        `is_peak_winter` derived from the tour_date week.
    is_bank_holiday_week : int, optional
        Optional override for the holiday flag (what-if in the UI).
    is_peak_winter : int, optional
        Optional override for peak winter flag.

    Returns
    -------
    dict
        Keys match CLASSIFICATION_FEATURE_COLUMNS.
    """
    year = int(calendar_row["year"])
    week_number = int(calendar_row["week_number"])
    month = int(calendar_row["month"])

    bh_flag = int(
        is_bank_holiday_week
        if is_bank_holiday_week is not None
        else calendar_row["is_bank_holiday_week"]
    )
    peak_flag = int(
        is_peak_winter
        if is_peak_winter is not None
        else calendar_row["is_peak_winter"]
    )

    features = {
        "region": region,
        "route_difficulty": route_difficulty,
        "weather_severity_bin": weather_severity_bin,
        "party_size": int(party_size),
        "lead_time_days": int(lead_time_days),
        "year": year,
        "week_number": week_number,
        "month": month,
        "is_bank_holiday_week": bh_flag,
        "is_peak_winter": peak_flag,
    }

    # Quick sanity check
    missing = [c for c in CLASSIFICATION_FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Missing expected classification feature keys: {missing}")

    return features
