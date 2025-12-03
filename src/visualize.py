# src/visualize.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from . import data_loaders

sns.set(style="whitegrid")


def bookings_seasonality_by_region() -> plt.Figure:
    """
    Create a line plot of weekly bookings over time by region.
    Used in the EDA Streamlit page.
    """
    weekly = data_loaders.load_weekly_regression_data()

    fig, ax = plt.subplots(figsize=(10, 4))
    for region, grp in weekly.groupby("region"):
        ax.plot(grp["week_start"], grp["bookings_count"], label=region)

    ax.set_title("Weekly Bookings Over Time by Region")
    ax.set_xlabel("Week start")
    ax.set_ylabel("Bookings count")
    ax.legend()
    fig.tight_layout()
    return fig


def holiday_uplift_barplot() -> plt.Figure:
    """
    Plot average bookings in bank holiday vs non-holiday weeks.
    """
    weekly = data_loaders.load_weekly_regression_data()
    summary = (
        weekly.groupby("is_bank_holiday_week", as_index=False)["bookings_count"]
        .mean()
        .rename(columns={"bookings_count": "avg_bookings"})
    )
    summary["holiday_label"] = summary["is_bank_holiday_week"].map(
        {0: "Non-holiday weeks", 1: "Bank holiday weeks"}
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=summary, x="holiday_label", y="avg_bookings", ax=ax)
    ax.set_title("Average Weekly Bookings – Holiday vs Non-holiday Weeks")
    ax.set_xlabel("")
    ax.set_ylabel("Average bookings")
    fig.tight_layout()
    return fig


def cancellation_rate_by_weather() -> plt.Figure:
    """
    Plot cancellation rate by weather severity bin.
    """
    bookings = data_loaders.load_bookings_classification_data()
    summary = (
        bookings.groupby("weather_severity_bin", as_index=False)["was_cancelled"]
        .mean()
        .rename(columns={"was_cancelled": "cancellation_rate"})
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(
        data=summary.sort_values("cancellation_rate"),
        x="weather_severity_bin",
        y="cancellation_rate",
        ax=ax,
    )
    ax.set_title("Cancellation Rate by Weather Severity")
    ax.set_xlabel("Weather severity")
    ax.set_ylabel("Cancellation rate")
    fig.tight_layout()
    return fig


def roc_pr_curves(
    y_true,
    y_proba,
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Produce ROC and Precision–Recall curves for display in the Model Report page.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label="ROC curve")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    fig_roc.tight_layout()

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
    ax_pr.plot(rec, prec, label="PR curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall Curve")
    fig_pr.tight_layout()

    return fig_roc, fig_pr
