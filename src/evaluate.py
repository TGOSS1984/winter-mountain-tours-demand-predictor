# src/evaluate.py

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute MAE, MAPE and R² for regression outputs.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(
        np.mean(
            np.abs((y_true - y_pred) / np.maximum(y_true, 1))
        )
        * 100
    )
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "MAPE": mape, "R2": float(r2)}


def classification_metrics(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute ROC AUC, confusion matrix and classification report
    at a given threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "roc_auc": float(roc_auc),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "report": report,
    }


def confusion_matrix_to_df(cm) -> pd.DataFrame:
    """
    Convert a 2x2 confusion matrix into a labelled DataFrame that
    can be displayed easily in Streamlit.
    """
    return pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )
