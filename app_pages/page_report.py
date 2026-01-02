# app_pages/page_report.py

import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBModel
import json
from pathlib import Path

from src import data_loaders, evaluate, visualize
from src.ui import inject_global_css


def _ensure_xgb_gpu_id(model):
    """
    Heroku CPU builds of xgboost still expect a `gpu_id` attribute on the
    XGBModel in some paths. Patch added.
    """
    if isinstance(model, Pipeline):
        est = model.steps[-1][1]
    else:
        est = model

    if isinstance(est, XGBModel) and not hasattr(est, "gpu_id"):
        est.gpu_id = -1

    return model


def app():
    inject_global_css()
    
    st.title("Model Report")

    st.markdown(
        """
        This page summarises performance of the two main models:

        - **Bookings Forecast (Regression)**  
        - **Cancellation Risk (Classification)**
        """
    )

    st.markdown("""
### 🔍 Model Performance Summary (LO4.2)

This project went through an **iterative modelling process**, reflecting how real-world analytics systems are refined as data quality and assumptions improve.

---

#### 📌 Initial Results (Early Synthetic Dataset)

**Bookings Forecast Model (Regression – Initial Version)**  
- Mean Absolute Error (MAE): **32.61 bookings**  
- Mean Absolute Percentage Error (MAPE): **119.64%**  
- R² Score: **-5.02**

These early results were impacted by limitations in the initial synthetic data generation:
- Weekly bookings could be **near-zero or negative** due to unbounded noise.
- Region-level demand differences were **under-represented**.
- Fewer regions were included, limiting variation.

While this version still demonstrated a complete ML pipeline end-to-end, the metrics highlighted issues with data realism rather than model setup.

---

#### ✅ Improved Results (Refined Synthetic Dataset)

After refining the synthetic data to better reflect real-world constraints:
- Bookings are generated from **positive base demand per region**.
- Noise is constrained to prevent negative values.
- Additional regions (**Peak District** and **Yorkshire Dales**) were added to increase realism and geographic coverage.

**Bookings Forecast Model (Regression – Tuned XGBoost)**  
- Mean Absolute Error (MAE): **~9.37 bookings**  
- Mean Absolute Percentage Error (MAPE): **~10.70%**  
- R² Score: **~0.80**

These results show a substantial improvement in model performance and stability, indicating that the regression model now explains the majority of variance in weekly demand while remaining interpretable and operationally useful.

---

#### ⚠️ Cancellation Risk Model (Classification)

**Cancellation Risk Model**  
- ROC AUC Score: **~0.63**

The classification model performs above random chance (0.5) and demonstrates moderate ability to separate cancelled vs non-cancelled bookings.  
It is suitable as a **prototype risk-ranking tool**, enabling Customer Service teams to prioritise proactive communication for higher-risk bookings.

---

#### 🧠 Interpretation

The improvement between the initial and refined results reflects:
- Better alignment between synthetic data assumptions and real-world constraints.
- Increased signal-to-noise ratio for learning.
- A more realistic operational scenario for staffing and planning decisions.

This iterative refinement mirrors real analytics workflows, where data quality and modelling assumptions are progressively improved rather than fixed from the outset.
""")

    # --- Load offline regression metrics (from notebooks) ---
    REG_METRICS_PATH = Path("models/v1_bookings_model_metrics.json")
    reg_metrics = None
    if REG_METRICS_PATH.exists():
        with open(REG_METRICS_PATH, "r", encoding="utf-8") as f:
            reg_metrics = json.load(f)
    else:
        st.warning("Offline regression metrics not found at models/v1_bookings_model_metrics.json")


    tab_reg, tab_clf = st.tabs(["Bookings Forecast", "Cancellation Risk"])

    # --- Regression tab ---
    with tab_reg:
        st.subheader("Weekly Bookings – Regression Model")

        st.markdown("**Offline test set metrics (from notebooks):**")

        if reg_metrics and "xgboost_tuned" in reg_metrics:
            xgb = reg_metrics["xgboost_tuned"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{xgb['MAE']:.2f}")
            with col2:
                st.metric("MAPE", f"{xgb['MAPE']:.2f}%")
            with col3:
                st.metric("R²", f"{xgb['R2']:.3f}")

            # Comparison Table
            rows = []
            for name in ["baseline", "linear_regression", "random_forest", "xgboost_tuned"]:
                if name in reg_metrics:
                    m = reg_metrics[name]
                    rows.append({"model": name, "MAE": m["MAE"], "MAPE": m["MAPE"], "R2": m["R2"]})

            if rows:
                st.markdown("**Model comparison (offline test set):**")
                st.table(pd.DataFrame(rows).set_index("model"))

            # tuned params
            if "best_params" in reg_metrics:
                with st.expander("Best tuned hyperparameters (XGBoost)"):
                    st.json(reg_metrics["best_params"])

        else:
            st.info("No offline regression metrics available yet. Re-run the regression notebook to regenerate metrics JSON.")


        st.caption(
            "Note: diagnostic plots (Actual vs Predicted, residuals) are generated in the modelling notebook "
            "and saved under reports/figures/ for reproducibility. The dashboard displays the offline test metrics "
            "as the source of truth."
        )


    # --- Classification tab ---
    with tab_clf:
        st.subheader("Cancellation Risk – Classification Model")

        train_clf, test_clf = data_loaders.load_classification_train_test()
        model = data_loaders.load_cancellation_model()
        model = _ensure_xgb_gpu_id(model)

        X_test = test_clf[
            [
                "region",
                "route_difficulty",
                "weather_severity_bin",
                "party_size",
                "lead_time_days",
                "year",
                "week_number",
                "month",
                "is_bank_holiday_week",
                "is_peak_winter",
            ]
        ]
        y_test = test_clf["was_cancelled"]

        y_proba = model.predict_proba(X_test)[:, 1]

        # Default threshold for the report
        metrics_05 = evaluate.classification_metrics(y_test, y_proba, threshold=0.5)
        cm_df = evaluate.confusion_matrix_to_df(metrics_05["confusion_matrix"])

        st.markdown("**ROC AUC (test set):**")
        st.metric(label="ROC AUC", value=f"{metrics_05['roc_auc']:.3f}")

        st.markdown("**Confusion matrix (threshold = 0.5):**")
        st.table(cm_df)

        st.markdown("**ROC & Precision–Recall curves**")
        fig_roc, fig_pr = visualize.roc_pr_curves(y_test, y_proba)
        col_roc, col_pr = st.columns(2)
        with col_roc:
            st.pyplot(fig_roc)
        with col_pr:
            st.pyplot(fig_pr)

        st.caption(
            "The ROC and PR curves show how well the model distinguishes between "
            "cancelled and non-cancelled bookings across different thresholds."
        )

