# app_pages/page_report.py

import streamlit as st
import pandas as pd

from src import data_loaders, evaluate, visualize


def app():
    st.title("Model Report")

    st.markdown(
        """
        This page summarises performance of the two main models:

        - **Bookings Forecast (Regression)**  
        - **Cancellation Risk (Classification)**
        """
    )

    tab_reg, tab_clf = st.tabs(["Bookings Forecast", "Cancellation Risk"])

    # --- Regression tab ---
    with tab_reg:
        st.subheader("Weekly Bookings – Regression Model")

        train_reg, test_reg = data_loaders.load_regression_train_test()
        model = data_loaders.load_bookings_model()

        X_test = test_reg[
            [
                "region",
                "weather_severity_bin",
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
        ]
        y_test = test_reg["bookings_count"]

        y_pred = model.predict(X_test)
        metrics = evaluate.regression_metrics(y_test, y_pred)

        st.markdown("**Test set metrics:**")
        st.table(pd.DataFrame(metrics, index=["XGBoost (tuned)"]))

        # Actual vs Predicted plot
        st.markdown("**Actual vs Predicted (test set)**")
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.scatter(y_test, y_pred, alpha=0.6)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--")
        ax1.set_xlabel("Actual bookings")
        ax1.set_ylabel("Predicted bookings")
        ax1.set_title("Actual vs Predicted – Weekly Bookings")
        st.pyplot(fig1)

        st.caption(
            "Points lying close to the diagonal indicate good agreement between "
            "predicted and actual weekly bookings."
        )

    # --- Classification tab ---
    with tab_clf:
        st.subheader("Cancellation Risk – Classification Model")

        train_clf, test_clf = data_loaders.load_classification_train_test()
        model = data_loaders.load_cancellation_model()

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

