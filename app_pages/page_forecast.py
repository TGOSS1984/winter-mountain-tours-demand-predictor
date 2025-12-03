# app_pages/page_forecast.py

import streamlit as st
import pandas as pd

from src import data_loaders, models


def app():
    st.title("Bookings Forecast")

    st.markdown(
        """
        Use this page to forecast **weekly bookings per region** and explore simple
        what-if scenarios around holidays and weather.
        """
    )

    weekly = data_loaders.load_weekly_regression_data()

    # Sidebar-style controls in a column
    col_controls, col_result = st.columns([1, 2])

    with col_controls:
        regions = sorted(weekly["region"].unique())
        region = st.selectbox("Region", options=regions)

        available_weeks = (
            weekly[weekly["region"] == region]["week_start"]
            .drop_duplicates()
            .sort_values()
        )
        week_start = st.selectbox(
            "Week (start date)",
            options=available_weeks,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )

        st.markdown("**What-if scenario**")

        is_holiday = st.checkbox("Treat this as a bank holiday week?")
        is_peak_winter = st.checkbox("Treat this as a peak winter week?")

        severity = st.selectbox(
            "Weather severity (scenario)",
            options=["mild", "moderate", "severe"],
            index=0,
        )

        scenario = {
            "is_bank_holiday_week": int(is_holiday),
            "is_peak_winter": int(is_peak_winter),
            "weather_severity_bin": severity,
        }

        if st.button("Run forecast"):
            result = models.forecast_weekly_bookings(
                region=region,
                week_start=pd.to_datetime(week_start),
                scenario=scenario,
            )
            st.session_state["forecast_result"] = result

    with col_result:
        st.subheader("Forecast result")

        result = st.session_state.get("forecast_result")
        if result is None:
            st.info("Select a region and week, adjust the scenario, then click **Run forecast**.")
        else:
            pred = result["prediction"]
            inputs = result["inputs"]

            st.metric(
                label=f"Forecast bookings for {region} (week starting {inputs['week_start'].strftime('%Y-%m-%d')})",
                value=f"{pred:.1f} bookings",
            )

            with st.expander("Inputs used for this forecast"):
                st.json(inputs)

            st.markdown(
                """
                This forecast uses the trained regression model (tuned XGBoost) based on:
                - Recent demand (lag features),
                - Calendar indicators (bank holidays, peak winter),
                - Weather severity and other weather metrics.

                Compare this forecast with historical patterns on the **EDA & Insights** page
                to judge whether the model behaviour seems reasonable.
                """
            )

