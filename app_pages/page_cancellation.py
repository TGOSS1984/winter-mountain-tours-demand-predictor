# app_pages/page_cancellation.py

import streamlit as st
import pandas as pd

from src import data_loaders, models, features


def _get_calendar_row_for_date(tour_date: pd.Timestamp):
    """
    Find the calendar fields (year, week_number, month, holiday flags)
    for a given tour date, using the weekly regression dataset.
    """
    weekly = data_loaders.load_weekly_regression_data()

    week_start = tour_date - pd.to_timedelta(tour_date.weekday(), unit="D")
    # Use first region's row for that week as calendar proxy
    row = weekly[weekly["week_start"] == week_start].iloc[0]
    return row


def app():
    st.title("Cancellation Risk")

    st.markdown(
        """
        Estimate **cancellation risk** for a single upcoming booking. This uses the
        trained classification model and returns a probability plus a high/low risk flag.
        """
    )

    with st.form("cancellation_form"):
        col1, col2 = st.columns(2)

        weekly = data_loaders.load_weekly_regression_data()
        regions = sorted(weekly["region"].unique())

        with col1:
            region = st.selectbox("Region", options=regions)
            tour_date = st.date_input("Tour date")
            party_size = st.slider("Party size", min_value=1, max_value=10, value=3)
            lead_time_days = st.slider(
                "Lead time (days between booking and tour)",
                min_value=1,
                max_value=90,
                value=21,
            )

        with col2:
            route_difficulty = st.selectbox(
                "Route difficulty",
                options=["easy", "moderate", "challenging"],
                index=1,
            )
            weather_severity_bin = st.selectbox(
                "Expected weather severity",
                options=["mild", "moderate", "severe"],
                index=1,
            )
            threshold = st.slider(
                "High-risk threshold (probability)",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
            )

        submitted = st.form_submit_button("Estimate risk")

    if not submitted:
        st.info("Fill in the booking details and click **Estimate risk**.")
        return

    # Build features from inputs
    tour_dt = pd.to_datetime(tour_date)
    calendar_row = _get_calendar_row_for_date(tour_dt)

    booking_features = features.build_classification_feature_dict(
        region=region,
        route_difficulty=route_difficulty,
        weather_severity_bin=weather_severity_bin,
        party_size=party_size,
        lead_time_days=lead_time_days,
        tour_date=tour_dt,
        calendar_row=calendar_row,
    )

    result = models.predict_cancellation_risk(
        booking_features=booking_features, threshold=threshold
    )

    prob = result["probability"]
    is_high_risk = result["is_high_risk"]

    st.subheader("Prediction")

    col_left, col_right = st.columns(2)
    with col_left:
        st.metric(
            label="Predicted cancellation probability",
            value=f"{prob:.1%}",
        )

    with col_right:
        st.metric(
            label="High-risk?",
            value="Yes" if is_high_risk else "No",
        )

    with st.expander("Features used for this prediction"):
        st.json(booking_features)

    st.markdown(
        """
        - The **threshold slider** controls how cautious the system is when flagging bookings as high-risk.  
        - A lower threshold will flag more bookings (higher recall, more false positives).  
        - A higher threshold will flag fewer bookings (lower recall, fewer false positives).
        """
    )

