# app_pages/page_eda.py

import streamlit as st
import pandas as pd

from src import data_loaders, visualize
from src.ui import inject_global_css


def app():
    inject_global_css()

    st.title("📊 EDA & Insights")

    st.markdown(
        """
        <div class="card">
        <p>
        Explore historical bookings and cancellations to understand the drivers of demand and risk:
        </p>
        <ul>
          <li>Seasonality by region</li>
          <li>Holiday uplift</li>
          <li>Weather-driven cancellations</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    weekly = data_loaders.load_weekly_regression_data()

    # Seasonality
    st.subheader("Seasonality of Bookings by Region")

    # Ensure datetime
    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])

    # Time grain selector
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        grain = st.selectbox("Time grain", ["Week", "Month", "Quarter"], index=1)
    with col_g2:
        use_range = st.checkbox("Limit date range", value=True)

    if use_range:
        date_min = weekly["week_start"].min()
        date_max = weekly["week_start"].max()
        start_dt, end_dt = st.slider(
            "Date range",
            min_value=date_min.to_pydatetime(),
            max_value=date_max.to_pydatetime(),
            value=(date_min.to_pydatetime(), date_max.to_pydatetime()),
        )
        weekly = weekly[
            (weekly["week_start"] >= pd.to_datetime(start_dt)) &
            (weekly["week_start"] <= pd.to_datetime(end_dt))
        ].copy()

    regions = sorted(weekly["region"].dropna().unique())
    selected_regions = st.multiselect(
        "Select region(s) to display",
        options=regions,
        default=regions,
    )


    if selected_regions:
        filtered = weekly[weekly["region"].isin(selected_regions)].copy()

        # Create a period column based on selected time grain
        if grain == "Week":
            filtered["period"] = filtered["week_start"]
        elif grain == "Month":
            filtered["period"] = filtered["week_start"].dt.to_period("M").dt.to_timestamp()
        else:  # Quarter
            filtered["period"] = filtered["week_start"].dt.to_period("Q").dt.to_timestamp()

        pivot = filtered.pivot_table(
            index="period",
            columns="region",
            values="bookings_count",
            aggfunc="sum",
        ).sort_index()

        st.line_chart(pivot)
        st.caption(
            "Use Month/Quarter to reduce noise and spot patterns faster, then switch back to Week to drill into detail."
        )

        st.caption(
            "Peaks in winter and holiday periods support the importance of calendar features "
            "and lagged demand as predictors."
        )
    else:
        st.warning("Please select at least one region to display the seasonality chart.")

    # Holiday uplift
    st.subheader("Holiday Uplift in Bookings")
    fig_holiday = visualize.holiday_uplift_barplot()
    st.pyplot(fig_holiday)
    st.caption(
        "Bank holiday weeks show a higher average bookings count than non-holiday weeks, "
        "supporting Hypothesis H1 (holiday uplift)."
    )

    # Cancellations vs weather
    st.subheader("Cancellation Rate by Weather Severity")
    fig_cancel_weather = visualize.cancellation_rate_by_weather()
    st.pyplot(fig_cancel_weather)
    st.caption(
        "Cancellation rates increase with weather severity, supporting Hypothesis H2 and justifying "
        "the inclusion of weather severity bins in the cancellation model."
    )


