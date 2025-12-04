# app_pages/page_eda.py

import streamlit as st

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
    st.subheader("Seasonality of Weekly Bookings by Region")

    regions = sorted(weekly["region"].unique())
    selected_regions = st.multiselect(
        "Select region(s) to display",
        options=regions,
        default=regions,
    )

    if selected_regions:
        filtered = weekly[weekly["region"].isin(selected_regions)].copy()
        pivot = filtered.pivot_table(
            index="week_start", columns="region", values="bookings_count", aggfunc="sum"
        ).sort_index()

        st.line_chart(pivot)
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


