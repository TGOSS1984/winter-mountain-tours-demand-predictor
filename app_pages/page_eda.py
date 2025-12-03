# app_pages/page_eda.py

import streamlit as st

from src import data_loaders, visualize


def app():
    st.title("EDA & Insights")

    st.markdown(
        """
        This page explores historical bookings and cancellations to answer questions like:
        - When are tours busiest?
        - How much uplift do bank holidays provide?
        - How do cancellations change with weather severity?
        """
    )

    weekly = data_loaders.load_weekly_regression_data()

    # --- Interactive seasonality line chart (Plotly-like feel using Streamlit) ---
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
            "Interactive line chart showing weekly bookings over time for the selected region(s). "
            "Peaks in winter and holiday periods support the importance of calendar features "
            "and lagged demand."
        )
    else:
        st.warning("Please select at least one region to display the seasonality chart.")

    # --- Pre-built matplotlib figures for concise insights ---
    st.subheader("Holiday Uplift in Bookings")

    fig_holiday = visualize.holiday_uplift_barplot()
    st.pyplot(fig_holiday)
    st.caption(
        "Bank holiday weeks show a higher average bookings count than non-holiday weeks, "
        "supporting Hypothesis H1 (holiday uplift)."
    )

    st.subheader("Cancellation Rate by Weather Severity")

    fig_cancel_weather = visualize.cancellation_rate_by_weather()
    st.pyplot(fig_cancel_weather)
    st.caption(
        "Cancellation rates increase with weather severity, supporting Hypothesis H2. "
        "This justifies including weather severity bins in the cancellation model."
    )

