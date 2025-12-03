# app_pages/page_forecast.py

import streamlit as st


def app():
    st.title("Bookings Forecast")
    st.markdown(
        "This page will provide a forecast of weekly bookings per region, "
        "with simple what-if options."
    )
