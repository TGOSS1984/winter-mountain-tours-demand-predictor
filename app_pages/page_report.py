# app_pages/page_report.py

import streamlit as st


def app():
    st.title("Model Report")
    st.markdown(
        "This page will summarise model performance, metrics and evaluation plots "
        "for both the bookings forecast and cancellation models."
    )
