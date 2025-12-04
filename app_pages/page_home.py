# app_pages/page_home.py

import streamlit as st
from src.ui import inject_global_css


def app():
    inject_global_css()

    st.title("🏔️ Winter Mountain Tour Demand & Cancellation Predictor")

    st.markdown(
        """
        <div class="card">
        <p>
        This dashboard is a predictive analytics prototype for <strong>UK winter mountain tours</strong>.
        It combines historical bookings, calendar features (bank holidays, peak winter) and
        synthetic weather data to:
        </p>
        <ul>
          <li>Forecast <strong>weekly bookings per region</strong>, and</li>
          <li>Predict <strong>cancellation risk</strong> for individual bookings.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
          <h3>Who is this for?</h3>
          <ul>
            <li><strong>Operations Lead</strong> – plan guide rosters and capacity by region.</li>
            <li><strong>Guides / Route Leaders</strong> – understand likely tour volumes.</li>
            <li><strong>Marketing</strong> – identify peak and low weeks for promotions.</li>
            <li><strong>Customer Service</strong> – prioritise high-risk bookings for proactive reminders.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
              <h4>Bookings Forecast Model</h4>
              <p>Regression model trained and evaluated on weekly region-level data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
              <h4>Cancellation Risk Model</h4>
              <p>Classification model with ROC AUC, confusion matrices and threshold tuning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Use the sidebar to navigate between **EDA & Insights**, "
        "**Bookings Forecast**, **Cancellation Risk**, **Model Report**, and **Data & Docs**."
    )

