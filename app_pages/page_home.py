# app_pages/page_home.py

import streamlit as st


def app():
    st.title("Winter Mountain Tour Demand & Cancellation Predictor")

    st.markdown(
        """
        This dashboard is a predictive analytics prototype for **UK winter mountain tours**.
        It combines historical bookings, calendar features (bank holidays, peak winter) and
        synthetic weather data to:

        - Forecast **weekly bookings per region**, and  
        - Predict **cancellation risk** for individual bookings.
        """
    )

    st.subheader("Who is this for?")
    st.markdown(
        """
        - **Operations Lead** – plan guide rosters and capacity by region.  
        - **Guides / Route Leaders** – understand likely tour volumes.  
        - **Marketing** – identify peak and low weeks for promotions.  
        - **Customer Service** – prioritise high-risk bookings for proactive reminders.
        """
    )

    st.subheader("Quick model status")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Bookings Forecast Model")
        st.success(
            "Regression model trained and evaluated. "
            "Error metrics and actual vs predicted plots are available on the **Model Report** page."
        )

    with col2:
        st.markdown("#### Cancellation Risk Model")
        st.success(
            "Classification model trained with ROC AUC and confusion matrices. "
            "Try it on the **Cancellation Risk** page."
        )

    st.info(
        "Use the sidebar to navigate between **EDA & Insights**, "
        "**Bookings Forecast**, **Cancellation Risk**, and **Model Report**."
    )
