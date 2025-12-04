# app_pages/page_data_docs.py

import streamlit as st
import pandas as pd

from src import data_loaders


def app():
    st.title("Data & Documentation")

    st.markdown(
        """
        This page provides supporting information for the project, including:
        - **Data lineage** from raw → cleaned → processed  
        - **Data dictionary** for key modelling fields  
        - **Notebook links** for cleaning, feature engineering, and modelling  
        - **Model artefacts & version info**  
        """
    )

    # -------------------------
    # Data Lineage
    # -------------------------
    st.subheader("Data Lineage")

    st.markdown(
        """
        **Source inputs:**
        - Synthetic booking-level dataset (internally generated)
        - Calendar data (UK bank holidays endpoint)
        - Synthetic weather data  
        
        **Transformation steps:**
        1. `01_cleaning.ipynb` — handling missing values, type coercion, data fixes  
        2. `02_feature_engineering.ipynb` — weekly aggregation, lag features, classification feature generation  
        3. Final processed datasets saved in **`data/processed/`**  
        """
    )

    # -------------------------
    # Data Dictionary
    # -------------------------
    st.subheader("Data Dictionary (Key Fields)")

    data_dict = {
        "region": "Tour region (categorical: Lake District, Snowdonia, Cairngorms)",
        "bookings_count": "Total number of tours booked in that region-week (regression target)",
        "was_cancelled": "Binary label for cancellation (classification target)",
        "week_start": "Monday of the ISO week",
        "year": "Calendar year extracted from tour_date",
        "week_number": "ISO week number",
        "month": "Month (1–12)",
        "is_bank_holiday_week": "1 if week includes a UK bank holiday",
        "is_peak_winter": "1 if winter peak (Dec–Mar)",
        "mean_temp_c": "Average temperature for week",
        "precip_mm": "Average precipitation",
        "snowfall_flag": "1 if snow observed in week",
        "weather_severity_bin": "Weather category: mild / moderate / severe",
        "lag_1w_bookings": "Bookings from previous week",
        "lag_4w_mean": "Rolling 4-week mean bookings",
        "lag_52w_bookings": "Bookings from same week last year",
        "party_size": "Number of participants",
        "lead_time_days": "Days between booking and tour date",
        "route_difficulty": "easy / moderate / challenging",
    }

    df_dict = pd.DataFrame.from_dict(
        data_dict, orient="index", columns=["Description"]
    )

    st.table(df_dict)

    # -------------------------
    # Sample Data Outputs
    # -------------------------
    st.subheader("Sample Data")

    weekly = data_loaders.load_weekly_regression_data()
    st.markdown("**Weekly regression dataset (top 10 rows):**")
    st.dataframe(weekly.head(10))

    bookings = data_loaders.load_bookings_classification_data()
    st.markdown("**Booking-level classification dataset (top 10 rows):**")
    st.dataframe(bookings.head(10))

    # -------------------------
    # Notebook Links (Displayed as text)
    # -------------------------
    st.subheader("Notebooks")

    st.markdown(
        """
        These notebooks contain the full CRISP-DM workflow:

        - `00_collect_endpoint_bank_holidays.ipynb`  
        - `01_cleaning.ipynb`  
        - `02_feature_engineering.ipynb`  
        - `03_model_regression.ipynb`  
        - `04_model_classification.ipynb`  

        To view them, open the GitHub repository.
        """
    )

    # -------------------------
    # Model Version Info
    # -------------------------
    st.subheader("Model Artefacts & Version Info")

    try:
        bookings_model = data_loaders.load_bookings_model()
        st.success("Bookings forecast model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading forecast model: {e}")

    try:
        cancel_model = data_loaders.load_cancellation_model()
        st.success("Cancellation model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading cancellation model: {e}")

    st.markdown(
        """
        **Current model versions:**  
        - `models/v1_bookings_model.pkl`  
        - `models/v1_cancel_model.pkl`  

        These files are stored in the repository so that the Streamlit app can
        load them directly without re-running the notebooks.
        """
    )
