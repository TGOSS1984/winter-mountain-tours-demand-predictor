import streamlit as st

from app_pages.multipage import MultiPage
from app_pages import (
    page_home,
    page_eda,
    page_forecast,
    page_cancellation,
    page_report,
    page_data_docs,
    page_weather_image,
    page_map,
    )


def main():

    st.set_page_config(
        page_title="Winter Mountain Tours – Predictive Analytics",
        page_icon="🏔️",
        layout="wide",
    )

    with st.sidebar:
        st.image("app_pages/assets/images/logo-white-text.webp", width=120)


    app = MultiPage(app_name="Winter Mountain Tour Demand & Cancellation Predictor")

    # Register pages
    app.add_page("🏔️ Home / Project Summary", page_home.app)
    app.add_page("📊 EDA & Insights", page_eda.app)
    app.add_page("📈 Bookings Forecast", page_forecast.app)
    app.add_page("📍 Map View", page_map.app)
    app.add_page("⚠️ Cancellation Risk", page_cancellation.app)
    app.add_page("🖼️ Weather from Image", page_weather_image.app)
    app.add_page("🔍 Model Report", page_report.app)
    app.add_page("Data & Docs", page_data_docs.app)
    



    # Run app
    app.run()


if __name__ == "__main__":
    main()

