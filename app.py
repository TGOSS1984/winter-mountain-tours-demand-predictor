import streamlit as st

def main():
    st.set_page_config(
        page_title="Winter Mountain Tours – Predictive Analytics",
        page_icon="🏔️",
        layout="wide",
    )

    st.title("Winter Mountain Tour Demand & Cancellation Predictor")
    st.markdown(
        """
        This app will forecast **weekly bookings** for each mountain region
        and predict **cancellation risk** for upcoming tours.

        Phase 0: ✅ Project scaffolded  
        Next steps: add **business understanding**, data design, and EDA.
        """
    )

    st.info(
        "App scaffolded successfully. Use `streamlit run app.py` to view it locally."
    )

if __name__ == "__main__":
    main()
