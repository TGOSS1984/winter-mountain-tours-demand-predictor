# src/ui.py

import streamlit as st


def inject_global_css() -> None:
    """
    Global CSS tweaks & font awesome link

    """
    st.markdown(
        """
    
        <style>
        /* Sidebar (navigation bar) background */
        section[data-testid="stSidebar"] {
            background-color: #404e64 !important;
        }

        /* Sidebar text contrast */
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Make the main content a bit wider and less cramped */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        /* Card-like look for st.container sections via .card class */
        .card {
            background-color: #404e64; /* matches dark theme secondary bg */
            border-radius: 0.75rem;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #f9fafb;
        }

        .card h2, .card h3, .card h4 {
            margin-top: 0.25rem;
        }

        /* Slightly tighter tables */
        .stTable, .dataframe {
            font-size: 0.9rem;
        }

        /* Metrics spacing */
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
