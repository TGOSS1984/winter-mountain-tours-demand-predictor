# app_pages/page_map.py

import pandas as pd
import plotly.express as px
import streamlit as st

from src import data_loaders
from src.geo import get_region_latlon
from src.ui import inject_global_css


def app():
    inject_global_css()
    st.title("📍 Map View: Regions Overview")

    st.markdown(
        """
        <div class="card">
        This page provides a geographic view of the operating regions used in the dataset.
        In later steps, this map will be extended to show <strong>forecasted weekly bookings</strong>
        and scenario-driven changes over time (e.g. bank holidays and weather severity).
        </div>
        """,
        unsafe_allow_html=True,
    )

    # load weekly data just to get the region labels used in the dataset.
    weekly = data_loaders.load_weekly_regression_data()
    regions = sorted(pd.Series(weekly["region"]).dropna().unique())

    rows = []
    for r in regions:
        lat, lon = get_region_latlon(r)
        rows.append({"region": r, "lat": lat, "lon": lon})

    df_map = pd.DataFrame(rows)

    fig = px.scatter_geo(
        df_map,
        lat="lat",
        lon="lon",
        hover_name="region",
        scope="europe",
        title="Operating Regions (centroid markers)",
    )
    fig.update_traces(marker=dict(size=12))

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Markers represent approximate region centroids (for visualisation only). "
        "This supports region-level planning rather than route-level navigation."
    )
