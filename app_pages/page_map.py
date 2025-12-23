# app_pages/page_map.py

import pandas as pd
import plotly.express as px
import streamlit as st

from src import data_loaders, models
from src.geo import get_region_latlon
from src.ui import inject_global_css


def app():
    inject_global_css()
    st.title("📍 Map View: Forecasted Bookings by Region")

    st.markdown(
        """
        <div class="card">
        Select a week to view <strong>forecasted bookings per region</strong> on a map.
        Marker <strong>size</strong> and <strong>colour</strong> represent forecast intensity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    weekly = data_loaders.load_weekly_regression_data()

    # Build list of weeks from the dataset
    available_weeks = (
    weekly["week_start"].dropna().drop_duplicates().sort_values().reset_index(drop=True)
    )

    # Default to last available week (often most relevant)
    default_idx = len(available_weeks) - 1 if len(available_weeks) > 0 else 0

    st.markdown("### 🗓️ Week selection")
    week_idx = st.slider(
        "Scrub through weeks",
        min_value=0,
        max_value=max(len(available_weeks) - 1, 0),
        value=st.session_state.get("map_week_idx", default_idx),
        step=1,
    )

    chosen_week = pd.to_datetime(available_weeks.iloc[week_idx])
    st.session_state["map_week_idx"] = week_idx

    # Optional fallback selector (useful for debugging)
    with st.expander("Alternative week picker (debug)", expanded=False):
        week_start_pick = st.selectbox(
            "Pick a week (start date)",
            options=available_weeks,
            index=week_idx,
            format_func=lambda d: pd.to_datetime(d).strftime("%Y-%m-%d"),
        )
        chosen_week = pd.to_datetime(week_start_pick)
        # sync slider to pick if user uses the selectbox
        try:
            st.session_state["map_week_idx"] = int(available_weeks[available_weeks == week_start_pick].index[0])
        except Exception:
            pass

    col_prev, col_next, col_jump = st.columns([1, 1, 2])
    with col_prev:
        if st.button("◀ Previous"):
            st.session_state["map_week_idx"] = max(0, st.session_state["map_week_idx"] - 1)
            st.rerun()

    with col_next:
        if st.button("Next ▶"):
            st.session_state["map_week_idx"] = min(len(available_weeks) - 1, st.session_state["map_week_idx"] + 1)
            st.rerun()

    with col_jump:
        st.caption("Tip: use the slider to scrub through weeks and watch demand shift by region.")



    st.markdown("#### Scenario (optional)")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        is_holiday = st.checkbox("Bank holiday week", value=False)
    with col_b:
        is_peak_winter = st.checkbox("Peak winter", value=False)
    with col_c:
        severity = st.selectbox(
            "Weather severity",
            options=["mild", "moderate", "severe"],
            index=0,
        )

    scenario = {
        "is_bank_holiday_week": int(is_holiday),
        "is_peak_winter": int(is_peak_winter),
        "weather_severity_bin": severity,
    }

    if st.button("Update map"):
        st.session_state["map_scenario"] = scenario

        chosen_scenario = st.session_state.get("map_scenario", scenario)


    regions = sorted(pd.Series(weekly["region"]).dropna().unique())

    rows = []
    errors = []

    for region in regions:
        try:
            lat, lon = get_region_latlon(region)

            result = models.forecast_weekly_bookings(
                region=region,
                week_start=chosen_week,
                scenario=chosen_scenario,
            )
            pred = float(result["prediction"])

            rows.append(
                {
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "forecast_bookings": max(pred, 0.0),
                }
            )
        except Exception as e:
            errors.append(f"{region}: {e}")

    df_map = pd.DataFrame(rows)

    if df_map.empty:
        st.error("No forecast results were generated. Check region mappings and model availability.")
        if errors:
            with st.expander("Debug details"):
                st.write(errors)
        return

    fig = px.scatter_geo(
        df_map,
        lat="lat",
        lon="lon",
        hover_name="region",
        hover_data={"forecast_bookings": ":.1f", "lat": False, "lon": False},
        size="forecast_bookings",
        color="forecast_bookings",
        scope="europe",
        title=f"Forecasted bookings (week starting {chosen_week.strftime('%Y-%m-%d')})",
        size_max=40,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Interpretation: larger/warmer markers indicate higher forecasted demand. "
        "This helps Operations quickly see where staffing pressure is highest for the selected week."
    )

    # Non-blocking diagnostics
    if errors:
        with st.expander("Regions skipped (debug)"):
            st.write(errors)

