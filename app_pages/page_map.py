# app_pages/page_map.py

import pandas as pd
import plotly.express as px
import streamlit as st
from functools import lru_cache

from src import data_loaders, models
from src.geo import get_region_latlon
from src.ui import inject_global_css

@lru_cache(maxsize=256)
def _cached_forecast(region: str, week_start_str: str, scenario_key: str) -> float:
    """
    Cached forecast wrapper to avoid re-running model inference repeatedly when
    scrubbing through weeks or re-rendering Streamlit.
    """
    week_start = pd.to_datetime(week_start_str)
    # scenario_key is already encoded; decode it back to dict safely
    # Format: "holiday=0|peak=1|sev=mild"
    parts = dict(p.split("=", 1) for p in scenario_key.split("|"))
    scenario = {
        "is_bank_holiday_week": int(parts["holiday"]),
        "is_peak_winter": int(parts["peak"]),
        "weather_severity_bin": parts["sev"],
    }

    result = models.forecast_weekly_bookings(
        region=region,
        week_start=week_start,
        scenario=scenario,
    )
    return float(result["prediction"])



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

    # Persist scenario automatically (no button needed)
    st.session_state["map_scenario"] = scenario
    chosen_scenario = st.session_state["map_scenario"]


    # Build stable keys for caching
    week_str = chosen_week.strftime("%Y-%m-%d")
    scenario_key = (
        f"holiday={chosen_scenario['is_bank_holiday_week']}"
        f"|peak={chosen_scenario['is_peak_winter']}"
        f"|sev={chosen_scenario['weather_severity_bin']}"
    )



    regions = sorted(pd.Series(weekly["region"]).dropna().unique())

    st.markdown("### 🧭 Region filter (optional)")
    selected_regions = st.multiselect(
        "Choose regions to display",
        options=regions,
        default=regions,
    )

    if not selected_regions:
        st.warning("Select at least one region to display the map.")
        return

    st.markdown("### ✅ Current view")
    st.write(
        f"**Week starting:** {week_str}  \n"
        f"**Scenario:** bank holiday={chosen_scenario['is_bank_holiday_week']}, "
        f"peak winter={chosen_scenario['is_peak_winter']}, "
        f"severity={chosen_scenario['weather_severity_bin']}"
    )


    rows = []
    errors = []

    for region in selected_regions:
        try:
            lat, lon = get_region_latlon(region)

            pred = _cached_forecast(region, week_str, scenario_key)


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

    col1, col2, col3 = st.columns(3)
    col1.metric("Min forecast", f"{df_map['forecast_bookings'].min():.1f}")
    col2.metric("Average forecast", f"{df_map['forecast_bookings'].mean():.1f}")
    col3.metric("Max forecast", f"{df_map['forecast_bookings'].max():.1f}")


    # 5E: quick summary of top regions
    if not df_map.empty:
        top = df_map.sort_values("forecast_bookings", ascending=False).head(3)
        st.markdown("### 🔥 Top forecasted demand (this week)")
        for i, row in enumerate(top.itertuples(index=False), start=1):
            st.write(f"{i}. **{row.region}** — {row.forecast_bookings:.1f} bookings")

    if df_map.empty:
        st.error("No forecast results were generated. Check region mappings and model availability.")
        if errors:
            with st.expander("Debug details"):
                st.write(errors)
        return

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        hover_name="region",
        hover_data={"forecast_bookings": ":.1f", "lat": False, "lon": False},
        size="forecast_bookings",
        color="forecast_bookings",
        size_max=40,
        zoom=4.6,
        center={"lat": 55.0, "lon": -3.5},
        title=f"Forecasted bookings (week starting {chosen_week.strftime('%Y-%m-%d')})",
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=50, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


    # Zoom map to UK & Ireland for clearer focus
    fig.update_geos(
        lataxis_range=[49.5, 59.7],   # south coast to north Scotland
        lonaxis_range=[-8.8, 2.2],    # west Ireland to east England
        showcountries=True,
        countrycolor="rgba(255,255,255,0.15)",
        showland=True,
        landcolor="rgba(255,255,255,0.04)",
    )

    st.caption(
        "Interpretation: larger/warmer markers indicate higher forecasted demand. "
        "This helps Operations quickly see where staffing pressure is highest for the selected week."
    )

    csv_bytes = df_map.sort_values("forecast_bookings", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download map data (CSV)",
        data=csv_bytes,
        file_name=f"forecast_map_{week_str}.csv",
        mime="text/csv",
    )


    # Non-blocking diagnostics
    if errors:
        with st.expander("Regions skipped (debug)"):
            st.write(errors)

