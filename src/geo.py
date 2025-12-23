# src/geo.py
"""
Geographic helpers for region-level mapping.

Lightwieght Addition
"""

from __future__ import annotations

from typing import Dict, Tuple


# Approximate region centroids (lat, lon) for visualisation purposes only.
# These are not used for navigation, only for presenting regional forecasts.
REGION_CENTROIDS: Dict[str, Tuple[float, float]] = {
    # England
    "lake_district": (54.4609, -3.0886),
    "peak_district": (53.3400, -1.7800),
    "yorkshire_dales": (54.2000, -2.2000),

    # Wales
    "snowdonia": (53.0685, -4.0763),

    # Scotland
    "highlands": (57.1200, -4.7100),
}


def get_region_latlon(region: str) -> Tuple[float, float]:
    """
    Return (lat, lon) for a given region key.

    Parameters
    ----------
    region : str
        Region name used across the dataset.

    Returns
    -------
    (lat, lon) : tuple[float, float]
    """
    key = (region or "").strip().lower()
    if key not in REGION_CENTROIDS:
        raise KeyError(
            f"Unknown region '{region}'. Add it to src/geo.py REGION_CENTROIDS."
        )
    return REGION_CENTROIDS[key]
