"""
UI package for SeeStar Photometry.
"""

from .app_helpers import (
    load_fits_preview,
    detect_candidate_stars,
    create_plotly_lightcurve,
    create_plotly_quality,
    compute_photometry_stats,
)

__all__ = [
    "load_fits_preview",
    "detect_candidate_stars",
    "create_plotly_lightcurve",
    "create_plotly_quality",
    "compute_photometry_stats",
]
