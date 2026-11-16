"""
Test aperture photometry and differential magnitude logic.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seestar_photometry.enhanced_photometry import EnhancedPhotometry
from seestar_photometry.ui.app_helpers import compute_photometry_stats


def _make_star_field():
    """Create synthetic test image with known fluxes."""
    height, width = 150, 150
    img = np.full((height, width), 50.0, dtype=np.float32)  # sky = 50

    # Star 1 (Target): center at (50, 50), flux ~ 5000
    y, x = np.ogrid[:height, :width]
    s1 = 500.0 * np.exp(-((x - 50)**2 + (y - 50)**2) / (2 * 2.0**2))
    # Star 2 (Reference): center at (100, 100), flux ~ 2500
    s2 = 250.0 * np.exp(-((x - 100)**2 + (y - 100)**2) / (2 * 2.0**2))

    return img + s1 + s2


def test_aperture_photometry_measure():
    img = _make_star_field()
    ep = EnhancedPhotometry(
        aperture_radius=8.0,
        sky_annulus_inner=12.0,
        sky_annulus_outer=20.0,
        auto_aperture=False
    )

    # Measure target star
    t_res = ep.measure_star(img, 50.0, 50.0)
    assert bool(t_res["valid"]) is True
    t_flux = t_res["flux"]
    t_sky = t_res["sky"]
    assert t_flux > 0
    assert abs(t_sky - 50.0) < 5.0

    # Measure reference star
    r_res = ep.measure_star(img, 100.0, 100.0)
    assert bool(r_res["valid"]) is True
    r_flux = r_res["flux"]
    r_sky = r_res["sky"]
    assert r_flux > 0
    assert abs(r_sky - 50.0) < 5.0

    # Differential magnitude = -2.5 * log10(target / reference)
    diff_mag = -2.5 * np.log10(t_flux / r_flux)
    assert -1.2 < diff_mag < -0.3


def test_photometry_stats_summary():
    df = pd.DataFrame({
        "frame": [0, 1, 2, 3],
        "diff_magnitude": [-2.5, -2.52, -2.48, -2.51],
        "time": ["2026-01-01T00:00:00", "2026-01-01T00:01:00", "2026-01-01T00:02:00", "2026-01-01T00:03:00"]
    })
    stats = compute_photometry_stats(df)
    assert stats["total_frames"] == 4
    assert stats["valid_frames"] == 4
    assert stats["success_rate"] == 100.0
    assert abs(stats["mean_mag"] - (-2.5025)) < 0.01
    assert "EXCELLENT" in stats["grade"]
