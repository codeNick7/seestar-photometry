"""
Test configuration defaults, presets, and backwards compatibility.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seestar_photometry.config import (
    PhotometryConfig,
    get_settings,
    get_preset_config,
    PRESETS,
)


def test_default_config():
    cfg = PhotometryConfig()
    assert cfg.aperture_radius == 8
    assert cfg.sky_annulus_inner == 12
    assert cfg.sky_annulus_outer == 20
    assert cfg.pixel_scale == 1.21
    assert cfg.auto_aperture is True
    assert cfg.refine_centroids is True


def test_backwards_compatibility_aliases():
    cfg = PhotometryConfig()
    assert cfg.photometry_input_dir == cfg.input_dir
    assert cfg.photometry_output_dir == cfg.output_dir
    assert cfg.photometry_aperture_radius == cfg.aperture_radius
    assert cfg.photometry_pixel_scale == cfg.pixel_scale


def test_presets():
    assert "seestar_s50" in PRESETS
    assert "dwarf_ii" in PRESETS
    assert "vespera" in PRESETS

    seestar_cfg = get_preset_config("seestar_s50")
    assert seestar_cfg.aperture_radius == 8
    assert seestar_cfg.pixel_scale == 1.21

    dwarf_cfg = get_preset_config("dwarf_ii")
    assert dwarf_cfg.aperture_radius == 6
    assert dwarf_cfg.pixel_scale == 1.45


def test_custom_overrides():
    custom_cfg = get_preset_config("seestar_s50", aperture_radius=15, pixel_scale=1.5)
    assert custom_cfg.aperture_radius == 15
    assert custom_cfg.pixel_scale == 1.5
