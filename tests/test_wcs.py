"""
Test WCS creation and coordinate utilities.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seestar_photometry.wcs_utils import (
    create_wcs_from_header,
    pixel_to_world,
    world_to_pixel,
    get_field_center,
)


def test_wcs_creation():
    header = {
        "RA": 47.042218,
        "DEC": 40.955648,
        "NAXIS1": 1920,
        "NAXIS2": 1080,
        "FOCALLEN": 250.0,
        "XPIXSZ": 2.9,
    }
    image_shape = (1080, 1920)
    wcs = create_wcs_from_header(header, image_shape)
    assert wcs is not None

    center_coords = get_field_center(header)
    assert center_coords is not None
    center_ra, center_dec = center_coords
    assert abs(center_ra - 47.0422) < 0.1
    assert abs(center_dec - 40.9556) < 0.1


def test_pixel_to_world_roundtrip():
    header = {
        "RA": 80.0,
        "DEC": 20.0,
        "NAXIS1": 1000,
        "NAXIS2": 1000,
        "SCALE": 1.21,
    }
    wcs = create_wcs_from_header(header, (1000, 1000))
    assert wcs is not None

    ra, dec = pixel_to_world(wcs, 500.0, 500.0)
    assert ra is not None
    assert abs(ra - 80.0) < 0.05
    assert abs(dec - 20.0) < 0.05

    px, py = world_to_pixel(wcs, ra, dec)
    assert abs(px - 500.0) < 0.1
    assert abs(py - 500.0) < 0.1
