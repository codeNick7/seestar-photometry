"""
Test that all modules and symbols in seestar_photometry can be imported cleanly.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_package_imports():
    import seestar_photometry
    assert hasattr(seestar_photometry, "__version__")
    assert seestar_photometry.__version__ == "1.0.0"


def test_core_classes_exported():
    from seestar_photometry import (
        FITSDataset,
        FrameAligner,
        StarSelector,
        PhotometryEngine,
        ResultsExporter,
        EnhancedAligner,
        EnhancedPhotometry,
        FrameQualityAnalyzer,
        MoonGlareAnalyzer,
        PhotometryConfig,
        get_settings,
    )
    assert FITSDataset is not None
    assert FrameAligner is not None
    assert PhotometryEngine is not None
    assert PhotometryConfig is not None


def test_ui_helpers_import():
    from seestar_photometry.ui import (
        load_fits_preview,
        detect_candidate_stars,
        compute_photometry_stats,
    )
    assert callable(load_fits_preview)
    assert callable(detect_candidate_stars)
    assert callable(compute_photometry_stats)
