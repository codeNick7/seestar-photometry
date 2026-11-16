"""
SeeStar Photometry
==================

High-performance differential photometry pipeline optimized for large FITS
datasets (5-10+ GB) from the ZWO SeeStar S50 and smart telescopes.

Features:
- Memory-efficient processing (single-frame memory mapping, low RAM overhead)
- Hybrid WCS + Astroalign + Cross-correlation star matching
- SNR-optimized aperture photometry with annular sky background subtraction
- Sub-pixel centroid refinement
- Adaptive aperture sizing based on measured stellar FWHM
- Real-time variable star identification via AAVSO VSX & SIMBAD
- Moon glare & illumination impact tracking via Skyfield ephemeris
- Interactive Matplotlib GUI & modern Streamlit Web UI
"""

__version__ = "1.0.0"
__author__ = "COSMOS / SeeStar Photometry Contributors"

from .config import (
    PhotometryConfig,
    get_settings,
    get_preset_config,
    settings,
    PRESETS,
)

from .photometry_engine import (
    FITSDataset,
    FrameAligner,
    StarSelector,
    PhotometryEngine,
    ResultsExporter,
    main,
)

from .alignment import (
    EnhancedAligner,
    select_best_reference_frame,
)

from .enhanced_photometry import (
    EnhancedPhotometry,
    measure_differential_photometry,
    measure_fwhm_from_stars,
)

from .frame_quality import (
    FrameQualityAnalyzer,
    QualityConfig,
    QualityMetrics,
)

from .moon_glare import (
    MoonGlareAnalyzer,
    MoonGlareConfig,
    MoonMetrics,
)

from .star_identification import (
    check_if_variable_star,
    query_variable_stars_vsx,
    query_stars_around_field,
)

from .wcs_utils import (
    create_wcs_from_header,
    pixel_to_world,
    world_to_pixel,
    get_field_center,
)

__all__ = [
    # Metadata
    "__version__",
    # Configuration
    "PhotometryConfig",
    "get_settings",
    "get_preset_config",
    "settings",
    "PRESETS",
    # Pipeline components
    "FITSDataset",
    "FrameAligner",
    "StarSelector",
    "PhotometryEngine",
    "ResultsExporter",
    "main",
    # Alignment
    "EnhancedAligner",
    "select_best_reference_frame",
    # Photometry
    "EnhancedPhotometry",
    "measure_differential_photometry",
    "measure_fwhm_from_stars",
    # Quality & Moon Glare
    "FrameQualityAnalyzer",
    "QualityConfig",
    "QualityMetrics",
    "MoonGlareAnalyzer",
    "MoonGlareConfig",
    "MoonMetrics",
    # Star ID & WCS
    "check_if_variable_star",
    "query_variable_stars_vsx",
    "query_stars_around_field",
    "create_wcs_from_header",
    "pixel_to_world",
    "world_to_pixel",
    "get_field_center",
]
