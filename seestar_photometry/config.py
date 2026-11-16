"""
Configuration module for SeeStar Photometry.
Provides default settings optimized for the ZWO SeeStar S50 smart telescope,
with support for environment variables, custom configuration, and presets.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class PhotometryConfig:
    """
    Configuration parameters for photometry processing.
    Defaults are optimized for ZWO SeeStar S50 (20mm aperture, 2-10 second exposures).
    """

    # Directories
    input_dir: str = field(
        default_factory=lambda: os.getenv("SEESTAR_INPUT_DIR", "data/input")
    )
    output_dir: str = field(
        default_factory=lambda: os.getenv("SEESTAR_OUTPUT_DIR", "data/output")
    )

    # Photometry Apertures (pixels)
    aperture_radius: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_APERTURE_RADIUS", "8"))
    )
    sky_annulus_inner: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_SKY_ANNULUS_INNER", "12"))
    )
    sky_annulus_outer: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_SKY_ANNULUS_OUTER", "20"))
    )

    # Display & Alignment
    downsample_factor: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_DOWNSAMPLE_FACTOR", "2"))
    )
    max_control_points: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_MAX_CONTROL_POINTS", "100"))
    )
    reference_frame_candidates: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_REF_FRAME_CANDIDATES", "20"))
    )

    # Export & Plotting
    plot_format: str = field(
        default_factory=lambda: os.getenv("SEESTAR_PLOT_FORMAT", "png")
    )
    plot_dpi: int = field(
        default_factory=lambda: int(os.getenv("SEESTAR_PLOT_DPI", "300"))
    )

    # Algorithmic enhancements
    auto_aperture: bool = field(
        default_factory=lambda: os.getenv("SEESTAR_AUTO_APERTURE", "true").lower() in ("true", "1", "yes")
    )
    refine_centroids: bool = field(
        default_factory=lambda: os.getenv("SEESTAR_REFINE_CENTROIDS", "true").lower() in ("true", "1", "yes")
    )
    detection_sigma: float = field(
        default_factory=lambda: float(os.getenv("SEESTAR_DETECTION_SIGMA", "3.0"))
    )
    pixel_scale: float = field(
        default_factory=lambda: float(os.getenv("SEESTAR_PIXEL_SCALE", "1.21"))
    )

    # Observer location (optional, for moon glare calculation)
    observer_lat: float = field(
        default_factory=lambda: float(os.getenv("SEESTAR_OBSERVER_LAT", "0.0"))
    )
    observer_lon: float = field(
        default_factory=lambda: float(os.getenv("SEESTAR_OBSERVER_LON", "0.0"))
    )

    # Backwards-compatibility aliases with COSMOS settings
    @property
    def photometry_input_dir(self) -> str:
        return self.input_dir

    @property
    def photometry_output_dir(self) -> str:
        return self.output_dir

    @property
    def photometry_aperture_radius(self) -> int:
        return self.aperture_radius

    @property
    def photometry_sky_annulus_inner(self) -> int:
        return self.sky_annulus_inner

    @property
    def photometry_sky_annulus_outer(self) -> int:
        return self.sky_annulus_outer

    @property
    def photometry_downsample_factor(self) -> int:
        return self.downsample_factor

    @property
    def photometry_max_control_points(self) -> int:
        return self.max_control_points

    @property
    def photometry_reference_frame_candidates(self) -> int:
        return self.reference_frame_candidates

    @property
    def photometry_plot_format(self) -> str:
        return self.plot_format

    @property
    def photometry_plot_dpi(self) -> int:
        return self.plot_dpi

    @property
    def photometry_auto_aperture(self) -> bool:
        return self.auto_aperture

    @property
    def photometry_refine_centroids(self) -> bool:
        return self.refine_centroids

    @property
    def photometry_detection_sigma(self) -> float:
        return self.detection_sigma

    @property
    def photometry_pixel_scale(self) -> float:
        return self.pixel_scale

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "aperture_radius": self.aperture_radius,
            "sky_annulus_inner": self.sky_annulus_inner,
            "sky_annulus_outer": self.sky_annulus_outer,
            "downsample_factor": self.downsample_factor,
            "max_control_points": self.max_control_points,
            "reference_frame_candidates": self.reference_frame_candidates,
            "plot_format": self.plot_format,
            "plot_dpi": self.plot_dpi,
            "auto_aperture": self.auto_aperture,
            "refine_centroids": self.refine_centroids,
            "detection_sigma": self.detection_sigma,
            "pixel_scale": self.pixel_scale,
            "observer_lat": self.observer_lat,
            "observer_lon": self.observer_lon,
        }


# Instrument Presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "seestar_s50": {
        "aperture_radius": 8,
        "sky_annulus_inner": 12,
        "sky_annulus_outer": 20,
        "pixel_scale": 1.21,
        "detection_sigma": 3.0,
        "auto_aperture": True,
        "refine_centroids": True,
    },
    "dwarf_ii": {
        "aperture_radius": 6,
        "sky_annulus_inner": 10,
        "sky_annulus_outer": 18,
        "pixel_scale": 1.45,
        "detection_sigma": 3.0,
        "auto_aperture": True,
        "refine_centroids": True,
    },
    "vespera": {
        "aperture_radius": 10,
        "sky_annulus_inner": 16,
        "sky_annulus_outer": 26,
        "pixel_scale": 1.15,
        "detection_sigma": 3.5,
        "auto_aperture": True,
        "refine_centroids": True,
    },
    "general": {
        "aperture_radius": 10,
        "sky_annulus_inner": 15,
        "sky_annulus_outer": 25,
        "pixel_scale": 1.0,
        "detection_sigma": 3.0,
        "auto_aperture": True,
        "refine_centroids": True,
    },
}


def get_preset_config(preset_name: str = "seestar_s50", **overrides) -> PhotometryConfig:
    """Create a PhotometryConfig from a known preset with optional overrides."""
    preset = PRESETS.get(preset_name.lower(), PRESETS["seestar_s50"]).copy()
    preset.update(overrides)
    return PhotometryConfig(**preset)


# Global default configuration instance
settings = PhotometryConfig()


def get_settings() -> PhotometryConfig:
    """Return the global default configuration instance."""
    return settings
