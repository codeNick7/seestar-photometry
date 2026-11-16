"""
Test synthetic frame quality and star detection.
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seestar_photometry.frame_quality import (
    FrameQualityAnalyzer,
    QualityConfig,
)
from seestar_photometry.photometry_engine import reduce_brightness_for_saturation


def _make_synthetic_image(width=200, height=200, background=100.0, noise_std=5.0):
    """Generate synthetic astronomical image with background noise and stars."""
    np.random.seed(42)
    img = np.random.normal(background, noise_std, (height, width)).astype(np.float32)

    def add_star(x0, y0, amplitude, sigma=2.0):
        y, x = np.ogrid[:height, :width]
        star = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        return star

    img += add_star(50, 50, amplitude=1500.0)
    img += add_star(120, 80, amplitude=800.0)
    img += add_star(160, 150, amplitude=2200.0)
    return np.clip(img, 0, 65535).astype(np.uint16)


def test_frame_quality_analyzer():
    img = _make_synthetic_image()
    analyzer = FrameQualityAnalyzer(QualityConfig(saturation_threshold=0.95))
    metrics = analyzer.analyze_frame(img, header={})

    assert metrics is not None
    assert metrics.quality_score >= 0
    assert not metrics.is_saturated


def test_saturation_reduction():
    img = _make_synthetic_image()
    # Artificially saturate pixels (65000 in uint16 range)
    img[45:55, 45:55] = 65000.0

    analyzer = FrameQualityAnalyzer(QualityConfig())
    metrics = analyzer.analyze_frame(img, header={})
    assert metrics.is_saturated

    reduced_img, scale = reduce_brightness_for_saturation(img, target_saturation_threshold=0.90)
    assert scale < 1.0
    assert np.max(reduced_img) < 65000.0
