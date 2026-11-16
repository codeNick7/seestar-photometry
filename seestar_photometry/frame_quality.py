"""
Frame Quality Analysis for Photometry
Detects cosmic rays, saturation, clouds, and measures image quality metrics
"""

import numpy as np
from scipy.ndimage import median_filter, binary_opening
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def create_dao_finder(fwhm: float = 3.5, threshold: float = 3.0,
                      brightest: Optional[int] = None,
                      peakmax: Optional[float] = None) -> DAOStarFinder:
    """
    Create DAOStarFinder with parameters optimized for low SNR images.

    Uses permissive sharpness and roundness parameters and lower threshold
    to handle faint stars from small aperture telescopes (e.g., SeeStar S50
    with 20mm aperture and 2-second exposures).

    Args:
        fwhm: Full width at half maximum estimate (default: 3.5 for slightly defocused/undersampled)
        threshold: Detection threshold in sigma above background (default: 3.0 for faint stars)
        brightest: Maximum number of brightest sources to return
        peakmax: Maximum peak value (None = no limit)

    Returns:
        Configured DAOStarFinder instance
    """
    return DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold,
        brightest=brightest,
        peakmax=peakmax,
        sharplo=0.05,  # Very relaxed for faint/undersampled stars
        sharphi=2.0,   # Allow broader range of PSF shapes
        roundlo=-2.0,  # Allow elongated stars (tracking errors)
        roundhi=2.0,   # Allow elongated stars
        exclude_border=True  # Exclude stars near edges
    )


@dataclass
class QualityConfig:
    """Configuration for frame quality analysis"""
    cosmic_ray_sigma: float = 5.0
    cosmic_ray_kernel: int = 5
    saturation_threshold: float = 0.95
    saturation_flag_fraction: float = 0.001
    cloud_fwhm_threshold: float = 1.5
    min_stars_for_fwhm: int = 3


@dataclass
class QualityMetrics:
    """Quality metrics for a single frame"""
    quality_score: float = 0.0          # 0-100 overall score
    cloud_score: float = 0.0            # 0-100 (0=clear, 100=cloudy)
    is_saturated: bool = False
    saturation_fraction: float = 0.0
    n_cosmic_rays: int = 0
    cosmic_ray_mask: Optional[np.ndarray] = None  # Boolean mask
    fwhm: Optional[float] = None        # Seeing quality (pixels)
    moon_distance_deg: Optional[float] = None  # Filled by moon analyzer


class FrameQualityAnalyzer:
    """
    Analyzes individual frames for quality metrics

    Features:
    - Cosmic ray detection and masking
    - Saturation detection
    - Cloud scoring based on FWHM and background
    - Overall quality scoring
    """

    def __init__(self, config: QualityConfig = None):
        """
        Initialize quality analyzer

        Args:
            config: Quality analysis configuration
        """
        self.config = config or QualityConfig()

        # Baseline metrics from reference frame (set during first analysis)
        self.baseline_variance = None
        self.baseline_fwhm = None
        self.baseline_n_stars = None

    def analyze_frame(self, image: np.ndarray, header=None) -> QualityMetrics:
        """
        Perform complete quality analysis on a frame

        Args:
            image: 2D numpy array of image data
            header: FITS header (optional)

        Returns:
            QualityMetrics object with all quality measurements
        """
        metrics = QualityMetrics()

        try:
            # Cosmic ray detection
            metrics.cosmic_ray_mask, metrics.n_cosmic_rays = self.detect_cosmic_rays(image)

            # Saturation detection
            metrics.is_saturated, metrics.saturation_fraction = self.detect_saturation(image)

            # FWHM measurement
            metrics.fwhm = self.measure_fwhm(image)

            # Cloud scoring (requires baseline from reference frame)
            if self.baseline_variance is None:
                # This is the reference frame - establish baseline
                self._establish_baseline(image, metrics.fwhm)
                metrics.cloud_score = 0.0  # Reference frame is "clear" by definition
            else:
                # Compare to baseline
                metrics.cloud_score = self.compute_cloud_score(image, metrics.fwhm)

            # Overall quality score (weighted composite)
            metrics.quality_score = self._compute_overall_quality(metrics)

        except Exception as e:
            logger.warning(f"Error in quality analysis: {e}")
            # Return default metrics on error
            pass

        return metrics

    def detect_cosmic_rays(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect cosmic ray hits using median filter and sigma clipping

        Algorithm:
        1. Apply median filter to smooth image
        2. Subtract to isolate outliers
        3. Flag pixels > N*sigma above median
        4. Morphological cleanup

        Args:
            image: 2D numpy array

        Returns:
            mask: Boolean array (True = cosmic ray)
            count: Number of cosmic ray pixels detected
        """
        try:
            # Convert to float to avoid overflow
            image_float = image.astype(np.float64)

            # Apply median filter
            kernel_size = self.config.cosmic_ray_kernel
            median_filtered = median_filter(image_float, size=kernel_size)

            # Difference image (highlights outliers)
            difference = image_float - median_filtered

            # Sigma clipping to get robust statistics
            mean, median, std = sigma_clipped_stats(difference, sigma=3.0)

            # Threshold for cosmic ray detection
            threshold = mean + self.config.cosmic_ray_sigma * std

            # Create mask
            cosmic_ray_mask = difference > threshold

            # Morphological cleanup (remove single-pixel noise)
            # Binary opening removes small isolated pixels
            cosmic_ray_mask = binary_opening(cosmic_ray_mask, structure=np.ones((2, 2)))

            count = np.sum(cosmic_ray_mask)

            return cosmic_ray_mask, int(count)

        except Exception as e:
            logger.error(f"Error detecting cosmic rays: {e}")
            # Return empty mask on error
            return np.zeros(image.shape, dtype=bool), 0

    def detect_saturation(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect saturated pixels in image

        Args:
            image: 2D numpy array

        Returns:
            is_saturated: True if significant saturation detected
            fraction: Fraction of saturated pixels (0.0-1.0)
        """
        try:
            # Determine max value from dtype
            if image.dtype == np.uint16:
                max_value = 65535
            elif image.dtype == np.uint8:
                max_value = 255
            elif image.dtype in [np.float32, np.float64]:
                # For float images, use 99.9th percentile as "max"
                max_value = np.percentile(image, 99.9)
            else:
                # Default fallback
                max_value = np.max(image)

            # Saturation threshold (95% of max value)
            saturation_threshold = self.config.saturation_threshold * max_value

            # Count saturated pixels
            saturated_pixels = np.sum(image >= saturation_threshold)
            total_pixels = image.size
            fraction = saturated_pixels / total_pixels

            # Flag if more than threshold fraction is saturated
            is_saturated = fraction > self.config.saturation_flag_fraction

            return is_saturated, fraction

        except Exception as e:
            logger.error(f"Error detecting saturation: {e}")
            return False, 0.0

    def measure_fwhm(self, image: np.ndarray) -> Optional[float]:
        """
        Measure Full Width at Half Maximum (FWHM) as seeing quality metric

        Uses DAOStarFinder to detect stars and measures FWHM using second moments.
        Optimized for low SNR images from small aperture telescopes.

        Args:
            image: 2D numpy array

        Returns:
            fwhm: Average FWHM in pixels, or None if measurement fails
        """
        try:
            # Get robust background statistics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)

            # Detect stars with lower threshold for faint stars
            daofind = create_dao_finder(
                fwhm=3.5,  # Slightly larger for undersampled stars
                threshold=3.0 * std,  # 3-sigma detection for faint stars
                brightest=15,  # Use more stars for better statistics
                peakmax=None  # No upper limit
            )

            sources = daofind(image - median)

            if sources is None or len(sources) < self.config.min_stars_for_fwhm:
                return None

            # Measure FWHM for each star using second moments
            fwhm_values = []
            box_size = 11
            half_box = box_size // 2

            x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
            y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
            for source in sources[:15]:  # Use up to 15 brightest
                x = int(round(source[x_col]))
                y = int(round(source[y_col]))

                # Check bounds
                if (x - half_box < 0 or x + half_box >= image.shape[1] or
                    y - half_box < 0 or y + half_box >= image.shape[0]):
                    continue

                # Extract cutout
                cutout = image[y - half_box:y + half_box + 1,
                              x - half_box:x + half_box + 1] - median

                # Clip negative values
                cutout = np.clip(cutout, 0, None)

                # Calculate second moments
                total = np.sum(cutout)
                if total <= 0:
                    continue

                y_grid, x_grid = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
                x_mean = np.sum(x_grid * cutout) / total
                y_mean = np.sum(y_grid * cutout) / total

                x_var = np.sum((x_grid - x_mean)**2 * cutout) / total
                y_var = np.sum((y_grid - y_mean)**2 * cutout) / total

                # Calculate FWHM from variance (FWHM = 2.355 * sigma)
                sigma = np.sqrt((x_var + y_var) / 2)
                fwhm = 2.355 * sigma

                # Sanity check: FWHM should be reasonable
                if 1.5 < fwhm < 12.0:
                    fwhm_values.append(fwhm)

            if not fwhm_values:
                return None

            # Use median for robustness
            avg_fwhm = np.median(fwhm_values)
            return float(avg_fwhm)

        except Exception as e:
            logger.debug(f"Error measuring FWHM: {e}")
            return None

    def compute_cloud_score(self, image: np.ndarray, fwhm: Optional[float]) -> float:
        """
        Compute cloud contamination score

        Combines multiple metrics:
        - Sky background variance (clouds reduce contrast)
        - FWHM degradation (clouds blur stars)
        - Star count reduction (clouds hide faint stars)

        Args:
            image: 2D numpy array
            fwhm: Measured FWHM for this frame

        Returns:
            cloud_score: 0-100 (0=clear, 100=cloudy)
        """
        try:
            # Get current frame metrics
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)
            variance = std ** 2

            # Count stars with relaxed quality criteria
            daofind = create_dao_finder(
                fwhm=3.0, 
                threshold=5.0 * std, 
                brightest=50
            )
            sources = daofind(image - median)
            n_stars = len(sources) if sources is not None else 0

            # Compare to baseline
            scores = []

            # Variance score (lower variance = clouds)
            if self.baseline_variance and self.baseline_variance > 0:
                variance_ratio = variance / self.baseline_variance
                variance_score = max(0, 100 * (1 - variance_ratio))
                scores.append((variance_score, 0.3))  # 30% weight

            # FWHM score (larger FWHM = worse seeing = possible clouds)
            if fwhm and self.baseline_fwhm and self.baseline_fwhm > 0:
                fwhm_ratio = fwhm / self.baseline_fwhm
                fwhm_score = 100 * max(0, fwhm_ratio - 1.0)  # >1.0 = degraded
                # Cap at threshold
                fwhm_score = min(fwhm_score, 100)
                scores.append((fwhm_score, 0.4))  # 40% weight

            # Star count score (fewer stars = clouds hiding them)
            if self.baseline_n_stars and self.baseline_n_stars > 0:
                star_ratio = n_stars / self.baseline_n_stars
                star_score = 100 * max(0, 1.0 - star_ratio)
                scores.append((star_score, 0.3))  # 30% weight

            # Weighted average
            if scores:
                total_weight = sum(w for _, w in scores)
                cloud_score = sum(s * w for s, w in scores) / total_weight
            else:
                cloud_score = 0.0

            return np.clip(cloud_score, 0, 100)

        except Exception as e:
            logger.debug(f"Error computing cloud score: {e}")
            return 0.0

    def _establish_baseline(self, image: np.ndarray, fwhm: Optional[float]):
        """
        Establish baseline metrics from reference frame

        Args:
            image: Reference frame image
            fwhm: Measured FWHM
        """
        try:
            # Background variance
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)
            self.baseline_variance = std ** 2

            # FWHM
            self.baseline_fwhm = fwhm if fwhm else 3.0

            # Star count
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std, brightest=50)
            sources = daofind(image - median)
            self.baseline_n_stars = len(sources) if sources is not None else 10

            logger.debug(f"Established quality baselines: variance={self.baseline_variance:.1f}, "
                        f"FWHM={self.baseline_fwhm:.2f}, n_stars={self.baseline_n_stars}")

        except Exception as e:
            logger.warning(f"Error establishing baseline: {e}")
            # Use defaults
            self.baseline_variance = 1000.0
            self.baseline_fwhm = 3.0
            self.baseline_n_stars = 10

    def _compute_overall_quality(self, metrics: QualityMetrics) -> float:
        """
        Compute overall quality score from individual metrics

        Args:
            metrics: Quality metrics object

        Returns:
            overall_score: 0-100 (higher = better quality)
        """
        score = 100.0

        # Penalize saturation
        if metrics.is_saturated:
            saturation_penalty = min(50, metrics.saturation_fraction * 1000)
            score -= saturation_penalty

        # Penalize clouds
        score -= metrics.cloud_score * 0.5  # Cloud score already 0-100

        # Penalize cosmic rays (moderate penalty)
        cosmic_ray_penalty = min(20, metrics.n_cosmic_rays * 0.1)
        score -= cosmic_ray_penalty

        # Clip to valid range
        return np.clip(score, 0, 100)
