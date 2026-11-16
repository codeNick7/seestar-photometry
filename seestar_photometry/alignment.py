"""
Enhanced Alignment Module for Photometry Pipeline

Provides robust frame alignment using multiple strategies:
1. Cross-correlation for initial shift detection
2. Centroid refinement using center of mass (robust for low SNR)
3. Sub-pixel accurate transformations
4. Optional WCS-based alignment when headers are available

Optimized for SeeStar S50 telescope with:
- Low SNR images (20mm aperture, 2-second exposures)
- Variable tracking quality
- Potential field rotation
"""

import sys
from pathlib import Path

# Enable relative imports when executed directly
if __package__ is None or __package__ == "":
    pkg_dir = Path(__file__).resolve().parent
    if str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))
    __package__ = "seestar_photometry"

import warnings
import numpy as np
from scipy import ndimage
from skimage.transform import AffineTransform
from skimage.registration import phase_cross_correlation
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from typing import Tuple, Optional, List, Dict
import logging

# Suppress fitting warnings
warnings.filterwarnings('ignore', message='.*fit may not have converged.*')
warnings.filterwarnings('ignore', category=UserWarning, module='photutils')
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function for centroid fitting"""
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    return offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))


class EnhancedAligner:
    """
    Enhanced frame alignment with multiple strategies for robustness.

    Designed to handle:
    - Low SNR images from small aperture telescopes
    - Field rotation and dithering
    - Variable tracking quality
    """

    def __init__(self, detection_sigma: float = 3.0, min_stars: int = 5):
        """
        Initialize enhanced aligner.

        Args:
            detection_sigma: Star detection threshold in sigma above background
            min_stars: Minimum stars required for star-based alignment
        """
        self.detection_sigma = detection_sigma
        self.min_stars = min_stars
        self.reference_image = None
        self.reference_stars = None
        self.reference_shape = None

    def set_reference(self, image: np.ndarray) -> int:
        """
        Set reference image and detect stars.

        Args:
            image: Reference image array

        Returns:
            Number of stars detected in reference
        """
        self.reference_image = image.astype(np.float64)
        self.reference_shape = image.shape
        self.reference_stars = self._detect_stars(image)

        n_stars = len(self.reference_stars) if self.reference_stars is not None else 0
        logger.info(f"Reference frame set with {n_stars} detected stars")
        return n_stars

    def _detect_stars(self, image: np.ndarray, max_stars: int = 100) -> Optional[np.ndarray]:
        """
        Detect stars with parameters optimized for low SNR images.

        Uses lower threshold and relaxed shape parameters for faint stars.

        Args:
            image: Image to detect stars in
            max_stars: Maximum number of stars to return

        Returns:
            Array of (x, y, flux) for each detected star, or None if detection fails
        """
        try:
            # Robust background estimation
            mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)

            # Use lower threshold for faint stars (SeeStar S50 specific)
            threshold = self.detection_sigma * std

            # Relaxed DAOStarFinder parameters for low SNR
            daofind = DAOStarFinder(
                fwhm=3.5,  # Slightly larger for undersampled stars
                threshold=threshold,
                brightest=max_stars,
                sharplo=0.02,  # Very relaxed for faint stars
                sharphi=3.0,   # Allow broader range
                roundlo=-3.0,  # Allow very elongated
                roundhi=3.0,
                exclude_border=True
            )

            sources = daofind(image - median)

            if sources is None or len(sources) < self.min_stars:
                # Try with even lower threshold
                daofind2 = DAOStarFinder(
                    fwhm=4.0,
                    threshold=2.0 * std,  # Even lower
                    brightest=max_stars,
                    sharplo=0.01,
                    sharphi=5.0,
                    roundlo=-5.0,
                    roundhi=5.0,
                    exclude_border=True
                )
                sources = daofind2(image - median)

                if sources is None or len(sources) < self.min_stars:
                    return None

            # Skip centroid refinement for speed - DAOStarFinder already provides good positions
            positions = []
            x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
            y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
            for source in sources:
                x, y = source[x_col], source[y_col]
                flux = source['flux']
                positions.append([x, y, flux])

            return np.array(positions)

        except Exception as e:
            logger.error(f"Star detection failed: {e}")
            return None

    def _refine_centroid(self, image: np.ndarray, x: float, y: float,
                         background: float, box_size: int = 11) -> Tuple[Optional[float], Optional[float]]:
        """
        Refine centroid position using center of mass (robust for low SNR).

        Args:
            image: Image array
            x, y: Initial centroid position
            background: Background level
            box_size: Size of box to fit (should be odd)

        Returns:
            Refined (x, y) position or (None, None) if fitting fails
        """
        try:
            half_box = box_size // 2

            # Extract cutout
            x_int, y_int = int(round(x)), int(round(y))

            # Check bounds
            if (x_int - half_box < 0 or x_int + half_box >= image.shape[1] or
                y_int - half_box < 0 or y_int + half_box >= image.shape[0]):
                return None, None

            cutout = image[y_int - half_box:y_int + half_box + 1,
                          x_int - half_box:x_int + half_box + 1] - background

            # Clip negative values
            cutout = np.clip(cutout, 0, None)

            # Check if there's enough signal
            total_flux = np.sum(cutout)
            if total_flux <= 0:
                return None, None

            # Use center of mass (fast and robust)
            y_grid, x_grid = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
            x_cen = np.sum(x_grid * cutout) / total_flux
            y_cen = np.sum(y_grid * cutout) / total_flux

            # Convert back to image coordinates
            x_refined = x_int - half_box + x_cen
            y_refined = y_int - half_box + y_cen

            # Sanity check - refined position should be close to original
            if abs(x_refined - x) > 3 or abs(y_refined - y) > 3:
                return None, None

            return x_refined, y_refined

        except Exception as e:
            logger.debug(f"Centroid refinement failed: {e}")
            return None, None

    def compute_alignment(self, target_image: np.ndarray,
                          method: str = 'auto') -> Dict:
        """
        Compute alignment transformation from target to reference frame.

        Args:
            target_image: Image to align
            method: Alignment method ('auto', 'cross_correlation', 'star_matching', 'hybrid')

        Returns:
            Dictionary with transformation matrix and metadata
        """
        if self.reference_image is None:
            raise ValueError("Reference image not set")

        target = target_image.astype(np.float64)

        result = {
            'matrix': np.eye(3),
            'aligned': False,
            'method': None,
            'shift_x': 0.0,
            'shift_y': 0.0,
            'rotation_deg': 0.0,
            'scale': 1.0,
            'n_matched_stars': 0,
            'rms_error': None
        }

        if method == 'auto':
            # Try methods in order of preference
            methods_to_try = ['hybrid', 'star_matching', 'cross_correlation']
        else:
            methods_to_try = [method]

        for m in methods_to_try:
            try:
                if m == 'cross_correlation':
                    result = self._align_cross_correlation(target)
                elif m == 'star_matching':
                    result = self._align_star_matching(target)
                elif m == 'hybrid':
                    result = self._align_hybrid(target)

                if result['aligned']:
                    logger.debug(f"Alignment successful using {m}")
                    break

            except Exception as e:
                logger.debug(f"Alignment method {m} failed: {e}")
                continue

        return result

    def _align_cross_correlation(self, target: np.ndarray) -> Dict:
        """
        Align using phase cross-correlation (fast, handles shifts well).

        Args:
            target: Target image

        Returns:
            Alignment result dictionary
        """
        try:
            # Use phase cross-correlation for sub-pixel shift detection
            shift, error, diffphase = phase_cross_correlation(
                self.reference_image,
                target,
                upsample_factor=10  # 0.1 pixel precision
            )

            # shift is (row_shift, col_shift) = (y, x)
            # This is how much to shift TARGET to align with REFERENCE
            shift_y, shift_x = shift

            # Create transformation matrix that maps TARGET coords to REFERENCE coords
            # (consistent with star_matching and astroalign conventions)
            # If target point + shift = reference point, then:
            # [[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]] @ [target_x, target_y, 1] = [ref_x, ref_y, 1]
            matrix = np.array([
                [1, 0, shift_x],
                [0, 1, shift_y],
                [0, 0, 1]
            ])

            return {
                'matrix': matrix,
                'aligned': True,
                'method': 'cross_correlation',
                'shift_x': shift_x,
                'shift_y': shift_y,
                'rotation_deg': 0.0,
                'scale': 1.0,
                'n_matched_stars': 0,
                'rms_error': error
            }

        except Exception as e:
            logger.warning(f"Cross-correlation alignment failed: {e}")
            return {'matrix': np.eye(3), 'aligned': False, 'method': 'cross_correlation'}

    def _align_star_matching(self, target: np.ndarray) -> Dict:
        """
        Align using matched star positions (handles rotation/scale).

        Args:
            target: Target image

        Returns:
            Alignment result dictionary
        """
        if self.reference_stars is None or len(self.reference_stars) < self.min_stars:
            return {'matrix': np.eye(3), 'aligned': False, 'method': 'star_matching'}

        try:
            # Detect stars in target
            target_stars = self._detect_stars(target)

            if target_stars is None or len(target_stars) < self.min_stars:
                return {'matrix': np.eye(3), 'aligned': False, 'method': 'star_matching'}

            # Match stars between reference and target
            matches = self._match_stars(self.reference_stars, target_stars)

            if len(matches) < self.min_stars:
                logger.warning(f"Only {len(matches)} star matches found")
                return {'matrix': np.eye(3), 'aligned': False, 'method': 'star_matching'}

            # Compute transformation from matched pairs
            ref_pts = np.array([[m[0], m[1]] for m in matches])
            tgt_pts = np.array([[m[2], m[3]] for m in matches])

            # Fit affine transformation
            tform = AffineTransform()
            success = tform.estimate(tgt_pts, ref_pts)

            if not success:
                return {'matrix': np.eye(3), 'aligned': False, 'method': 'star_matching'}

            # Calculate residuals
            transformed = tform(tgt_pts)
            residuals = np.sqrt(np.sum((transformed - ref_pts)**2, axis=1))
            rms_error = np.sqrt(np.mean(residuals**2))

            # Extract transformation parameters
            rotation = np.degrees(np.arctan2(tform.params[1, 0], tform.params[0, 0]))
            scale = np.sqrt(tform.params[0, 0]**2 + tform.params[1, 0]**2)

            return {
                'matrix': tform.params,
                'aligned': True,
                'method': 'star_matching',
                'shift_x': tform.params[0, 2],
                'shift_y': tform.params[1, 2],
                'rotation_deg': rotation,
                'scale': scale,
                'n_matched_stars': len(matches),
                'rms_error': rms_error
            }

        except Exception as e:
            logger.warning(f"Star matching alignment failed: {e}")
            return {'matrix': np.eye(3), 'aligned': False, 'method': 'star_matching'}

    def _align_hybrid(self, target: np.ndarray) -> Dict:
        """
        Hybrid alignment: cross-correlation for initial shift, then star matching for refinement.

        Best for SeeStar S50 data with potential large shifts + rotation.

        Args:
            target: Target image

        Returns:
            Alignment result dictionary
        """
        try:
            # Step 1: Get initial shift using cross-correlation
            cc_result = self._align_cross_correlation(target)

            if not cc_result['aligned']:
                # Fall back to star matching only
                return self._align_star_matching(target)

            # Step 2: Apply initial shift to target
            shift_x = cc_result['shift_x']
            shift_y = cc_result['shift_y']

            # Create shifted version of target
            shifted_target = ndimage.shift(target, [shift_y, shift_x], order=3, mode='constant')

            # Step 3: Detect stars in pre-aligned image
            target_stars = self._detect_stars(shifted_target)

            if target_stars is None or len(target_stars) < self.min_stars:
                # Return cross-correlation result
                return cc_result

            # Step 4: Match stars for fine alignment (rotation/scale)
            if self.reference_stars is None:
                return cc_result

            matches = self._match_stars(self.reference_stars, target_stars)

            if len(matches) < self.min_stars:
                return cc_result

            # Compute fine transformation
            ref_pts = np.array([[m[0], m[1]] for m in matches])
            tgt_pts = np.array([[m[2], m[3]] for m in matches])

            tform_fine = AffineTransform()
            success = tform_fine.estimate(tgt_pts, ref_pts)

            if not success:
                return cc_result

            # Combine transformations: T_total = T_fine * T_shift
            shift_matrix = np.array([
                [1, 0, shift_x],
                [0, 1, shift_y],
                [0, 0, 1]
            ])

            combined_matrix = tform_fine.params @ shift_matrix

            # Calculate residuals
            # tgt_pts are already in the shifted image coordinate system,
            # so we just apply the fine transform to compare with reference
            transformed = tform_fine(tgt_pts)
            residuals = np.sqrt(np.sum((transformed - ref_pts)**2, axis=1))
            rms_error = np.sqrt(np.mean(residuals**2))

            # Extract combined transformation parameters
            rotation = np.degrees(np.arctan2(combined_matrix[1, 0], combined_matrix[0, 0]))
            scale = np.sqrt(combined_matrix[0, 0]**2 + combined_matrix[1, 0]**2)

            return {
                'matrix': combined_matrix,
                'aligned': True,
                'method': 'hybrid',
                'shift_x': combined_matrix[0, 2],
                'shift_y': combined_matrix[1, 2],
                'rotation_deg': rotation,
                'scale': scale,
                'n_matched_stars': len(matches),
                'rms_error': rms_error
            }

        except Exception as e:
            logger.warning(f"Hybrid alignment failed: {e}")
            # Fall back to simpler methods
            return self._align_star_matching(target)

    def _match_stars(self, ref_stars: np.ndarray, tgt_stars: np.ndarray,
                     max_distance: float = 10.0) -> List[Tuple[float, float, float, float]]:
        """
        Match stars between reference and target using nearest neighbor with brightness weighting.

        Args:
            ref_stars: Reference star positions (x, y, flux)
            tgt_stars: Target star positions (x, y, flux)
            max_distance: Maximum matching distance in pixels

        Returns:
            List of (ref_x, ref_y, tgt_x, tgt_y) matched pairs
        """
        matches = []

        # Sort by flux (brightest first)
        ref_sorted = ref_stars[np.argsort(-ref_stars[:, 2])]
        tgt_sorted = tgt_stars[np.argsort(-tgt_stars[:, 2])]

        used_tgt = set()

        for ref_star in ref_sorted:
            ref_x, ref_y, ref_flux = ref_star

            best_match = None
            best_dist = max_distance
            best_idx = -1

            for i, tgt_star in enumerate(tgt_sorted):
                if i in used_tgt:
                    continue

                tgt_x, tgt_y, tgt_flux = tgt_star

                # Distance criterion
                dist = np.sqrt((ref_x - tgt_x)**2 + (ref_y - tgt_y)**2)

                if dist < best_dist:
                    # Also check flux ratio (should be similar within factor of 3)
                    flux_ratio = max(ref_flux, tgt_flux) / max(min(ref_flux, tgt_flux), 1e-10)

                    if flux_ratio < 3.0:
                        best_dist = dist
                        best_match = (ref_x, ref_y, tgt_x, tgt_y)
                        best_idx = i

            if best_match is not None:
                matches.append(best_match)
                used_tgt.add(best_idx)

        return matches

    def transform_coordinates(self, x: float, y: float, transform_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Transform coordinates from reference frame to target frame.

        Args:
            x, y: Position in reference frame
            transform_matrix: 3x3 affine transformation matrix

        Returns:
            Transformed (x, y) position in target frame
        """
        try:
            # For coordinate transformation from reference to target,
            # we need the inverse transformation
            inv_matrix = np.linalg.inv(transform_matrix)

            coords = np.array([x, y, 1.0])
            transformed = inv_matrix @ coords

            return transformed[0], transformed[1]

        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, returning original coordinates")
            return x, y


def select_best_reference_frame(dataset, n_candidates: int = 20) -> Tuple[int, Dict]:
    """
    Select the best reference frame from the dataset.

    Criteria:
    1. Number of detected stars (more = better tracking)
    2. Star brightness / SNR (brighter = better for matching)
    3. FWHM / sharpness (smaller = better seeing)
    4. Low saturation

    Args:
        dataset: FITSDataset object
        n_candidates: Number of frames to evaluate

    Returns:
        (best_index, evaluation_metrics)
    """
    from .frame_quality import FrameQualityAnalyzer, QualityConfig

    n_candidates = min(n_candidates, len(dataset))
    evaluations = []

    quality_analyzer = FrameQualityAnalyzer(QualityConfig())

    print(f"\nEvaluating first {n_candidates} frames for best reference...")

    for i in range(n_candidates):
        try:
            image, header, filename = dataset.load_frame(i)

            # Quality metrics
            quality_metrics = quality_analyzer.analyze_frame(image, header)

            # Star detection
            mean, median, std = sigma_clipped_stats(image, sigma=3.0)
            daofind = DAOStarFinder(
                fwhm=3.5,
                threshold=3.0 * std,  # Lower threshold
                brightest=100,
                sharplo=0.05,
                sharphi=2.0,
                roundlo=-2.0,
                roundhi=2.0
            )
            sources = daofind(image - median)
            n_stars = len(sources) if sources is not None else 0

            # Sharpness metric (Laplacian variance)
            laplacian = ndimage.laplace(image)
            sharpness = np.var(laplacian)

            # Combined score
            # Weight: stars (40%), sharpness (30%), quality (30%)
            star_score = min(n_stars / 50.0, 1.0) * 100  # Max 100 at 50+ stars
            sharpness_score = min(sharpness / 1e6, 1.0) * 100

            # Penalize saturation
            quality_score = quality_metrics.quality_score
            if quality_metrics.is_saturated:
                quality_score *= 0.5

            combined_score = (0.4 * star_score +
                            0.3 * sharpness_score +
                            0.3 * quality_score)

            evaluations.append({
                'index': i,
                'filename': filename,
                'n_stars': n_stars,
                'sharpness': sharpness,
                'quality_score': quality_score,
                'combined_score': combined_score,
                'is_saturated': quality_metrics.is_saturated,
                'fwhm': quality_metrics.fwhm
            })

            print(f"  Frame {i:3d}: {n_stars:3d} stars, "
                  f"sharpness={sharpness:.1e}, "
                  f"quality={quality_score:.1f}, "
                  f"score={combined_score:.1f}")

        except Exception as e:
            logger.warning(f"Error evaluating frame {i}: {e}")
            continue

    if not evaluations:
        logger.error("No frames could be evaluated")
        return 0, {}

    # Select best frame
    best = max(evaluations, key=lambda x: x['combined_score'])

    print(f"\n✓ Selected frame {best['index']} as reference")
    print(f"  Stars: {best['n_stars']}, Sharpness: {best['sharpness']:.1e}, Score: {best['combined_score']:.1f}")

    return best['index'], best
