#!/usr/bin/env python3
"""
Fast Photometry Tool for Large FITS Files
Optimized for processing 5-10 GB datasets from SeeStarS50 telescope

Features:
- Memory-efficient processing of large FITS files
- Star-matching alignment across multiple observation sequences
- Interactive target and reference star selection
- Aperture photometry with annular background subtraction
- Differential photometry for atmospheric correction
- Export plots and CSV data
"""

import sys
from pathlib import Path

# Enable relative imports when executed directly
if __package__ is None or __package__ == "":
    pkg_dir = Path(__file__).resolve().parent
    if str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))
    __package__ = "seestar_photometry"

import os
import re
import warnings
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
import astroalign as aa
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
from tqdm import tqdm

# Suppress minor warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')

# Suppress numpy warnings about empty arrays
np.seterr(invalid='ignore', divide='ignore')


def reduce_brightness_for_saturation(image: np.ndarray, 
                                     target_saturation_threshold: float = 0.90,
                                     min_scaling_factor: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    Reduce brightness of saturated frame to avoid saturation
    
    Calculates a scaling factor that brings the maximum pixel values
    below the saturation threshold, then applies linear scaling.
    
    This preserves relative brightness relationships (linear scaling),
    so differential photometry ratios remain valid.
    
    Args:
        image: Input image array (typically float64 after load_frame)
        target_saturation_threshold: Target max brightness (default: 0.90 = 90% of max)
        min_scaling_factor: Minimum scaling factor (default: 0.1 = don't reduce below 10%)
        
    Returns:
        scaled_image: Image with reduced brightness
        scale_factor: Applied scaling factor (for reference)
    """
    try:
        # Determine the maximum value in the image
        max_pixel_value = np.max(image)
        
        # For float images (which is what we get from load_frame),
        # use the 99.9th percentile as a reference for "normal" max
        # This avoids outliers from cosmic rays or hot pixels
        reference_max = np.percentile(image, 99.9)
        
        # Calculate target maximum (below saturation threshold)
        # We want to scale so that the max value is at target_saturation_threshold
        # of the reference max
        target_max = target_saturation_threshold * reference_max
        
        # Calculate scaling factor
        # We want: max_pixel_value * scale_factor <= target_max
        # So: scale_factor <= target_max / max_pixel_value
        # But use reference_max for more stable scaling
        if max_pixel_value > target_max:
            scale_factor = target_max / max_pixel_value
        else:
            # Already below threshold, but might still have some saturation
            # Scale based on reference_max to be safe
            scale_factor = target_max / reference_max if reference_max > 0 else 1.0
        
        # Don't brighten, only dim
        scale_factor = min(scale_factor, 1.0)
        
        # Apply minimum scaling factor limit (don't reduce too much)
        scale_factor = max(scale_factor, min_scaling_factor)
        
        # Apply linear scaling
        scaled_image = image * scale_factor
        
        # Ensure we don't create negative values
        scaled_image = np.clip(scaled_image, 0, None)
        
        return scaled_image, scale_factor
        
    except Exception as e:
        logger.error(f"Error reducing brightness: {e}")
        # Return original image if scaling fails
        return image, 1.0


def robust_clean_lightcurve(
    times,
    magnitudes,
    sigma_global: float = 3.5,
    sigma_local: float = 3.0,
    window_size: int = 25,
    bin_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust cleaning of lightcurve using MAD + rolling clipping.
    Scientifically safe — no manual shaping.

    Args:
        times: Array-like of observation times (numeric or datetime).
        magnitudes: Array-like of differential magnitudes.
        sigma_global: MAD-based global clipping threshold (default 3.5).
        sigma_local: Rolling-window local clipping threshold (default 3.0).
        window_size: Rolling window size for local clipping (default 25).
        bin_size: If set, bin clean points into groups of this size and return binned time/mag (default None).

    Returns:
        clean_times: Times after outlier removal (and optional binning).
        clean_mags: Magnitudes after outlier removal (and optional binning).
    """
    times = np.asarray(times)
    mags = np.asarray(magnitudes, dtype=float)

    # Global MAD clipping
    median = np.nanmedian(mags)
    mad = np.nanmedian(np.abs(mags - median))
    robust_std = 1.4826 * mad if mad > 0 else np.nanstd(mags)
    if robust_std <= 0 or np.isnan(robust_std):
        robust_std = np.nanstd(mags) or 1e-6
    global_mask = np.abs(mags - median) < sigma_global * robust_std

    times_g = times[global_mask]
    mags_g = mags[global_mask]

    # Local rolling clipping
    df = pd.DataFrame({"time": times_g, "mag": mags_g})
    df = df.sort_values("time").reset_index(drop=True)

    rolling_median = df["mag"].rolling(window_size, center=True).median()
    rolling_std = df["mag"].rolling(window_size, center=True).std()
    local_mask = np.abs(df["mag"].values - rolling_median.values) < sigma_local * rolling_std.values
    local_mask = np.where(np.isnan(local_mask), True, local_mask)

    df_clean = df.loc[local_mask]

    clean_times = df_clean["time"].values
    clean_mags = df_clean["mag"].values

    # Optional binning
    if bin_size is not None and bin_size > 1 and len(clean_mags) >= bin_size:
        n_bins = len(clean_mags) // bin_size
        clean_times = clean_times[: n_bins * bin_size]
        clean_mags = clean_mags[: n_bins * bin_size]
        clean_times = clean_times.reshape(n_bins, bin_size).mean(axis=1)
        clean_mags = clean_mags.reshape(n_bins, bin_size).mean(axis=1)

    return clean_times, clean_mags


class FITSDataset:
    """
    Memory-efficient FITS file handler
    Uses memory mapping to avoid loading entire dataset into RAM
    """

    def __init__(self, input_dir: str):
        self.input_dir = Path(input_dir)
        self.fits_files = sorted(self.input_dir.glob('*.fit*'))

        if not self.fits_files:
            raise FileNotFoundError(f"No FITS files found in {input_dir}")

        print(f"Found {len(self.fits_files)} FITS files")

        # Get metadata from first file
        self.reference_header = None
        self.image_shape = None
        self._load_reference_metadata()

    def _load_reference_metadata(self):
        """Load metadata from first file without loading image data"""
        # Try memory mapping first, fall back if not supported
        try:
            with fits.open(self.fits_files[0], memmap=True) as hdul:
                self.reference_header = hdul[0].header.copy()
                self.image_shape = hdul[0].data.shape
        except (ValueError, OSError):
            # BZERO/BSCALE present or other memmap issue - use regular loading
            with fits.open(self.fits_files[0], memmap=False) as hdul:
                self.reference_header = hdul[0].header.copy()
                self.image_shape = hdul[0].data.shape
                print("Note: Using non-memory-mapped mode (FITS has scaling keywords)")

        print(f"Image shape: {self.image_shape}")

    def load_frame(self, index: int, memmap: bool = True) -> Tuple[np.ndarray, fits.Header, str]:
        """
        Load a single frame efficiently

        Args:
            index: Frame index
            memmap: Use memory mapping (faster, less RAM)

        Returns:
            image: Luminance image (2D array)
            header: FITS header
            filename: Original filename
        """
        filepath = self.fits_files[index]

        # Try with memmap first, fall back if needed
        try:
            with fits.open(filepath, memmap=memmap) as hdul:
                data = hdul[0].data.copy() if memmap else hdul[0].data
                header = hdul[0].header
        except (ValueError, OSError):
            # BZERO/BSCALE present - load without memmap
            with fits.open(filepath, memmap=False) as hdul:
                data = hdul[0].data
                header = hdul[0].header

        # Convert RGB to luminance (standard photometric conversion)
        if len(data.shape) == 3:
            # RGB: Y = 0.299*R + 0.587*G + 0.114*B
            luminance = (0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2])
        else:
            luminance = data.copy() if not isinstance(data, np.ndarray) else data

        # Convert to float for processing
        luminance = luminance.astype(np.float64)

        return luminance, header, filepath.name

    def get_observation_time(self, header: fits.Header, filename: str) -> datetime:
        """
        Extract observation time from FITS header or filename

        Priority:
        1. DATE-OBS header keyword
        2. JD/MJD header keywords
        3. Filename parsing
        """
        # Try standard FITS keywords
        if 'DATE-OBS' in header:
            try:
                return Time(header['DATE-OBS'], format='fits').datetime
            except:
                pass

        if 'JD' in header:
            try:
                return Time(header['JD'], format='jd').datetime
            except:
                pass

        if 'MJD' in header:
            try:
                return Time(header['MJD'], format='mjd').datetime
            except:
                pass

        # Try to parse from filename (common patterns)
        # Example: image_2024-01-15_20-30-45.fits
        date_patterns = [
            r'(\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2}[-_]\d{2})',
            r'(\d{8}[-_]\d{6})',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1).replace('_', '-')
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
                except:
                    pass

        # Fallback: use file modification time
        print(f"Warning: Could not parse time from {filename}, using file mtime")
        return datetime.fromtimestamp(self.fits_files[0].stat().st_mtime)

    def __len__(self):
        return len(self.fits_files)


class FrameAligner:
    """
    Enhanced star-matching alignment with multiple strategies.
    Uses hybrid alignment (cross-correlation + star matching) for robustness.
    Stores only transformation matrices (tiny memory footprint)
    """

    def __init__(self, dataset: FITSDataset):
        self.dataset = dataset
        self.reference_index = 0
        self.reference_image = None
        self.reference_wcs = None
        self.reference_header = None
        self.transformations = {}  # Store transformation matrices
        self.use_wcs = True  # Prefer WCS-based alignment when available
        # Brightness reduction settings
        self.enable_brightness_reduction = True
        self.target_saturation_threshold = 0.90
        self.min_scaling_factor = 0.1
        # Enhanced alignment settings
        self.use_enhanced_alignment = True  # Use new hybrid alignment
        self._enhanced_aligner = None  # Lazy-loaded enhanced aligner

    def set_reference(self, index: int = 0):
        """Set reference frame for alignment"""
        self.reference_index = index
        image, header, _ = self.dataset.load_frame(index)

        # Apply brightness reduction if enabled
        if self.enable_brightness_reduction:
            # Check for saturation using quality analyzer
            from .frame_quality import FrameQualityAnalyzer, QualityConfig
            quality_analyzer = FrameQualityAnalyzer(QualityConfig())
            quality_metrics = quality_analyzer.analyze_frame(image, header)

            if quality_metrics.is_saturated:
                image, scale_factor = reduce_brightness_for_saturation(
                    image,
                    self.target_saturation_threshold,
                    self.min_scaling_factor
                )
                logger.debug(f"Reference frame {index}: Reduced brightness by factor {scale_factor:.3f}")

        self.reference_image = image
        self.reference_header = header

        # Initialize enhanced aligner
        if self.use_enhanced_alignment:
            try:
                from .alignment import EnhancedAligner
                self._enhanced_aligner = EnhancedAligner(
                    detection_sigma=3.0,  # Lower threshold for faint stars
                    min_stars=5
                )
                n_stars = self._enhanced_aligner.set_reference(image)
                logger.info(f"Enhanced aligner initialized with {n_stars} reference stars")
            except Exception as e:
                logger.warning(f"Enhanced aligner initialization failed: {e}, falling back to standard")
                self._enhanced_aligner = None

        # Try to create WCS from header
        try:
            from .wcs_utils import create_wcs_from_header
            self.reference_wcs = create_wcs_from_header(header, self.reference_image.shape)
            if self.reference_wcs:
                print(f"✓ Reference frame set to index {index} with WCS")
            else:
                print(f"✓ Reference frame set to index {index} (no WCS)")
        except Exception as e:
            print(f"✓ Reference frame set to index {index} (WCS creation failed: {e})")
            self.reference_wcs = None

    def select_best_reference(self, n_candidates: int = 10):
        """
        Intelligently select the best reference frame from first N frames

        Criteria:
        1. Number of detected stars (more = better)
        2. Image sharpness (higher = better focus)
        3. Contrast (higher = better seeing conditions)

        Args:
            n_candidates: Number of initial frames to evaluate (default: 10)
        """
        n_candidates = min(n_candidates, len(self.dataset))
        print(f"\nEvaluating first {n_candidates} frames to select best reference...")

        # Suppress warnings during reference selection
        import warnings
        from astropy.utils.exceptions import AstropyWarning
        fq_logger = logging.getLogger('seestar_photometry.frame_quality')
        old_log_level = fq_logger.level
        fq_logger.setLevel(logging.WARNING)
        warnings.filterwarnings('ignore', category=AstropyWarning)
        warnings.filterwarnings('ignore', message='.*NoDetectionsWarning.*')

        best_score = -np.inf
        best_index = 0

        scores = []

        for i in range(n_candidates):
            try:
                image, header, filename = self.dataset.load_frame(i)
                
                # Apply brightness reduction if enabled (for consistent evaluation)
                if self.enable_brightness_reduction:
                    from .frame_quality import FrameQualityAnalyzer, QualityConfig
                    quality_analyzer = FrameQualityAnalyzer(QualityConfig())
                    quality_metrics = quality_analyzer.analyze_frame(image, header)
                    if quality_metrics.is_saturated:
                        image, _ = reduce_brightness_for_saturation(
                            image,
                            self.target_saturation_threshold,
                            self.min_scaling_factor
                        )

                # Metric 1: Image sharpness (variance of Laplacian)
                # Higher variance = sharper image = better focus
                from scipy import ndimage
                laplacian = ndimage.laplace(image)
                sharpness = np.var(laplacian)

                # Metric 2: Number of stars detected
                # More stars = better seeing and tracking
                from photutils.detection import DAOStarFinder
                from astropy.stats import sigma_clipped_stats

                mean, median, std = sigma_clipped_stats(image, sigma=3.0)
                # Use relaxed quality criteria to avoid warnings
                from .frame_quality import create_dao_finder
                daofind = create_dao_finder(fwhm=3.0, threshold=5.*std)
                sources = daofind(image - median)
                n_stars = len(sources) if sources is not None else 0

                # Metric 3: Contrast (std/mean ratio)
                # Higher contrast = better image quality
                contrast = std / mean if mean > 0 else 0

                # Combined score (normalized and weighted)
                # Weight stars heavily, then sharpness, then contrast
                score = (n_stars / 100.0) * 0.5 + (sharpness / 1e6) * 0.3 + contrast * 0.2

                scores.append({
                    'index': i,
                    'filename': filename,
                    'sharpness': sharpness,
                    'n_stars': n_stars,
                    'contrast': contrast,
                    'score': score
                })

                print(f"  Frame {i:2d}: {n_stars:3d} stars, sharpness={sharpness:.1e}, score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_index = i

            except Exception as e:
                print(f"  Frame {i:2d}: Error evaluating - {e}")
                continue

        # Restore logging level and warnings
        fq_logger.setLevel(old_log_level)
        warnings.resetwarnings()
        
        print(f"\n✓ Selected frame {best_index} as reference (best quality)")
        if scores:
            best = scores[best_index]
            print(f"  Stars: {best['n_stars']}, Sharpness: {best['sharpness']:.1e}, Score: {best['score']:.3f}")

        self.set_reference(best_index)
        return best_index

    def compute_alignments(self, max_control_points: int = 50, cache_dir: str = 'output'):
        """
        Compute alignment transformations for all frames
        Uses star-matching, stores only transformation matrices
        """
        print(f"\nComputing alignments for {len(self.dataset)} frames...")

        if self.reference_image is None:
            self.set_reference(0)

        self.transformations = {}
        failed_frames = []
        
        # Suppress warnings and reduce logging during alignment for clean progress bar
        import warnings
        from astropy.utils.exceptions import AstropyWarning
        fq_logger = logging.getLogger('seestar_photometry.frame_quality')
        wcs_logger = logging.getLogger('seestar_photometry.wcs_utils')
        old_log_level = fq_logger.level
        old_wcs_log_level = wcs_logger.level
        fq_logger.setLevel(logging.WARNING)
        wcs_logger.setLevel(logging.WARNING)
        # Suppress warnings during alignment
        warnings.filterwarnings('ignore', category=AstropyWarning)
        warnings.filterwarnings('ignore', message='.*NoDetectionsWarning.*')

        for i in tqdm(range(len(self.dataset)), desc="Aligning frames", ncols=80):
            if i == self.reference_index:
                # Identity transformation for reference frame
                self.transformations[i] = {
                    'matrix': np.eye(3),
                    'aligned': True
                }
                continue

            try:
                # Load frame
                target_image, header, _ = self.dataset.load_frame(i)

                # Apply brightness reduction if enabled (helps with alignment)
                if self.enable_brightness_reduction:
                    from .frame_quality import FrameQualityAnalyzer, QualityConfig
                    quality_analyzer = FrameQualityAnalyzer(QualityConfig())
                    quality_metrics = quality_analyzer.analyze_frame(target_image, header)
                    if quality_metrics.is_saturated:
                        target_image, _ = reduce_brightness_for_saturation(
                            target_image,
                            self.target_saturation_threshold,
                            self.min_scaling_factor
                        )

                alignment_result = None

                # Try enhanced alignment first (most robust)
                if self.use_enhanced_alignment and self._enhanced_aligner is not None:
                    try:
                        alignment_result = self._enhanced_aligner.compute_alignment(
                            target_image, method='hybrid'
                        )
                        if alignment_result['aligned']:
                            self.transformations[i] = {
                                'matrix': alignment_result['matrix'],
                                'aligned': True,
                                'method': alignment_result['method'],
                                'n_stars': alignment_result.get('n_matched_stars', 0),
                                'rms_error': alignment_result.get('rms_error'),
                                'shift_x': alignment_result.get('shift_x', 0),
                                'shift_y': alignment_result.get('shift_y', 0),
                                'rotation_deg': alignment_result.get('rotation_deg', 0),
                            }
                            continue
                    except Exception as e:
                        logger.debug(f"Enhanced alignment failed for frame {i}: {e}")
                        alignment_result = None

                # Try WCS-based alignment if enhanced failed
                wcs_matrix = None
                if alignment_result is None and self.use_wcs and self.reference_wcs is not None:
                    wcs_matrix = self.compute_wcs_transformation(header, target_image.shape)

                if wcs_matrix is not None:
                    # WCS alignment successful
                    self.transformations[i] = {
                        'matrix': wcs_matrix,
                        'aligned': True,
                        'method': 'wcs'
                    }
                else:
                    # Fall back to astroalign star matching
                    transf, (source_list, target_list) = aa.find_transform(
                        target_image,
                        self.reference_image,
                        max_control_points=max_control_points
                    )

                    self.transformations[i] = {
                        'matrix': transf.params,
                        'aligned': True,
                        'n_stars': len(source_list),
                        'method': 'astroalign'
                    }

            except Exception:
                # Silently track failed frames (no scrolling output)
                failed_frames.append(i)
                # Store identity transformation as fallback
                self.transformations[i] = {
                    'matrix': np.eye(3),
                    'aligned': False
                }

        # Restore logging levels and warnings
        fq_logger.setLevel(old_log_level)
        wcs_logger.setLevel(old_wcs_log_level)
        warnings.resetwarnings()

        n_aligned = sum(1 for t in self.transformations.values() if t['aligned'])
        n_failed = len(failed_frames)

        # Finalize WCS logging (clear the inline status)
        from .wcs_utils import finalize_wcs_logging
        finalize_wcs_logging()

        print(f"\n✓ Successfully aligned {n_aligned}/{len(self.dataset)} frames")
        if n_failed > 0:
            print(f"  Skipped {n_failed} frames (tracking issues, clouds, or field rotation)")

        # Save alignment cache
        self.save_alignment_cache(cache_dir)

    def compute_wcs_transformation(self, target_header: fits.Header, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Compute transformation matrix using WCS from FITS headers

        Args:
            target_header: FITS header of target frame
            target_shape: Shape of target image

        Returns:
            3x3 transformation matrix or None if WCS unavailable
        """
        if self.reference_wcs is None:
            return None

        try:
            from .wcs_utils import create_wcs_from_header

            # Create WCS for target frame
            target_wcs = create_wcs_from_header(target_header, target_shape)
            if target_wcs is None:
                return None

            # Compute transformation matrix from WCS
            # Strategy: Sample grid points in reference frame, convert to world coords,
            # then convert to target frame pixel coords

            # Get image center and corners
            ref_h, ref_w = self.reference_image.shape

            # Sample points: corners and center
            ref_points = np.array([
                [ref_w/2, ref_h/2],      # center
                [0, 0],                   # top-left
                [ref_w, 0],               # top-right
                [0, ref_h],               # bottom-left
                [ref_w, ref_h]            # bottom-right
            ])

            # Convert reference pixel coords to world coords
            world_coords = self.reference_wcs.pixel_to_world_values(ref_points[:, 0], ref_points[:, 1])

            # Convert world coords to target pixel coords
            target_points = target_wcs.world_to_pixel_values(world_coords[0], world_coords[1])
            target_points = np.array([target_points[0], target_points[1]]).T

            # Compute affine transformation from target to reference points
            # (consistent with other alignment methods: matrix @ target = reference)
            from skimage.transform import AffineTransform

            tform = AffineTransform()
            tform.estimate(target_points, ref_points)

            return tform.params

        except Exception as e:
            logger.debug(f"WCS transformation failed: {e}")
            return None

    def save_alignment_cache(self, cache_dir: str = 'output'):
        """Save alignment transformations to cache file"""
        cache_path = Path(cache_dir) / 'alignment_cache.npz'
        cache_path.parent.mkdir(exist_ok=True)

        # Prepare data for saving
        matrices = []
        aligned_flags = []
        indices = []

        for i, trans in self.transformations.items():
            indices.append(i)
            matrices.append(trans['matrix'])
            aligned_flags.append(trans['aligned'])

        np.savez_compressed(
            cache_path,
            reference_index=self.reference_index,
            indices=np.array(indices),
            matrices=np.array(matrices),
            aligned=np.array(aligned_flags)
        )

        print(f"  Alignment cache saved to: {cache_path}")

    def load_alignment_cache(self, cache_dir: str = 'output') -> bool:
        """
        Load alignment transformations from cache file
        Returns True if successful, False otherwise
        """
        cache_path = Path(cache_dir) / 'alignment_cache.npz'

        if not cache_path.exists():
            return False

        try:
            data = np.load(cache_path)

            # Restore transformations
            self.reference_index = int(data['reference_index'])
            self.transformations = {}

            for idx, matrix, aligned in zip(data['indices'], data['matrices'], data['aligned']):
                self.transformations[int(idx)] = {
                    'matrix': matrix,
                    'aligned': bool(aligned)
                }

            # Load reference image
            self.reference_image, _, _ = self.dataset.load_frame(self.reference_index)

            n_aligned = sum(1 for t in self.transformations.values() if t['aligned'])
            print(f"✓ Loaded alignment cache: {n_aligned}/{len(self.transformations)} frames aligned")
            print(f"  Reference frame: {self.reference_index}")

            return True

        except Exception as e:
            print(f"  Warning: Could not load alignment cache: {e}")
            return False

    def transform_coordinates(self, x: float, y: float, frame_index: int) -> Tuple[float, float]:
        """
        Transform coordinates from reference frame to target frame

        The stored matrix M maps target coords to reference coords: M @ target = reference
        To go from reference to target, we need: target = M^(-1) @ reference

        Args:
            x, y: Coordinates in reference frame
            frame_index: Target frame index

        Returns:
            Transformed x, y coordinates in target frame
        """
        if frame_index not in self.transformations:
            return x, y

        matrix = self.transformations[frame_index]['matrix']

        # Apply inverse transformation to get coordinates in target frame
        # Matrix maps target->reference, so inverse maps reference->target
        try:
            inv_matrix = np.linalg.inv(matrix)
            coords = np.array([x, y, 1])
            transformed = inv_matrix @ coords
            x_t, y_t = transformed[0], transformed[1]

            # Validate transformed coordinates are reasonable
            # (should not have moved more than ~50% of image size)
            if self.reference_image is not None:
                h, w = self.reference_image.shape
                if abs(x_t - x) > w * 0.5 or abs(y_t - y) > h * 0.5:
                    logger.debug(f"Large coordinate shift for frame {frame_index}: "
                               f"({x:.1f}, {y:.1f}) -> ({x_t:.1f}, {y_t:.1f})")

            return x_t, y_t
        except Exception as e:
            logger.debug(f"Coordinate transformation failed for frame {frame_index}: {e}")
            return x, y


class StarSelector:
    """
    Interactive GUI for selecting target and reference stars
    Uses downsampled image for faster display
    """

    def __init__(self, reference_image: np.ndarray, downsample_factor: int = 2,
                 wcs=None, header=None):
        self.original_image = reference_image
        self.downsample_factor = downsample_factor
        self.wcs = wcs  # WCS for variable star checking
        self.header = header  # FITS header

        # Downsample for display (faster rendering)
        from scipy.ndimage import zoom
        self.display_image = zoom(reference_image, 1/downsample_factor, order=1)

        # Normalize for display using ZScale + Asinh stretch (professional astronomy standard)
        # Asinh stretch shows both faint and bright stars clearly - used by HST, SDSS, etc.
        from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize

        # Find optimal display limits with ZScale
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(self.display_image)

        # Safety check: ensure vmin and vmax are different
        if vmax - vmin < 1e-10:
            vmin = np.min(self.display_image)
            vmax = np.max(self.display_image)

        # Apply asinh stretch for wide dynamic range
        # a=0.1 provides good balance between faint and bright stars
        norm = ImageNormalize(
            self.display_image,
            interval=interval,
            stretch=AsinhStretch(a=0.1),
            clip=True
        )

        self.display_image_norm = norm(self.display_image)

        self.target_pos = None
        self.reference_stars = []
        self.aperture_radius = 10  # pixels
        self.fig = None
        self.ax = None
        self.variable_warnings = []  # Track variable star warnings

    def select_stars(self, aperture_radius: int = 10,
                     sky_annulus_inner: int = 15,
                     sky_annulus_outer: int = 25) -> Dict:
        """
        Interactive star selection

        Returns:
            Dictionary with target and reference star positions (in original image coordinates)
        """
        self.aperture_radius = aperture_radius
        self.sky_inner = sky_annulus_inner
        self.sky_outer = sky_annulus_outer

        # Create interactive plot with clear instructions
        self.fig = plt.figure(figsize=(14, 11))

        # Main image axes
        self.ax = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
        self.ax.imshow(self.display_image_norm, cmap='gray', origin='lower')
        self.ax.set_title('Star Selection - Click to Choose Your Stars',
                         fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel('X position (pixels)', fontsize=10)
        self.ax.set_ylabel('Y position (pixels)', fontsize=10)

        # Instruction box at bottom
        instructions_ax = plt.subplot2grid((10, 1), (9, 0))
        instructions_ax.axis('off')
        instructions_text = (
            "STEP 1: Click on your TARGET STAR (the one you want to measure)\n"
            "STEP 2: Click on 1-3 REFERENCE STARS (bright, isolated stars nearby)\n"
            "         → Choose stars similar brightness to your target\n"
            "         → Avoid edges, saturated, or crowded stars\n"
            "STEP 3: Press 'q' or close window when done\n\n"
            "Right-click to undo last selection"
        )
        instructions_ax.text(0.05, 0.5, instructions_text,
                           transform=instructions_ax.transAxes,
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           family='monospace')

        # Status text (will update as user clicks)
        self.status_text = self.ax.text(0.02, 0.98, 'Waiting for target selection...',
                                       transform=self.ax.transAxes,
                                       fontsize=11, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                                       fontweight='bold')

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.show()

        # Scale coordinates back to original image
        if self.target_pos is None:
            raise ValueError("No target star selected!")

        target_original = (
            self.target_pos[0] * self.downsample_factor,
            self.target_pos[1] * self.downsample_factor
        )

        references_original = [
            (x * self.downsample_factor, y * self.downsample_factor)
            for x, y in self.reference_stars
        ]

        return {
            'target': target_original,
            'references': references_original,
            'aperture_radius': aperture_radius,
            'sky_inner': sky_annulus_inner,
            'sky_outer': sky_annulus_outer
        }

    def _check_variable_star(self, x: float, y: float) -> Optional[Dict]:
        """Check if clicked position is a known variable star"""
        if self.wcs is None:
            return None

        try:
            from .wcs_utils import pixel_to_world
            from .star_identification import check_if_variable_star

            # Convert display coords to original image coords
            orig_x = x * self.downsample_factor
            orig_y = y * self.downsample_factor

            # Convert pixel to world coordinates
            ra, dec = pixel_to_world(self.wcs, orig_x, orig_y)
            if ra is None or dec is None:
                return None

            # Check if it's a known variable
            var_info = check_if_variable_star(ra, dec, tolerance_arcsec=60.0)
            return var_info

        except Exception as e:
            logger.debug(f"Variable star check failed: {e}")
            return None

    def _on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # Left click
            if self.target_pos is None:
                # First click: select target
                self.target_pos = (x, y)
                self._draw_target(x, y)
                self._update_status()
                print(f"✓ Target selected at ({x:.1f}, {y:.1f})")
            else:
                # Subsequent clicks: add reference stars
                # Draw marker IMMEDIATELY (non-blocking)
                ref_num = len(self.reference_stars) + 1
                self.reference_stars.append((x, y))
                self._draw_reference(x, y, len(self.reference_stars))
                self._update_status()
                print(f"✓ Reference star {len(self.reference_stars)} selected at ({x:.1f}, {y:.1f})")

                # Check if it's a variable star in background (non-blocking)
                import threading
                def check_variable_async():
                    var_info = self._check_variable_star(x, y)
                    if var_info:
                        warning_msg = f"⚠ WARNING: Reference star {ref_num} may be variable: {var_info.get('name', 'Unknown')}"
                        print(warning_msg)
                        print(f"  Type: {var_info.get('type', 'Unknown')}, Separation: {var_info.get('separation_arcsec', '?'):.1f}\"")
                        print("  Consider selecting a different reference star!")
                        self.variable_warnings.append((ref_num, var_info))

                # Run variable star check in background thread
                thread = threading.Thread(target=check_variable_async, daemon=True)
                thread.start()

        elif event.button == 3:  # Right click - remove last
            if self.reference_stars:
                self.reference_stars.pop()
                self._redraw_all()
                self._update_status()
                print("↶ Last reference star removed")
            elif self.target_pos is not None:
                self.target_pos = None
                self._redraw_all()
                self._update_status()
                print("↶ Target star removed")

        elif event.button == 2:  # Middle click - finish
            if self.target_pos and len(self.reference_stars) >= 1:
                plt.close(self.fig)
            else:
                print("⚠ Need at least 1 target and 1 reference star!")

    def _on_key(self, event):
        """Handle key presses"""
        if event.key == 'q':
            if self.target_pos and len(self.reference_stars) >= 1:
                plt.close(self.fig)
            else:
                print("⚠ Need at least 1 target and 1 reference star before finishing!")

    def _update_status(self):
        """Update status text based on current selections"""
        if self.target_pos is None:
            status = 'Waiting for target selection...'
            color = 'yellow'
        elif len(self.reference_stars) == 0:
            status = '✓ Target selected! Now select 1-3 reference stars'
            color = 'lightblue'
        elif len(self.reference_stars) == 1:
            status = '✓ Good! Add 1-2 more references (or press \'q\' if done)'
            color = 'lightgreen'
        else:
            status = f'✓ Excellent! {len(self.reference_stars)} reference stars selected. Press \'q\' when ready'
            color = 'lightgreen'

        self.status_text.set_text(status)
        self.status_text.set_bbox(dict(boxstyle='round', facecolor=color, alpha=0.8))
        self.fig.canvas.draw()

    def _draw_target(self, x, y):
        """Draw target star marker"""
        circle1 = Circle((x, y), self.aperture_radius / self.downsample_factor,
                         fill=False, color='red', linewidth=2, label='Target')
        circle2 = Circle((x, y), self.sky_inner / self.downsample_factor,
                         fill=False, color='red', linewidth=1, linestyle='--')
        circle3 = Circle((x, y), self.sky_outer / self.downsample_factor,
                         fill=False, color='red', linewidth=1, linestyle='--')
        self.ax.add_patch(circle1)
        self.ax.add_patch(circle2)
        self.ax.add_patch(circle3)
        self.ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
        self.fig.canvas.draw()

    def _draw_reference(self, x, y, index):
        """Draw reference star marker"""
        circle1 = Circle((x, y), self.aperture_radius / self.downsample_factor,
                         fill=False, color='cyan', linewidth=2)
        circle2 = Circle((x, y), self.sky_inner / self.downsample_factor,
                         fill=False, color='cyan', linewidth=1, linestyle='--')
        circle3 = Circle((x, y), self.sky_outer / self.downsample_factor,
                         fill=False, color='cyan', linewidth=1, linestyle='--')
        self.ax.add_patch(circle1)
        self.ax.add_patch(circle2)
        self.ax.add_patch(circle3)
        self.ax.plot(x, y, 'c+', markersize=12, markeredgewidth=2)
        self.ax.text(x+5, y+5, str(index), color='cyan', fontsize=10, fontweight='bold')
        self.fig.canvas.draw()

    def _redraw_all(self):
        """Redraw all selections"""
        self.ax.clear()
        self.ax.imshow(self.display_image_norm, cmap='gray', origin='lower')
        self.ax.set_title('Star Selection - Click to Choose Your Stars',
                         fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel('X position (pixels)', fontsize=10)
        self.ax.set_ylabel('Y position (pixels)', fontsize=10)

        # Recreate status text
        self.status_text = self.ax.text(0.02, 0.98, '',
                                       transform=self.ax.transAxes,
                                       fontsize=11, verticalalignment='top',
                                       fontweight='bold')

        if self.target_pos:
            self._draw_target(*self.target_pos)

        for i, (x, y) in enumerate(self.reference_stars, 1):
            self._draw_reference(x, y, i)

        self.fig.canvas.draw()


class PhotometryEngine:
    """
    Enhanced aperture photometry with SNR optimization.
    Processes frames one at a time for memory efficiency.

    Features:
    - Adaptive aperture sizing based on FWHM
    - Sub-pixel centroid refinement
    - SNR calculation and reporting
    - Robust sky background estimation
    """

    def __init__(self, dataset: FITSDataset, aligner: FrameAligner, star_data: Dict):
        self.dataset = dataset
        self.aligner = aligner
        self.star_data = star_data

        self.target_pos = star_data['target']
        self.reference_positions = star_data['references']
        self.aperture_radius = star_data['aperture_radius']
        self.sky_inner = star_data['sky_inner']
        self.sky_outer = star_data['sky_outer']

        # Results storage
        self.results = []

        # Quality analyzers
        from .frame_quality import FrameQualityAnalyzer, QualityConfig
        from .moon_glare import MoonGlareAnalyzer, MoonGlareConfig

        self.quality_analyzer = FrameQualityAnalyzer(QualityConfig())
        self.moon_analyzer = MoonGlareAnalyzer(MoonGlareConfig())
        self.quality_metrics_list = []  # Store per-frame quality metrics
        self.moon_metrics_list = []  # Store per-frame moon metrics

        # Brightness reduction settings
        self.enable_brightness_reduction = True  # Enable automatic brightness reduction for saturated frames
        self.target_saturation_threshold = 0.90  # Target max brightness (90% of max) after reduction
        self.min_scaling_factor = 0.1  # Minimum scaling factor (don't reduce below 10%)

        # Enhanced photometry settings
        self.use_enhanced_photometry = True  # Use enhanced photometry with SNR
        self.auto_aperture = True  # Automatically optimize aperture based on FWHM
        self.refine_centroids = True  # Use sub-pixel centroid refinement
        self._enhanced_photometer = None  # Lazy-loaded enhanced photometer

    def process_all_frames(self):
        """
        Process all frames with enhanced aperture photometry
        Memory-efficient: loads one frame at a time
        Only processes successfully aligned frames

        Features:
        - SNR calculation for each measurement
        - Adaptive aperture sizing based on FWHM
        - Sub-pixel centroid refinement
        """
        # Initialize enhanced photometer if enabled
        if self.use_enhanced_photometry:
            try:
                from .enhanced_photometry import EnhancedPhotometry
                self._enhanced_photometer = EnhancedPhotometry(
                    aperture_radius=self.aperture_radius if not self.auto_aperture else None,
                    sky_annulus_inner=self.sky_inner,
                    sky_annulus_outer=self.sky_outer,
                    auto_aperture=self.auto_aperture
                )
                logger.info("Enhanced photometry enabled with SNR calculation")
            except Exception as e:
                logger.warning(f"Enhanced photometry initialization failed: {e}")
                self._enhanced_photometer = None

        # Get list of successfully aligned frames only
        aligned_frames = [i for i, t in self.aligner.transformations.items() if t['aligned']]
        n_aligned = len(aligned_frames)
        n_total = len(self.dataset)

        print(f"\nPerforming aperture photometry on {n_aligned} successfully aligned frames")
        print(f"(Skipping {n_total - n_aligned} frames that failed alignment)")

        failed_photometry = []
        brightness_reduced_frames = []  # Track frames with brightness reduction

        progress_bar = tqdm(aligned_frames, desc="Photometry", ncols=80)
        for i in progress_bar:
            try:
                # Load frame
                image, header, filename = self.dataset.load_frame(i)
                obs_time = self.dataset.get_observation_time(header, filename)

                # Quality analysis
                quality_metrics = self.quality_analyzer.analyze_frame(image, header)

                # Moon analysis (if WCS available)
                moon_metrics = None
                if self.aligner.reference_wcs is not None:
                    target_ra, target_dec = self._get_target_coordinates()
                    if target_ra is not None and target_dec is not None:
                        lat, lon = self._get_observation_location(header)
                        moon_metrics = self.moon_analyzer.analyze_moon_impact(
                            obs_time, target_ra, target_dec, lat, lon
                        )
                        if moon_metrics:
                            quality_metrics.moon_distance_deg = moon_metrics.moon_distance_deg

                # Store quality metrics
                self.quality_metrics_list.append(quality_metrics)
                self.moon_metrics_list.append(moon_metrics)

                # Apply brightness reduction if saturation detected
                brightness_scale_factor = 1.0
                if self.enable_brightness_reduction and quality_metrics.is_saturated:
                    image, brightness_scale_factor = reduce_brightness_for_saturation(
                        image,
                        self.target_saturation_threshold,
                        self.min_scaling_factor
                    )
                    brightness_reduced_frames.append((i, brightness_scale_factor))

                # Transform star positions to this frame's coordinates
                target_x, target_y = self.aligner.transform_coordinates(
                    self.target_pos[0], self.target_pos[1], i
                )

                # Use enhanced photometry if available
                target_snr = np.nan
                ref_snr_avg = np.nan
                diff_mag_err = np.nan

                if self._enhanced_photometer is not None:
                    # Use enhanced photometry with SNR calculation
                    from .enhanced_photometry import measure_differential_photometry

                    # Transform reference positions
                    ref_positions_transformed = []
                    for ref_x, ref_y in self.reference_positions:
                        ref_x_t, ref_y_t = self.aligner.transform_coordinates(ref_x, ref_y, i)
                        ref_positions_transformed.append((ref_x_t, ref_y_t))

                    # Perform differential photometry with SNR
                    phot_result = measure_differential_photometry(
                        image,
                        (target_x, target_y),
                        ref_positions_transformed,
                        fwhm=quality_metrics.fwhm,
                        aperture_radius=None if self.auto_aperture else self.aperture_radius
                    )

                    target_flux = phot_result['target_flux']
                    target_snr = phot_result['target_snr']
                    target_sky = phot_result['target_measurement']['sky'] if phot_result['target_measurement'] else np.nan
                    ref_flux_avg = phot_result['ref_flux_avg']
                    ref_snr_avg = phot_result['ref_snr_avg']
                    # Calculate ref_sky_avg only if there are valid measurements
                    valid_ref_skies = [m['sky'] for m in phot_result['ref_measurements'] if m['valid'] and m['sky'] is not None]
                    ref_sky_avg = np.mean(valid_ref_skies) if valid_ref_skies else np.nan
                    diff_mag = phot_result['diff_magnitude']
                    diff_mag_err = phot_result['diff_mag_err']

                    # Update target position if centroid was refined
                    if phot_result['target_measurement'] and phot_result['target_measurement']['centroid_refined']:
                        target_x = phot_result['target_measurement']['x']
                        target_y = phot_result['target_measurement']['y']

                else:
                    # Fallback to basic photometry
                    target_flux, target_sky = self._measure_star(
                        image, target_x, target_y
                    )

                    # Measure reference stars
                    ref_fluxes = []
                    ref_skies = []
                    for ref_x, ref_y in self.reference_positions:
                        ref_x_t, ref_y_t = self.aligner.transform_coordinates(ref_x, ref_y, i)
                        flux, sky = self._measure_star(image, ref_x_t, ref_y_t)
                        if flux is not None and flux > 0:
                            ref_fluxes.append(flux)
                            ref_skies.append(sky)

                    # Calculate differential photometry
                    if ref_fluxes and target_flux is not None and target_flux > 0:
                        ref_flux_avg = np.mean(ref_fluxes)
                        ref_sky_avg = np.mean(ref_skies)

                        if ref_flux_avg > 0:
                            diff_mag = -2.5 * np.log10(target_flux / ref_flux_avg)
                        else:
                            diff_mag = np.nan
                    else:
                        ref_flux_avg = np.nan
                        ref_sky_avg = np.nan
                        diff_mag = np.nan
                        target_flux = target_flux if target_flux is not None else np.nan
                        target_sky = target_sky if target_sky is not None else np.nan

                # Store results with quality metrics and SNR
                self.results.append({
                    'frame': i,
                    'filename': filename,
                    'time': obs_time,
                    'target_x': target_x,
                    'target_y': target_y,
                    'target_flux': target_flux,
                    'target_sky': target_sky,
                    'target_snr': target_snr,
                    'ref_flux_avg': ref_flux_avg,
                    'ref_sky_avg': ref_sky_avg,
                    'ref_snr_avg': ref_snr_avg,
                    'diff_magnitude': diff_mag,
                    'diff_mag_err': diff_mag_err,
                    # Quality metrics
                    'quality_score': quality_metrics.quality_score,
                    'cloud_score': quality_metrics.cloud_score,
                    'is_saturated': quality_metrics.is_saturated,
                    'saturation_fraction': quality_metrics.saturation_fraction,
                    'n_cosmic_rays': quality_metrics.n_cosmic_rays,
                    'fwhm': quality_metrics.fwhm,
                    'moon_distance_deg': quality_metrics.moon_distance_deg,
                    'moon_glare_level': moon_metrics.glare_level if moon_metrics else None,
                    'moon_phase': moon_metrics.moon_phase if moon_metrics else None,
                    # Brightness reduction
                    'brightness_scale_factor': brightness_scale_factor,
                })

            except Exception:
                # Silently track failed frames (no scrolling output)
                failed_photometry.append(i)
                continue

        n_successful = len(self.results)
        print(f"\n✓ Successfully measured {n_successful}/{n_aligned} aligned frames")
        if n_successful < n_aligned:
            n_failed_phot = n_aligned - n_successful
            print(f"  Skipped {n_failed_phot} frames (stars out of bounds or measurement issues)")
        
        # Print summary of brightness reduction if any frames were reduced
        if brightness_reduced_frames:
            n_reduced = len(brightness_reduced_frames)
            avg_scale = np.mean([scale for _, scale in brightness_reduced_frames])
            print(f"  Applied brightness reduction to {n_reduced} saturated frame(s) (avg scale: {avg_scale:.3f})")

    def _measure_star(self, image: np.ndarray, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Measure star flux with aperture photometry and annular background subtraction

        Returns:
            flux: Background-subtracted flux
            sky: Sky background level
        """
        try:
            # Check if position is within image bounds
            if (x < 0 or y < 0 or
                x >= image.shape[1] or y >= image.shape[0]):
                return None, None

            # Create apertures
            aperture = CircularAperture((x, y), r=self.aperture_radius)
            annulus = CircularAnnulus((x, y),
                                     r_in=self.sky_inner,
                                     r_out=self.sky_outer)

            # Measure aperture flux
            phot_table = aperture_photometry(image, aperture)
            aperture_sum = phot_table['aperture_sum'][0]

            # Measure sky background
            annulus_table = aperture_photometry(image, annulus)
            annulus_sum = annulus_table['aperture_sum'][0]
            annulus_area = annulus.area
            aperture_area = aperture.area

            # Sky background per pixel
            sky_per_pixel = annulus_sum / annulus_area

            # Background-subtracted flux
            flux = aperture_sum - (sky_per_pixel * aperture_area)

            return flux, sky_per_pixel

        except Exception as e:
            return None, None

    def _get_target_coordinates(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get target star RA/DEC from WCS

        Returns:
            (ra, dec) in degrees, or (None, None) if WCS not available
        """
        try:
            if self.aligner.reference_wcs is not None:
                from .wcs_utils import pixel_to_world
                ra, dec = pixel_to_world(
                    self.aligner.reference_wcs,
                    self.target_pos[0],
                    self.target_pos[1]
                )
                return ra, dec
            return None, None
        except Exception as e:
            logger.warning(f"Error getting target coordinates: {e}")
            return None, None

    def _get_observation_location(self, header) -> Tuple[float, float]:
        """
        Extract observer lat/lon from FITS header or use defaults

        Args:
            header: FITS header

        Returns:
            (lat, lon) in degrees
        """
        # Try various FITS keywords for site location
        lat = header.get('SITELAT', header.get('LAT-OBS', header.get('LATITUDE', 20.0)))
        lon = header.get('SITELONG', header.get('LONG-OBS', header.get('LONGITUDE', 77.0)))

        # Convert to float
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            # Use defaults (approximate India location)
            lat, lon = 20.0, 77.0

        return lat, lon

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame(self.results)


class ResultsExporter:
    """
    Export plots and CSV data.
    Applies robust statistical cleaning (MAD + rolling clipping) to the lightcurve;
    plots and CSV use cleaned magnitudes.
    """

    def __init__(self, results_df: pd.DataFrame, output_dir: str,
                 target_name: str = None, target_ra: float = None, target_dec: float = None,
                 sigma_global: float = 3.5, sigma_local: float = 3.0,
                 window_size: int = 30, bin_size: Optional[int] = None):
        self.df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.target_name = target_name
        self.target_ra = target_ra
        self.target_dec = target_dec

        # Robust cleaning: compute clean_times, clean_mags for plots and CSV
        valid = ~np.isnan(self.df['diff_magnitude'])
        times_raw = self.df.loc[valid, 'time'].values
        mags_raw = self.df.loc[valid, 'diff_magnitude'].values
        if len(times_raw) > 0:
            clean_times, clean_mags = robust_clean_lightcurve(
                times_raw, mags_raw,
                sigma_global=sigma_global,
                sigma_local=sigma_local,
                window_size=window_size,
                bin_size=bin_size,
            )
            self.clean_times = pd.Series(pd.to_datetime(clean_times)) if len(clean_times) > 0 else pd.Series(dtype='datetime64[ns]')
            self.clean_mags = np.asarray(clean_mags, dtype=float)
        else:
            self.clean_times = pd.Series(dtype='datetime64[ns]')
            self.clean_mags = np.array([], dtype=float)

    def _create_compressed_time_axis(self, times, gap_threshold_hours):
        """
        Create a compressed time axis that removes gaps

        Returns:
            compressed_indices: Array of indices for plotting (gaps removed)
            tick_positions: Positions for x-axis ticks
            tick_labels: Corresponding time labels
            gap_positions: Positions where visual gap markers should be shown
            gap_labels: Labels showing time before and after each gap
        """
        if len(times) == 0:
            return np.array([]), [], [], [], []

        times_np = times.values
        time_diffs = np.diff(times_np).astype('timedelta64[s]').astype(float) / 3600.0
        gap_indices = np.where(time_diffs > gap_threshold_hours)[0]

        # Create compressed index (sequential, gaps removed)
        compressed_indices = np.arange(len(times))

        # Identify segment boundaries for tick placement
        segment_starts = [0] + (gap_indices + 1).tolist()
        segment_ends = (gap_indices + 1).tolist() + [len(times)]

        tick_positions = []
        tick_labels = []
        gap_positions = []
        gap_labels = []  # Labels showing times at gaps

        for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            # Add tick at start of segment (but not if it's right after a gap)
            if i == 0 or start - segment_ends[i-1] > 1:
                tick_positions.append(start)
                tick_labels.append(times.iloc[start].strftime('%m/%d\n%H:%M'))

            # Handle gap at end of segment
            if end < len(times):
                # Gap position (between this segment end and next segment start)
                gap_pos = end - 0.5
                gap_positions.append(gap_pos)

                # Create label showing time before and after gap
                time_before = times.iloc[end - 1].strftime('%H:%M')
                time_after = times.iloc[end].strftime('%H:%M')
                date_before = times.iloc[end - 1].strftime('%m/%d')
                date_after = times.iloc[end].strftime('%m/%d')

                # If dates are different, show date change
                if date_before != date_after:
                    gap_label = f'{date_before}\n{time_before}  →  {time_after}\n{date_after}'
                else:
                    gap_label = f'{time_before}\n  →\n{time_after}'

                gap_labels.append(gap_label)

        # Add final tick
        if len(times) - 1 not in tick_positions:
            tick_positions.append(len(times) - 1)
            tick_labels.append(times.iloc[-1].strftime('%m/%d\n%H:%M'))

        return compressed_indices, tick_positions, tick_labels, gap_positions, gap_labels

    def _plot_with_gaps(self, ax, times, values, gap_threshold_hours, **plot_kwargs):
        """
        Plot data with compressed time axis (gaps removed)

        Merges observation segments by removing time gaps entirely.

        Args:
            ax: Matplotlib axis to plot on
            times: Pandas datetime series
            values: Numpy array of values to plot
            gap_threshold_hours: Gap threshold in hours
            **plot_kwargs: Additional arguments passed to ax.plot()
        """
        if len(times) == 0:
            return

        # Create compressed index for x-axis
        compressed_idx, tick_pos, tick_labels, gap_pos, gap_labels = self._create_compressed_time_axis(
            times, gap_threshold_hours)

        # Plot using compressed indices
        ax.plot(compressed_idx, values, '.-', **plot_kwargs)

        # Store for axis formatting (will be applied later)
        if not hasattr(ax, '_compressed_time_data'):
            ax._compressed_time_data = {
                'tick_positions': tick_pos,
                'tick_labels': tick_labels,
                'gap_positions': gap_pos,
                'gap_labels': gap_labels
            }

    def _fill_between_with_gaps(self, ax, times, y1, y2, gap_threshold_hours, **fill_kwargs):
        """
        Fill between two lines with compressed time axis

        Args:
            ax: Matplotlib axis
            times: Pandas datetime series
            y1: Lower bound values
            y2: Upper bound values
            gap_threshold_hours: Gap threshold in hours
            **fill_kwargs: Additional arguments passed to ax.fill_between()
        """
        if len(times) == 0:
            return

        # Create compressed index
        compressed_idx, _, _, _, _ = self._create_compressed_time_axis(times, gap_threshold_hours)

        # Use compressed index for x-axis
        ax.fill_between(compressed_idx, y1, y2, **fill_kwargs)

    def plot_lightcurve(self, format: str = 'png', dpi: int = 300):
        """
        Plot magnitude vs time with proper date/time axis

        Args:
            format: 'png' or 'jpeg'
            dpi: Resolution
        """
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Use statistically cleaned lightcurve
        times = self.clean_times
        magnitudes = self.clean_mags

        if len(times) == 0:
            print("Warning: No valid measurements for plotting")
            return

        # Determine time span and appropriate tick interval
        time_span_hours = (times.max() - times.min()).total_seconds() / 3600

        if time_span_hours < 12:
            # Single night: 1 hour ticks
            major_locator = mdates.HourLocator(interval=1)
            minor_locator = mdates.MinuteLocator(interval=15)
            date_format = mdates.DateFormatter('%H:%M')
            xlabel = 'Time (UTC)'
        elif time_span_hours < 48:
            # 1-2 days: 3 hour ticks
            major_locator = mdates.HourLocator(interval=3)
            minor_locator = mdates.HourLocator(interval=1)
            date_format = mdates.DateFormatter('%m/%d %H:%M')
            xlabel = 'Date and Time (UTC)'
        else:
            # Multiple days: show date with hours
            major_locator = mdates.HourLocator(interval=6)
            minor_locator = mdates.HourLocator(interval=3)
            date_format = mdates.DateFormatter('%m/%d\n%H:%M')
            xlabel = 'Date and Time (UTC)'

        # Plot 1: Differential Magnitude vs Time
        # Detect gaps in data (e.g., daytime breaks) and plot segments separately
        gap_threshold_hours = 1.0  # Consider gaps > 1 hour as breaks
        self._plot_with_gaps(ax1, times, magnitudes, gap_threshold_hours,
                            color='b', linewidth=1, markersize=3, alpha=0.7)
        ax1.set_ylabel('Differential Magnitude', fontsize=11)

        # Build title with target info
        title = 'Differential Photometry Light Curve'
        if self.target_name:
            title = f'{self.target_name} - {title}'
        ax1.set_title(title, fontsize=13, fontweight='bold')

        # Add RA/DEC annotation if available
        if self.target_ra is not None and self.target_dec is not None:
            # Convert RA to hours:minutes:seconds
            ra_hours = self.target_ra / 15.0
            ra_h = int(ra_hours)
            ra_m = int((ra_hours - ra_h) * 60)
            ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60

            # Convert DEC to degrees:arcminutes:arcseconds
            dec_sign = '+' if self.target_dec >= 0 else '-'
            dec_abs = abs(self.target_dec)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60

            coord_text = f'RA: {ra_h:02d}h {ra_m:02d}m {ra_s:05.2f}s  |  Dec: {dec_sign}{dec_d:02d}° {dec_m:02d}\' {dec_s:05.2f}"'
            ax1.text(0.98, 0.97, coord_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Magnitudes increase downward

        # Apply compressed time axis formatting
        if hasattr(ax1, '_compressed_time_data'):
            data = ax1._compressed_time_data
            ax1.set_xticks(data['tick_positions'])
            ax1.set_xticklabels(data['tick_labels'])

            # Draw visual gap markers with time labels
            for i, gap_pos in enumerate(data['gap_positions']):
                ax1.axvline(gap_pos, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                # Add gap label showing times before and after
                ylim = ax1.get_ylim()
                y_range = ylim[1] - ylim[0]
                gap_label = data['gap_labels'][i] if i < len(data['gap_labels']) else '||'
                ax1.text(gap_pos, ylim[0] + y_range * 0.02, gap_label,
                        ha='center', va='bottom', fontsize=7, color='gray',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        else:
            # Fallback to date formatting if compressed data not available
            ax1.xaxis.set_major_locator(major_locator)
            ax1.xaxis.set_minor_locator(minor_locator)
            ax1.xaxis.set_major_formatter(date_format)

        # Add statistics
        if len(magnitudes) > 0:
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            ax1.axhline(mean_mag, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_mag:.3f}')
            # Use gap-aware fill_between
            self._fill_between_with_gaps(ax1, times, mean_mag - std_mag, mean_mag + std_mag,
                                        gap_threshold_hours, alpha=0.2, color='red',
                                        label=f'±1σ: {std_mag:.3f}')
            ax1.legend(loc='best')

        # Plot 2: Amplitude (variation from mean)
        if len(magnitudes) > 0:
            mean_mag = np.mean(magnitudes)
            amplitude = magnitudes - mean_mag
            # Plot with gap detection (same as above)
            self._plot_with_gaps(ax2, times, amplitude, gap_threshold_hours,
                                color='g', linewidth=1, markersize=3, alpha=0.7)
            ax2.axhline(0, color='k', linestyle='-', alpha=0.3)

            # Update xlabel to indicate compressed time if gaps detected
            times_np = times.values
            time_diffs = np.diff(times_np).astype('timedelta64[s]').astype(float) / 3600.0
            n_gaps = np.sum(time_diffs > gap_threshold_hours)
            if n_gaps > 0:
                xlabel = f'{xlabel} (gaps removed)'

            ax2.set_xlabel(xlabel, fontsize=11)
            ax2.set_ylabel('Amplitude (mag)', fontsize=11)
            ax2.set_title('Magnitude Variation from Mean', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Apply compressed time axis formatting
            if hasattr(ax2, '_compressed_time_data'):
                data = ax2._compressed_time_data
                ax2.set_xticks(data['tick_positions'])
                ax2.set_xticklabels(data['tick_labels'])

                # Draw visual gap markers (without labels on bottom plot - already shown on top)
                for gap_pos in data['gap_positions']:
                    ax2.axvline(gap_pos, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            else:
                # Fallback to date formatting
                ax2.xaxis.set_major_locator(major_locator)
                ax2.xaxis.set_minor_locator(minor_locator)
                ax2.xaxis.set_major_formatter(date_format)

        # Rotate labels for readability
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f'lightcurve.{format}'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Light curve saved to: {output_path}")

        # Check if gaps were detected
        times_np = times.values
        time_diffs = np.diff(times_np).astype('timedelta64[s]').astype(float) / 3600.0
        n_gaps = np.sum(time_diffs > gap_threshold_hours)
        if n_gaps > 0:
            print(f"  Note: {n_gaps} time gap(s) detected (>{gap_threshold_hours}h) - plot segments separated")

        plt.close()

    def plot_magnitude_vs_frame(self, format: str = 'png', dpi: int = 300):
        """
        Plot magnitude vs time (alternative view with different styling)
        """
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(14, 6))

        # Use statistically cleaned lightcurve
        times = self.clean_times
        magnitudes = self.clean_mags

        if len(times) == 0:
            print("Warning: No valid measurements for plotting")
            return

        # Determine time span and tick interval
        time_span_hours = (times.max() - times.min()).total_seconds() / 3600

        if time_span_hours < 12:
            major_locator = mdates.HourLocator(interval=1)
            minor_locator = mdates.MinuteLocator(interval=15)
            date_format = mdates.DateFormatter('%H:%M')
            xlabel = 'Time (UTC)'
        elif time_span_hours < 48:
            major_locator = mdates.HourLocator(interval=3)
            minor_locator = mdates.HourLocator(interval=1)
            date_format = mdates.DateFormatter('%m/%d %H:%M')
            xlabel = 'Date and Time (UTC)'
        else:
            major_locator = mdates.HourLocator(interval=6)
            minor_locator = mdates.HourLocator(interval=3)
            date_format = mdates.DateFormatter('%m/%d\n%H:%M')
            xlabel = 'Date and Time (UTC)'

        # Use compressed time axis
        gap_threshold_hours = 1.0
        compressed_idx, tick_pos, tick_labels, gap_pos, gap_labels = self._create_compressed_time_axis(
            times, gap_threshold_hours)

        # Plot with scatter for better visibility (using compressed index)
        ax.scatter(compressed_idx, magnitudes, c='blue', s=15, alpha=0.6, edgecolors='none')

        # Add connecting lines with gap detection
        self._plot_with_gaps(ax, times, magnitudes, gap_threshold_hours,
                            color='b', linewidth=0.5, alpha=0.3)

        # Apply compressed axis formatting
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)

        # Draw gap markers with labels
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        for i, gap_position in enumerate(gap_pos):
            ax.axvline(gap_position, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            # Add gap label
            gap_label = gap_labels[i] if i < len(gap_labels) else '||'
            ax.text(gap_position, ylim[0] + y_range * 0.02, gap_label,
                   ha='center', va='bottom', fontsize=7, color='gray',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

        # Update xlabel
        n_gaps = len(gap_pos)
        if n_gaps > 0:
            xlabel = f'{xlabel} (gaps removed)'

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Differential Magnitude', fontsize=11)

        # Build title with target info
        title = 'All Measurements - Scatter Plot'
        if self.target_name:
            title = f'{self.target_name} - {title}'
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Add RA/DEC annotation if available
        if self.target_ra is not None and self.target_dec is not None:
            # Convert RA to hours:minutes:seconds
            ra_hours = self.target_ra / 15.0
            ra_h = int(ra_hours)
            ra_m = int((ra_hours - ra_h) * 60)
            ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60

            # Convert DEC to degrees:arcminutes:arcseconds
            dec_sign = '+' if self.target_dec >= 0 else '-'
            dec_abs = abs(self.target_dec)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60

            coord_text = f'RA: {ra_h:02d}h {ra_m:02d}m {ra_s:05.2f}s  |  Dec: {dec_sign}{dec_d:02d}° {dec_m:02d}\' {dec_s:05.2f}"'
            ax.text(0.98, 0.97, coord_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # Rotate labels (already using compressed time axis with custom ticks)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        output_path = self.output_dir / f'magnitude_vs_time_scatter.{format}'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Scatter plot saved to: {output_path}")
        plt.close()

    def export_csv(self):
        """Export statistically cleaned photometry data to CSV (clean_mags, not raw)."""
        output_path = self.output_dir / 'photometry_results.csv'

        # Export cleaned lightcurve: time and diff_magnitude (cleaned)
        if len(self.clean_times) == 0:
            # Fallback: no valid/cleaned points
            df_export = pd.DataFrame(columns=['time', 'diff_magnitude'])
        else:
            times_iso = pd.Series(self.clean_times).apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x))
            df_export = pd.DataFrame({
                'time': times_iso.values,
                'diff_magnitude': self.clean_mags,
            })
        df_export.to_csv(output_path, index=False)
        print(f"CSV data exported to: {output_path} (cleaned lightcurve)")

    def export_summary(self):
        """Export summary statistics with helpful recommendations (uses cleaned magnitudes)."""
        mags = self.clean_mags

        if len(mags) == 0:
            print("Warning: No valid measurements for summary")
            return

        # Calculate statistics
        mean_mag = np.mean(mags)
        std_mag = np.std(mags)
        min_mag = np.min(mags)
        max_mag = np.max(mags)
        amplitude = max_mag - min_mag
        n_frames = len(self.df)
        n_valid = len(mags)
        success_rate = (n_valid / n_frames) * 100 if n_frames > 0 else 0

        # Generate recommendations
        recommendations = []

        if std_mag < 0.01:
            quality = "EXCELLENT"
            recommendations.append("Your data quality is exceptional! Very low scatter.")
        elif std_mag < 0.03:
            quality = "GOOD"
            recommendations.append("Good data quality with acceptable scatter.")
        elif std_mag < 0.05:
            quality = "FAIR"
            recommendations.append("Moderate scatter detected.")
            recommendations.append("Consider: checking aperture size or adding more reference stars.")
        else:
            quality = "NEEDS IMPROVEMENT"
            recommendations.append("High scatter in measurements.")
            recommendations.append("Try: adjusting aperture size, choosing different reference stars,")
            recommendations.append("     or checking for clouds/poor seeing conditions.")

        if amplitude > 0.1:
            recommendations.append(f"Significant variability detected (amplitude: {amplitude:.3f} mag)!")
            recommendations.append("Your target star may be variable - interesting result!")
        elif amplitude > 0.05:
            recommendations.append(f"Moderate variability detected (amplitude: {amplitude:.3f} mag).")
        else:
            recommendations.append(f"Star appears relatively constant (amplitude: {amplitude:.3f} mag).")

        if success_rate < 90:
            recommendations.append(f"Note: {100-success_rate:.1f}% of frames failed alignment.")
            recommendations.append("This might indicate tracking issues or very different field orientations.")

        # Write summary file
        output_path = self.output_dir / 'RESULTS_SUMMARY.txt'
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("              PHOTOMETRY ANALYSIS RESULTS\n")
            f.write("=" * 70 + "\n\n")

            f.write("DATA QUALITY ASSESSMENT\n")
            f.write("-" * 70 + "\n")
            f.write(f"Overall Quality: {quality}\n")
            f.write(f"Measurement Precision (std): {std_mag:.6f} magnitudes\n")
            f.write(f"Success Rate: {success_rate:.1f}% ({n_valid}/{n_frames} frames)\n\n")

            f.write("MEASUREMENTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Frames Processed........... {n_frames}\n")
            f.write(f"Valid Measurements............... {n_valid}\n")
            f.write(f"Mean Differential Magnitude...... {mean_mag:.6f}\n")
            f.write(f"Standard Deviation............... {std_mag:.6f}\n")
            f.write(f"Minimum Magnitude................ {min_mag:.6f}\n")
            f.write(f"Maximum Magnitude................ {max_mag:.6f}\n")
            f.write(f"Total Amplitude (max-min)........ {amplitude:.6f}\n\n")

            f.write("WHAT THIS MEANS\n")
            f.write("-" * 70 + "\n")
            for rec in recommendations:
                f.write(f"• {rec}\n")
            f.write("\n")

            f.write("NEXT STEPS\n")
            f.write("-" * 70 + "\n")
            f.write("1. Check lightcurve.png - Look for patterns or trends\n")
            f.write("2. Review magnitude_vs_frame.png - Identify any bad frames\n")
            f.write("3. Examine photometry_results.csv - For detailed analysis\n\n")

            if quality in ["EXCELLENT", "GOOD"]:
                f.write("4. Share your results! This data is publication/presentation ready.\n")
            else:
                f.write("4. Consider re-running with adjusted parameters if needed.\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"\n📊 Results summary saved to: {output_path}")

        # Also print to console
        print("\n" + "=" * 70)
        print("              ANALYSIS COMPLETE - RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n✓ Data Quality: {quality}")
        print(f"✓ Measurement Precision: {std_mag:.6f} magnitudes")
        print(f"✓ Variability Amplitude: {amplitude:.6f} magnitudes")
        print(f"✓ Success Rate: {success_rate:.1f}%")
        print("\nKey Findings:")
        for rec in recommendations[:3]:  # Show first 3 recommendations
            print(f"  • {rec}")
        print("\n" + "=" * 70)

    def export_quality_report(self):
        """Export comprehensive quality analysis report"""
        output_path = self.output_dir / 'quality_report.txt'

        # Check if quality metrics exist
        if 'quality_score' not in self.df.columns:
            print("Note: Quality metrics not available in this dataset")
            return

        # Calculate quality statistics
        n_frames = len(self.df)
        n_high_quality = len(self.df[self.df['quality_score'] >= 80])
        n_moderate_quality = len(self.df[(self.df['quality_score'] >= 60) & (self.df['quality_score'] < 80)])
        n_poor_quality = len(self.df[self.df['quality_score'] < 60])

        # Saturation analysis
        saturated_frames = self.df[self.df['is_saturated'] == True]
        n_saturated = len(saturated_frames)
        
        # Check if brightness reduction was applied
        has_brightness_reduction = 'brightness_scale_factor' in self.df.columns
        if has_brightness_reduction:
            reduced_frames = self.df[self.df['brightness_scale_factor'] < 1.0]
            n_reduced = len(reduced_frames)
            if n_reduced > 0:
                avg_scale = reduced_frames['brightness_scale_factor'].mean()
                min_scale = reduced_frames['brightness_scale_factor'].min()

        # Cosmic ray statistics
        total_cosmic_rays = self.df['n_cosmic_rays'].sum()
        avg_cosmic_rays = self.df['n_cosmic_rays'].mean()
        frames_with_many_crs = len(self.df[self.df['n_cosmic_rays'] > 10])

        # Cloud conditions
        n_clear = len(self.df[self.df['cloud_score'] < 20])
        n_partly_cloudy = len(self.df[(self.df['cloud_score'] >= 20) & (self.df['cloud_score'] < 50)])
        n_cloudy = len(self.df[self.df['cloud_score'] >= 50])

        # Moon analysis (if available)
        has_moon_data = 'moon_distance_deg' in self.df.columns and not self.df['moon_distance_deg'].isna().all()
        if has_moon_data:
            avg_moon_dist = self.df['moon_distance_deg'].mean()
            avg_moon_phase = self.df['moon_phase'].mean()
            high_glare_frames = len(self.df[self.df['moon_glare_level'] == 'high'])

        # FWHM/seeing analysis
        if 'fwhm' in self.df.columns and not self.df['fwhm'].isna().all():
            valid_fwhm = self.df['fwhm'].dropna()
            best_seeing = valid_fwhm.min()
            worst_seeing = valid_fwhm.max()
            avg_fwhm = valid_fwhm.mean()
            best_frame_idx = valid_fwhm.idxmin()
            worst_frame_idx = valid_fwhm.idxmax()
            has_fwhm = True
        else:
            has_fwhm = False

        # Write report
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("           FRAME QUALITY ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Frames Analyzed:           {n_frames}\n")
            f.write(f"High Quality Frames (>80):       {n_high_quality} ({n_high_quality/n_frames*100:.1f}%)\n")
            f.write(f"Moderate Quality (60-80):        {n_moderate_quality} ({n_moderate_quality/n_frames*100:.1f}%)\n")
            f.write(f"Poor Quality (<60):              {n_poor_quality} ({n_poor_quality/n_frames*100:.1f}%)\n\n")

            f.write("SATURATION WARNINGS\n")
            f.write("-" * 70 + "\n")
            if n_saturated > 0:
                f.write(f"Frames with Saturation (original): {n_saturated}\n")
                
                # Check if brightness reduction was applied
                if has_brightness_reduction and n_reduced > 0:
                    f.write(f"Brightness Reduction Applied:     {n_reduced} frames\n")
                    f.write(f"  Average scale factor:           {avg_scale:.3f}\n")
                    f.write(f"  Minimum scale factor:           {min_scale:.3f}\n")
                    f.write("\n✓ Brightness reduction was automatically applied to saturated frames\n")
                    f.write("  to mitigate saturation effects during photometry.\n")
                    f.write("  (Saturation values shown are from original frames before reduction)\n")
                else:
                    f.write("\n⚠ WARNING: Saturated frames detected but brightness reduction not applied!\n")
                    f.write("Consider reducing exposure time or using ND filter.\n")
                
                f.write("\nSaturated Frames (original):\n")
                for idx, row in saturated_frames.head(50).iterrows():  # Limit to first 50 to avoid huge reports
                    scale_info = ""
                    if has_brightness_reduction and row.get('brightness_scale_factor', 1.0) < 1.0:
                        scale_info = f" [reduced by {row['brightness_scale_factor']:.3f}]"
                    f.write(f"  Frame {row['frame']}: {row['saturation_fraction']*100:.1f}% saturated pixels{scale_info}\n")
                if len(saturated_frames) > 50:
                    f.write(f"  ... and {len(saturated_frames) - 50} more frames\n")
            else:
                f.write("✓ No saturation detected - excellent!\n")
            f.write("\n")

            f.write("COSMIC RAY DETECTIONS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Cosmic Rays Removed:       {int(total_cosmic_rays)}\n")
            f.write(f"Average per Frame:               {avg_cosmic_rays:.1f}\n")
            f.write(f"Frames with >10 Hits:            {frames_with_many_crs}\n")
            if frames_with_many_crs > 0:
                f.write("\nFrames with high cosmic ray counts:\n")
                high_cr_frames = self.df[self.df['n_cosmic_rays'] > 10].head(10)
                for idx, row in high_cr_frames.iterrows():
                    f.write(f"  Frame {row['frame']}: {row['n_cosmic_rays']} cosmic rays\n")
            f.write("\n")

            f.write("CLOUD CONDITIONS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Clear Conditions (score <20):    {n_clear} frames ({n_clear/n_frames*100:.1f}%)\n")
            f.write(f"Partly Cloudy (20-50):          {n_partly_cloudy} frames ({n_partly_cloudy/n_frames*100:.1f}%)\n")
            f.write(f"Cloudy (>50):                   {n_cloudy} frames ({n_cloudy/n_frames*100:.1f}%)\n")
            if n_cloudy > n_frames * 0.1:
                f.write("\n⚠ Significant cloud contamination detected!\n")
                f.write("Consider filtering frames with cloud_score > 50.\n")
            f.write("\n")

            if has_moon_data:
                f.write("MOON GLARE ANALYSIS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Moon Phase (average):            {avg_moon_phase:.1f}%\n")
                f.write(f"Moon Distance (average):         {avg_moon_dist:.1f}°\n")
                if high_glare_frames > 0:
                    f.write(f"High Glare Warning Frames:       {high_glare_frames}\n\n")
                    f.write("Frames with HIGH moon glare:\n")
                    high_glare = self.df[self.df['moon_glare_level'] == 'high'].head(10)
                    for idx, row in high_glare.iterrows():
                        f.write(f"  Frame {row['frame']}: Moon {row['moon_distance_deg']:.1f}° away, "
                               f"{row['moon_phase']:.0f}% phase\n")
                    f.write("\n⚠ Moon proximity may affect background measurements.\n")
                else:
                    f.write("✓ No significant moon glare detected.\n")
                f.write("\n")

            if has_fwhm:
                f.write("FWHM ANALYSIS (Seeing Quality)\n")
                f.write("-" * 70 + "\n")
                f.write(f"Best Seeing:                     {best_seeing:.2f} pixels (Frame {best_frame_idx})\n")
                f.write(f"Worst Seeing:                    {worst_seeing:.2f} pixels (Frame {worst_frame_idx})\n")
                f.write(f"Average FWHM:                    {avg_fwhm:.2f} pixels\n")
                if avg_fwhm < 3.0:
                    f.write("\n✓ Excellent seeing conditions!\n")
                elif avg_fwhm < 4.0:
                    f.write("\n✓ Good seeing conditions.\n")
                elif avg_fwhm < 5.0:
                    f.write("\n⚠ Moderate seeing - some turbulence present.\n")
                else:
                    f.write("\n⚠ Poor seeing - significant atmospheric turbulence.\n")
                f.write("\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")
            recommendations = []

            if n_saturated > n_frames * 0.05:
                if has_brightness_reduction and n_reduced > 0:
                    recommendations.append(f"• {n_saturated} frames were originally saturated - brightness reduction was applied to {n_reduced} frames")
                else:
                    recommendations.append(f"• {n_saturated} frames saturated - consider excluding these from analysis or enable brightness reduction")
            if n_cloudy > n_frames * 0.2:
                recommendations.append(f"• {n_cloudy} frames cloudy - filter by cloud_score < 50 for best results")
            if has_moon_data and high_glare_frames > 0:
                recommendations.append(f"• Moon glare affected {high_glare_frames} frames - differential photometry should mitigate")
            if n_high_quality > n_frames * 0.8:
                recommendations.append("• Excellent data quality overall - suitable for publication")
            elif n_high_quality > n_frames * 0.6:
                recommendations.append("• Good data quality - suitable for analysis")
            else:
                recommendations.append("• Consider filtering frames with quality_score > 60")

            if recommendations:
                for rec in recommendations:
                    f.write(rec + "\n")
            else:
                f.write("• Data quality is excellent - no specific concerns\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"\n📊 Quality report saved to: {output_path}")


def main(input_dir: Optional[str] = None):
    """
    Main photometry workflow
    
    Args:
        input_dir: Optional path to input folder containing FITS files.
                   If not provided, uses settings.photometry_input_dir or default.
    """
    print("\n" + "=" * 70)
    print("          🔭 SIMPLE PHOTOMETRY TOOL 🔭")
    print("     Measure Your Star's Brightness Over Time")
    print("=" * 70)
    print("\nMade for astronomers, by astronomers - no complexity, just results!")
    print("Processing optimized for large datasets (5-10 GB)\n")

    # Import configuration from SeeStar Photometry settings
    try:
        try:
            from .config import get_settings
        except ImportError:
            from seestar_photometry.config import get_settings

        cfg = get_settings()
        default_input_dir = cfg.input_dir
        OUTPUT_DIR = cfg.output_dir
        APERTURE_RADIUS = cfg.aperture_radius
        SKY_ANNULUS_INNER = cfg.sky_annulus_inner
        SKY_ANNULUS_OUTER = cfg.sky_annulus_outer
        DOWNSAMPLE_FACTOR = cfg.downsample_factor
        MAX_CONTROL_POINTS = cfg.max_control_points
        REFERENCE_FRAME_CANDIDATES = cfg.reference_frame_candidates
        PLOT_FORMAT = cfg.plot_format
        PLOT_DPI = cfg.plot_dpi
        print("✓ Configuration loaded from SeeStar Photometry settings")
    except Exception as e:
        # Fallback to defaults if settings not available
        default_input_dir = 'data/input'
        OUTPUT_DIR = 'data/output'
        APERTURE_RADIUS = 8
        SKY_ANNULUS_INNER = 12
        SKY_ANNULUS_OUTER = 20
        DOWNSAMPLE_FACTOR = 2
        MAX_CONTROL_POINTS = 100
        REFERENCE_FRAME_CANDIDATES = 20
        PLOT_FORMAT = 'png'
        PLOT_DPI = 300
        print(f"⚠ Using default configuration ({e})")

    # Use provided input_dir if given, otherwise use settings/default
    INPUT_DIR = input_dir if input_dir is not None else default_input_dir
    if input_dir is not None:
        print(f"✓ Using provided input directory: {INPUT_DIR}")

    # Step 1: Load FITS dataset
    print("\n" + "─" * 70)
    print("STEP 1/6: Loading your FITS files")
    print("─" * 70)
    dataset = FITSDataset(INPUT_DIR)
    print(f"✓ Loaded {len(dataset)} frames successfully")

    # Step 2: Compute alignments
    print("\n" + "─" * 70)
    print("STEP 2/6: Finding the best reference frame and aligning images")
    print("─" * 70)

    aligner = FrameAligner(dataset)

    # Check for existing alignment cache
    cache_loaded = aligner.load_alignment_cache(OUTPUT_DIR)

    if cache_loaded:
        print("  Using cached alignment (saves time!)")
        print("  Delete 'output/alignment_cache.npz' to force re-alignment")
    else:
        print("This ensures all your images are perfectly aligned for analysis...")
        # Intelligently select best reference frame from first N frames
        aligner.select_best_reference(n_candidates=REFERENCE_FRAME_CANDIDATES)
        aligner.compute_alignments(max_control_points=MAX_CONTROL_POINTS, cache_dir=OUTPUT_DIR)

    print("✓ Alignment ready!")

    # Step 3: Interactive star selection
    print("\n" + "─" * 70)
    print("STEP 3/6: Select your stars (Interactive)")
    print("─" * 70)
    print("A window will open showing your image.")
    print("Follow the on-screen instructions to select stars.")
    print("Don't worry - you can undo mistakes with right-click!")
    if aligner.reference_wcs:
        print("✓ Variable star checking enabled - warnings will appear if you select a known variable!\n")
    else:
        print("\n")
    input("Press ENTER to open the star selection window...")

    selector = StarSelector(aligner.reference_image,
                          downsample_factor=DOWNSAMPLE_FACTOR,
                          wcs=aligner.reference_wcs,
                          header=aligner.reference_header)
    star_data = selector.select_stars(
        aperture_radius=APERTURE_RADIUS,
        sky_annulus_inner=SKY_ANNULUS_INNER,
        sky_annulus_outer=SKY_ANNULUS_OUTER
    )

    print(f"\n✓ Great! Target selected at: ({star_data['target'][0]:.0f}, {star_data['target'][1]:.0f})")
    print(f"✓ Using {len(star_data['references'])} reference star(s) for differential photometry")

    # Step 4: Perform photometry on all frames
    print("\n" + "─" * 70)
    print("STEP 4/6: Measuring star brightness in all frames")
    print("─" * 70)
    print("This is where the magic happens! Measuring photometry...")
    print("(This may take a while for large datasets - grab a coffee! ☕)\n")
    photometry = PhotometryEngine(dataset, aligner, star_data)
    photometry.process_all_frames()
    print("\n✓ Photometry complete!")

    # Step 5: Get results
    print("\n" + "─" * 70)
    print("STEP 5/6: Analyzing your data")
    print("─" * 70)
    results_df = photometry.get_results_dataframe()
    print(f"✓ Analyzed {len(results_df)} measurements")

    # Step 6: Export results
    print("\n" + "─" * 70)
    print("STEP 6/6: Creating plots and exporting results")
    print("─" * 70)
    exporter = ResultsExporter(results_df, OUTPUT_DIR)

    print("Creating light curve plots...")
    exporter.plot_lightcurve(format=PLOT_FORMAT, dpi=PLOT_DPI)
    exporter.plot_magnitude_vs_frame(format=PLOT_FORMAT, dpi=PLOT_DPI)

    print("Exporting data to CSV...")
    exporter.export_csv()

    print("Generating summary report...")
    exporter.export_summary()

    print("Generating quality report...")
    exporter.export_quality_report()

    print("\n" + "=" * 70)
    print("                    🎉 ALL DONE! 🎉")
    print("=" * 70)
    print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
    print("\n📊 Open RESULTS_SUMMARY.txt to see your analysis!")
    print("📈 Check lightcurve.png to visualize your star's brightness")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Photometry Tool - Process FITS images for differential photometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python photometry_engine.py
  python photometry_engine.py --input /path/to/fits/folder
  python photometry_engine.py -i ./my_images
        """
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Input folder containing FITS files (default: uses settings.photometry_input_dir or data/photometry/input)'
    )
    
    args = parser.parse_args()
    main(input_dir=args.input)
