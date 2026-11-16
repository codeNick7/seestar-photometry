"""
Enhanced Photometry Module

Optimized aperture photometry for low SNR images from small telescopes.

Features:
- Dynamic aperture sizing based on measured FWHM
- Sub-pixel centroid tracking for each frame
- SNR calculation and optimal aperture selection
- Sigma-clipped sky background estimation
- Ensemble photometry with weighted averaging

Optimized for SeeStar S50:
- 20mm aperture (narrow)
- 2-second exposures (short)
- Variable tracking quality
"""

import warnings
import numpy as np
from scipy import ndimage
from astropy.stats import sigma_clipped_stats, sigma_clip
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_com
from typing import Tuple, Optional, Dict, List
import logging

# Suppress centroid fitting warnings
warnings.filterwarnings('ignore', message='.*fit may not have converged.*')
warnings.filterwarnings('ignore', category=UserWarning, module='photutils')

logger = logging.getLogger(__name__)


class EnhancedPhotometry:
    """
    Enhanced aperture photometry with SNR optimization.

    Provides:
    - Adaptive aperture sizing based on PSF width
    - Sub-pixel centroid refinement
    - Robust background estimation
    - SNR calculation and reporting
    """

    def __init__(self,
                 aperture_radius: Optional[float] = None,
                 sky_annulus_inner: Optional[float] = None,
                 sky_annulus_outer: Optional[float] = None,
                 auto_aperture: bool = True):
        """
        Initialize enhanced photometry.

        Args:
            aperture_radius: Fixed aperture radius in pixels (or None for auto)
            sky_annulus_inner: Inner sky annulus radius
            sky_annulus_outer: Outer sky annulus radius
            auto_aperture: If True, optimize aperture for maximum SNR
        """
        self.aperture_radius = aperture_radius
        self.sky_annulus_inner = sky_annulus_inner
        self.sky_annulus_outer = sky_annulus_outer
        self.auto_aperture = auto_aperture

        # Default aperture parameters (will be overridden if auto_aperture=True)
        self._default_aperture = 8.0
        self._default_sky_inner = 12.0
        self._default_sky_outer = 20.0

        # Measured PSF FWHM for each frame
        self.measured_fwhm = None

    def measure_star(self, image: np.ndarray,
                     x: float, y: float,
                     fwhm: Optional[float] = None,
                     refine_centroid: bool = True) -> Dict:
        """
        Measure star flux with enhanced techniques.

        Args:
            image: 2D image array
            x, y: Initial star position
            fwhm: Measured FWHM (for aperture sizing)
            refine_centroid: If True, refine position before measurement

        Returns:
            Dictionary with flux, sky, snr, and metadata
        """
        result = {
            'flux': None,
            'flux_err': None,
            'sky': None,
            'sky_err': None,
            'snr': None,
            'x': x,
            'y': y,
            'aperture_radius': None,
            'centroid_refined': False,
            'valid': False
        }

        try:
            # Check bounds
            if not self._check_bounds(image, x, y, margin=25):
                return result

            # Refine centroid if requested
            if refine_centroid:
                x_refined, y_refined = self._refine_centroid(image, x, y)
                if x_refined is not None:
                    result['x'] = x_refined
                    result['y'] = y_refined
                    result['centroid_refined'] = True
                    x, y = x_refined, y_refined

            # Determine aperture parameters
            if self.auto_aperture and fwhm is not None:
                aperture_r, sky_inner, sky_outer = self._optimal_aperture(fwhm)
            else:
                aperture_r = self.aperture_radius or self._default_aperture
                sky_inner = self.sky_annulus_inner or self._default_sky_inner
                sky_outer = self.sky_annulus_outer or self._default_sky_outer

            result['aperture_radius'] = aperture_r

            # Create apertures
            aperture = CircularAperture((x, y), r=aperture_r)
            annulus = CircularAnnulus((x, y), r_in=sky_inner, r_out=sky_outer)

            # Robust sky estimation using sigma-clipped statistics
            sky_per_pixel, sky_std = self._estimate_sky_robust(image, x, y, sky_inner, sky_outer)

            if sky_per_pixel is None:
                return result

            result['sky'] = sky_per_pixel
            result['sky_err'] = sky_std

            # Measure aperture sum
            phot = aperture_photometry(image, aperture)
            aperture_sum = phot['aperture_sum'][0]
            aperture_area = aperture.area

            # Background-subtracted flux
            flux = aperture_sum - (sky_per_pixel * aperture_area)
            result['flux'] = flux

            # Calculate flux uncertainty
            # Using CCD equation: sigma_flux = sqrt(flux + n_pix * (sky_var + readnoise^2))
            # For simplicity, assume readnoise dominated by sky noise
            read_noise = 10.0  # Typical for CMOS sensors
            sky_variance = sky_std ** 2

            # Total noise in aperture
            noise_squared = abs(flux) + aperture_area * (sky_variance + read_noise**2)
            flux_err = np.sqrt(noise_squared)
            result['flux_err'] = flux_err

            # Signal-to-noise ratio
            if flux_err > 0:
                snr = flux / flux_err
                result['snr'] = max(0, snr)  # Can't be negative

            result['valid'] = flux > 0 and result['snr'] is not None and result['snr'] > 0

        except Exception as e:
            logger.debug(f"Measurement failed at ({x:.1f}, {y:.1f}): {e}")

        return result

    def _check_bounds(self, image: np.ndarray, x: float, y: float, margin: int = 15) -> bool:
        """Check if position is within valid image bounds.

        Uses a smaller margin (15 pixels) to allow measurements closer to edges.
        The sky annulus (outer radius ~20 pixels) will handle edge cases gracefully.
        """
        h, w = image.shape
        in_bounds = (margin <= x < w - margin and margin <= y < h - margin)
        if not in_bounds:
            logger.debug(f"Position ({x:.1f}, {y:.1f}) out of bounds for image {w}x{h} with margin {margin}")
        return in_bounds

    def _refine_centroid(self, image: np.ndarray, x: float, y: float,
                        box_size: int = 15) -> Tuple[Optional[float], Optional[float]]:
        """
        Refine centroid position using center of mass (fast, robust for low SNR).

        Uses weighted center of mass which is more robust than Gaussian fitting
        for faint stars with low SNR.

        Args:
            image: Image array
            x, y: Initial position
            box_size: Size of extraction box

        Returns:
            Refined (x, y) or (None, None) if failed
        """
        try:
            half_box = box_size // 2
            x_int, y_int = int(round(x)), int(round(y))

            # Check bounds
            if (x_int - half_box < 0 or x_int + half_box >= image.shape[1] or
                y_int - half_box < 0 or y_int + half_box >= image.shape[0]):
                return None, None

            # Extract cutout
            cutout = image[y_int - half_box:y_int + half_box + 1,
                          x_int - half_box:x_int + half_box + 1].copy()

            # Subtract background using sigma-clipped median
            _, median, _ = sigma_clipped_stats(cutout, sigma=3.0, maxiters=3)
            cutout = cutout - median

            # Clip negative values
            cutout = np.clip(cutout, 0, None)

            # Check if there's enough signal
            total_flux = np.sum(cutout)
            if total_flux <= 0:
                return None, None

            # Use center of mass (fast and robust for low SNR)
            y_grid, x_grid = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
            x_cen = np.sum(x_grid * cutout) / total_flux
            y_cen = np.sum(y_grid * cutout) / total_flux

            # Check if centroid is within reasonable bounds
            if (x_cen < 1 or x_cen > cutout.shape[1] - 2 or
                y_cen < 1 or y_cen > cutout.shape[0] - 2):
                return None, None

            # Convert to image coordinates
            x_refined = x_int - half_box + x_cen
            y_refined = y_int - half_box + y_cen

            # Sanity check - shouldn't move too far (max 3 pixels)
            if abs(x_refined - x) > 3 or abs(y_refined - y) > 3:
                return None, None

            return x_refined, y_refined

        except Exception as e:
            logger.debug(f"Centroid refinement failed: {e}")
            return None, None

    def _optimal_aperture(self, fwhm: float) -> Tuple[float, float, float]:
        """
        Calculate optimal aperture size based on FWHM.

        For point sources, optimal aperture is typically:
        - Aperture radius = 1.5-2.0 × FWHM for maximum SNR
        - Sky annulus inner = 2.5-3.0 × FWHM (avoid star light)
        - Sky annulus outer = 4.0-5.0 × FWHM

        Args:
            fwhm: Measured FWHM in pixels

        Returns:
            (aperture_radius, sky_inner, sky_outer)
        """
        # Constrain FWHM to reasonable range
        fwhm = np.clip(fwhm, 2.0, 10.0)

        # Optimal aperture for maximum SNR (1.5× FWHM is theoretical optimum)
        # Use slightly larger (1.8×) to capture more flux with some seeing variation
        aperture_radius = 1.8 * fwhm

        # Sky annulus (far enough to avoid star wings)
        sky_inner = 2.5 * fwhm
        sky_outer = 4.0 * fwhm

        # Minimum sizes to ensure enough pixels
        aperture_radius = max(aperture_radius, 5.0)
        sky_inner = max(sky_inner, aperture_radius + 3.0)
        sky_outer = max(sky_outer, sky_inner + 5.0)

        return aperture_radius, sky_inner, sky_outer

    def _estimate_sky_robust(self, image: np.ndarray, x: float, y: float,
                             r_in: float, r_out: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Robust sky background estimation using sigma-clipped statistics.

        Args:
            image: Image array
            x, y: Star position
            r_in, r_out: Annulus radii

        Returns:
            (sky_per_pixel, sky_std) or (None, None) if failed
        """
        try:
            # Create mask for annulus region
            y_grid, x_grid = np.ogrid[:image.shape[0], :image.shape[1]]
            r_squared = (x_grid - x)**2 + (y_grid - y)**2

            annulus_mask = (r_squared >= r_in**2) & (r_squared <= r_out**2)

            # Extract pixels in annulus
            sky_pixels = image[annulus_mask]

            if len(sky_pixels) < 10:
                return None, None

            # Sigma-clipped statistics for robust background
            # 3-sigma clipping removes stars and cosmic rays in annulus
            clipped = sigma_clip(sky_pixels, sigma=3.0, maxiters=5)
            sky_mean = np.mean(clipped.data[~clipped.mask])
            sky_std = np.std(clipped.data[~clipped.mask])

            return sky_mean, sky_std

        except Exception as e:
            logger.debug(f"Sky estimation failed: {e}")
            return None, None

    def find_optimal_aperture_curve(self, image: np.ndarray, x: float, y: float,
                                    r_min: float = 3.0, r_max: float = 20.0,
                                    n_steps: int = 20) -> Dict:
        """
        Generate curve of growth to find optimal aperture.

        Measures flux at different aperture sizes to find the radius
        that maximizes SNR.

        Args:
            image: Image array
            x, y: Star position
            r_min, r_max: Aperture radius range
            n_steps: Number of aperture sizes to test

        Returns:
            Dictionary with aperture_radii, fluxes, snrs, and optimal_radius
        """
        radii = np.linspace(r_min, r_max, n_steps)
        fluxes = []
        snrs = []

        # Estimate sky using large annulus
        sky_per_pixel, sky_std = self._estimate_sky_robust(image, x, y, r_max + 5, r_max + 15)

        if sky_per_pixel is None:
            return {'optimal_radius': self._default_aperture}

        read_noise = 10.0

        for r in radii:
            try:
                aperture = CircularAperture((x, y), r=r)
                phot = aperture_photometry(image, aperture)
                aperture_sum = phot['aperture_sum'][0]
                area = aperture.area

                flux = aperture_sum - (sky_per_pixel * area)
                noise = np.sqrt(abs(flux) + area * (sky_std**2 + read_noise**2))
                snr = flux / noise if noise > 0 else 0

                fluxes.append(flux)
                snrs.append(max(0, snr))

            except Exception:
                fluxes.append(np.nan)
                snrs.append(0)

        # Find optimal radius (maximum SNR)
        snrs_array = np.array(snrs)
        valid_mask = ~np.isnan(snrs_array) & (snrs_array > 0)

        if not np.any(valid_mask):
            optimal_radius = self._default_aperture
        else:
            optimal_idx = np.argmax(snrs_array[valid_mask])
            valid_radii = radii[valid_mask]
            optimal_radius = valid_radii[optimal_idx]

        return {
            'aperture_radii': radii.tolist(),
            'fluxes': fluxes,
            'snrs': snrs,
            'optimal_radius': optimal_radius,
            'max_snr': max(snrs) if snrs else 0
        }


def measure_differential_photometry(image: np.ndarray,
                                    target_pos: Tuple[float, float],
                                    ref_positions: List[Tuple[float, float]],
                                    fwhm: Optional[float] = None,
                                    aperture_radius: Optional[float] = None) -> Dict:
    """
    Measure differential photometry for target relative to reference stars.

    Args:
        image: Image array
        target_pos: (x, y) of target star
        ref_positions: List of (x, y) for reference stars
        fwhm: Measured FWHM for aperture sizing
        aperture_radius: Fixed aperture (overrides auto-sizing)

    Returns:
        Dictionary with differential magnitude, SNR, and all measurements
    """
    photometer = EnhancedPhotometry(
        aperture_radius=aperture_radius,
        auto_aperture=(aperture_radius is None)
    )

    result = {
        'diff_magnitude': np.nan,
        'diff_mag_err': np.nan,
        'target_flux': np.nan,
        'target_snr': np.nan,
        'ref_flux_avg': np.nan,
        'ref_snr_avg': np.nan,
        'n_valid_refs': 0,
        'target_measurement': None,
        'ref_measurements': []
    }

    # Measure target
    target_meas = photometer.measure_star(image, target_pos[0], target_pos[1], fwhm=fwhm)
    result['target_measurement'] = target_meas

    if not target_meas['valid']:
        logger.debug(f"Target measurement failed at ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
        return result

    result['target_flux'] = target_meas['flux']
    result['target_snr'] = target_meas['snr']

    # Measure references
    ref_fluxes = []
    ref_snrs = []

    for ref_x, ref_y in ref_positions:
        ref_meas = photometer.measure_star(image, ref_x, ref_y, fwhm=fwhm)
        result['ref_measurements'].append(ref_meas)

        if ref_meas['valid'] and ref_meas['flux'] > 0:
            ref_fluxes.append(ref_meas['flux'])
            ref_snrs.append(ref_meas['snr'])

    result['n_valid_refs'] = len(ref_fluxes)

    if not ref_fluxes:
        logger.debug(f"No valid reference measurements from {len(ref_positions)} positions")
        return result

    # Weighted average of reference fluxes (weight by SNR^2)
    weights = np.array(ref_snrs) ** 2
    ref_flux_avg = np.average(ref_fluxes, weights=weights)
    ref_snr_avg = np.mean(ref_snrs)

    result['ref_flux_avg'] = ref_flux_avg
    result['ref_snr_avg'] = ref_snr_avg

    # Differential magnitude
    if ref_flux_avg > 0 and target_meas['flux'] > 0:
        diff_mag = -2.5 * np.log10(target_meas['flux'] / ref_flux_avg)
        result['diff_magnitude'] = diff_mag

        # Magnitude error propagation
        # sigma_mag ≈ 1.0857 / SNR for small errors
        target_mag_err = 1.0857 / target_meas['snr'] if target_meas['snr'] > 0 else np.inf
        ref_mag_err = 1.0857 / ref_snr_avg if ref_snr_avg > 0 else np.inf
        diff_mag_err = np.sqrt(target_mag_err**2 + ref_mag_err**2)
        result['diff_mag_err'] = diff_mag_err

    return result


def measure_fwhm_from_stars(image: np.ndarray, n_stars: int = 10) -> Optional[float]:
    """
    Measure average FWHM from brightest stars in the image.

    Args:
        image: Image array
        n_stars: Number of stars to use for measurement

    Returns:
        Median FWHM in pixels, or None if measurement fails
    """
    try:
        from photutils.detection import DAOStarFinder
        from photutils.psf import fit_2dgaussian

        # Background statistics
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)

        # Find stars
        daofind = DAOStarFinder(
            fwhm=4.0,
            threshold=5.0 * std,
            brightest=n_stars * 2,  # Find extra in case some fail
            sharplo=0.1,
            sharphi=2.0
        )

        sources = daofind(image - median)

        if sources is None or len(sources) < 3:
            return None

        # Measure FWHM for each star
        fwhm_values = []

        x_col = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
        y_col = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
        for source in sources[:n_stars]:
            x, y = source[x_col], source[y_col]

            # Extract cutout
            box_size = 15
            half_box = box_size // 2
            x_int, y_int = int(round(x)), int(round(y))

            if (x_int - half_box < 0 or x_int + half_box >= image.shape[1] or
                y_int - half_box < 0 or y_int + half_box >= image.shape[0]):
                continue

            cutout = image[y_int - half_box:y_int + half_box + 1,
                          x_int - half_box:x_int + half_box + 1] - median

            # Fit 2D Gaussian to get FWHM
            try:
                # Simple method: use second moments
                y_grid, x_grid = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
                cutout_positive = np.clip(cutout, 0, None)
                total = np.sum(cutout_positive)

                if total > 0:
                    x_mean = np.sum(x_grid * cutout_positive) / total
                    y_mean = np.sum(y_grid * cutout_positive) / total

                    x_var = np.sum((x_grid - x_mean)**2 * cutout_positive) / total
                    y_var = np.sum((y_grid - y_mean)**2 * cutout_positive) / total

                    sigma_x = np.sqrt(x_var)
                    sigma_y = np.sqrt(y_var)

                    # FWHM = 2.355 * sigma (for Gaussian)
                    fwhm = 2.355 * np.sqrt(sigma_x * sigma_y)

                    if 1.5 < fwhm < 15.0:  # Reasonable range
                        fwhm_values.append(fwhm)

            except Exception:
                continue

        if fwhm_values:
            return np.median(fwhm_values)
        return None

    except Exception as e:
        logger.warning(f"FWHM measurement failed: {e}")
        return None
