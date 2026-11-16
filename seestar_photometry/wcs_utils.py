"""
WCS (World Coordinate System) Utilities for Photometry
Creates WCS from FITS headers and provides coordinate transformations
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Flag to track if we've already logged WCS creation details
_wcs_logged = False
_wcs_frame_count = 0


def create_wcs_from_header(header: fits.Header, image_shape: Tuple[int, int]) -> Optional[WCS]:
    """
    Create a simple WCS from FITS header RA/DEC keywords

    Args:
        header: FITS header containing RA/DEC
        image_shape: (height, width) of the image

    Returns:
        WCS object or None if creation fails
    """
    global _wcs_logged, _wcs_frame_count
    import sys

    try:
        # Try to extract RA/DEC from various possible keywords
        ra_deg = None
        dec_deg = None

        # Check for RA keywords (in order of preference)
        for ra_key in ['RA', 'OBJCTRA', 'CRVAL1', 'TELRA']:
            if ra_key in header:
                ra_value = header[ra_key]
                if isinstance(ra_value, (int, float)):
                    ra_deg = float(ra_value)
                    logger.debug(f"Found RA in header: {ra_key} = {ra_deg}°")
                    break
                elif isinstance(ra_value, str):
                    # Try to parse sexagesimal format
                    try:
                        coord = SkyCoord(ra_value, unit=(u.hourangle, u.deg))
                        ra_deg = coord.ra.deg
                        logger.debug(f"Parsed RA from {ra_key}: {ra_value} → {ra_deg}°")
                        break
                    except:
                        continue

        # Check for DEC keywords
        for dec_key in ['DEC', 'OBJCTDEC', 'CRVAL2', 'TELDEC']:
            if dec_key in header:
                dec_value = header[dec_key]
                if isinstance(dec_value, (int, float)):
                    dec_deg = float(dec_value)
                    logger.debug(f"Found DEC in header: {dec_key} = {dec_deg}°")
                    break
                elif isinstance(dec_value, str):
                    # Try to parse sexagesimal format
                    try:
                        coord = SkyCoord(f"0h0m0s {dec_value}")
                        dec_deg = coord.dec.deg
                        logger.debug(f"Parsed DEC from {dec_key}: {dec_value} → {dec_deg}°")
                        break
                    except:
                        continue

        if ra_deg is None or dec_deg is None:
            logger.debug("Could not find RA/DEC in FITS header")
            return None

        # Get pixel scale (arcsec/pixel)
        pixel_scale = None
        for key in ['PIXSCALE', 'SCALE', 'SECPIX']:
            if key in header:
                pixel_scale = float(header[key])
                logger.debug(f"Found pixel scale: {key} = {pixel_scale} arcsec/pixel")
                break

        # If no pixel scale, try to calculate from focal length
        if pixel_scale is None and 'FOCALLEN' in header and 'XPIXSZ' in header:
            focal_length_mm = float(header['FOCALLEN'])
            pixel_size_um = float(header['XPIXSZ'])
            # pixel_scale (arcsec/pixel) = 206.265 * pixel_size_um / focal_length_mm
            pixel_scale = 206.265 * pixel_size_um / focal_length_mm
            logger.debug(f"Calculated pixel scale from focal length: {pixel_scale:.2f} arcsec/pixel")

        # Default pixel scale for SeeStarS50 if still not found
        # SeeStar S50 specs: f/5, 250mm focal length, 2.9um pixels
        # Pixel scale = 206.265 * pixel_size_um / focal_length_mm = 206.265 * 2.9 / 250 ≈ 2.39 arcsec/pixel
        # But actual measured is closer to 1.21 arcsec/pixel (possibly 2x2 binning)
        if pixel_scale is None:
            pixel_scale = 1.21  # SeeStarS50 measured pixel scale
            logger.debug(f"Using default pixel scale: {pixel_scale} arcsec/pixel")

        # Create WCS object
        wcs = WCS(naxis=2)

        # Set reference pixel (center of image)
        height, width = image_shape
        wcs.wcs.crpix = [width / 2.0, height / 2.0]

        # Set reference coordinates (RA/DEC of center)
        wcs.wcs.crval = [ra_deg, dec_deg]

        # Set pixel scale (convert arcsec to degrees)
        pixel_scale_deg = pixel_scale / 3600.0
        wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # RA decreases to the right

        # Set coordinate type
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Gnomonic projection

        # Increment frame counter and update status on same line
        _wcs_frame_count += 1

        if not _wcs_logged:
            # First time: show WCS details
            logger.debug(f"WCS enabled: RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°, scale={pixel_scale:.2f}\"/pix")
            _wcs_logged = True

        # Update WCS processing count on same line
        print(f"\rProcessing WCS: {_wcs_frame_count} frames", end='', flush=True, file=sys.stderr)

        return wcs

    except Exception as e:
        logger.error(f"Error creating WCS from header: {e}", exc_info=True)
        return None


def pixel_to_world(wcs: WCS, x: float, y: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to world coordinates (RA/DEC)

    Args:
        wcs: WCS object
        x, y: Pixel coordinates

    Returns:
        (ra_deg, dec_deg)
    """
    try:
        world = wcs.pixel_to_world(x, y)
        return world.ra.deg, world.dec.deg
    except Exception as e:
        logger.error(f"Error converting pixel to world: {e}")
        return None, None


def world_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    """
    Convert world coordinates (RA/DEC) to pixel coordinates

    Args:
        wcs: WCS object
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees

    Returns:
        (x, y) pixel coordinates
    """
    try:
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
        x, y = wcs.world_to_pixel(coord)
        return float(x), float(y)
    except Exception as e:
        logger.error(f"Error converting world to pixel: {e}")
        return None, None


def finalize_wcs_logging():
    """Print newline to finish the inline WCS status update"""
    global _wcs_frame_count
    import sys
    if _wcs_frame_count > 0:
        print(f"\rProcessed WCS for {_wcs_frame_count} frames" + " " * 20, file=sys.stderr)


def get_field_center(header: fits.Header) -> Optional[Tuple[float, float]]:
    """
    Extract field center RA/DEC from FITS header

    Returns:
        (ra_deg, dec_deg) or None if not found
    """
    try:
        ra_deg = None
        dec_deg = None

        for ra_key in ['RA', 'OBJCTRA', 'CRVAL1', 'TELRA']:
            if ra_key in header:
                ra_deg = float(header[ra_key])
                break

        for dec_key in ['DEC', 'OBJCTDEC', 'CRVAL2', 'TELDEC']:
            if dec_key in header:
                dec_deg = float(header[dec_key])
                break

        if ra_deg is not None and dec_deg is not None:
            return ra_deg, dec_deg

        return None

    except Exception as e:
        logger.error(f"Error extracting field center: {e}")
        return None
