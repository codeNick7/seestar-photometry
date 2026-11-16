"""
Star Identification and Variable Star Checking
Integrates with SIMBAD star catalog and AAVSO VSX
"""

import logging
import requests
import socket
from typing import List, Dict, Tuple, Optional
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)

_ONLINE_STATUS = None


def is_network_available() -> bool:
    """Quick check if internet connectivity and DNS are available (cached)"""
    global _ONLINE_STATUS
    if _ONLINE_STATUS is not None:
        return _ONLINE_STATUS
    try:
        socket.gethostbyname("vsx.aavso.org")
        _ONLINE_STATUS = True
    except Exception:
        _ONLINE_STATUS = False
    return _ONLINE_STATUS

# Import star field catalog service
try:
    from .catalog import get_star_field
    STAR_CATALOG_AVAILABLE = True
except ImportError:
    try:
        from seestar_photometry.catalog import get_star_field
        STAR_CATALOG_AVAILABLE = True
    except ImportError:
        logger.warning("Star field catalog service not available")
        STAR_CATALOG_AVAILABLE = False


def query_stars_around_field(ra_deg: float, dec_deg: float, radius_deg: float = 5.0, max_mag: float = 12.0) -> List[Dict]:
    """
    Query stars around a field center using SIMBAD star catalog

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        radius_deg: Search radius in degrees
        max_mag: Maximum magnitude

    Returns:
        List of star dictionaries with ra, dec, magnitude
    """
    if not STAR_CATALOG_AVAILABLE:
        logger.warning("Star catalog not available, returning empty list")
        return []

    try:
        stars = get_star_field(ra_deg, dec_deg, radius_deg, max_mag)
        logger.info(f"Found {len(stars)} stars around RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°")
        return stars
    except Exception as e:
        logger.error(f"Error querying stars: {e}", exc_info=True)
        return []


def query_variable_stars_vsx(ra_deg: float, dec_deg: float, radius_deg: float = 5.0) -> List[Dict]:
    """
    Query AAVSO Variable Star Index (VSX) for known variable stars

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        radius_deg: Search radius in degrees

    Returns:
        List of variable star dictionaries with ra, dec, name, type, period
    """
    if not is_network_available():
        return []
    try:
        # AAVSO VSX API endpoint
        # Documentation: https://www.aavso.org/vsx/index.php?view=api

        # VSX uses RA in hours, DEC in degrees, and radius in arcminutes
        ra_hours = ra_deg / 15.0
        radius_arcmin = radius_deg * 60.0

        # VSX API query (using their public search interface)
        # Format: cone search with RA/DEC center and radius
        url = "https://www.aavso.org/vsx/index.php"
        params = {
            'view': 'query.votable',
            'coords': f"{ra_hours:.6f} {dec_deg:+.6f}",
            'size': str(radius_arcmin),
            'unit': '3',  # 3 = arcminutes
            'format': 'json'
        }

        logger.info(f"Querying AAVSO VSX for variables around RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°")

        # Note: VSX API can be slow or rate-limited
        response = requests.get(url, params=params, timeout=2, headers={'User-Agent': 'SeeStar-Photometry/1.0'})

        if response.status_code != 200:
            logger.warning(f"VSX query returned status {response.status_code}")
            return []

        # Parse response (format depends on VSX API response)
        # For now, return empty - full implementation would parse VOTable or JSON
        logger.debug("VSX response parsing not fully implemented")
        return []

    except requests.Timeout:
        logger.warning("VSX query timed out")
        return []
    except Exception as e:
        logger.error(f"Error querying VSX: {e}", exc_info=True)
        return []


def query_variable_stars_simbad(ra_deg: float, dec_deg: float, radius_deg: float = 5.0) -> List[Dict]:
    """
    Query SIMBAD for known variable stars (fallback if VSX unavailable)

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        radius_deg: Search radius in degrees

    Returns:
        List of variable star dictionaries
    """
    if not is_network_available():
        return []
    try:
        # SIMBAD coordinate query for variable stars
        radius_arcmin = radius_deg * 60.0
        coord_str = f"{ra_deg:.6f} {dec_deg:+.6f}"

        params = {
            'Coord': coord_str,
            'Radius': f"{radius_arcmin:.1f}",
            'Radius.unit': 'arcmin',
            'output.format': 'ASCII',
            'output.max': 200,
            'obj.otype': 'V*'  # Variable stars only
        }

        url = "https://simbad.cds.unistra.fr/simbad/sim-coo"

        logger.debug(f"Querying SIMBAD for variable stars")
        response = requests.get(url, params=params, timeout=2,
                              headers={'User-Agent': 'SeeStar-Photometry/1.0'})

        if response.status_code != 200:
            logger.debug(f"SIMBAD query returned status {response.status_code}")
            return []

        # Parse SIMBAD response for variable stars
        variables = []
        lines = response.text.split('\n')

        for line in lines:
            if '|' in line and 'V*' in line:
                # Basic parsing - would need refinement
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    # Extract coordinates and name
                    # This is a simplified parser
                    try:
                        variables.append({
                            'name': parts[1] if len(parts) > 1 else 'Unknown',
                            'type': 'Variable',
                            'note': 'From SIMBAD'
                        })
                    except:
                        continue

        logger.debug(f"Found {len(variables)} variable stars in SIMBAD")
        return variables

    except Exception as e:
        logger.error(f"Error querying SIMBAD for variables: {e}")
        return []


# Curated database of primary variable stars for instant offline verification in the field
OFFLINE_KNOWN_VARIABLES = [
    {"name": "Algol (β Persei)", "ra": 47.0422, "dec": 40.9556, "type": "EA/SD (Eclipsing)", "period": 2.867328},
    {"name": "ρ Persei", "ra": 45.9619, "dec": 38.8402, "type": "SRB (Semiregular)", "period": 50.0},
    {"name": "δ Cephei", "ra": 337.2875, "dec": 58.4152, "type": "DCEP (Classical Cepheid)", "period": 5.366341},
    {"name": "β Lyrae (Sheliak)", "ra": 282.5201, "dec": 33.3627, "type": "EB (Eclipsing Binary)", "period": 12.94061},
    {"name": "η Aquilae", "ra": 298.1396, "dec": 1.0097, "type": "DCEP (Cepheid)", "period": 7.17664},
    {"name": "Mira (ο Ceti)", "ra": 34.7214, "dec": -2.9776, "type": "M (Mira Variable)", "period": 331.96},
    {"name": "Betelgeuse (α Orionis)", "ra": 88.7929, "dec": 7.4071, "type": "SRC (Pulsating Supergiant)", "period": 420.0},
    {"name": "χ Cygni", "ra": 296.9069, "dec": 32.9142, "type": "M (Mira)", "period": 408.0},
    {"name": "R Coronae Borealis", "ra": 237.1188, "dec": 28.3033, "type": "RCB", "period": 44.0},
]


def check_if_variable_star(ra_deg: float, dec_deg: float, tolerance_arcsec: float = 30.0, **kwargs) -> Optional[Dict]:
    """
    Check if a given position matches a known variable star

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        tolerance_arcsec: Match tolerance in arcseconds

    Returns:
        Variable star info dict if matched, None otherwise
    """
    tolerance_arcsec = kwargs.get('search_radius_arcsec', tolerance_arcsec)
    try:
        target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

        # 1. Fast check against offline known variable stars catalog
        for var in OFFLINE_KNOWN_VARIABLES:
            var_coord = SkyCoord(ra=var['ra'] * u.deg, dec=var['dec'] * u.deg)
            sep = target_coord.separation(var_coord).arcsec
            if sep <= tolerance_arcsec:
                match = dict(var)
                match['separation_arcsec'] = float(sep)
                return match

        # 2. Online queries (with short timeout in case offline)
        radius_deg = (tolerance_arcsec / 3600.0) * 2
        variables = []
        try:
            variables = query_variable_stars_vsx(ra_deg, dec_deg, radius_deg)
        except Exception:
            pass

        if not variables:
            try:
                variables = query_variable_stars_simbad(ra_deg, dec_deg, radius_deg)
            except Exception:
                pass

        if not variables:
            return None

        # Find closest match
        closest = None
        min_sep = tolerance_arcsec

        for var in variables:
            if 'ra' in var and 'dec' in var:
                var_coord = SkyCoord(ra=var['ra']*u.deg, dec=var['dec']*u.deg)
                sep = target_coord.separation(var_coord).arcsec

                if sep < min_sep:
                    min_sep = sep
                    closest = var
                    closest['separation_arcsec'] = sep

        return closest

    except Exception as e:
        logger.error(f"Error checking variable star: {e}")
        return None
        logger.error(f"Error checking variable star: {e}")
        return None


def identify_stars_in_field(ra_deg: float, dec_deg: float,
                            star_positions: List[Tuple[float, float]],
                            wcs) -> List[Dict]:
    """
    Identify stars at given pixel positions using WCS and catalog

    Args:
        ra_deg: Field center RA
        dec_deg: Field center DEC
        star_positions: List of (x, y) pixel coordinates
        wcs: WCS object for coordinate conversion

    Returns:
        List of star identification dictionaries
    """
    identifications = []

    try:
        from .wcs_utils import pixel_to_world

        for i, (x, y) in enumerate(star_positions):
            # Convert pixel to world coordinates
            star_ra, star_dec = pixel_to_world(wcs, x, y)

            if star_ra is None or star_dec is None:
                identifications.append({
                    'index': i,
                    'pixel_x': x,
                    'pixel_y': y,
                    'identified': False
                })
                continue

            # Check if it's a known variable star
            var_info = check_if_variable_star(star_ra, star_dec)

            identifications.append({
                'index': i,
                'pixel_x': x,
                'pixel_y': y,
                'ra': star_ra,
                'dec': star_dec,
                'is_variable': var_info is not None,
                'variable_info': var_info,
                'identified': True
            })

    except Exception as e:
        logger.error(f"Error identifying stars: {e}", exc_info=True)

    return identifications
