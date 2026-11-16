"""
Star Field Service
Fetches real star catalog data around a celestial object for finder charts
Uses SIMBAD coordinate query API for reliable star data
"""

import logging
from typing import List, Dict, Any
import requests
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# SIMBAD API endpoint for coordinate queries
SIMBAD_BASE_URL = "http://simbad.u-strasbg.fr/simbad/sim-coo"
SIMBAD_VOTABLE_URL = "http://simbad.u-strasbg.fr/simbad/sim-coo"


def get_star_field(ra_deg: float, dec_deg: float, radius_deg: float = 5.0, max_mag: float = 6.5) -> List[Dict[str, Any]]:
    """
    Query stars around a given RA/DEC coordinate using SIMBAD coordinate query API.
    SIMBAD is more reliable than VizieR for coordinate-based queries.

    Args:
        ra_deg: Right Ascension in degrees
        dec_deg: Declination in degrees
        radius_deg: Search radius in degrees (default: 5 degrees for finder scope)
        max_mag: Maximum magnitude to include (default: 6.5, midpoint between city and dark skies)

    Returns:
        List of star dictionaries with ra, dec, magnitude, and other properties
    """
    try:
        # Convert radius to arcminutes for SIMBAD
        radius_arcmin = radius_deg * 60.0
        
        # SIMBAD coordinate query - format: "RA DEC" in degrees
        coord_str = f"{ra_deg:.6f} {dec_deg:+.6f}"
        
        # Query SIMBAD using ASCII format (easier to parse)
        params = {
            'Coord': coord_str,
            'Radius': f"{radius_arcmin:.1f}",
            'Radius.unit': 'arcmin',
            'output.format': 'ASCII',  # ASCII is easier to parse
            'output.max': 1000,  # Maximum number of objects (increased for better finder charts)
        }
        
        url = f"{SIMBAD_VOTABLE_URL}?{urlencode(params)}"
        
        logger.info(f"Querying SIMBAD for stars around RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°, radius={radius_deg}°")
        logger.debug(f"SIMBAD URL: {url}")
        
        response = requests.get(url, timeout=60, headers={'User-Agent': 'SeeStar-Photometry/1.0'})
        response.raise_for_status()
        
        logger.debug(f"SIMBAD response status: {response.status_code}, length: {len(response.text)}")
        logger.debug(f"First 500 chars of response: {response.text[:500]}")
        
        # Parse ASCII response (more reliable than VOTable for this use case)
        stars = _parse_simbad_ascii(response.text, max_mag)

        # If ASCII parsing didn't work, try VOTable parsing
        if len(stars) == 0 and ('VOTable' in response.text or '<?xml' in response.text):
            logger.debug("Trying VOTable parsing as fallback")
            stars = _parse_simbad_votable(response.text, max_mag)

        if len(stars) == 0:
            logger.error(f"PARSER FAILED: No stars parsed from SIMBAD response.")
            logger.error(f"Response length: {len(response.text)} chars")
            logger.error(f"Response preview: {response.text[:500]}")
            # Save response for debugging
            try:
                with open('/tmp/simbad_debug.txt', 'w') as f:
                    f.write(response.text)
                logger.error(f"Full response saved to /tmp/simbad_debug.txt")
            except Exception as e:
                logger.error(f"Could not save debug file: {e}")

        logger.info(f"Retrieved {len(stars)} stars from SIMBAD")
        return stars
        
    except requests.RequestException as e:
        logger.error(f"Error querying SIMBAD: {e}")
        # Try VizieR as fallback
        return _try_vizier_fallback(ra_deg, dec_deg, radius_deg, max_mag)
    except Exception as e:
        logger.error(f"Unexpected error in get_star_field: {e}", exc_info=True)
        # Try VizieR as fallback
        return _try_vizier_fallback(ra_deg, dec_deg, radius_deg, max_mag)


def _parse_simbad_votable(votable_xml: str, max_mag: float) -> List[Dict[str, Any]]:
    """Parse SIMBAD VOTable XML response"""
    stars = []
    
    try:
        root = ET.fromstring(votable_xml)
        
        # Find namespace
        ns = {'vot': 'http://www.ivoa.net/xml/VOTable/v1.3'}
        
        # Find all TR (table row) elements
        for tr in root.findall('.//vot:TR', ns):
            ra_val = None
            dec_val = None
            mag_val = None
            
            # Get TD (table data) elements
            tds = tr.findall('vot:TD', ns)
            
            # SIMBAD VOTable typically has: RA, DEC, and magnitude columns
            # We need to find the column definitions first
            for i, td in enumerate(tds):
                # Get the value
                value = td.text.strip() if td.text else None
                if not value or value == '':
                    continue
                
                # Try to determine column type by value format
                # RA is typically in hours (0-24) or degrees
                # DEC is typically in degrees (-90 to +90)
                # Magnitude is a decimal number
                
                try:
                    num_val = float(value)
                    
                    # Heuristic: if value is between 0-24, likely RA in hours
                    if 0 <= num_val <= 24 and ra_val is None:
                        ra_val = num_val * 15.0  # Convert hours to degrees
                    # If value is between -90 and 90, likely DEC
                    elif -90 <= num_val <= 90 and dec_val is None:
                        dec_val = num_val
                    # If value is positive and reasonable magnitude range
                    elif 0 <= num_val <= 20 and mag_val is None:
                        mag_val = num_val
                except ValueError:
                    # Try parsing as sexagesimal format (HH:MM:SS or DD:MM:SS)
                    if ':' in value:
                        parts = value.split(':')
                        if len(parts) >= 3:
                            try:
                                h = float(parts[0])
                                m = float(parts[1])
                                s = float(parts[2])
                                
                                # If hours format (0-24), it's RA
                                if 0 <= h < 24:
                                    ra_val = (h + m/60.0 + s/3600.0) * 15.0
                                # Otherwise might be DEC
                                elif dec_val is None:
                                    sign = -1 if h < 0 else 1
                                    dec_val = sign * (abs(h) + m/60.0 + s/3600.0)
                            except ValueError:
                                pass
                    continue
            
            # Only add if we have both RA and DEC
            if ra_val is not None and dec_val is not None:
                # Use default magnitude if not found
                if mag_val is None:
                    mag_val = 6.0
                
                # Filter by magnitude
                if mag_val <= max_mag:
                    stars.append({
                        'ra': ra_val,
                        'dec': dec_val,
                        'magnitude': mag_val,
                    })
        
        # If VOTable parsing didn't work, try simpler approach
        if len(stars) == 0:
            # Try parsing as ASCII if VOTable failed
            stars = _parse_simbad_ascii(votable_xml, max_mag)
            
    except ET.ParseError as e:
        logger.warning(f"Failed to parse SIMBAD VOTable XML: {e}")
        # Try ASCII parsing as fallback
        stars = _parse_simbad_ascii(votable_xml, max_mag)
    except Exception as e:
        logger.error(f"Error parsing SIMBAD VOTable: {e}", exc_info=True)
        stars = []
    
    return stars


def _parse_simbad_ascii(response_text: str, max_mag: float) -> List[Dict[str, Any]]:
    """
    Parse SIMBAD ASCII response with pipe-delimited format
    Format: # |dist|identifier|typ|coord1 (RA DEC)|Mag U|Mag B|Mag V|Mag R|Mag I|spec|#bib|#not
    """
    stars = []
    lines = response_text.split('\n')

    # Look for pipe-delimited table format
    coord_col_idx = -1
    type_col_idx = -1
    mag_v_idx = -1
    mag_b_idx = -1
    mag_r_idx = -1
    header_found = False

    logger.debug(f"Parsing SIMBAD ASCII response: {len(lines)} lines")

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Skip initial lines until we find the header
        if 'coord1' in line_stripped.lower():
            # Parse header to find column indices
            parts = [p.strip() for p in line_stripped.split('|')]
            logger.debug(f"Header line found at line {i}: {len(parts)} columns")
            logger.debug(f"Header parts: {parts}")
            for idx, part in enumerate(parts):
                part_lower = part.lower()
                logger.debug(f"  Column {idx}: '{part}' (lower: '{part_lower}')")
                if 'coord1' in part_lower:
                    coord_col_idx = idx
                elif part_lower == 'typ':  # Exact match to avoid matching 'spec. type'
                    type_col_idx = idx
                elif 'mag v' in part_lower:
                    mag_v_idx = idx
                elif 'mag b' in part_lower:
                    mag_b_idx = idx
                elif 'mag r' in part_lower:
                    mag_r_idx = idx
            header_found = True
            logger.info(f"Header parsed: coord_col={coord_col_idx}, type_col={type_col_idx}, mag_v={mag_v_idx}, mag_b={mag_b_idx}, mag_r={mag_r_idx}")
            continue

        # Skip separator lines
        if line_stripped.startswith('-') or line_stripped.startswith('='):
            continue

        # Parse data lines (pipe-delimited)
        if '|' in line_stripped and header_found:
            parts = [p.strip() for p in line_stripped.split('|')]

            # Safety check
            if len(parts) <= coord_col_idx:
                continue

            try:
                # Filter by object type - only include stellar objects
                # Common stellar types: *, PM* (proper motion star), V* (variable star),
                # ** (double star), SB* (spectroscopic binary), EB* (eclipsing binary)
                # Exclude: X (X-ray source), ? (unknown), OpC (open cluster), etc.
                if type_col_idx >= 0 and type_col_idx < len(parts):
                    obj_type = parts[type_col_idx].strip()
                    stellar_types = ['*', 'PM*', 'V*', '**', 'SB*', 'EB*', 'Y*O', 'Em*', 'Er*', 'dS*']
                    if not any(obj_type.startswith(t) for t in stellar_types):
                        continue  # Skip non-stellar objects

                # Extract coordinate string (format: "HH MM SS.S +DD MM SS.S")
                coord_str = parts[coord_col_idx].strip()
                if not coord_str or coord_str == '~':
                    continue

                # Parse sexagesimal coordinates
                # Format: "03 46 57.35 +24 06 12.9"
                coord_parts = coord_str.split()
                if len(coord_parts) < 6:
                    continue

                # Parse RA (first 3 parts: HH MM SS.S)
                ra_h = float(coord_parts[0])
                ra_m = float(coord_parts[1])
                ra_s = float(coord_parts[2])
                ra_deg = (ra_h + ra_m/60.0 + ra_s/3600.0) * 15.0

                # Parse DEC (last 3 parts: +/-DD MM SS.S)
                dec_str = coord_parts[3]
                dec_sign = -1 if dec_str.startswith('-') else 1
                dec_d = abs(float(coord_parts[3]))
                dec_m = float(coord_parts[4])
                dec_s = float(coord_parts[5])
                dec_deg = dec_sign * (dec_d + dec_m/60.0 + dec_s/3600.0)

                # Extract magnitude (prefer V, then B, then R)
                mag_val = None
                if mag_v_idx >= 0 and mag_v_idx < len(parts):
                    mag_str = parts[mag_v_idx].strip()
                    if mag_str and mag_str != '~':
                        try:
                            mag_val = float(mag_str)
                        except ValueError:
                            pass

                if mag_val is None and mag_b_idx >= 0 and mag_b_idx < len(parts):
                    mag_str = parts[mag_b_idx].strip()
                    if mag_str and mag_str != '~':
                        try:
                            mag_val = float(mag_str)
                        except ValueError:
                            pass

                if mag_val is None and mag_r_idx >= 0 and mag_r_idx < len(parts):
                    mag_str = parts[mag_r_idx].strip()
                    if mag_str and mag_str != '~':
                        try:
                            mag_val = float(mag_str)
                        except ValueError:
                            pass

                # Skip objects without a valid magnitude - they're likely catalog entries
                if mag_val is None:
                    continue

                # Validate and add star
                if 0 <= ra_deg <= 360 and -90 <= dec_deg <= 90:
                    if mag_val <= max_mag:
                        stars.append({
                            'ra': ra_deg,
                            'dec': dec_deg,
                            'magnitude': mag_val,
                        })

                        if len(stars) <= 5:
                            logger.debug(f"Parsed star {len(stars)}: RA={ra_deg:.4f}°, DEC={dec_deg:.4f}°, Mag={mag_val:.2f}")

            except (ValueError, IndexError) as e:
                if i < 30:  # Only log first few errors
                    logger.debug(f"Error parsing line {i}: {line_stripped[:60]}..., error: {e}")
                continue

    return stars


def _try_vizier_fallback(ra_deg: float, dec_deg: float, radius_deg: float, max_mag: float) -> List[Dict[str, Any]]:
    """Try VizieR as fallback if SIMBAD fails"""
    try:
        # Use CDS Portal API (more reliable than VizieR direct)
        from urllib.parse import quote
        
        ra_hours = ra_deg / 15.0
        ra_h = int(ra_hours)
        ra_m = int((ra_hours - ra_h) * 60)
        ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
        
        dec_sign = '+' if dec_deg >= 0 else '-'
        dec_abs = abs(dec_deg)
        dec_d = int(dec_abs)
        dec_m = int((dec_abs - dec_d) * 60)
        dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
        
        # CDS Portal query
        coord = f"{ra_h:02d}h{ra_m:02d}m{ra_s:05.2f}s {dec_sign}{dec_d:02d}d{dec_m:02d}m{dec_s:05.2f}s"
        url = f"http://cdsportal.u-strasbg.fr/cgi-bin/VizieR-3?-source=I/259/tyc2&-c={quote(coord)}&-c.rs={radius_deg}&-out.max=200"
        
        logger.info(f"Trying VizieR fallback: {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Parse response (similar to before)
        # For now, return empty and let user know we need a better solution
        logger.warning("VizieR fallback attempted but parsing not fully implemented")
        return []
        
    except Exception as e:
        logger.error(f"VizieR fallback also failed: {e}")
        return []



def get_star_field_simple(ra_deg: float, dec_deg: float, radius_deg: float = 5.0, max_mag: float = 9.0) -> List[Dict[str, Any]]:
    """
    Simplified fallback: Use SIMBAD API for bright stars only.
    This is a backup if VizieR is unavailable.
    """
    try:
        # SIMBAD query for stars around the object
        # Note: SIMBAD has rate limits, so use sparingly
        simbad_url = f"http://simbad.u-strasbg.fr/simbad/sim-coo"
        params = {
            'Coord': f"{ra_deg}d {dec_deg}d",
            'Radius': f"{radius_deg}d",
            'Radius.unit': 'arcmin',
            'output.format': 'ASCII',
        }
        
        # For now, return empty list - we'll use VizieR as primary
        # This can be implemented if needed as fallback
        return []
    except Exception as e:
        logger.error(f"Error in SIMBAD fallback: {e}")
        return []

