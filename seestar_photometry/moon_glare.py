"""
Moon Glare Analysis for Photometry
Calculates moon position and estimates glare impact on observations
"""

import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MoonGlareConfig:
    """Configuration for moon glare analysis"""
    enable_glare_removal: bool = False  # Phase 3 - experimental
    glare_sigma: float = 50.0  # Glare spread in pixels


@dataclass
class MoonMetrics:
    """Moon position and glare metrics"""
    moon_distance_deg: float = 0.0      # Angular separation from target
    moon_phase: float = 0.0             # 0-100% illumination
    moon_altitude: float = 0.0          # Degrees above horizon
    moon_ra: float = 0.0                # Moon RA (degrees)
    moon_dec: float = 0.0               # Moon DEC (degrees)
    glare_level: str = "none"           # "none", "low", "medium", "high"
    glare_intensity: float = 0.0        # Quantitative 0-100


class MoonGlareAnalyzer:
    """
    Analyzes moon position and estimates glare impact

    Uses Skyfield ephemeris (reuses existing code from weather.py)
    Calculates angular separation and estimates glare level
    """

    def __init__(self, config: MoonGlareConfig = None):
        """
        Initialize moon glare analyzer

        Args:
            config: Moon glare configuration
        """
        self.config = config or MoonGlareConfig()

        # Lazy-load Skyfield to avoid import overhead
        self._eph = None
        self._ts = None

        # Track if we've already logged errors to avoid spam
        self._error_logged = False

    def analyze_moon_impact(
        self,
        obs_time: datetime,
        target_ra: float,
        target_dec: float,
        observer_lat: float,
        observer_lon: float
    ) -> Optional[MoonMetrics]:
        """
        Analyze moon's impact on observation

        Args:
            obs_time: Observation datetime (UTC)
            target_ra: Target RA in degrees
            target_dec: Target DEC in degrees
            observer_lat: Observer latitude in degrees
            observer_lon: Observer longitude in degrees

        Returns:
            MoonMetrics object, or None if calculation fails
        """
        try:
            # Load ephemeris
            if self._eph is None:
                self._load_ephemeris()

            # Get moon position
            moon_ra, moon_dec, moon_alt, moon_phase = self._get_moon_position(
                obs_time, observer_lat, observer_lon
            )

            # Calculate angular separation
            distance_deg = self.calculate_angular_separation(
                target_ra, target_dec, moon_ra, moon_dec
            )

            # Estimate glare level
            glare_level, glare_intensity = self.estimate_glare_level(
                distance_deg, moon_phase, moon_alt
            )

            return MoonMetrics(
                moon_distance_deg=distance_deg,
                moon_phase=moon_phase,
                moon_altitude=moon_alt,
                moon_ra=moon_ra,
                moon_dec=moon_dec,
                glare_level=glare_level,
                glare_intensity=glare_intensity
            )

        except Exception as e:
            if not self._error_logged:
                logger.warning(f"Moon analysis disabled - error: {e}")
                self._error_logged = True
            return None

    def calculate_angular_separation(
        self,
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float
    ) -> float:
        """
        Calculate angular separation between two celestial coordinates

        Uses spherical geometry (haversine formula)

        Args:
            ra1, dec1: First coordinate (degrees)
            ra2, dec2: Second coordinate (degrees)

        Returns:
            separation: Angular distance in degrees
        """
        # Convert to radians
        ra1_rad = np.radians(ra1)
        dec1_rad = np.radians(dec1)
        ra2_rad = np.radians(ra2)
        dec2_rad = np.radians(dec2)

        # Haversine formula
        d_ra = ra2_rad - ra1_rad
        d_dec = dec2_rad - dec1_rad

        a = (np.sin(d_dec / 2) ** 2 +
             np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(d_ra / 2) ** 2)

        c = 2 * np.arcsin(np.sqrt(a))

        # Convert back to degrees
        separation = np.degrees(c)

        return separation

    def estimate_glare_level(
        self,
        distance_deg: float,
        moon_phase: float,
        moon_alt: float
    ) -> Tuple[str, float]:
        """
        Estimate moon glare level and intensity

        Model based on:
        - Distance: Inverse square law
        - Phase: Full moon worst, new moon none
        - Altitude: Higher = brighter

        Args:
            distance_deg: Angular separation in degrees
            moon_phase: Moon illumination (0-100%)
            moon_alt: Moon altitude in degrees

        Returns:
            level: "none", "low", "medium", "high"
            intensity: Quantitative 0-100 scale
        """
        # Moon below horizon: no glare
        if moon_alt < 0:
            return "none", 0.0

        # Distance factor (inverse square, normalized to 20° reference)
        # Closer = worse glare
        distance_factor = (20.0 / max(distance_deg, 1.0)) ** 2

        # Phase factor (full moon = worst)
        phase_factor = moon_phase / 100.0

        # Altitude factor (higher = brighter)
        altitude_factor = 0.1 + 0.9 * (moon_alt / 90.0)

        # Combined glare intensity (0-100 scale)
        glare_intensity = distance_factor * phase_factor * altitude_factor * 100.0

        # Clip to valid range
        glare_intensity = np.clip(glare_intensity, 0, 100)

        # Classify by distance thresholds
        if distance_deg > 60:
            level = "none"
        elif distance_deg > 40:
            level = "low"
        elif distance_deg > 20:
            level = "medium"
        else:
            level = "high"

        return level, float(glare_intensity)

    def _load_ephemeris(self):
        """Load Skyfield ephemeris data"""
        try:
            from skyfield.api import load, wgs84
            from skyfield import almanac

            # Load ephemeris (JPL DE421)
            loader = load('de421.bsp')
            self._eph = loader
            self._ts = load.timescale()

            logger.debug("Loaded Skyfield ephemeris for moon calculations")

        except Exception as e:
            logger.error(f"Error loading ephemeris: {e}")
            raise

    def _get_moon_position(
        self,
        obs_time: datetime,
        lat: float,
        lon: float
    ) -> Tuple[float, float, float, float]:
        """
        Get moon RA/DEC, altitude, and phase at observation time

        Args:
            obs_time: Observation time (UTC)
            lat: Observer latitude (degrees)
            lon: Observer longitude (degrees)

        Returns:
            moon_ra: Moon RA in degrees
            moon_dec: Moon DEC in degrees
            moon_alt: Moon altitude in degrees
            moon_phase: Moon illumination percentage (0-100)
        """
        try:
            from skyfield.api import wgs84

            # Ensure UTC
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
            else:
                obs_time = obs_time.astimezone(timezone.utc)

            # Create time object
            t = self._ts.from_datetime(obs_time)

            # Get celestial bodies
            earth = self._eph['earth']
            moon = self._eph['moon']
            sun = self._eph['sun']

            # Observer location (earth + topocentric offset)
            site = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
            observer = earth + site

            # Moon position as seen from observer
            astrometric = observer.at(t).observe(moon)
            apparent = astrometric.apparent()

            # RA/DEC
            ra, dec, distance = apparent.radec()
            moon_ra = ra._degrees
            moon_dec = dec._degrees

            # Altitude/Azimuth
            alt, az, distance = apparent.altaz()
            moon_alt = alt.degrees

            # Moon phase (illumination fraction)
            # Calculate using sun-earth-moon geometry
            e = earth.at(t)
            s = e.observe(sun).apparent()
            m = e.observe(moon).apparent()

            # Phase angle between sun and moon
            phase_angle = s.separation_from(m).degrees

            # Convert phase angle to illumination fraction
            # 0° = full moon (100%), 180° = new moon (0%)
            moon_illumination = (1 + np.cos(np.radians(phase_angle))) / 2
            moon_phase = moon_illumination * 100.0

            return moon_ra, moon_dec, moon_alt, moon_phase

        except Exception as e:
            logger.error(f"Error getting moon position: {e}")
            # Return defaults
            return 0.0, 0.0, -90.0, 0.0
