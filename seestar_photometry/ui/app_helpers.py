"""
Helper utilities for SeeStar Photometry Web UI.
Handles FITS image loading, astronomical stretches, candidate star detection,
and interactive visualization creation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize, LogStretch, LinearStretch
from scipy.ndimage import zoom


def load_fits_preview(
    fits_path: str | Path,
    downsample_factor: int = 2,
    stretch: str = "asinh",
    contrast: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load a FITS file and create a stretched 2D display image.

    Args:
        fits_path: Path to FITS file
        downsample_factor: Factor by which to downsample (>=1)
        stretch: 'asinh', 'zscale', 'log', or 'linear'
        contrast: Asinh stretch parameter

    Returns:
        tuple of (display_image_normalized, original_2d_image, header_metadata)
    """
    path = Path(fits_path)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path, memmap=True) as hdul:
        # Extract primary or first image HDU
        hdu = None
        for h in hdul:
            if h.data is not None and len(h.data.shape) >= 2:
                hdu = h
                break
        if hdu is None:
            raise ValueError(f"No 2D image data found in {path}")

        data = hdu.data.astype(np.float32)
        header = dict(hdu.header)

        # Handle 3D / RGB data (channels first or last)
        if len(data.shape) == 3:
            if data.shape[0] == 3:
                # Luminance conversion: Y = 0.299R + 0.587G + 0.114B
                image = 0.299 * data[0] + 0.587 * data[1] + 0.114 * data[2]
            elif data.shape[2] == 3:
                image = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
            else:
                image = np.mean(data, axis=0)
        else:
            image = data

    # Clean NaNs and Infs
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Downsample for faster UI rendering
    if downsample_factor > 1:
        display_img = zoom(image, 1.0 / downsample_factor, order=1)
    else:
        display_img = image.copy()

    # Apply astronomical interval and stretch
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(display_img)
    if vmax - vmin < 1e-9:
        vmin = float(np.min(display_img))
        vmax = float(np.max(display_img))
        if vmax <= vmin:
            vmax = vmin + 1.0

    if stretch.lower() == "asinh":
        stretch_obj = AsinhStretch(a=contrast)
    elif stretch.lower() == "log":
        stretch_obj = LogStretch()
    elif stretch.lower() == "linear":
        stretch_obj = LinearStretch()
    else:
        # Default Zscale interval with linear stretch
        stretch_obj = LinearStretch()

    norm = ImageNormalize(display_img, vmin=vmin, vmax=vmax, stretch=stretch_obj, clip=True)
    norm_img = norm(display_img)

    metadata = {
        "filename": path.name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "telescope": header.get("TELESCOP", header.get("INSTRUME", "SeeStar S50")),
        "object": header.get("OBJECT", "Unknown"),
        "exposure": header.get("EXPTIME", header.get("EXPOSURE", "N/A")),
        "filter": header.get("FILTER", "IRCUT"),
        "date_obs": header.get("DATE-OBS", header.get("DATE", "N/A")),
        "ra": header.get("RA", header.get("OBJCTRA", "N/A")),
        "dec": header.get("DEC", header.get("OBJCTDEC", "N/A")),
    }

    return norm_img, image, metadata


def detect_candidate_stars(
    image: np.ndarray,
    sigma: float = 3.0,
    fwhm: float = 4.0,
    max_stars: int = 50
) -> List[Dict[str, float]]:
    """
    Detect brightest isolated candidate stars in the frame using DAOStarFinder.

    Returns:
        List of dicts with keys: 'id', 'x', 'y', 'flux', 'peak'
    """
    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder

        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * std)
        sources = daofind(image - median)

        if sources is None or len(sources) == 0:
            return []

        # Sort by flux descending
        sources.sort("flux", reverse=True)
        candidates = []
        for i, row in enumerate(sources[:max_stars]):
            candidates.append({
                "id": i + 1,
                "x": float(row["xcentroid"]),
                "y": float(row["ycentroid"]),
                "flux": float(row["flux"]),
                "peak": float(row["peak"]),
            })
        return candidates
    except Exception:
        return []


def create_plotly_lightcurve(
    df: pd.DataFrame,
    target_name: Optional[str] = None
) -> Any:
    """
    Create an interactive Plotly light curve chart.
    Falls back to None if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plot_df = df.copy()
        if "time" in plot_df.columns:
            plot_df["datetime"] = pd.to_datetime(plot_df["time"], errors="coerce")
            x_col = "datetime"
            x_title = "Observation Time (UTC)"
        else:
            x_col = "frame"
            x_title = "Frame Number"

        # Filter valid differential magnitudes
        valid_df = plot_df[plot_df["diff_magnitude"].notna()].copy()
        if len(valid_df) == 0:
            return None

        mean_mag = valid_df["diff_magnitude"].mean()
        std_mag = valid_df["diff_magnitude"].std()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f"Differential Magnitude vs Time (Mean: {mean_mag:.3f} mag, σ: {std_mag:.3f})",
                "Brightness Variation / Amplitude (Δmag)"
            ),
            row_heights=[0.65, 0.35]
        )

        # Invert y-axis for astronomical magnitudes (smaller is brighter)
        fig.add_trace(
            go.Scatter(
                x=valid_df[x_col],
                y=valid_df["diff_magnitude"],
                mode="markers+lines",
                marker=dict(size=6, color="#00d2ff", line=dict(width=1, color="#ffffff")),
                line=dict(width=1, color="rgba(0, 210, 255, 0.4)"),
                name="Differential Mag",
                customdata=np.stack((valid_df["frame"], valid_df["filename"]), axis=-1),
                hovertemplate="<b>Frame %{customdata[0]}</b><br>" +
                              "File: %{customdata[1]}<br>" +
                              "Time: %{x}<br>" +
                              "Diff Mag: %{y:.4f}<extra></extra>"
            ),
            row=1, col=1
        )

        # Add mean magnitude line
        fig.add_hline(
            y=mean_mag, line_dash="dash", line_color="#ffaa00",
            annotation_text=f"Mean: {mean_mag:.3f}",
            annotation_position="bottom right",
            row=1, col=1
        )

        # Lower plot: Amplitude variation (magnitude - mean)
        amp_variation = valid_df["diff_magnitude"] - mean_mag
        fig.add_trace(
            go.Bar(
                x=valid_df[x_col],
                y=amp_variation,
                marker=dict(
                    color=amp_variation,
                    colorscale="Viridis",
                    showscale=False
                ),
                name="Δmag (Variation)",
                hovertemplate="Time: %{x}<br>Δmag: %{y:.4f}<extra></extra>"
            ),
            row=2, col=1
        )

        title_text = f"Light Curve: {target_name}" if target_name else "Differential Photometry Light Curve"
        fig.update_layout(
            title=title_text,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            hovermode="closest",
            height=600,
            margin=dict(l=60, r=40, t=60, b=40),
        )

        # Invert primary magnitude axis
        fig.update_yaxes(title_text="Diff Magnitude (mag, inverted)", autorange="reversed", row=1, col=1)
        fig.update_yaxes(title_text="Variation (mag)", row=2, col=1)
        fig.update_xaxes(title_text=x_title, row=2, col=1)

        return fig
    except ImportError:
        return None


def create_plotly_quality(quality_df: pd.DataFrame) -> Any:
    """
    Create an interactive Plotly chart of frame quality and star count.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if len(quality_df) == 0:
            return None

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Frame Quality Score (0-100)", "Detected Star Count & FWHM"),
            row_heights=[0.5, 0.5]
        )

        x = quality_df.get("frame", np.arange(len(quality_df)))

        # Quality score trace
        if "quality_score" in quality_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x, y=quality_df["quality_score"],
                    mode="lines+markers",
                    name="Quality Score",
                    line=dict(color="#2ecc71", width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            fig.add_hline(y=80, line_dash="dash", line_color="#27ae60", annotation_text="High Quality", row=1, col=1)
            fig.add_hline(y=60, line_dash="dash", line_color="#e67e22", annotation_text="Moderate", row=1, col=1)

        # Stars trace
        if "n_stars" in quality_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x, y=quality_df["n_stars"],
                    mode="lines",
                    name="Star Count",
                    line=dict(color="#3498db", width=2)
                ),
                row=2, col=1
            )

        # FWHM trace
        if "fwhm" in quality_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x, y=quality_df["fwhm"],
                    mode="lines",
                    name="FWHM (pixels)",
                    line=dict(color="#e74c3c", width=1.5, dash="dot"),
                    yaxis="y4"
                ),
                row=2, col=1
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            height=500,
            margin=dict(l=60, r=40, t=50, b=40),
        )
        fig.update_yaxes(title_text="Quality (0-100)", row=1, col=1)
        fig.update_yaxes(title_text="Star Count", row=2, col=1)
        fig.update_xaxes(title_text="Frame Number", row=2, col=1)

        return fig
    except ImportError:
        return None


def compute_photometry_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key photometric summary statistics from results DataFrame."""
    valid = df[df["diff_magnitude"].notna()]
    total_frames = len(df)
    valid_frames = len(valid)
    success_rate = (valid_frames / total_frames * 100) if total_frames > 0 else 0.0

    if valid_frames == 0:
        return {
            "total_frames": total_frames,
            "valid_frames": 0,
            "success_rate": 0.0,
            "mean_mag": None,
            "std_mag": None,
            "min_mag": None,
            "max_mag": None,
            "amplitude": None,
            "grade": "NO VALID DATA",
        }

    mags = valid["diff_magnitude"]
    mean_mag = float(mags.mean())
    std_mag = float(mags.std()) if valid_frames > 1 else 0.0
    min_mag = float(mags.min())
    max_mag = float(mags.max())
    amplitude = max_mag - min_mag

    if std_mag < 0.05:
        grade = "EXCELLENT (<0.05 mag scatter)"
    elif std_mag < 0.15:
        grade = "GOOD (<0.15 mag scatter)"
    elif std_mag < 0.35:
        grade = "MODERATE (<0.35 mag scatter)"
    else:
        grade = "HIGH SCATTER (needs check)"

    return {
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "success_rate": round(success_rate, 1),
        "mean_mag": round(mean_mag, 4),
        "std_mag": round(std_mag, 4),
        "min_mag": round(min_mag, 4),
        "max_mag": round(max_mag, 4),
        "amplitude": round(amplitude, 4),
        "grade": grade,
    }
