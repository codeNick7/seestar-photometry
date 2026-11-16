"""
SeeStar Photometry - Interactive Web Dashboard
A modern scientific web interface for processing and analyzing SeeStar S50 FITS photometry data.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

from seestar_photometry.config import PhotometryConfig, PRESETS, get_preset_config
from seestar_photometry.ui.app_helpers import (
    load_fits_preview,
    detect_candidate_stars,
    create_plotly_lightcurve,
    compute_photometry_stats,
)
from seestar_photometry.star_identification import query_variable_stars_vsx, check_if_variable_star
from seestar_photometry.wcs_utils import create_wcs_from_header, pixel_to_world
from seestar_photometry.photometry_engine import (
    FITSDataset,
    FrameAligner,
    PhotometryEngine,
    ResultsExporter,
)

# Page configuration
st.set_page_config(
    page_title="SeeStar Photometry",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark astronomical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🔭 SeeStar Photometry</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Precision differential photometry & light-curve analysis for SeeStar S50 smart telescopes</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Sidebar: Instrument & Pipeline Settings
    # ─────────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        preset_choice = st.selectbox(
            "Instrument Preset",
            options=["SeeStar S50 (Default)", "Dwarf II", "Vespera", "Custom"],
            index=0,
            help="Pre-calibrated aperture and scale defaults for smart telescopes"
        )

        preset_key_map = {
            "SeeStar S50 (Default)": "seestar_s50",
            "Dwarf II": "dwarf_ii",
            "Vespera": "vespera",
            "Custom": "general",
        }
        current_preset = PRESETS[preset_key_map[preset_choice]]

        st.subheader("Data Paths")
        cosmos_input = Path("/Volumes/codeNick7/workspace/COSMOS-Observation-Planner/backend/photometry/input")
        if cosmos_input.exists() and not list((ROOT_DIR / "data" / "input").glob("*.fit*")):
            default_input = str(cosmos_input)
        else:
            default_input = str(ROOT_DIR / "data" / "input")
        default_output = str(ROOT_DIR / "data" / "output")

        # Check if sample data exists
        sample_path = ROOT_DIR / "data" / "sample"
        use_sample = st.checkbox("Load Sample Algol Dataset", value=False, help="Use pre-packaged Algol results for demonstration")

        if use_sample and sample_path.exists():
            input_dir = str(sample_path)
            output_dir = str(sample_path)
            st.info(f"Using sample data from `{input_dir}`")
        else:
            input_dir = st.text_input("FITS Input Directory", value=default_input)
            output_dir = st.text_input("Output Directory", value=default_output)

        st.subheader("Photometry Apertures")
        col_ap1, col_ap2 = st.columns(2)
        with col_ap1:
            aperture_radius = st.number_input("Aperture (px)", min_value=3, max_value=40, value=current_preset["aperture_radius"])
            sky_inner = st.number_input("Sky Inner (px)", min_value=5, max_value=60, value=current_preset["sky_annulus_inner"])
        with col_ap2:
            sky_outer = st.number_input("Sky Outer (px)", min_value=8, max_value=80, value=current_preset["sky_annulus_outer"])
            downsample = st.selectbox("Display Downsample", options=[1, 2, 4], index=1)

        st.subheader("Algorithmic Options")
        auto_aperture = st.checkbox("Auto-optimize aperture via FWHM", value=current_preset.get("auto_aperture", True))
        refine_centroids = st.checkbox("Sub-pixel centroid refinement", value=current_preset.get("refine_centroids", True))
        detection_sigma = st.slider("Star detection sigma (σ)", min_value=1.5, max_value=8.0, value=float(current_preset.get("detection_sigma", 3.0)), step=0.5)
        pixel_scale = st.number_input("Pixel Scale (arcsec/px)", min_value=0.2, max_value=5.0, value=float(current_preset.get("pixel_scale", 1.21)), step=0.05)

    # ─────────────────────────────────────────────────────────────────────────────
    # Main Tabs
    # ─────────────────────────────────────────────────────────────────────────────
    tab_run, tab_curve, tab_eclipse, tab_data, tab_guide = st.tabs([
        "🚀 Run Pipeline & Select Stars",
        "📈 Light Curve & Results",
        "🌑 Variable Star / Eclipse Analysis",
        "📁 Data Table & Exports",
        "📖 SeeStar S50 Guide",
    ])

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 1: Run Pipeline & Star Selection
    # ─────────────────────────────────────────────────────────────────────────────
    with tab_run:
        input_path = Path(input_dir)
        fits_files = []
        if input_path.exists():
            for ext in ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS"):
                fits_files.extend(list(input_path.glob(ext)))
            fits_files = sorted(list(set(fits_files)))

        st.subheader("1. File Scan & Reference Frame")
        if not fits_files:
            st.warning(f"No FITS files found in `{input_dir}`. Please place `.fit` or `.fits` files in the input folder.")
            st.markdown("""
            **Quick tip:**
            - You can connect your SeeStar S50 to your PC/Mac via USB-C or Wi-Fi.
            - Copy the raw exposure frames (e.g. `Light_*.fit`) to `data/input/`.
            - Or enable **Load Sample Algol Dataset** in the sidebar to test immediately!
            """)
        else:
            st.success(f"Found **{len(fits_files)}** FITS frames in `{input_dir}`")
            ref_idx = st.slider("Select frame for star inspection", min_value=0, max_value=len(fits_files) - 1, value=0)
            selected_file = fits_files[ref_idx]

            # Stretch controls
            col_st1, col_st2, col_st3 = st.columns([1, 1, 2])
            with col_st1:
                stretch_type = st.selectbox("Stretch Method", options=["asinh", "zscale", "log", "linear"], index=0)
            with col_st2:
                cmap_choice = st.selectbox("Colormap", options=["gray", "magma", "viridis", "bone"], index=0)
            with col_st3:
                contrast_val = st.slider("Asinh Contrast Factor", min_value=0.01, max_value=0.5, value=0.1, step=0.02)

            with st.spinner("Rendering FITS preview..."):
                display_img, raw_img, meta = load_fits_preview(
                    selected_file, downsample_factor=downsample, stretch=stretch_type, contrast=contrast_val
                )

            # Metadata bar
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("Object", meta["object"])
            mcol2.metric("Exposure", f"{meta['exposure']} s" if meta['exposure'] != "N/A" else "N/A")
            mcol3.metric("Filter", meta["filter"])
            mcol4.metric("Dimensions", f"{meta['width']} × {meta['height']}")
            mcol5.metric("Telescope", meta["telescope"])

            # Candidate star auto-detection
            candidates = detect_candidate_stars(raw_img, sigma=detection_sigma, max_stars=30)

            st.subheader("2. Target & Reference Star Selection")
            col_sel1, col_sel2 = st.columns([1, 1])

            with col_sel1:
                st.markdown("**Target Star Coordinates (pixels in original image)**")
                default_tx = candidates[0]["x"] if candidates else raw_img.shape[1] / 2
                default_ty = candidates[0]["y"] if candidates else raw_img.shape[0] / 2
                tx = st.number_input("Target X", min_value=0.0, max_value=float(raw_img.shape[1]), value=float(default_tx), step=1.0)
                ty = st.number_input("Target Y", min_value=0.0, max_value=float(raw_img.shape[0]), value=float(default_ty), step=1.0)

            with col_sel2:
                st.markdown("**Reference Star 1 Coordinates**")
                default_r1x = candidates[1]["x"] if len(candidates) > 1 else (raw_img.shape[1] / 2 + 50)
                default_r1y = candidates[1]["y"] if len(candidates) > 1 else (raw_img.shape[0] / 2 + 50)
                r1x = st.number_input("Ref 1 X", min_value=0.0, max_value=float(raw_img.shape[1]), value=float(default_r1x), step=1.0)
                r1y = st.number_input("Ref 1 Y", min_value=0.0, max_value=float(raw_img.shape[0]), value=float(default_r1y), step=1.0)

            # Optional Reference Star 2
            with st.expander("➕ Add Reference Star 2 (Optional but recommended)"):
                default_r2x = candidates[2]["x"] if len(candidates) > 2 else 0.0
                default_r2y = candidates[2]["y"] if len(candidates) > 2 else 0.0
                use_ref2 = st.checkbox("Enable Reference Star 2", value=len(candidates) > 2)
                r2x = st.number_input("Ref 2 X", min_value=0.0, max_value=float(raw_img.shape[1]), value=float(default_r2x), step=1.0)
                r2y = st.number_input("Ref 2 Y", min_value=0.0, max_value=float(raw_img.shape[0]), value=float(default_r2y), step=1.0)

            # Safety check: Query AAVSO VSX to verify reference star is NOT variable
            if st.button("🛡️ Check Reference Stars Against AAVSO VSX Variable Catalog"):
                with st.spinner("Checking variable star catalog..."):
                    # Check WCS
                    wcs = None
                    try:
                        with fits.open(selected_file, memmap=False) as hd:
                            wcs = create_wcs_from_header(dict(hd[0].header), raw_img.shape)
                    except Exception:
                        pass

                    if wcs is not None:
                        ra_r1, dec_r1 = pixel_to_world(wcs, r1x, r1y)
                        if ra_r1 is not None and dec_r1 is not None:
                            var_info = check_if_variable_star(ra_r1, dec_r1, tolerance_arcsec=30.0)
                            if var_info:
                                st.session_state["vsx_status"] = ("error", f"⚠️ Warning: Reference Star 1 matches known variable star **{var_info.get('name')}** (Type: {var_info.get('type')})! Choose another star.")
                            else:
                                st.session_state["vsx_status"] = ("success", f"✓ Reference Star 1 (RA {ra_r1:.4f}°, Dec {dec_r1:.4f}°) is safe (no known variable match).")
                        else:
                            st.session_state["vsx_status"] = ("warning", "Could not convert pixel coordinates to celestial coordinates.")
                    else:
                        st.session_state["vsx_status"] = ("info", "WCS coordinates not detected in FITS header. Automatic variable checking requires RA/DEC.")

            if "vsx_status" in st.session_state:
                status_type, msg = st.session_state["vsx_status"]
                if status_type == "error":
                    st.error(msg)
                elif status_type == "success":
                    st.success(msg)
                elif status_type == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)

            # Render overlay plot with apertures
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(display_img, cmap=cmap_choice, origin="lower")
            ax.set_title(f"Aperture Preview: Frame #{ref_idx} ({selected_file.name})", color="white")
            ax.axis("on")

            # Scale coordinates to display image scale
            scale_disp = 1.0 / downsample
            # Target (Green)
            ax.add_patch(Circle((tx * scale_disp, ty * scale_disp), aperture_radius * scale_disp, edgecolor="#00ff66", facecolor="none", lw=1.5, label="Target Aperture"))
            ax.add_patch(Circle((tx * scale_disp, ty * scale_disp), sky_inner * scale_disp, edgecolor="#ffbb00", facecolor="none", ls="--", lw=1, label="Sky Inner"))
            ax.add_patch(Circle((tx * scale_disp, ty * scale_disp), sky_outer * scale_disp, edgecolor="#ffbb00", facecolor="none", ls=":", lw=1, label="Sky Outer"))
            ax.text(tx * scale_disp + 10, ty * scale_disp + 10, "Target", color="#00ff66", fontsize=10, fontweight="bold")

            # Ref 1 (Cyan)
            ax.add_patch(Circle((r1x * scale_disp, r1y * scale_disp), aperture_radius * scale_disp, edgecolor="#00d2ff", facecolor="none", lw=1.5, label="Ref 1 Aperture"))
            ax.text(r1x * scale_disp + 10, r1y * scale_disp + 10, "Ref 1", color="#00d2ff", fontsize=10, fontweight="bold")

            # Ref 2 (Purple)
            if use_ref2 and (r2x > 0 or r2y > 0):
                ax.add_patch(Circle((r2x * scale_disp, r2y * scale_disp), aperture_radius * scale_disp, edgecolor="#d070ff", facecolor="none", lw=1.5, label="Ref 2 Aperture"))
                ax.text(r2x * scale_disp + 10, r2y * scale_disp + 10, "Ref 2", color="#d070ff", fontsize=10, fontweight="bold")

            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            st.pyplot(fig)

            # Run button
            st.subheader("3. Execute Photometry Analysis")
            if st.button("🚀 Run Full Photometry Pipeline", type="primary"):
                ref_list = [[r1x, r1y]]
                if use_ref2 and (r2x > 0 or r2y > 0):
                    ref_list.append([r2x, r2y])

                star_data = {
                    "target": [tx, ty],
                    "references": ref_list,
                }

                out_p = Path(output_dir)
                out_p.mkdir(parents=True, exist_ok=True)

                progress_bar = st.progress(0, text="Initializing dataset...")
                status_box = st.empty()

                try:
                    dataset = FITSDataset(input_dir)
                    status_box.info(f"Loaded {len(dataset)} frames. Initializing alignment...")
                    progress_bar.progress(15, text="Selecting reference frame...")

                    aligner = FrameAligner(dataset)
                    aligner.select_best_reference(n_candidates=min(20, len(dataset)))

                    progress_bar.progress(35, text="Aligning frames with star-matching & WCS...")
                    aligner.compute_alignments(max_control_points=100, cache_dir=output_dir)

                    progress_bar.progress(60, text="Measuring aperture & differential photometry...")
                    photometry = PhotometryEngine(dataset, aligner, star_data)
                    photometry.process_all_frames()

                    progress_bar.progress(85, text="Generating light curves and reports...")
                    results_df = photometry.get_results_dataframe()

                    exporter = ResultsExporter(results_df, output_dir)
                    exporter.plot_lightcurve(format="png", dpi=300)
                    exporter.plot_magnitude_vs_frame(format="png", dpi=300)
                    exporter.export_csv()
                    exporter.export_summary()
                    exporter.export_quality_report()

                    progress_bar.progress(100, text="Completed!")
                    status_box.success("🎉 Photometry analysis completed successfully! Check the 'Light Curve & Results' tab.")
                except Exception as ex:
                    status_box.error(f"Error during execution: {ex}")

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 2: Light Curve & Results
    # ─────────────────────────────────────────────────────────────────────────────
    with tab_curve:
        st.subheader("Light Curve & Statistical Metrics")
        results_csv = Path(output_dir) / "photometry_results.csv"

        if not results_csv.exists():
            st.info("No photometry results found in output directory yet. Run the pipeline in Tab 1 or load sample data from sidebar.")
        else:
            df = pd.read_csv(results_csv)
            stats = compute_photometry_stats(df)

            kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
            kcol1.metric("Total Frames", stats["total_frames"])
            kcol2.metric("Valid Frames", f"{stats['valid_frames']} ({stats['success_rate']}%)")
            kcol3.metric("Mean Diff Mag", f"{stats['mean_mag']:.3f}" if stats['mean_mag'] else "N/A")
            kcol4.metric("Scatter / Precision (σ)", f"±{stats['std_mag']:.3f} mag" if stats['std_mag'] else "N/A")
            kcol5.metric("Peak-to-Peak (Δmag)", f"{stats['amplitude']:.3f} mag" if stats['amplitude'] else "N/A")

            st.caption(f"Data Quality Grade: **{stats['grade']}**")

            # Plotly interactive chart
            fig_curve = create_plotly_lightcurve(df)
            if fig_curve is not None:
                st.plotly_chart(fig_curve, use_container_width=True)
            else:
                st.warning("Could not render Plotly chart. Rendering static plot:")
                lc_png = Path(output_dir) / "lightcurve.png"
                if lc_png.exists():
                    st.image(str(lc_png))

            # Matplotlib static plots preview
            with st.expander("📸 High-Resolution Publication Plots"):
                pcol1, pcol2 = st.columns(2)
                lc_file = Path(output_dir) / "lightcurve.png"
                sc_file = Path(output_dir) / "magnitude_vs_time_scatter.png"
                if lc_file.exists():
                    with pcol1:
                        st.image(str(lc_file), caption="Differential Light Curve (Publication PNG)")
                if sc_file.exists():
                    with pcol2:
                        st.image(str(sc_file), caption="Magnitude vs Time Scatter")

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 3: Variable Star & Eclipse Analysis
    # ─────────────────────────────────────────────────────────────────────────────
    with tab_eclipse:
        st.subheader("🌑 Variable Star & Eclipse Minimum Predictor")
        st.markdown("Analyze brightness dips to calculate eclipse ingress/egress, minimum times, and predict future transit events.")

        results_csv = Path(output_dir) / "photometry_results.csv"
        if not results_csv.exists():
            st.info("Run photometry analysis first to generate results for eclipse prediction.")
        else:
            df_ecl = pd.read_csv(results_csv)
            valid_ecl = df_ecl[df_ecl["diff_magnitude"].notna()].copy()

            if "time" in valid_ecl.columns:
                valid_ecl["datetime"] = pd.to_datetime(valid_ecl["time"])

                # Find eclipse minimum (deepest point = faintest target flux / highest differential magnitude)
                # Note: diff_magnitude = -2.5 * log10(target / ref)
                # A fainter target produces a less negative / higher numerical magnitude.
                # However, in differential photometry, depending on target vs ref brightness:
                min_idx = valid_ecl["diff_magnitude"].idxmax()
                max_dimming_time = valid_ecl.loc[min_idx, "datetime"]
                max_dimming_mag = valid_ecl.loc[min_idx, "diff_magnitude"]

                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric("Observed Minimum Time (UTC)", max_dimming_time.strftime("%Y-%m-%d %H:%M:%S"))
                ecol2.metric("Observed Minimum Magnitude", f"{max_dimming_mag:.3f} mag")
                ecol3.metric("Eclipse Depth (Δmag)", f"{max_dimming_mag - valid_ecl['diff_magnitude'].min():.3f} mag")

                st.subheader("Future Eclipse Prediction")
                target_choice = st.selectbox("Orbital Period Preset", options=["Algol (β Persei) - 2.867328 d", "Custom Target"])
                if "Algol" in target_choice:
                    period_days = 2.867328
                else:
                    period_days = st.number_input("Target Orbital Period (days)", min_value=0.1, max_value=100.0, value=2.867328, step=0.01)

                next_min_time = max_dimming_time + pd.Timedelta(days=period_days)
                st.success(f"🗓️ **Predicted Next Minimum:** `{next_min_time.strftime('%Y-%m-%d %H:%M:%S UTC')}` (Period: {period_days:.6f} days)")

                # Top 10 deepest measurements table
                st.subheader("Deepest Eclipse Measurements")
                display_cols = [c for c in ["frame", "filename", "time", "diff_magnitude", "target_flux", "ref_flux_avg"] if c in valid_ecl.columns]
                st.dataframe(
                    valid_ecl.nlargest(10, "diff_magnitude")[display_cols],
                    use_container_width=True
                )

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 4: Data Table & Exports
    # ─────────────────────────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Photometry Results Table")
        results_csv = Path(output_dir) / "photometry_results.csv"
        if results_csv.exists():
            df_full = pd.read_csv(results_csv)
            st.dataframe(df_full, use_container_width=True)

            dcol1, dcol2, dcol3 = st.columns(3)
            with dcol1:
                st.download_button(
                    label="📥 Download CSV Results",
                    data=df_full.to_csv(index=False),
                    file_name="photometry_results.csv",
                    mime="text/csv",
                )
            with dcol2:
                summary_file = Path(output_dir) / "RESULTS_SUMMARY.txt"
                if summary_file.exists():
                    st.download_button(
                        label="📄 Download Analysis Summary",
                        data=summary_file.read_text(),
                        file_name="RESULTS_SUMMARY.txt",
                        mime="text/plain",
                    )
            with dcol3:
                quality_file = Path(output_dir) / "quality_report.txt"
                if quality_file.exists():
                    st.download_button(
                        label="📊 Download Quality Report",
                        data=quality_file.read_text(),
                        file_name="quality_report.txt",
                        mime="text/plain",
                    )

    # ─────────────────────────────────────────────────────────────────────────────
    # TAB 5: SeeStar S50 Guide
    # ─────────────────────────────────────────────────────────────────────────────
    with tab_guide:
        st.subheader("📘 Best Practices for SeeStar S50 Photometry")
        st.markdown("""
        The **ZWO SeeStar S50** is a 50mm f/5 apochromatic smart telescope with an IMX462 sensor (pixel scale: 1.21 arcsec/pixel).
        To achieve research-grade photometric precision with SeeStar S50 data:

        ### 1. Exposure & Gain
        - **Exposure Duration:** 2 to 10 seconds per sub-exposure. Shorter exposures (2-5s) help prevent bright target stars (like Algol at mag 2.1) from saturating the sensor.
        - **Filter Selection:** Always observe in **IRCUT** (luminance) mode for stellar photometry rather than LP (dual-band light pollution) filter.
        - **Saturation Checking:** The pipeline automatically applies brightness reduction if pixels exceed the saturation threshold (60,000 ADU).

        ### 2. Star Selection Tips
        - **Target Star:** Choose well-isolated targets without blended companions.
        - **Reference Stars:**
          - Pick 1 to 3 non-variable reference stars of similar brightness and color.
          - Use the built-in **AAVSO VSX check** button in the app to ensure your chosen reference is not a known variable!
          - Spread reference stars across the field rather than clustered together.

        ### 3. Apertures & FWHM
        - Typical SeeStar stellar FWHM is **3.5 - 5.5 pixels**.
        - Standard aperture radius: **8 - 10 pixels**.
        - Inner sky annulus: **12 - 15 pixels**.
        - Outer sky annulus: **20 - 25 pixels**.
        - Leave **Auto-optimize aperture via FWHM** enabled to let the pipeline dynamically adapt to seeing conditions!
        """)


if __name__ == "__main__":
    main()
