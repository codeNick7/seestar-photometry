#!/usr/bin/env python3
"""
Regenerate Photometry Plots with Custom Metadata
================================================

Re-plots light curves and magnitude scatter charts with custom target name,
celestial coordinates (RA/DEC), plot formatting, and high-DPI publication styling.

Usage:
  python regenerate_plots.py
  python regenerate_plots.py --csv data/output/photometry_results.csv --target "Algol (β Per)" --ra 47.0422 --dec 40.9556
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seestar_photometry.photometry_engine import ResultsExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate photometry plots with target metadata and coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/output/photometry_results.csv",
        help="Path to photometry_results.csv (default: data/output/photometry_results.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save regenerated plots (default: same as CSV directory)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Target Star",
        help="Target star name for chart headers (e.g., 'Algol (β Per)')"
    )
    parser.add_argument(
        "--ra",
        type=float,
        default=None,
        help="Target Right Ascension in degrees (e.g., 47.042218)"
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Target Declination in degrees (e.g., 40.955648)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "jpeg"],
        help="Image format (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for raster formats (default: 300)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    # Check fallback in sample directory if default output not found
    if not csv_path.exists():
        sample_csv = ROOT_DIR / "data" / "sample" / "photometry_results.csv"
        if sample_csv.exists():
            print(f"ℹ️ {csv_path} not found. Using sample dataset at {sample_csv}")
            csv_path = sample_csv
        else:
            print(f"❌ Error: Could not find {csv_path}")
            print("Please run the photometry pipeline first or provide a valid CSV path with --csv.")
            sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("📈 REGENERATING PHOTOMETRY PLOTS")
    print("=" * 70)
    print(f"Input CSV:   {csv_path}")
    print(f"Output Dir:  {out_dir}")
    print(f"Target:      {args.target}")
    if args.ra is not None and args.dec is not None:
        print(f"Coordinates: RA {args.ra:.6f}°, Dec {args.dec:.6f}°")
    print(f"Format:      {args.format.upper()} @ {args.dpi} DPI")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    valid_count = df["diff_magnitude"].notna().sum()
    print(f"\nFound {len(df)} total rows ({valid_count} valid measurements).")

    exporter = ResultsExporter(
        df,
        str(out_dir),
        target_name=args.target,
        target_ra=args.ra,
        target_dec=args.dec,
    )

    print("\nRegenerating light curve plot...")
    exporter.plot_lightcurve(format=args.format, dpi=args.dpi)

    print("Regenerating scatter plot...")
    exporter.plot_magnitude_vs_frame(format=args.format, dpi=args.dpi)

    print("\n✓ Plots generated successfully!")
    print(f"  • Light curve: {out_dir / f'lightcurve.{args.format}'}")
    print(f"  • Scatter:     {out_dir / f'magnitude_vs_time_scatter.{args.format}'}")


if __name__ == "__main__":
    main()
