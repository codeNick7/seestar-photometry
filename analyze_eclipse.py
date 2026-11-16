#!/usr/bin/env python3
"""
Variable Star & Eclipse Analysis Script
=======================================

Analyzes light curves from differential photometry to determine eclipse minima,
calculate transit depths, and predict upcoming minimum occurrences.

Usage:
  python analyze_eclipse.py
  python analyze_eclipse.py --csv data/output/photometry_results.csv --target "Algol" --period 2.867328
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze variable star eclipse minima & predict future eclipses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/output/photometry_results.csv",
        help="Path to photometry_results.csv (default: data/output/photometry_results.csv)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Algol (β Per)",
        help="Target name (default: 'Algol (β Per)')"
    )
    parser.add_argument(
        "--period",
        type=float,
        default=2.867328,
        help="Orbital period in days (default: 2.867328 days for Algol)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        sample_csv = ROOT_DIR / "data" / "sample" / "photometry_results.csv"
        if sample_csv.exists():
            print(f"ℹ️ {csv_path} not found. Using sample dataset at {sample_csv}\n")
            csv_path = sample_csv
        else:
            print(f"❌ Error: Could not find {csv_path}")
            sys.exit(1)

    print("=" * 70)
    print(f"🔭 ECLIPSE & VARIABLE STAR ANALYSIS: {args.target}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        print("❌ Error: CSV missing 'time' column.")
        sys.exit(1)

    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    valid = df[df["diff_magnitude"].notna()].copy()

    if len(valid) == 0:
        print("❌ Error: No valid magnitude measurements found.")
        sys.exit(1)

    # In differential photometry: diff_mag = -2.5 * log10(target_flux / ref_flux)
    # When target dims during eclipse, target_flux decreases, so diff_mag increases (less negative or higher).
    # Find the maximum dimming point:
    deepest_idx = valid["diff_magnitude"].idxmax()
    eclipse_time = valid.loc[deepest_idx, "datetime"]
    deepest_mag = valid.loc[deepest_idx, "diff_magnitude"]
    brightest_mag = valid["diff_magnitude"].min()
    depth = deepest_mag - brightest_mag

    print(f"\n📊 Observed Eclipse Minimum:")
    print(f"  • Deepest Magnitude:   {deepest_mag:.4f} mag")
    print(f"  • Baseline Magnitude:  {brightest_mag:.4f} mag")
    print(f"  • Transit Depth (Δm):  {depth:.4f} mag")
    print(f"  • Minimum Time (UTC):  {eclipse_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Timezones for convenient reference
    try:
        import zoneinfo
        tz_ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        tz_est = zoneinfo.ZoneInfo("America/New_York")
        print(f"  • Minimum Time (IST):  {eclipse_time.astimezone(tz_ist).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"  • Minimum Time (EST):  {eclipse_time.astimezone(tz_est).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception:
        pass

    # Predict future eclipses
    next_eclipse_utc = eclipse_time + timedelta(days=args.period)
    next_next_eclipse_utc = eclipse_time + timedelta(days=2 * args.period)

    print(f"\n🗓️ Predicted Future Eclipses (Period = {args.period:.6f} days):")
    print(f"  • Next Minimum (+1 cycle):  {next_eclipse_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  • Next Minimum (+2 cycles): {next_next_eclipse_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Light curve statistics
    print(f"\n📈 Light Curve Summary:")
    print(f"  • Total frames analyzed:     {len(df)}")
    print(f"  • Valid photometric points:  {len(valid)} ({len(valid)/len(df)*100:.1f}%)")
    print(f"  • Mean Differential Mag:     {valid['diff_magnitude'].mean():.4f}")
    print(f"  • Standard Deviation (σ):    ±{valid['diff_magnitude'].std():.4f} mag")

    # Top deepest points
    print(f"\n🔍 Top 5 Deepest Eclipse Measurements:")
    deepest_5 = valid.nlargest(5, "diff_magnitude")
    for idx, row in deepest_5.iterrows():
        frame_str = f"Frame {int(row['frame']):3d} | " if "frame" in row else f"Row {idx:3d} | "
        time_str = row['datetime'].strftime('%H:%M:%S') + " UTC"
        mag_str = f"Mag: {row['diff_magnitude']:.4f}"
        flux_str = f" | Flux: {row['target_flux']:.1f}" if "target_flux" in row else ""
        print(f"  {frame_str}{time_str} | {mag_str}{flux_str}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
