#!/usr/bin/env python3
"""
SeeStar Photometry CLI Runner
=============================

Command-line entry point for running high-performance differential photometry
on FITS datasets from SeeStar S50 and other telescopes.

Usage:
  python run_photometry.py
  python run_photometry.py --input /path/to/fits/folder
  python run_photometry.py -i ./data/input -o ./data/output --preset seestar_s50
"""

import sys
import argparse
from pathlib import Path

# Ensure package is discoverable
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from seestar_photometry.photometry_engine import main
from seestar_photometry.config import PRESETS


def parse_args():
    parser = argparse.ArgumentParser(
        description="SeeStar Photometry - Process FITS images for differential photometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_photometry.py
  python run_photometry.py --input data/input --output data/output
  python run_photometry.py -i /Volumes/Seestar/Algol_2026 -o ./results
  python run_photometry.py --preset seestar_s50
        """
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input folder containing FITS files (default: data/input)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output folder for results, plots, and CSV (default: data/output)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default="seestar_s50",
        help="Instrument configuration preset (default: seestar_s50)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(input_dir=args.input)
