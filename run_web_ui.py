#!/usr/bin/env python3
"""
Launcher script for SeeStar Photometry Web UI.
Starts the Streamlit application on http://localhost:8501.
"""

import sys
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    app_path = root_dir / "app.py"

    print("=" * 60)
    print("🔭 Launching SeeStar Photometry Web Dashboard...")
    print("=" * 60)

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("\n❌ Streamlit is not installed in the current environment.")
        print("Please install it with:")
        print("   pip install streamlit plotly")
        print("\nOr install all project requirements:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
    ] + sys.argv[1:]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped SeeStar Photometry Dashboard.")


if __name__ == "__main__":
    main()
