#!/bin/zsh
# Run SeeStar Photometry tool with native Apple Silicon ARM64 Python
# Ensures proper architecture for compiled libraries (astropy, photutils, scipy)

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    VENV_ACTIVATE="source venv/bin/activate"
elif [ -d ".venv" ]; then
    VENV_ACTIVATE="source .venv/bin/activate"
else
    VENV_ACTIVATE="true"
fi

echo "🔭 Running SeeStar Photometry on Apple Silicon (ARM64)..."
arch -arm64 zsh -c "$VENV_ACTIVATE && python3 run_photometry.py $@"
