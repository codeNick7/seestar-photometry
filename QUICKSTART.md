# ⚡ SeeStar Photometry Quickstart Guide

Get up and running with SeeStar S50 differential photometry in **5 minutes**!

---

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/seestar-photometry.git
cd seestar-photometry

# Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Or on Apple Silicon Mac, run `./run_photometry_arm64.sh` after activating.)*

---

## 2. Prepare Your FITS Files

1. Connect your SeeStar S50 to your computer via USB-C or Wi-Fi (station mode).
2. Copy your raw sub-exposure FITS files (`Light_*.fit` or `*.fits`) into `data/input/`:

```
seestar-photometry/
├── data/
│   ├── input/
│   │   ├── Light_Algol_5.0s_IRCUT_20260305-192019.fit
│   │   ├── Light_Algol_5.0s_IRCUT_20260305-192034.fit
│   │   └── ...
│   └── output/
```

---

## 3. Option A: Launch Interactive Web UI (Easiest)

```bash
python run_web_ui.py
# or
streamlit run app.py
```

1. Open your browser to **http://localhost:8501**.
2. Select your image stretch (**Asinh** recommended).
3. Choose your **Target** star and **Reference** stars (or use candidate auto-detection).
4. Click **🛡️ Check Reference Stars** to verify against the AAVSO variable catalog.
5. Click **🚀 Run Full Photometry Pipeline**.
6. Switch to the **📈 Light Curve & Results** tab to explore your interactive light curve!

---

## 4. Option B: Command Line (CLI)

```bash
python run_photometry.py --input data/input
```

1. Press `ENTER` when prompted.
2. The interactive star selection viewer opens:
   - **Click 1:** Target star (green circle)
   - **Click 2 & 3:** Bright, non-variable reference stars (cyan/purple circles)
   - **Right-Click:** Undo last pick if you clicked the wrong star
   - **Key `q`:** Finish selection and start processing
3. Watch the progress bar as frames are aligned and measured.

---

## 5. Review Results

All outputs are saved to `data/output/`:

- `lightcurve.png` – Publication-ready differential light curve with $\pm 1\sigma$ band
- `magnitude_vs_time_scatter.png` – Magnitude vs time scatter chart
- `photometry_results.csv` – Timestamped magnitudes and net star fluxes
- `RESULTS_SUMMARY.txt` – Data quality score, mean magnitude, and amplitude
- `quality_report.txt` – FWHM, star count, and saturation evaluation

---

## 6. Analyze Eclipses & Predict Next Events

For eclipsing binaries (e.g. Algol) or exoplanet transits:
```bash
python analyze_eclipse.py --csv data/output/photometry_results.csv --target "Algol" --period 2.867328
```
This prints the exact observed minimum time (UTC & local), transit depth ($\Delta m$), and predicted future eclipse dates!

---

## 7. Need to Re-Plot with Custom Target Name or High DPI?

```bash
python regenerate_plots.py \
  --csv data/output/photometry_results.csv \
  --target "Algol (β Persei)" \
  --ra 47.042218 \
  --dec 40.955648 \
  --dpi 300
```
