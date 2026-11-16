#!/usr/bin/env python3
"""
Automated UI capture script for SeeStar Photometry Web Dashboard.
Uses Playwright to interact with the running Streamlit app, capture high-resolution
screenshots at key workflow stages, and record a demo video walkthrough.
"""

import time
import os
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = ROOT_DIR / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_capture():
    print(f"Capturing screenshots to: {SCREENSHOTS_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        # Create context with video recording enabled
        context = browser.new_context(
            viewport={"width": 1440, "height": 960},
            device_scale_factor=2,  # High-DPI Retina screenshots
            record_video_dir=str(SCREENSHOTS_DIR / "raw_video"),
            record_video_size={"width": 1440, "height": 960}
        )

        page = context.new_page()

        # Step 1: Open App
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        page.wait_for_selector(".main-header", timeout=20000)
        time.sleep(3)

        # -------------------------------------------------------------
        # Stage 1: Pipeline Setup & FITS Frame Inspection
        # -------------------------------------------------------------
        print("Capturing Stage 1: Pipeline Setup & FITS Inspection...")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)
        page.screenshot(
            path=str(SCREENSHOTS_DIR / "01_pipeline_setup_fits_preview.png"),
            full_page=False
        )

        # -------------------------------------------------------------
        # Stage 2: Star Selection & Live Aperture Overlay
        # -------------------------------------------------------------
        print("Capturing Stage 2: Star Selection & Apertures Preview...")
        # Scroll down to reveal star coordinate inputs and the aperture preview plot
        img_locator = page.locator("img").first
        img_locator.scroll_into_view_if_needed()
        time.sleep(2)
        page.screenshot(
            path=str(SCREENSHOTS_DIR / "02_star_selection_apertures.png"),
            full_page=False
        )

        # -------------------------------------------------------------
        # Stage 3: AAVSO VSX Variable Star Catalog Verification
        # -------------------------------------------------------------
        print("Capturing Stage 3: AAVSO VSX Variable Star Safety Verification...")
        vsx_btn = page.locator("button:has-text('Check Reference Stars Against AAVSO VSX')")
        vsx_btn.scroll_into_view_if_needed()
        time.sleep(1)
        vsx_btn.click()
        print("Waiting for VSX verification response...")
        time.sleep(6)

        # Ensure the notification alert is scrolled into view nicely
        vsx_btn.scroll_into_view_if_needed()
        time.sleep(1)
        page.screenshot(
            path=str(SCREENSHOTS_DIR / "03_aavso_vsx_safety_check.png"),
            full_page=False
        )

        # -------------------------------------------------------------
        # Stage 4: Light Curve & Statistical Metrics
        # -------------------------------------------------------------
        print("Capturing Stage 4: Differential Light Curve & Metrics...")
        page.get_by_role("tab", name="Light Curve & Results").click()
        time.sleep(3)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)

        page.screenshot(
            path=str(SCREENSHOTS_DIR / "04_light_curve_and_metrics.png"),
            full_page=False
        )

        # -------------------------------------------------------------
        # Stage 5: High-Resolution Publication Plots Expander
        # -------------------------------------------------------------
        print("Capturing Stage 5: Publication Plots...")
        expander = page.locator("summary:has-text('High-Resolution Publication Plots')")
        if expander.count() > 0:
            expander.scroll_into_view_if_needed()
            time.sleep(1)
            expander.click()
            time.sleep(2)
            page.screenshot(
                path=str(SCREENSHOTS_DIR / "05_publication_plots.png"),
                full_page=False
            )

        # -------------------------------------------------------------
        # Stage 6: Variable Star / Eclipse Analysis & Transit Predictor
        # -------------------------------------------------------------
        print("Capturing Stage 6: Variable Star & Eclipse Analysis...")
        page.get_by_role("tab", name="Variable Star / Eclipse Analysis").click()
        time.sleep(3)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)

        page.screenshot(
            path=str(SCREENSHOTS_DIR / "06_variable_star_eclipse_analysis.png"),
            full_page=False
        )

        # -------------------------------------------------------------
        # Stage 7: Data Table & Publication Exports
        # -------------------------------------------------------------
        print("Capturing Stage 7: Data Table & Exports...")
        page.get_by_role("tab", name="Data Table & Exports").click()
        time.sleep(3)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)

        page.screenshot(
            path=str(SCREENSHOTS_DIR / "07_data_table_and_exports.png"),
            full_page=False
        )

        # -------------------------------------------------------------
        # Stage 8: SeeStar S50 Best Practices Guide
        # -------------------------------------------------------------
        print("Capturing Stage 8: SeeStar S50 Guide...")
        page.get_by_role("tab", name="SeeStar S50 Guide").click()
        time.sleep(3)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1.5)

        page.screenshot(
            path=str(SCREENSHOTS_DIR / "08_seestar_best_practices_guide.png"),
            full_page=False
        )

        # Close page and context to finalize the video file
        time.sleep(2)
        page_video = page.video
        video_path = page_video.path() if page_video else None
        page.close()
        context.close()
        browser.close()

        if video_path and os.path.exists(video_path):
            final_webm = SCREENSHOTS_DIR / "app_demo_walkthrough.webm"
            shutil.copy(video_path, final_webm)
            print(f"Recorded raw video saved to: {final_webm}")

            final_mp4 = SCREENSHOTS_DIR / "app_demo_walkthrough.mp4"
            final_gif = SCREENSHOTS_DIR / "app_demo_walkthrough.gif"

            print("Converting video to MP4...")
            os.system(f"ffmpeg -y -i '{final_webm}' -c:v libx264 -pix_fmt yuv420p -preset fast '{final_mp4}' 2>/dev/null")

            print("Generating animated walkthrough GIF...")
            os.system(f"ffmpeg -y -i '{final_webm}' -vf 'fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer' '{final_gif}' 2>/dev/null")

            raw_dir = SCREENSHOTS_DIR / "raw_video"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)

    print("All captures and conversions completed successfully!")


if __name__ == "__main__":
    run_capture()
