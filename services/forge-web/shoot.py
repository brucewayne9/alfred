"""Screenshot the Forge dashboard (demo server must be running on :8099)."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "shots"
SHOTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:8099"
TABS = ["create", "queue", "library", "distribution", "intelligence"]


def launch(p):
    attempts = [
        {},
        {"executable_path": "/usr/bin/chromium-browser"},
        {"executable_path": "/snap/bin/chromium"},
    ]
    last = None
    for kw in attempts:
        try:
            return p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-color-profile=srgb"],
                **kw,
            )
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


with sync_playwright() as p:
    b = launch(p)
    # desktop
    dctx = b.new_context(viewport={"width": 1440, "height": 940}, device_scale_factor=2)
    dpg = dctx.new_page()
    for t in TABS:
        dpg.goto(f"{BASE}/#{t}", wait_until="networkidle")
        dpg.wait_for_timeout(1000)
        dpg.screenshot(path=str(SHOTS / f"desktop-{t}.png"))
        print("shot desktop", t)
    dctx.close()
    # mobile
    mctx = b.new_context(viewport={"width": 420, "height": 1480}, device_scale_factor=3, is_mobile=True)
    mpg = mctx.new_page()
    for t in ["create", "queue", "intelligence"]:
        mpg.goto(f"{BASE}/#{t}", wait_until="networkidle")
        mpg.wait_for_timeout(1000)
        mpg.screenshot(path=str(SHOTS / f"mobile-{t}.png"), full_page=True)
        print("shot mobile", t)
    mctx.close()
    b.close()
print("DONE")
