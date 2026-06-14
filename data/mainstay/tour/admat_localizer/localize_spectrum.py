#!/usr/bin/env python3
"""Spectrum Center (Charlotte) — two-night stand Nov 3 + 4.
NO spec sheet was provided (only a logo) -> standard social/web set; flag to Mike."""
import os
from PIL import Image
import localize2 as L

L.LOGO = Image.open("/tmp/admat_test/spectrum_white.png").convert("RGBA")
L.LOGO = L.LOGO.crop(L.LOGO.getbbox())
L.CITY = "CHARLOTTE, NC"
L.LOGO_FRAC = 0.68          # wide horizontal logo
OUT = "/tmp/admat_test/out_spectrum/Localized Ad Assets"

CAT = {"SOC":"SOCIAL","WEB":"WEBSITE"}
SIZES = [
 ("SOC","SOCIAL-1x1-1080x1080",1080,1080,True,"none"),
 ("SOC","SOCIAL-4x5-1080x1350",1080,1350,True,"none"),
 ("SOC","SOCIAL-9x16-1080x1920",1080,1920,True,"none"),
 ("SOC","SOCIAL-16x9-1920x1080",1920,1080,True,"url"),
 ("WEB","WEB-1200x628",1200,628,True,"url"),
]
NIGHTS = [("03","NOVEMBER 3"), ("04","NOVEMBER 4")]
if __name__ == "__main__":
    n=0
    for cat,base,w,h,dated,cta in SIZES:
        folder=f"{OUT}/{CAT[cat]}"; os.makedirs(folder, exist_ok=True)
        for nn,dl in NIGHTS:
            L.compose(w,h,dl,dated,cta).save(f"{folder}/RodWave-DLD-SpectrumCenter-{base}-{nn}.png"); n+=1
    print(f"rendered {n} Spectrum assets")
