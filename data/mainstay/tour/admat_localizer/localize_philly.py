#!/usr/bin/env python3
"""Philadelphia / Xfinity Mobile Arena localizer — two-night stand (Sep 12 + 14).
Website formats are clean art+title (their site auto-populates logo/date)."""
import os
from PIL import Image
import localize2 as L

XF = Image.open("/tmp/admat_test/xfinity_white.png").convert("RGBA")
XF = XF.crop(XF.getbbox())
L.LOGO = XF
L.CITY = "PHILADELPHIA, PA"
L.LOGO_FRAC = 0.60
OUT = "/tmp/admat_test/out_philly/Localized Ad Assets"

CAT = {"WEB":"WEBSITE", "SOC":"SOCIAL"}
SIZES = [
 # website: no venue logo / no date (site auto-populates) -> art + title only
 ("WEB","WEBSITE-1440x535",1440,535,False,"none"),
 ("WEB","WEBSITE-760x460",760,460,False,"none"),
 ("WEB","WEBSITE-600x400",600,400,False,"none"),
 # social: include date + venue
 ("SOC","SOCIAL-1080x1350-PORTRAIT",1080,1350,True,"none"),
 ("SOC","SOCIAL-1920x1080",1920,1080,True,"none"),
 ("SOC","SOCIAL-1080x1920-STORY",1080,1920,True,"none"),
 ("SOC","SOCIAL-1200x628",1200,628,True,"none"),
]
NIGHTS = [("12","SEPTEMBER 12"), ("14","SEPTEMBER 14")]
if __name__ == "__main__":
    n=0
    for cat,base,w,h,dated,cta in SIZES:
        folder=f"{OUT}/{CAT[cat]}"; os.makedirs(folder, exist_ok=True)
        if dated:
            for nn,dl in NIGHTS:
                L.compose(w,h,dl,True,cta).save(f"{folder}/RodWave-DLD-Xfinity-{base}-{nn}.png"); n+=1
        else:
            L.compose(w,h,"",False,cta).save(f"{folder}/RodWave-DLD-Xfinity-{base}.png"); n+=1
    print(f"rendered {n} Philadelphia assets")
