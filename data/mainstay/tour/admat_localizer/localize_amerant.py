#!/usr/bin/env python3
"""Amerant Bank Arena (Ft. Lauderdale) — two-night stand Nov 14 + 15."""
import os
from PIL import Image
import localize2 as L

L.LOGO = Image.open("/tmp/admat_test/amerant_white.png").convert("RGBA")
L.LOGO = L.LOGO.crop(L.LOGO.getbbox())
L.CITY = "FT. LAUDERDALE, FL"
L.LOGO_FRAC = 0.50
OUT = "/tmp/admat_test/out_amerant/Localized Ad Assets"

CAT = {"WEB":"WEBSITE","SG":"SEATGEEK","MARQ":"OUTDOOR MARQUEES",
       "SOC":"SOCIAL MEDIA","JUMBO":"JUMBOTRON","VW":"VIDEO WALL"}
SIZES = [
 ("WEB","WEBSITE-ShowScroll-1440x696",1440,696,True,"url"),
 ("WEB","WEBSITE-Thumbnail-490x490",490,490,True,"none"),
 ("SG","SEATGEEK-1800x1560",1800,1560,True,"none"),
 ("SG","SEATGEEK-2000x2000",2000,2000,True,"none"),
 ("SG","SEATGEEK-1528x864",1528,864,True,"none"),
 ("SG","SEATGEEK-1184x758",1184,758,True,"none"),
 ("SG","SEATGEEK-600x400",600,400,True,"none"),
 ("MARQ","OUTDOORMARQUEE-Size1-1344x960",1344,960,True,"url"),
 ("MARQ","OUTDOORMARQUEE-Size2-1056x768",1056,768,True,"url"),
 ("SOC","SOCIAL-16x9-1920x1080",1920,1080,True,"none"),
 ("SOC","SOCIAL-9x16-1080x1920",1080,1920,True,"none"),
 ("SOC","SOCIAL-4x5-1080x1350",1080,1350,True,"none"),
 ("SOC","SOCIAL-1x1-1080x1080",1080,1080,True,"none"),
 ("JUMBO","JUMBOTRON-1920x1080",1920,1080,True,"url"),
 ("VW","VIDEOWALL-2880x810",2880,810,True,"url"),
]
NIGHTS = [("14","NOVEMBER 14"), ("15","NOVEMBER 15")]
if __name__ == "__main__":
    n=0
    for cat,base,w,h,dated,cta in SIZES:
        folder=f"{OUT}/{CAT[cat]}"; os.makedirs(folder, exist_ok=True)
        for nn,dl in NIGHTS:
            L.compose(w,h,dl,dated,cta).save(f"{folder}/RodWave-DLD-Amerant-{base}-{nn}.png"); n+=1
    print(f"rendered {n} Amerant assets")
