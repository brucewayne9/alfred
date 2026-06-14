#!/usr/bin/env python3
"""TEMPLATE driver (v6) — copy to localize_<city>.py and fill the marked spots.
Reuses the shared engine (localize2.py) and its v6 layouts so you inherit every
fix learned on real arenas. Run:  python3 localize_<city>.py

WHAT EACH 'kind' DOES (pick per asset from the venue's spec sheet):
  imagery -> compose_grouped : photo formats. ONE tight left lockup (ROD WAVE H1,
             DON'T LOOK DOWN H2, then date/venue/CTA right under), off his face.
  strip   -> render_strip    : THIN word banners (leaderboards, email ads, marquee).
             Single auto-fit baseline; never stacks. Date optional; logo usually off.
  ribbon  -> render_ribbon   : LED ribbon boards. side_clear for zones by blue lines.
  inice   -> render_in_ice   : NHL virtual in-ice. Cream-only (NO gold/yellow).
  plain   -> compose         : legacy top/bottom engine layout (rarely needed now).
"""
import os
from PIL import Image
import localize2 as L

# 1) VENUE LOGO — rasterize the DARK-GROUND / white-reverse EPS at high res:
#    gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pngalpha -r600 -sOutputFile=logo.png in.eps
#    Dark logos vanish on the wood wall — always use the reverse/dark-ground mark.
L.LOGO = Image.open("<PACK>/<venue>_dark_trim.png").convert("RGBA")
L.LOGO = L.LOGO.crop(L.LOGO.getbbox())

# 2) Identity (match the master routing labels) + logo width tuning.
L.CITY  = "CITY, ST"
L.VENUE = "VENUE NAME"
L.LOGO_FRAC = 0.62          # square-ish ~0.44 ; very wide horizontal mark ~0.66
DATE = "MONTH DD"           # ONE date per set. Two-night arenas = run twice (-NN per night).
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_<city>", "Localized Ad Assets")

# 3) SIZES from the spec sheet: (category, base_name, w, h, dated, cta, kind)
#    Honor spec notes: "no date/location" -> dated=False ; "logo/name only" -> strip name_only ;
#    skip slots that aren't ours (e.g. a Ticketmaster season logo) and note them to Mike.
CAT = {"WEB":"WEBSITE + MISC", "SOC":"SOCIAL", "DIG":"DIGITAL", "PRINT":"PRINT"}
SIZES = [
 # ("WEB","WEBSITE-EventHeader-1420x500",1420,500,False,"none","imagery"),   # generic per spec
 # ("SOC","SOCIAL-Instagram-1080x1080",1080,1080,True,"none","imagery"),
 # ("DIG","DIGITAL-Leaderboard-728x90",728,90,True,"url","strip"),
 # ("DIG","DIGITAL-InIce-1800x1000",1800,1000,False,"none","inice"),
 # ("PRINT","PRINT-Poster-8_5x11-2550x3300",2550,3300,True,"url","imagery"),  # inches @300 DPI
]
# Per-strip fine print (optional): base_name -> render_strip kwargs.
STRIP_KW = {
 # "DIGITAL-CHWebAd2-320x50": dict(dated=True, with_url=False, with_logo=False),
 # "MARQUEE-WadeAve-350x112": dict(name_only=True, with_logo=False),   # venue adds the date
}
# Ribbon boards (LED): base_name -> dict(w,h,side_clear)
RIBBONS = {
 # "DIGITAL-DED-Zone2and4-3900x180": dict(w=3900, h=180, side_clear=140),
}

if __name__ == "__main__":
    n = 0
    for cat, base, w, h, dated, cta, kind in SIZES:
        folder = f"{OUT}/{CAT[cat]}"; os.makedirs(folder, exist_ok=True)
        if   kind == "strip":   c = L.render_strip(w, h, DATE, **STRIP_KW.get(base, {}))
        elif kind == "ribbon":  rk = RIBBONS[base]; c = L.render_ribbon(rk["w"], rk["h"], DATE, rk.get("side_clear", 0))
        elif kind == "inice":   c = L.render_in_ice(w, h)
        elif kind == "plain":   c = L.compose(w, h, DATE, dated, cta)
        else:                   c = L.compose_grouped(w, h, DATE, dated, cta)
        save_kw = {"dpi": (L.PRINT_DPI, L.PRINT_DPI)} if cat == "PRINT" else {}
        c.save(f"{folder}/RodWave-DLD-<Venue>-{base}.png", **save_kw); n += 1
    print(f"rendered {n} assets")
