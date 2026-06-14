#!/usr/bin/env python3
"""Raleigh / Lenovo Center localizer — Rod Wave 'Don't Look Down', SEP 15.
Reuses the house-style engine (localize2.py); swaps venue logo + city + date.
Adds three exotic handlers required by the Lenovo Center spec sheet:
  - Virtual In-Ice (1800x1000): NO gold/yellow (NHL ice rule) -> cream-only palette.
  - DED ribbon boards (4100x180, 3900x180): single-row ticker; 3900 zones keep a
    140px clear margin on the side edges that border the blue lines.
  - Wade Ave marquee + tiny strips: name/title only (venue adds the date).
One date only -> one set (no -NN suffix).
"""
import os
from PIL import Image, ImageDraw, ImageOps
import localize2 as L

PACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raleigh_pack")

# --- venue overrides ---------------------------------------------------------
# FullColor Horizontal on DARK ground: red Lenovo box + white "Center" — reads
# clean on the dark wood wall (verified). Very wide mark -> larger logo fraction.
L.LOGO = Image.open(os.path.join(PACK, "lenovo_horiz_dark_trim.png")).convert("RGBA")
L.LOGO = L.LOGO.crop(L.LOGO.getbbox())
L.CITY = "RALEIGH, NC"
L.LOGO_FRAC = 0.66
DATE = "SEPTEMBER 15"
VEN  = "LENOVO CENTER"
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_raleigh", "Localized Ad Assets")

# --- standard sizes (category, base_name, w, h, dated, cta, grouped) ----------
# grouped=True  -> imagery format: one tight left-center lockup (ROD WAVE H1,
#                  DON'T LOOK DOWN H2, then date/venue/CTA pulled right under it).
# grouped=False -> thin word-only banner: keep the proven top/bottom engine layout.
CAT = {"WEB":"WEBSITE + MISC", "SOC":"SOCIAL", "DIG":"DIGITAL", "PRINT":"PRINT"}
SIZES = [
 # Website + Misc
 ("WEB","WEBSITE-EventPageHeader-1420x500",1420,500,False,"none",True),       # generic, no date (spec)
 ("WEB","WEBSITE-EventPageHeader-DATED-1420x500",1420,500,True,"button",True),# dated + Buy Tickets (Mike's add)
 ("WEB","WEBSITE-EventListingThumb-864x524",864,524,True,"none",True),
 ("WEB","WEBSITE-EventOverlayPopup-900x550",900,550,True,"url",True),
 ("WEB","MARQUEE-WadeAve-350x112",350,112,False,"none",False),            # thin: logo/name only
 # Social
 ("SOC","SOCIAL-Facebook-1200x628",1200,628,True,"none",True),
 ("SOC","SOCIAL-TwitterX-1920x1080",1920,1080,True,"none",True),
 ("SOC","SOCIAL-Instagram-1080x1080",1080,1080,True,"none",True),
 ("SOC","SOCIAL-InstagramLongFeed-1080x1350",1080,1350,True,"none",True),
 ("SOC","SOCIAL-InstagramStories-1080x1920",1080,1920,True,"none",True),
 # Digital
 ("DIG","DIGITAL-CHWebAd1-300x250",300,250,True,"button",True),
 ("DIG","DIGITAL-CHWebAd2-320x50",320,50,False,"none",False),             # thin strip (spec)
 ("DIG","DIGITAL-CHWebAd3-728x90",728,90,True,"url",False),               # thin leaderboard
 ("DIG","DIGITAL-HurricanesEmail-600x100",600,100,True,"url",False),      # thin strip
 # Print (inches @ 300 DPI — high-end)
 ("PRINT","PRINT-UpcomingEventsFull-11x17-3300x5100",3300,5100,True,"url",True),
 ("PRINT","PRINT-UpcomingEventsSuites-8_5x11-2550x3300",2550,3300,True,"url",True),
 ("PRINT","PRINT-IndividualEvent-8_5x11-2550x3300",2550,3300,True,"url",True),
]
PRINT_DPI = 300

# --- grouped imagery layout: one tight left-center lockup --------------------
def _stack(w, dated, cta, date_line, max_logo_h):
    """Build ONE consolidated left-aligned lockup at width w: ROD WAVE (H1, big)
    over DON'T LOOK DOWN/TOUR (H2), then CITY/DATE/logo and CTA pulled right under
    it with tight gaps — so the whole block reads as a single unit."""
    parts = [L.title_block(w, int(w*2))]            # big name (width-driven, not height-capped)
    if dated:
        parts.append(L.localized_block(w, date_line, with_logo=True, align="left",
                                       max_logo_h=max_logo_h))
    if cta == "url":    parts.append(L.cta_url_block(w))
    elif cta == "button": parts.append(L.button_block(w))
    g_title = int(w*0.075)                          # gap below the title lockup
    g = int(w*0.05)                                 # gap between the lower rows
    H = parts[0].height + sum((g_title if i==1 else g) + p.height
                              for i, p in enumerate(parts) if i > 0)
    img = Image.new("RGBA", (w, H), (0,0,0,0)); y = 0
    for i, p in enumerate(parts):
        if i > 0: y += g_title if i == 1 else g
        img.alpha_composite(p, (0, y)); y += p.height
    return img

def compose_grouped(tw, th, date_line, dated, cta):
    ratio = tw/th
    if ratio > 1.18:                                # wide: Rod right, lockup in left lane
        c, pw, tf = L.banner_build(tw, th); margin = max(6, int(min(tw,th)*0.02))
        mx = int(tw*0.05); lane = int(tw*0.50); valign = "center"; mlh = int(th*0.20)
    elif ratio < 0.66:                              # tall: lockup grouped low-left, clear of Rod
        c, tf = L.tall_build(tw, th); margin = max(8, int(tw*0.03))
        mx = int(tw*0.07); lane = int(tw*0.70); valign = "bottom"; mlh = int(th*0.075)
    else:                                           # square-ish: lockup left, vertically centered
        c, tf = L.cover_build(tw, th); margin = max(8, int(min(tw,th)*0.025))
        mx = int(tw*0.05); lane = int(tw*0.56); valign = "center"; mlh = int(th*0.13)
    mask = L.rod_mask((tw, th), tf, margin)
    w = lane; blk = None; px = mx; py = 0
    for _ in range(16):
        blk = _stack(w, dated, cta, date_line, mlh)
        if blk.height > th - 2*int(th*0.04):        # too tall -> shrink
            w = int(w*0.92); continue
        if valign == "center":  py = (th - blk.height)//2
        elif valign == "bottom": py = th - blk.height - int(th*0.06)
        else:                    py = int(th*0.06)
        py = max(int(th*0.04), min(py, th - blk.height - int(th*0.04)))
        if not L._overlaps(mask, px, py, blk): break
        moved = False                               # nudge up to clear his face/body
        for dy in range(int(th*0.02), int(th*0.45), max(1, int(th*0.02))):
            ny = py - dy
            if ny < int(th*0.03): break
            if not L._overlaps(mask, px, ny, blk): py = ny; moved = True; break
        if moved: break
        w = int(w*0.92)                             # else shrink and retry
    L.paste(c, blk, px, py)
    return c

# --- exotic: Virtual In-Ice 1800x1000 (NO gold/yellow per NHL ice rule) -------
def render_in_ice(w=1800, h=1000):
    """Banner build with a cream-only palette (no gold), large title + venue logo,
    no date. Saved PNG. Restores the palette afterward."""
    g_old, c_old = L.GOLD, L.CREAM
    L.GOLD = L.CREAM = (238, 232, 214)            # kill gold/yellow for the ice
    try:
        c = compose_grouped(w, h, "", False, "none")   # grouped title lockup, no date/CTA
    finally:
        L.GOLD, L.CREAM = g_old, c_old
    # large venue logo, lower-left wood lane
    lane_w = int(w * 0.46)
    lg_w = int(lane_w * 0.86)
    lg = L.LOGO.resize((lg_w, int(L.LOGO.height * lg_w / L.LOGO.width)), Image.LANCZOS)
    c.paste(lg, (int(w*0.045), int(h - lg.height - h*0.10)), lg)
    return c

# --- exotic: DED ribbon boards (single-row ticker) ---------------------------
def render_ribbon(w, h, side_clear=0):
    """One full-width ribbon: wood field + ROD WAVE / DON'T LOOK DOWN / info lockup
    + venue logo. side_clear keeps content out of the left/right edge bands
    (zones 2&4 border the blue lines: 140px clear)."""
    cell = L.wood_panel(w, h)
    cell = Image.alpha_composite(cell.convert("RGBA"), Image.new("RGBA",(w,h),(0,0,0,70)))
    d = ImageDraw.Draw(cell)
    x0 = side_clear + int(h*0.18)
    x1 = w - side_clear - int(h*0.18)
    base = h - int(h*0.24)
    f_n = L.fnt(L.F_NAME,  int(h*0.46))           # ROD WAVE (Druk Wide)
    f_t = L.fnt(L.F_TITLE, int(h*0.40))           # DON'T LOOK DOWN
    f_m = L.fnt(L.F_MED,   int(h*0.30))           # info
    x = x0
    d.text((x, base - f_n.getbbox("ROD WAVE")[3]), "ROD WAVE", font=f_n, fill=L.GOLD)
    x += L.tw_(f_n, "ROD WAVE") + int(h*0.5)
    d.text((x, base - f_t.getbbox("DON'T LOOK DOWN")[3]), "DON'T LOOK DOWN", font=f_t, fill=L.CREAM)
    x += L.tw_(f_t, "DON'T LOOK DOWN") + int(h*0.5)
    info = f"{DATE}  ·  {VEN}  ·  {L.CITY}  ·  {L.URL}"
    my = (h - f_m.getbbox(info)[3]) // 2
    d.text((x, my), info, font=f_m, fill=L.CREAM)
    # venue logo pinned to the right (inside the clear band)
    lg_h = int(h*0.56); lg_w = int(L.LOGO.width * lg_h / L.LOGO.height)
    lg = L.LOGO.resize((lg_w, lg_h), Image.LANCZOS)
    lx = x1 - lg_w
    if lx > x + 20:
        cell.paste(lg, (lx, (h-lg_h)//2), lg)
    return cell.convert("RGB")

# --- thin horizontal strips: ONE auto-fit baseline lockup (no vertical jumble) -
def render_strip(w, h, dated=True, with_url=True, name_only=False, with_logo=True):
    """Single-line lockup: ROD WAVE · DON'T LOOK DOWN [· DATE · CITY · URL] + logo
    right. Every segment is measured and the whole row is scaled down until it
    fits the width — so short strips never overlap or stack."""
    pad = max(6, int(h*0.18))
    logo_h = int(h*0.56) if with_logo else 0
    logo_w = int(L.LOGO.width*logo_h/L.LOGO.height) if with_logo else 0
    if with_logo and logo_w > int(w*0.26):        # cap the very wide mark on strips
        logo_w = int(w*0.26); logo_h = int(logo_w*L.LOGO.height/L.LOGO.width)
    gap_logo = int(h*0.45) if with_logo else 0
    text_avail = w - 2*pad - (logo_w + gap_logo if with_logo else 0)
    s_name, s_tour, s_info = h*0.52, h*0.42, h*0.34
    SEP = "   ·   "
    segs = [("ROD WAVE", L.F_NAME, s_name, L.GOLD),
            ("DON’T LOOK DOWN", L.F_TITLE, s_tour, L.CREAM)]
    if dated and not name_only:
        info = f"{DATE}  ·  {L.CITY}" + (f"  ·  {L.URL}" if with_url else "")
        segs.append((info, L.F_MED, s_info, L.CREAM))
    def layout(scale):
        out = []; total = 0
        fsep = L.fnt(L.F_MED, max(4, int(s_info*scale)))
        for i, (t, fp, sz, col) in enumerate(segs):
            f = L.fnt(fp, max(4, int(sz*scale))); tw = L.tw_(f, t)
            out.append((t, f, col, tw)); total += tw
            if i < len(segs)-1: total += L.tw_(fsep, SEP)
        return total, out, fsep
    scale = 1.0
    for _ in range(48):
        total, rendered, fsep = layout(scale)
        if total <= text_avail or scale < 0.12: break
        scale *= 0.94
    c = Image.alpha_composite(L.wood_panel(w, h).convert("RGBA"),
                              Image.new("RGBA", (w, h), (0,0,0,70)))
    d = ImageDraw.Draw(c)
    base = int(h*0.72); x = pad; sepw = L.tw_(fsep, SEP)
    for i, (t, f, col, tw) in enumerate(rendered):
        d.text((x, base - f.getbbox(t)[3]), t, font=f, fill=col); x += tw
        if i < len(rendered)-1:
            d.text((x, base - fsep.getbbox(SEP)[3]), SEP, font=fsep, fill=L.GOLD); x += sepw
    if with_logo:
        lg = L.LOGO.resize((logo_w, logo_h), Image.LANCZOS)
        c.paste(lg, (w - pad - logo_w, (h-logo_h)//2), lg)
    return c.convert("RGB")

# per-strip recipe (base_name -> render_strip kwargs); routed before grouped/compose
THIN = {
 "DIGITAL-CHWebAd2-320x50":      dict(dated=True,  with_url=False, with_logo=False),  # tiny: name + date
 "DIGITAL-CHWebAd3-728x90":      dict(dated=True,  with_url=True,  with_logo=False),
 "DIGITAL-HurricanesEmail-600x100": dict(dated=True, with_url=True, with_logo=False),
 "MARQUEE-WadeAve-350x112":      dict(name_only=True, with_logo=False),               # name only (venue adds date)
}

if __name__ == "__main__":
    n = 0
    for cat, base, w, h, dated, cta, grouped in SIZES:
        folder = f"{OUT}/{CAT[cat]}"; os.makedirs(folder, exist_ok=True)
        if base in THIN:
            c = render_strip(w, h, **THIN[base])
        elif grouped:
            c = compose_grouped(w, h, DATE, dated, cta)
        else:
            c = L.compose(w, h, DATE, dated, cta)
        save_kw = {"dpi": (PRINT_DPI, PRINT_DPI)} if cat == "PRINT" else {}
        c.save(f"{folder}/RodWave-DLD-LenovoCenter-{base}.png", **save_kw); n += 1

    # exotic — In-Ice (no gold)
    dfolder = f"{OUT}/{CAT['DIG']}"; os.makedirs(dfolder, exist_ok=True)
    render_in_ice(1800, 1000).save(f"{dfolder}/RodWave-DLD-LenovoCenter-DIGITAL-VirtualInIce-NOGOLD-1800x1000.png"); n += 1

    # exotic — DED ribbon boards
    render_ribbon(4100, 180).save(f"{dfolder}/RodWave-DLD-LenovoCenter-DIGITAL-DED-Zone1and5-4100x180.png"); n += 1
    render_ribbon(3900, 180, side_clear=140).save(f"{dfolder}/RodWave-DLD-LenovoCenter-DIGITAL-DED-Zone2and4-3900x180.png"); n += 1

    print(f"rendered {n} Raleigh / Lenovo Center assets")
