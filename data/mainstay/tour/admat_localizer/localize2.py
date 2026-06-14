#!/usr/bin/env python3
"""Rod Wave 'Don't Look Down' — per-arena localizer v5 (team-faithful).
Mirrors the team's house style: full-width title lockup, ATLANTA, GA / MONTH DAY
block, full-color arena logo, per-night sets, category folders + naming.
"""
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# Engine assets (base_clean.png, fonts/) resolve next to this file so the kit is
# self-contained and survives /tmp wipes. Override with ADMAT_ROOT if needed.
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("ADMAT_ROOT") or (ENGINE_DIR if os.path.exists(f"{ENGINE_DIR}/base_clean.png") else "/tmp/admat_test")
BASE = Image.open(f"{ROOT}/base_clean.png").convert("RGB")
MW, MH = BASE.size
FT = f"{ROOT}/fonts/Druk_Collection"
# Fonts matched to the master PSD: title=Druk-Medium, ROD WAVE/TOUR=Druk Wide Medium,
# dates/cities=Druk Heavy, body=Druk Medium  (NOT Druk-Super / Druk Text)
F_TITLE = f"{FT}/Druk/Druk-Medium.otf"        # DON'T LOOK DOWN (tour)
F_NAME  = f"{FT}/Druk Wide/DrukWide-Medium.otf"  # ROD WAVE (hero)
F_WIDE  = f"{FT}/Druk Wide/DrukWide-Medium.otf"  # TOUR
F_HEAVY = f"{FT}/Druk/Druk-Heavy.otf"          # date / city
F_MED   = f"{FT}/Druk/Druk-Medium.otf"         # venue / cta

GOLD  = (203, 170, 112)
CREAM = (238, 232, 214)
WOOD  = (54, 45, 34)
FACE_CX, FACE_CY = 0.76, 0.235
MASTER_RATIO = MW / MH
WOOD_SRC = BASE.crop((46, 0, 360, MH))
# Rod's figure as a filled polygon in normalized master coords (right side =
# him). Text must never enter this region (plus a margin). Conservative/generous.
ROD_POLY = [(1.00,0.03),(0.75,0.05),(0.61,0.11),(0.56,0.19),(0.535,0.29),
            (0.51,0.37),(0.45,0.47),(0.37,0.59),(0.31,0.73),(0.27,0.88),
            (0.25,1.00),(1.00,1.00)]

def rod_mask(size, tf, margin):
    """Binary mask (255=Rod) at canvas `size`, polygon via transform tf, dilated by margin."""
    from PIL import ImageFilter
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon([tf(nx,ny) for nx,ny in ROD_POLY], fill=255)
    if margin>0: m = m.filter(ImageFilter.MaxFilter(margin*2+1)) if margin<=10 else m.filter(ImageFilter.GaussianBlur(margin)).point(lambda v:255 if v>30 else 0)
    return m
LOGO = Image.open(f"{ROOT}/logo_color.png").convert("RGBA")
CITY = "ATLANTA, GA"
VENUE = "VENUE"                 # venue name used in ribbon/strip info lines
URL  = "OFFICIAL-RODWAVE.COM"
PRINT_DPI = 300                 # high-end print default (save PRINT files with dpi=(PRINT_DPI,)*2)

def fnt(p, s): return ImageFont.truetype(p, max(4, int(s)))
def tw_(f, t): return f.getbbox(t)[2]
def off_(f, t): return f.getbbox(t)[1]
def fit_font(path, text, maxw, start, mn=6):
    s = start
    while s > mn:
        if tw_(fnt(path, s), text) <= maxw: return fnt(path, s)
        s -= 1
    return fnt(path, mn)

def paste(c, img, x, y): c.paste(img, (int(x), int(y)), img)

# ---------------- component blocks (RGBA) -----------------------------------
def title_block(w, max_h):
    """Artist-first lockup: ROD WAVE is the hero (top, biggest); DON'T LOOK DOWN
    (the tour) sits below it, smaller; TOUR is a small tag. Sized to w x max_h."""
    rw_wfit = fit_font(F_NAME, "ROD WAVE", w, int(w*0.42)).size   # hero in Druk Wide
    rw = max(10, min(rw_wfit, int(max_h / 1.95)))         # hero size, height-capped to fit stack
    # tour title clearly smaller than the name; Druk Medium; fit width, cap at 0.60*hero
    dfit = fit_font(F_TITLE, "DON'T LOOK DOWN", w, int(rw*0.80)).size
    if dfit < rw*0.42:                                     # too small on one line -> wrap
        dld = min(int(rw*0.60), fit_font(F_TITLE, "DON'T LOOK", w, int(rw*0.70)).size)
        dld_lines = ["DON'T LOOK", "DOWN"]
    else:
        dld = min(int(rw*0.60), dfit); dld_lines = ["DON'T LOOK DOWN"]
    tour = max(8, int(rw*0.28))
    fr = fnt(F_NAME, rw); fd = fnt(F_TITLE, dld); ftn = fnt(F_WIDE, tour)
    rlh = int(rw*0.82); dlh = int(dld*0.82); tlh = int(tour*1.5)
    H = rlh + int(rw*0.07) + dlh*len(dld_lines) + int(dld*0.18) + tlh + 8
    img = Image.new("RGBA", (w, H), (0,0,0,0)); d = ImageDraw.Draw(img); y = 0
    d.text((0, y - off_(fr, "ROD WAVE")), "ROD WAVE", font=fr, fill=GOLD)
    y += rlh + int(rw*0.07)
    for ln in dld_lines:
        d.text((0, y - off_(fd, ln)), ln, font=fd, fill=CREAM); y += dlh
    y += int(dld*0.18)
    d.text((0, y - off_(ftn, "TOUR")), "TOUR", font=ftn, fill=GOLD)
    bb = img.getbbox()
    return img.crop((0, 0, w, bb[3]))

LOGO_FRAC = 0.60          # logo width as fraction of block width (Birmingham overrides)
def localized_block(w, date_line, with_logo=True, align="left", max_logo_h=None):
    df = fit_font(F_HEAVY, date_line, w, int(w*0.20))
    cf = fnt(F_HEAVY, int(df.size*0.74))
    ch = int(cf.size*0.80); dh = int(df.size*0.80)
    logo_w = int(w*LOGO_FRAC); logo_h = int(LOGO.height * logo_w/LOGO.width) if with_logo else 0
    if with_logo and max_logo_h and logo_h > max_logo_h:      # cap tall logos
        logo_h = int(max_logo_h); logo_w = int(logo_h * LOGO.width/LOGO.height)
    gap = int(df.size*0.32)
    img = Image.new("RGBA", (w, ch + dh + (gap+logo_h if with_logo else 0) + int(df.size*0.4)), (0,0,0,0))
    d = ImageDraw.Draw(img); y = 0
    def X(tw): return (w-tw)//2 if align=="center" else 0
    d.text((X(tw_(cf,CITY)), y - off_(cf, CITY)), CITY, font=cf, fill=GOLD); y += int(ch*1.12)
    d.text((X(tw_(df,date_line)), y - off_(df, date_line)), date_line, font=df, fill=GOLD); y += dh + gap
    if with_logo:
        lg = LOGO.resize((logo_w, logo_h), Image.LANCZOS); paste(img, lg, X(logo_w), y); y += logo_h
    return img.crop((0,0,w,y))

def cta_url_block(w):
    f = fit_font(F_MED, f"TICKETS AVAILABLE ON:  {URL}", w, int(w*0.05))
    h = int(f.size*1.0)
    img = Image.new("RGBA", (w, int(h*2.4)), (0,0,0,0)); d = ImageDraw.Draw(img)
    d.text((0, 0), "TICKETS AVAILABLE ON:", font=f, fill=GOLD)
    d.text((0, int(h*1.15)), URL, font=f, fill=GOLD)
    return img.crop(img.getbbox())

def button_block(w):
    bw, bh = int(w*0.62), int(w*0.155)
    f = fit_font(F_HEAVY, "BUY TICKETS", int(bw*0.8), int(bh*0.6))
    img = Image.new("RGBA", (bw, bh), (0,0,0,0)); d = ImageDraw.Draw(img)
    d.rounded_rectangle([0,0,bw-1,bh-1], radius=int(bh*0.16), fill=(244,240,232))
    tw = tw_(f, "BUY TICKETS"); ty = (bh-(f.getbbox("BUY TICKETS")[3]))//2
    d.text(((bw-tw)//2, ty), "BUY TICKETS", font=f, fill=(26,22,18))
    return img

# ---------------- canvas builders -------------------------------------------
def wood_panel(pw, ph):
    strip = WOOD_SRC.resize((max(1,int(WOOD_SRC.width*ph/MH)), ph), Image.LANCZOS)
    sw = strip.width; panel = Image.new("RGB",(pw,ph),WOOD); x, flip = pw, False
    while x > 0:
        x -= sw; panel.paste(ImageOps.mirror(strip) if flip else strip,(x,0)); flip=not flip
    g = Image.new("L",(pw,1))
    for i in range(pw): g.putpixel((i,0), int(60+150*(i/pw)))
    return Image.composite(panel, Image.new("RGB",(pw,ph),(0,0,0)), g.resize((pw,ph)))

def cover_build(tw, th):
    """Cover-crop biased to KEEP the left wood column (text lane), face stays in frame."""
    sc = max(tw/MW, th/MH); sw, sh = round(MW*sc), round(MH*sc)
    img = BASE.resize((sw,sh), Image.LANCZOS)
    x0 = 0
    if FACE_CX*sw > tw - 0.06*tw:                 # face would fall off right -> shift just enough
        x0 = min(max(int(FACE_CX*sw + 0.18*sw) - tw, 0), sw-tw)
    y0 = min(max(int(FACE_CY*sh)-int(0.26*th),0), sh-th)
    tf = lambda nx,ny: (nx*sw - x0, ny*sh - y0)
    return img.crop((x0,y0,x0+tw,y0+th)), tf

def vfill(rod, tw, gap_h, dark=0.0):
    """Vertical wood fill (gap_h tall, tw wide) sampled from rod's clean top planks."""
    band = rod.crop((0, 6, tw, 6+min(54, rod.height-8))).resize((tw, max(1,gap_h)), Image.LANCZOS)
    if dark>0: band = Image.blend(band, Image.new("RGB",band.size,(0,0,0)), dark)
    return band

def tall_build(tw, th):
    """Fit Rod by WIDTH in the upper-middle; clean wood band above (title) and below (block)."""
    rh = round(MH*tw/MW); rod = BASE.resize((tw,rh), Image.LANCZOS)
    y_off = int(th*0.11)
    c = Image.new("RGB",(tw,th), WOOD)
    if y_off>0: c.paste(vfill(rod,tw,y_off), (0,0))
    c.paste(rod,(0,y_off))
    bs = y_off+rh
    if bs<th: c.paste(vfill(rod,tw,th-bs,dark=0.5),(0,bs))
    tf = lambda nx,ny: (nx*tw, ny*rh + y_off)
    return c, tf

def banner_build(tw, th):
    sh=th; sw=round(MW*th/MH); rod=BASE.resize((sw,sh),Image.LANCZOS).convert("RGBA")
    canvas=wood_panel(tw,th); fw=max(8,int(sw*0.06)); m=Image.new("L",(sw,sh),255); px=m.load()
    for i in range(fw):
        for y in range(sh): px[i,y]=int(255*i/fw)
    rod.putalpha(m); canvas.paste(rod,(tw-sw,0),rod)
    tf = lambda nx,ny: (nx*sw + (tw-sw), ny*th)
    return canvas.convert("RGB"), (tw-sw), tf

# ---------------- clearance-aware placement ---------------------------------
import numpy as _np
def _overlaps(mask, x, y, blk):
    if y<0: return True
    region = mask.crop((int(x),int(y),int(x)+blk.width,int(y)+blk.height))
    mr=_np.asarray(region); ta=_np.asarray(blk.split()[-1])
    if mr.shape!=ta.shape: return False
    return bool(((mr>128)&(ta>40)).any())

def place(c, mask, make, *, x=0, w=0, top=None, bottom=None, center_x=None,
          min_top=None, shrink=0.9, tries=12):
    """Render make(w); shrink width until it clears the Rod mask (and stays below
    min_top if given); paste; return (blk,py)."""
    bw=w
    blk=make(bw); px=py=0
    for _ in range(tries):
        blk=make(bw)
        px = int(center_x-blk.width//2) if center_x is not None else x
        py = top if top is not None else bottom-blk.height
        ok = not _overlaps(mask,px,py,blk)
        if min_top is not None and py < min_top: ok=False
        if ok: break
        bw=int(bw*shrink)
    paste(c,blk,px,py); return blk,py

# ---------------- layout dispatch -------------------------------------------
def compose(tw, th, date_line, dated, cta):
    ratio = tw/th
    if ratio > 1.18:                               # ---- WIDE BANNER ----
        c, pw, tf = banner_build(tw, th)
        mask = rod_mask((tw,th), tf, margin=max(6,int(min(tw,th)*0.02)))
        mx=int(tw*0.04); title_w=max(120,min(int(pw*0.96),int(tw*0.60)))
        # title from top; block + cta packed bottom-up, never overlapping each other or the title
        tb,ty = place(c,mask,lambda w: title_block(w,int(th*0.30)), x=mx, w=title_w, top=int(th*0.08))
        title_bottom = ty + tb.height
        yb = th - int(th*0.05)
        if cta=="url": mk=lambda w: cta_url_block(w); cw=min(title_w,int(tw*0.34))
        elif cta=="button": mk=lambda w: button_block(int(w)); cw=int(tw*0.18)
        else: mk=None
        if mk is not None:
            cb,cy=place(c,mask,mk, x=mx, w=cw, bottom=yb); yb=cy-int(th*0.03)
        if dated:
            place(c,mask,lambda w: localized_block(w,date_line,max_logo_h=int(th*0.17)),
                  x=mx, w=min(title_w,int(tw*0.32)), bottom=yb, min_top=title_bottom+int(th*0.03))
        return c
    elif ratio < 0.66:                             # ---- TALL ----
        c, tf = tall_build(tw, th)
        mask = rod_mask((tw,th), tf, margin=max(8,int(tw*0.03)))
        mx=int(tw*0.06); zw=tw-2*mx
        place(c,mask,lambda w: title_block(w,int(th*0.15)), x=mx, w=zw, top=int(th*0.035))
        yb=int(th*0.965)
        if cta=="url":
            cb,cy=place(c,mask,lambda w: cta_url_block(w), w=zw, center_x=tw//2, bottom=yb); yb=cy-int(th*0.012)
        elif cta=="button":
            cb,cy=place(c,mask,lambda w: button_block(int(w)), w=int(tw*0.5), center_x=tw//2, bottom=yb); yb=cy-int(th*0.012)
        if dated:
            place(c,mask,lambda w: localized_block(w,date_line,align="center",max_logo_h=int(th*0.085)), w=zw, center_x=tw//2, bottom=yb)
        return c
    # ---- POSTER (square / near) ----
    c, tf = cover_build(tw, th)
    mask = rod_mask((tw,th), tf, margin=max(8,int(min(tw,th)*0.025)))
    mx=int(tw*0.05); zw=tw-2*mx
    place(c,mask,lambda w: title_block(w,int(th*0.27)), x=mx, w=zw, top=int(th*0.045))
    bw=int(tw*0.56); yb=int(th*0.95)
    if cta=="button":
        cb,cy=place(c,mask,lambda w: button_block(int(w*0.75)), x=mx, w=int(tw*0.5), bottom=yb); yb=cy-int(th*0.02)
    elif cta=="url":
        cb,cy=place(c,mask,lambda w: cta_url_block(w), x=mx, w=bw, bottom=yb); yb=cy-int(th*0.015)
    if dated:
        place(c,mask,lambda w: localized_block(w,date_line,max_logo_h=int(th*0.13)), x=mx, w=bw, bottom=yb)
    return c

# ===================================================================
# v6 reusable layouts — learned from the Raleigh/Lenovo build (2026-06-14).
# Use these instead of bare compose() so every arena inherits the fixes:
#   - imagery formats  -> compose_grouped() : ONE tight left lockup (Mike's spec:
#       ROD WAVE H1, DON'T LOOK DOWN H2, then date/venue/CTA pulled right under it,
#       bigger + tighter, kept off his face by the mask). NOT title-top/date-bottom.
#   - thin word banners -> render_strip()   : single auto-fit baseline; never stacks.
#   - ribbon boards     -> render_ribbon()  : one ticker row; side_clear for LED zones.
#   - NHL in-ice        -> render_in_ice()  : cream-only (no gold/yellow) per ice rule.
# ===================================================================
from PIL import ImageDraw as _ImageDraw  # local alias (ImageDraw already imported)

def _stack_block(w, date_line, dated, cta, max_logo_h):
    """ONE consolidated left-aligned lockup at width w: ROD WAVE (H1) over
    DON'T LOOK DOWN/TOUR (H2), then CITY/DATE/logo + CTA tight underneath."""
    parts = [title_block(w, int(w*2))]                 # big name (width-driven)
    if dated:
        parts.append(localized_block(w, date_line, with_logo=True, align="left",
                                     max_logo_h=max_logo_h))
    if cta == "url":    parts.append(cta_url_block(w))
    elif cta == "button": parts.append(button_block(w))
    g_title = int(w*0.075); g = int(w*0.05)
    H = parts[0].height + sum((g_title if i == 1 else g) + p.height
                              for i, p in enumerate(parts) if i > 0)
    img = Image.new("RGBA", (w, H), (0,0,0,0)); y = 0
    for i, p in enumerate(parts):
        if i > 0: y += g_title if i == 1 else g
        img.alpha_composite(p, (0, y)); y += p.height
    return img

def compose_grouped(tw, th, date_line, dated, cta):
    """Imagery layout: render the whole lockup as ONE tight cluster on the left,
    vertically centered (low-left on tall), shrunk/nudged to clear Rod's face+body."""
    ratio = tw/th
    if ratio > 1.18:
        c, pw, tf = banner_build(tw, th); margin = max(6, int(min(tw,th)*0.02))
        mx = int(tw*0.05); lane = int(tw*0.50); valign = "center"; mlh = int(th*0.20)
    elif ratio < 0.66:
        c, tf = tall_build(tw, th); margin = max(8, int(tw*0.03))
        mx = int(tw*0.07); lane = int(tw*0.70); valign = "bottom"; mlh = int(th*0.075)
    else:
        c, tf = cover_build(tw, th); margin = max(8, int(min(tw,th)*0.025))
        mx = int(tw*0.05); lane = int(tw*0.56); valign = "center"; mlh = int(th*0.13)
    mask = rod_mask((tw, th), tf, margin)
    w = lane; blk = None; px = mx; py = 0
    for _ in range(16):
        blk = _stack_block(w, date_line, dated, cta, mlh)
        if blk.height > th - 2*int(th*0.04):
            w = int(w*0.92); continue
        if valign == "center":   py = (th - blk.height)//2
        elif valign == "bottom": py = th - blk.height - int(th*0.06)
        else:                    py = int(th*0.06)
        py = max(int(th*0.04), min(py, th - blk.height - int(th*0.04)))
        if not _overlaps(mask, px, py, blk): break
        moved = False
        for dy in range(int(th*0.02), int(th*0.45), max(1, int(th*0.02))):
            ny = py - dy
            if ny < int(th*0.03): break
            if not _overlaps(mask, px, ny, blk): py = ny; moved = True; break
        if moved: break
        w = int(w*0.92)
    paste(c, blk, px, py)
    return c

def render_strip(w, h, date_line, dated=True, with_url=True, name_only=False, with_logo=False):
    """Thin word-banner: ROD WAVE · DON'T LOOK DOWN [· DATE · CITY · URL] on ONE
    baseline, auto-scaled to fit width so nothing ever stacks/overlaps. Default
    no logo (the wide mark crowds tiny strips; it's on the imagery assets already);
    set with_logo=True only where the spec demands the venue mark."""
    pad = max(6, int(h*0.18))
    logo_h = int(h*0.56) if with_logo else 0
    logo_w = int(LOGO.width*logo_h/LOGO.height) if with_logo else 0
    if with_logo and logo_w > int(w*0.26):
        logo_w = int(w*0.26); logo_h = int(logo_w*LOGO.height/LOGO.width)
    gap_logo = int(h*0.45) if with_logo else 0
    text_avail = w - 2*pad - (logo_w + gap_logo if with_logo else 0)
    s_name, s_tour, s_info = h*0.52, h*0.42, h*0.34
    SEP = "   ·   "
    segs = [("ROD WAVE", F_NAME, s_name, GOLD),
            ("DON’T LOOK DOWN", F_TITLE, s_tour, CREAM)]
    if dated and not name_only:
        info = f"{date_line}  ·  {CITY}" + (f"  ·  {URL}" if with_url else "")
        segs.append((info, F_MED, s_info, CREAM))
    def layout(scale):
        out = []; total = 0
        fsep = fnt(F_MED, max(4, int(s_info*scale)))
        for i, (t, fp, sz, col) in enumerate(segs):
            f = fnt(fp, max(4, int(sz*scale))); tw = tw_(f, t)
            out.append((t, f, col, tw)); total += tw
            if i < len(segs)-1: total += tw_(fsep, SEP)
        return total, out, fsep
    scale = 1.0
    for _ in range(48):
        total, rendered, fsep = layout(scale)
        if total <= text_avail or scale < 0.12: break
        scale *= 0.94
    c = Image.alpha_composite(wood_panel(w, h).convert("RGBA"),
                              Image.new("RGBA", (w, h), (0,0,0,70)))
    d = ImageDraw.Draw(c); base = int(h*0.72); x = pad; sepw = tw_(fsep, SEP)
    for i, (t, f, col, tw) in enumerate(rendered):
        d.text((x, base - f.getbbox(t)[3]), t, font=f, fill=col); x += tw
        if i < len(rendered)-1:
            d.text((x, base - fsep.getbbox(SEP)[3]), SEP, font=fsep, fill=GOLD); x += sepw
    if with_logo:
        lg = LOGO.resize((logo_w, logo_h), Image.LANCZOS)
        c.paste(lg, (w - pad - logo_w, (h-logo_h)//2), lg)
    return c.convert("RGB")

def render_ribbon(w, h, date_line, side_clear=0):
    """Single full-width ribbon-board row: ROD WAVE / DON'T LOOK DOWN + info line
    + venue logo (right). side_clear keeps content off the edge bands (LED zones
    bordering blue lines need ~140px)."""
    c = Image.alpha_composite(wood_panel(w, h).convert("RGBA"),
                              Image.new("RGBA", (w, h), (0,0,0,70)))
    d = ImageDraw.Draw(c)
    x0 = side_clear + int(h*0.18); x1 = w - side_clear - int(h*0.18)
    base = h - int(h*0.24)
    f_n = fnt(F_NAME, int(h*0.46)); f_t = fnt(F_TITLE, int(h*0.40)); f_m = fnt(F_MED, int(h*0.30))
    x = x0
    d.text((x, base - f_n.getbbox("ROD WAVE")[3]), "ROD WAVE", font=f_n, fill=GOLD)
    x += tw_(f_n, "ROD WAVE") + int(h*0.5)
    d.text((x, base - f_t.getbbox("DON’T LOOK DOWN")[3]), "DON’T LOOK DOWN", font=f_t, fill=CREAM)
    x += tw_(f_t, "DON’T LOOK DOWN") + int(h*0.5)
    info = f"{date_line}  ·  {VENUE}  ·  {CITY}  ·  {URL}"
    d.text((x, (h - f_m.getbbox(info)[3])//2), info, font=f_m, fill=CREAM)
    lg_h = int(h*0.56); lg_w = int(LOGO.width*lg_h/LOGO.height)
    lg = LOGO.resize((lg_w, lg_h), Image.LANCZOS); lx = x1 - lg_w
    if lx > x + 20: c.paste(lg, (lx, (h-lg_h)//2), lg)
    return c.convert("RGB")

def render_in_ice(w, h, with_logo=True):
    """NHL virtual in-ice: cream-only palette (NO gold/yellow per ice rules),
    grouped title left, large venue logo lower-left, no date."""
    global GOLD, CREAM
    g_old, c_old = GOLD, CREAM
    GOLD = CREAM = (238, 232, 214)
    try:
        c = compose_grouped(w, h, "", False, "none")
    finally:
        GOLD, CREAM = g_old, c_old
    if with_logo:
        lg_w = int(w*0.46*0.86); lg = LOGO.resize((lg_w, int(LOGO.height*lg_w/LOGO.width)), Image.LANCZOS)
        c.paste(lg, (int(w*0.045), int(h - lg.height - h*0.10)), lg)
    return c

# ---------------- size table (category, base name, w,h, dated, cta) ----------
CAT = {
 "WEB":"WEBSITE", "TM":"TICKETMASTER ACCOUNT MANAGER", "SOC":"SOCIAL",
 "EMAIL":"EMAIL", "UPC":"UPCOMING EVENTS PRESENTED BY TICKETMASTER",
 "MARQ":"DIGITAL OUTDOOR MARQUEES", "LED":"IN-ARENA LED BOARDS", "PRINT":"PRINT",
}
SIZES = [
 ("WEB","WEBSITE-835x470",835,470,False,"none"),
 ("WEB","WEBSITE-640x360",640,360,False,"none"),
 ("WEB","WEBSITE-1440x500",1440,500,False,"none"),
 ("WEB","WEBSITE-1000x750",1000,750,True,"button"),
 ("TM","TICKETMASTERACCOUNTMANAGER-600x340",600,340,False,"none"),
 ("SOC","SOCIAL-1200x627",1200,627,True,"none"),
 ("SOC","SOCIAL-1000x1000",1000,1000,True,"none"),
 ("SOC","SOCIAL-1080x1920-IG-STORY",1080,1920,True,"none"),
 ("EMAIL","EMAIL-600x400",600,400,True,"button"),
 ("EMAIL","EMAIL-600x250",600,250,True,"button"),
 ("UPC","UPCOMINGEVENTS-TICKETMASTER-1920x1080",1920,1080,False,"none"),
 ("MARQ","DIGITALOUTDOORMARQUEES-540x1090",540,1090,True,"url"),
 ("MARQ","DIGITALOUTDOORMARQUEES-540x1260",540,1260,True,"url"),
 ("MARQ","DIGITALOUTDOORMARQUEES-900x1260",900,1260,True,"url"),
 ("LED","INARENALEDBOARDS-InBowlCornerBoards-1152x792",1152,792,True,"url"),
 ("LED","INARENALEDBOARDS-InBowlCornerBoards-2080x684",2080,684,True,"url"),
 ("LED","INARENALEDBOARDS-ConcourseTVs-1920x1080",1920,1080,True,"url"),
 ("LED","INARENALEDBOARDS-ConcourseTVs-260x760",260,760,True,"url"),
 ("PRINT","PRINT-5x5-1575x1575",1575,1575,True,"none"),
]
NIGHTS = [("18","NOVEMBER 18"), ("19","NOVEMBER 19")]
OUT = f"{ROOT}/out2/Localized Ad Assets"
if __name__ == "__main__":
    n=0
    for cat,base,w,h,dated,cta in SIZES:
        folder=f"{OUT}/{CAT[cat]}"; os.makedirs(folder, exist_ok=True)
        if dated:
            for nn,dl in NIGHTS:
                c=compose(w,h,dl,True,cta)
                c.save(f"{folder}/RodWave-DLD-Statefarm-{base}-{nn}.png"); n+=1
        else:
            c=compose(w,h,"",False,cta)
            c.save(f"{folder}/RodWave-DLD-Statefarm-{base}.png"); n+=1
    print(f"rendered {n} assets")
