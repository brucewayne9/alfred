#!/usr/bin/env python3
"""Birmingham / Legacy Arena (BJCC) localizer — reuses the house-style engine,
swaps venue logo + city + date, and adds the 360 ribbon-board renderer."""
import os
from PIL import Image, ImageDraw, ImageOps
import localize2 as L

# --- venue overrides ---------------------------------------------------------
LEGACY = Image.open("/tmp/admat_test/bham/2. UPLOAD - Your Logo + Ad Specs Here/"
                    "onedrive/Legacy_Primary_Mark_B&W_Reverse NO BACKGROUND.png").convert("RGBA")
LEGACY = LEGACY.crop(LEGACY.getbbox())
L.LOGO = LEGACY
L.CITY = "BIRMINGHAM, AL"
L.LOGO_FRAC = 0.44        # Legacy mark is ~square; smaller share so it doesn't tower
DATE = "SEPTEMBER 17"
OUT = "/tmp/admat_test/out_bham/Localized Ad Assets"

# --- ribbon board (2410x72 cell, tiled to 12048x144) -------------------------
def ribbon_cell(w=2410, h=72):
    """Single repeating ribbon unit: wood strip + horizontal show-info lockup."""
    strip = L.WOOD_SRC.resize((max(1,int(L.WOOD_SRC.width*h/L.MH)), h), Image.LANCZOS)
    cell = Image.new("RGB",(w,h)); x,flip=w,False
    while x>0:
        x-=strip.width; cell.paste(ImageOps.mirror(strip) if flip else strip,(x,0)); flip=not flip
    cell = Image.alpha_composite(cell.convert("RGBA"), Image.new("RGBA",(w,h),(0,0,0,70)))
    d = ImageDraw.Draw(cell)
    pad=int(h*0.18)
    f_t = L.fnt(L.F_TITLE, int(h*0.56))    # ROD WAVE — name leads, bigger
    f_b = L.fnt(L.F_TITLE, int(h*0.40))    # DON'T LOOK DOWN — smaller
    f_m = L.fnt(L.F_MED, int(h*0.32))      # info
    base = h - int(h*0.20)                  # common baseline
    x = pad
    d.text((x, base - f_t.getbbox("ROD WAVE")[3]),"ROD WAVE",font=f_t,fill=L.GOLD); x+=L.tw_(f_t,"ROD WAVE")+int(h*0.5)
    d.text((x, base - f_b.getbbox("DON'T LOOK DOWN")[3]),"DON'T LOOK DOWN",font=f_b,fill=L.CREAM); x+=L.tw_(f_b,"DON'T LOOK DOWN")+int(h*0.5)
    info=f"{DATE}  ·  LEGACY ARENA  ·  BIRMINGHAM, AL  ·  {L.URL}"
    my=(h-int(h*0.34))//2 - L.off_(f_m,"A")//2
    d.text((x,my),info,font=f_m,fill=L.CREAM)
    return cell.convert("RGB")

def ribbon_full(cellw=2410, cellh=72, W=12048, H=144):
    cell = ribbon_cell(cellw, cellh)
    full = Image.new("RGB",(W,H))
    for row in range(H//cellh):
        x=0
        while x < W:
            full.paste(cell,(x,row*cellh)); x+=cellw
    return full

# --- size set ----------------------------------------------------------------
CAT = {"SOC":"SOCIAL","SCR":"DIGITAL SCREENS","MARQ":"OUTDOOR DIGITAL MARQUEE",
       "RIB":"RIBBON BOARD"}
SIZES = [
 ("SOC","SOCIAL-1080x1920-STORY",1080,1920,"none"),
 ("SOC","SOCIAL-1080x1350-PORTRAIT",1080,1350,"none"),
 ("SOC","SOCIAL-1080x1080-SQUARE",1080,1080,"none"),
 ("SCR","DIGITALSCREENS-WEBSITE-1920x1080",1920,1080,"url"),
 ("MARQ","OUTDOORMARQUEE-840x476",840,476,"url"),
]
if __name__ == "__main__":
    n=0
    for cat,base,w,h,cta in SIZES:
        folder=f"{OUT}/{CAT[cat]}"; os.makedirs(folder,exist_ok=True)
        c=L.compose(w,h,DATE,True,cta)
        c.save(f"{folder}/RodWave-DLD-Legacy-{base}.png"); n+=1
    # ribbon
    rfolder=f"{OUT}/{CAT['RIB']}"; os.makedirs(rfolder,exist_ok=True)
    ribbon_cell().save(f"{rfolder}/RodWave-DLD-Legacy-RIBBON-CELL-2410x72.png")
    ribbon_full().save(f"{rfolder}/RodWave-DLD-Legacy-RIBBON-FULL-12048x144.png")
    n+=2
    print(f"rendered {n} Birmingham assets")
