# RUNBOOK — Localize the Rod Wave "Don't Look Down" Ad-Mat for an Arena
### For a backup agent (e.g. Lucius) to run in a pinch when Alfred is busy.

You are producing a localized tour ad-mat set for ONE arena. Work into the **testing
folder only** — NEVER deliver to an arena or email anyone. Mike reviews everything.

## 0. Prerequisites (check first; if missing, tell Mike — do not improvise)
- Python 3 with: `Pillow`, `psd-tools`, `numpy`. Tools: `pdftoppm`, `ghostscript` (for .ai/.eps logos).
- The engine kit: `~/alfred/data/mainstay/tour/admat_localizer/` (has `localize2.py`,
  `base_clean.png`, `fonts/`, example drivers, `localize_TEMPLATE.py`).
  If you are NOT on server 105, get the kit from Nextcloud:
  `MainStay/Don't Look Down Tour/Marketing Assets/00 - MASTER AD-MAT KIT/`
  (`base_clean.png` + `Druk_Collection.zip`). The renderer only needs `base_clean.png` + the fonts.
- Read `RULES.txt` in the master kit. Those rules are LOCKED — follow them exactly.

## 1. Get the arena pack
Mike gives you the arena's "UPLOAD - Your Logo + Ad Specs Here" Nextcloud share link.
Download + unzip it. Find: the **spec sheet** (PDF / DOCX / PNG table) and the **logo files**.

## 2. Read the spec sheet
List every required pixel size and note which formats say "NO date / NO logo"
(website/auto-populated ones). `pdftotext -layout file.pdf -` for PDFs; for a PNG
spec table, read it as an image.

## 3. Prep the venue logo
Use the venue's DARK-GROUND / WHITE / reverse / mono-white mark (dark logos vanish on the
wood wall). High-res rasterize the vector:
- `gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pngalpha -r600 -sOutputFile=<venue>_dark.png in.eps`
  then crop to bbox (`.crop(im.getbbox())`). Save `<venue>_dark_trim.png`.

## 4. Confirm city + date(s)
Match the master routing's city label (e.g. Amerant Bank Arena = "FT. LAUDERDALE, FL").
If the arena has TWO tour dates, build a SEPARATE set per date — NEVER two dates on one ad-mat.

## 5. Write the driver (engine v6 — use the new layouts)
Copy `localize_TEMPLATE.py` → `localize_<city>.py`. Set `L.LOGO`, `L.CITY`, `L.VENUE`,
`L.LOGO_FRAC` (square ~0.44 / wide horizontal ~0.66), `DATE`, and the `SIZES` table.
**Tag each asset with a `kind` and the engine routes it** (this is the key v6 upgrade):
- `imagery` → `L.compose_grouped` — photo formats. ONE tight LEFT cluster: ROD WAVE (H1) over
  DON'T LOOK DOWN (H2), then city/date/logo/CTA pulled right under it, bigger + tighter,
  auto-kept off his face/body. (Do NOT use bare `compose` for imagery — it strands the date at
  the bottom; Mike rejected that look.)
- `strip` → `L.render_strip` — THIN word banners (leaderboards, email ads, marquees). One
  auto-fitting baseline, never stacked. Drop the logo on tiny strips (it crowds the words).
- `ribbon` → `L.render_ribbon(w,h,DATE,side_clear=N)` — LED ribbon boards (N≈140px where a zone
  borders the blue lines).
- `inice` → `L.render_in_ice` — NHL virtual in-ice: cream-only, NO gold/yellow.
Print at 300 DPI: render at inches×300 (8.5×11→2550×3300, 11×17→3300×5100) and save with
`dpi=(300,300)`. Skip slots that aren't ours (e.g. a "Ticketmaster season logo") and
dimensionless spec lines — note them, don't fabricate.

## 6. Render + QA (do not skip)
`python3 localize_<city>.py`. Build a contact sheet and eyeball every asset against RULES:
- ROD WAVE on top and BIGGER than "DON'T LOOK DOWN"; imagery = one grouped left cluster.
- Thin banners = one clean line, no overlap.
- Fonts: Druk-Medium / Druk-Wide-Medium / Druk-Heavy (crisp, not chunky).
- NO text on the artist's face or body. Venue logo prominent. URL = OFFICIAL-RODWAVE.COM.
- One date per file. Respect "no date/logo" formats (offer a dated variant too).
Fix and re-render anything off.

## 7. Deliver to the testing folder ONLY → then Mike approves
Upload to Nextcloud `Resizing Testing/[City - Venue]/Localized Ad Assets/<CATEGORY>/` with a
`_CONTACT-SHEET`, make a public share link, and email Mike a summary + the contact sheet.
**STOP. Do not put anything in an arena's download folder. Do not email any arena. That is Mike's call.**
Only AFTER Mike approves does Alfred drop the set into the arena's real DOWNLOAD folder, which
auto-fires the venue "assets landed" email via `scripts/arena_drop_notifier.py` (Mike + Dre are
CC'd on every such notice).

## Escalate to Mike if
- A required font or the layered PSD is missing.
- The spec sheet has an exotic format you can't place cleanly.
- You're unsure of the date or city label.
Better to ask than to ship something off-brand.
