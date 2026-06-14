---
name: rodwave-dld-admat
description: Use when localizing the Rod Wave "Don't Look Down" tour ad-mat for a specific arena, venue, or market — producing every ad size that arena requires from the master PSD, with their venue logo, city, and tour date(s). Triggers on requests like "localize the Don't Look Down ad-mat for Brooklyn", "make the Rod Wave ad-mat for [arena]", "resize the tour creative for [city]".
---

# Rod Wave "Don't Look Down" — Arena Ad-Mat Localizer

## Overview
Turns the master "Don't Look Down" tour ad-mat into a complete, per-arena localized set:
strip the full date list, drop in ONE arena + ONE date + their venue logo, and render
every size on that arena's spec sheet. Proven on State Farm, Legacy, Xfinity Mobile,
Amerant, and Spectrum.

## Inputs needed
1. The arena's spec sheet + logo (usually their Nextcloud "UPLOAD - Your Logo + Ad Specs Here" folder — get the share link).
2. The city label + tour date(s). Pull dates from the master routing / Grey Matter; convert to the day if asked. If the arena has TWO tour dates, build a SEPARATE set per date (never combine two dates on one ad-mat).

## Engine location
`~/alfred/data/mainstay/tour/admat_localizer/` — the renderer (`localize2.py`, **v6**), the clean
Rod-on-wood plate (`base_clean.png`, 2160×2700), the Druk fonts, `localize_TEMPLATE.py`, and
example per-arena drivers (`localize_raleigh.py` is the current reference; also `_bham`, `_philly`,
`_amerant`, `_spectrum`). Read `RULES.md` in this skill folder before rendering — those rules are
LOCKED by Mike.

## Delivery flow (Mike-approved — DO NOT skip the review gate)
Render → upload to the **test folder** (`Resizing Testing/[City - Venue]/…`) + make a public share
link → **email Mike a summary + the contact sheet** → **he approves** → only THEN deliver to the
arena's real DOWNLOAD folder (`…/1. DOWNLOAD - Finished Assets From Ground Rush/Localized Ad Assets/<CAT>/`).
Dropping into the DOWNLOAD folder auto-fires the venue "assets just landed" email via
`scripts/arena_drop_notifier.py` (run it once to send immediately instead of waiting on cron).
NEVER deliver to an arena before Mike says go.

## Workflow
1. **Get the arena pack** — list/download their UPLOAD folder via the Nextcloud client; read the spec sheet (PDF/DOCX/PNG). List EVERY required size and each format's notes ("no date", "logo only", "avoid gold", "Nx edge clearance", "can omit details").
2. **Pick + rasterize the logo** — use their DARK-GROUND / white-reverse mark (dark logos vanish on the wood). Rasterize EPS at high res: `gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pngalpha -r600 -sOutputFile=logo.png in.eps`, then crop to bbox.
3. **Confirm city + date(s)** — match the master routing label. ONE date per set; two-night arenas = run twice (`-NN` per night).
4. **Write a tiny driver** — copy `localize_TEMPLATE.py`; set `L.LOGO`, `L.CITY`, `L.VENUE`, `L.LOGO_FRAC`, `DATE`, and the `SIZES` table. Tag each asset with a **kind** (see Quick reference) — the engine routes it. Custom code only for truly exotic formats.
5. **Render + QA** — render all; build a contact sheet; eyeball the hard ratios AND the thin strips. Verify `RULES.md`: name-first, off his face/body, grouped lockup on imagery, one line on thin banners, correct URL, one date.
6. **Verify against the spec sheet** — before telling Mike it's ready, cross-check every produced size against their sheet; report exact matches + any caveats (skipped non-ours slots, dimensionless items, print DPI).

## Quick reference — pick a `kind` per asset
| kind | use for | engine call |
|---|---|---|
| `imagery` | any photo format (social, posters, web headers, print) | `compose_grouped` — ONE tight left lockup |
| `strip` | thin word banners (leaderboards, email ads, marquees) | `render_strip` — single auto-fit baseline |
| `ribbon` | LED ribbon boards | `render_ribbon(…, side_clear=N)` |
| `inice` | NHL virtual in-ice | `render_in_ice` — cream-only, no gold |
| `plain` | legacy top/bottom split (rarely needed) | `compose` |

## Lessons baked into v6 (so they don't recur)
- **Imagery = one grouped cluster, NOT title-top/date-bottom.** Mike's spec: ROD WAVE big (H1) over DON'T LOOK DOWN (H2), then city/date/logo/CTA pulled tight right underneath, left-aligned, vertically centered — bigger and closer together. `compose_grouped` does this and keeps it off his face/body via the mask. (Old `compose` stranded the date at the bottom — don't use it for imagery.)
- **Thin banners must be ONE baseline, never stacked.** Short strips (e.g. 320×50, 728×90, 600×100) jumbled when the engine stacked title+date+CTA vertically. `render_strip` lays it on one line and auto-shrinks to fit. **Drop the venue logo on tiny strips** (the wide mark crowds the words; it's already on the imagery assets) — keep it only where the spec demands it (ribbons).
- **Honor the spec's per-asset notes.** "Generic / no date or location" headers render dateless — but OFFER Mike a dated + Buy-Tickets variant alongside. "Logo/name only" marquees = `strip(name_only=True)`. "Can omit details" = drop the URL on the tiniest strips.
- **NHL in-ice bans gold/yellow** — use `render_in_ice` (cream-only). Never the gold title there.
- **LED zones bordering blue lines need edge clearance** (Lenovo asked 140px) — `render_ribbon(side_clear=140)`.
- **Skip slots that aren't ours** (e.g. a "Ticketmaster season logo" zone) and **dimensionless items** (a spec line with no size) — don't fabricate; tell Mike what you skipped and why.
- **Print = 300 DPI for high-end.** Render print at inches×300 (8.5×11 → 2550×3300, 11×17 → 3300×5100) and save with `dpi=(300,300)`. The master plate is 2160×2700, so 8.5×11 is near-native crisp; 11×17 is a ~1.9× photo upscale (sharp text/logo, soft-ish photo) — flag that and offer to source a higher-res Rod plate if gallery-grade is needed.

## Common mistakes
- Using Druk-Super / Druk-Text (chunky/soft) instead of the real **Druk-Medium / Druk-Wide-Medium / Druk-Heavy**.
- Two dates on one ad-mat — always one date per file (`-NN` per night).
- Dark venue logo on the dark wall — use the dark-ground/white-reverse mark.
- Imagery rendered with the old split layout instead of the grouped cluster.
- Thin strips with stacked/overlapping words, or a giant logo crowding them.
- Delivering to an arena before Mike approves the test-folder proof.
