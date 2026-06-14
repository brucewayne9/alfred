# Don't Look Down Ad-Mat — LOCKED RULES (Mike, 2026-06-13)

These are not suggestions. Mike set them across the State Farm / Birmingham iterations.

## Hierarchy
- **ROD WAVE (artist name) is the hero: ABOVE the tour name AND bigger than it.**
- "DON'T LOOK DOWN" (the tour) sits below, smaller, with a small "TOUR" tag.
- Never tuck Rod Wave's name small in a corner.

## Type (exact fonts from the master PSD)
- Title "DON'T LOOK DOWN" = **Druk-Medium**
- "ROD WAVE" / "TOUR" = **Druk-Wide-Medium**
- Dates / cities = **Druk-Heavy**
- NOT Druk-Super (too chunky) or Druk-Text (wrong optical cut).

## Palette
- Title/name accent = warm gold `(203,170,112)`; body = cream `(238,232,214)`; wood fill `(54,45,34)`.
- Tour palette stays CONSTANT across arenas; only the venue logo/name/date change.

## Dates
- **One date per ad-mat.** Two-night arenas get two SEPARATE sets (`-NN` per night). Never combine.

## Imagery (does NOT apply to ribbon/ticker formats)
- **No text on or near Rod's face or body.** Title in the clean wood band; date block in clean wood (bottom-left on posters, centered-bottom on tall, left lane on banners). The engine enforces this with a silhouette mask — verify anyway.
- Rod's face is never cropped off.

## Venue logo
- **Prominent**, in clean wood. Use the venue's WHITE / reverse / mono-white mark (dark logos vanish on the wood). Keep brand color only if it reads clean on dark.

## CTA
- Ticket URL = **OFFICIAL-RODWAVE.COM** (official-dash-rodwave). NOT rodwave-official.
- On-sale URL goes on large display formats (LED, marquee, video wall, jumbotron); "BUY TICKETS" button where a spec demands it; nothing on pure social.
- Respect "no date/logo/text" instructions on website/auto-populated formats.

## City labels
- Match the master ad-mat's routing labels (e.g. Amerant Bank Arena in Sunrise FL is labeled **FT. LAUDERDALE, FL** per the master).

## Layout (LOCKED 2026-06-14, from the Raleigh build)
- **Imagery formats = ONE grouped lockup, left, vertically centered.** ROD WAVE is the H1
  (big), DON'T LOOK DOWN/TOUR the H2 directly under it, then CITY / DATE / venue logo / CTA
  pulled tight right beneath — as a single cluster. Bigger and closer together. NEVER the old
  title-at-top / date-stranded-at-bottom split. Keep the whole group clear of his face AND body.
  (Engine: `compose_grouped`.)
- **Thin word banners = ONE baseline, never stacked.** ROD WAVE · DON'T LOOK DOWN [· DATE · CITY
  · URL] on a single auto-fitting line. No vertical stacking, no overlap. Drop the venue logo on
  tiny strips (it crowds the words; it's on the imagery already). (Engine: `render_strip`.)
- **Print = 300 DPI** for anything going to a printer (inches × 300; save with the DPI flag).
- **NHL virtual in-ice = cream-only, no gold/yellow** (ice rule). (Engine: `render_in_ice`.)
- **LED ribbon zones that border blue lines** keep the spec's edge clearance (Lenovo = 140px).
- **Generic "no date/location" headers** render clean — but always also offer Mike a dated +
  Buy-Tickets variant alongside the clean one.
- **Don't fabricate** non-tour slots (e.g. a Ticketmaster season-logo zone) or dimensionless
  spec lines — skip them and tell Mike what/why.
