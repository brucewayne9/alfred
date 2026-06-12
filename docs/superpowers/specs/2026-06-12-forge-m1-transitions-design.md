# Forge M1 — Transitions (Design Spec)

**Date:** 2026-06-12
**Status:** Draft for review
**Parent arc:** [Forge Studio — Milestone Arc](2026-06-12-forge-studio-arc.md)
**Branch:** `feat/forge-studio`

---

## Goal

Give Forge montages CapCut / Instagram-Edits-style transitions, in two phases:

- **Phase 1 — Curated transition menu.** A tasteful, named picker with an "Auto" default,
  applied per montage. Buildable now.
- **Phase 2 — Beat-sync.** Snap cuts and transitions to the music's beat — the "Edits feel."
  Designed here, built after Phase 1 ships.

**North star:** Instagram Edits' *restraint + rhythm*, not a 50-effect drawer.

## Current state (grounded)

- `core/forge/renderers/film_montage.py` → `_concat_xfade(seg_paths, out, fade=0.18)` already
  chains ffmpeg `xfade` between segments, but **hardcodes `transition=fade`**. Segments are cut
  `fade` seconds long to absorb the overlap so captions stay synced. Single-segment falls back to
  copy.
- `core/forge/renderers/multi_montage.py` (operator-pick montage) uses `_concat_segments` —
  **hard cuts, no transition.**
- `render(params: dict, out_path)` is the renderer entry; `params` already carries config
  (`caption_style`, `aspect`, `sources`, …). `core/forge/handlers.py::_run_remix_format` passes
  `params` straight through from the web job.
- All segments are pre-normalized to 1080×1920 / 30fps / yuv420p — `xfade` needs matching
  inputs, so every transition type works without extra conditioning.

**Implication:** the menu is a *parameter*, not new video tech. ffmpeg `xfade` ships ~50 types,
all sharing the same `duration`/`offset` mechanic `_concat_xfade` already computes.

## Phase 1 — Curated menu

### Transition catalog
A single source-of-truth map (new `core/forge/transitions.py`):

| Picker label | `xfade` transition | Notes |
|---|---|---|
| **Auto** *(default)* | resolved at render (see below) | no-decision path |
| Cut | *(none — straight concat)* | hard cut |
| Dissolve | `fade` | today's default |
| Whip | `slideleft` | directional; alternate L/R per boundary |
| Wipe | `wipeleft` | directional |
| Zoom | `zoomin` | |
| Flash | `fadewhite` | |
| Blur | `smoothleft` | |
| Glitch | `pixelize` | |

Catalog entry = `{key, label, xfade, directional, default_duration}`. Unknown/missing key →
fall back to `Dissolve`. `Cut` routes to the hard-cut concat (skip xfade entirely).

### Code changes
1. **`core/forge/transitions.py`** (new) — the catalog + `resolve(key) -> spec` +
   `pick_auto(params) -> key`.
2. **`_concat_xfade`** — accept `transition: str = "fade"` and `duration` override; interpolate
   into the filter string instead of the literal `fade`. Directional types alternate
   left/right per boundary for visual variety within one style.
3. **`film_montage.render`** — read `params.get("transition")`, resolve via catalog; `Cut` →
   existing hard-cut path; else xfade with the resolved type.
4. **`multi_montage.render`** — **DEFERRED to its own sub-task (M1.2).** Discovered during build:
   `_concat_xfade` is video-only (`-an`), and `multi_montage`'s interview path extracts voice audio
   from the concatenated body while its captions are placed at *cumulative segment offsets*. Routing
   it through xfade naively would (a) drop the body's voice audio and (b) desync those caption
   offsets by the per-boundary overlap. Correct support needs an audio-preserving concat
   (`acrossfade` or post-mux) **and** recomputed caption offsets — built and tested separately so
   the existing operator-pick montage isn't regressed. `multi_montage` keeps hard cuts until then.
5. **`handlers.py`** — no logic change; `transition` rides in `params` already. Validate/whitelist
   the key against the catalog (reject unknown to a safe default).
6. **forge-web UI** — add a transition `<select>` to the montage create panel, populated from the
   catalog (single endpoint `GET /api/forge/transitions`), defaulting to **Auto**. One choice per
   montage.

### "Auto" logic (Phase 1)
Deterministic, conservative: default to **Dissolve**; if the montage is high-energy (music BPM
known or `params.energy == "high"`) pick **Whip**; audio-only/talking-head → **Dissolve**. Auto
never selects Flash/Glitch in Phase 1 (those stay deliberate opt-ins). Refined in Phase 2 once
beat data exists.

### Guardrail (enforced in code, not just docs)
One transition style per montage. No per-cut style mixing in the UI. Default surface = Auto / Cut /
Dissolve front-and-center; flashy types present but visually secondary.

## Phase 2 — Beat-sync (design only)

- **Beat detection:** `librosa.beat.beat_track` on the resolved music bed → tempo + beat-onset
  timestamps. Add `core/forge/beats.py`; reuse the bed already resolved in `multi_montage`
  (`_resolve_bed`) / `film_montage`.
- **Snap:** adjust each segment's cut length so boundaries land on the nearest beat (respect a
  min/max clip length so it doesn't produce strobing micro-cuts). Place the xfade `offset` on the
  beat.
- **UI:** one toggle — **"Auto-edit to the beat."** Off = Phase 1 behavior.
- **Energy-aware Auto:** with BPM known, Auto may vary transition by section energy (still within
  the restraint guardrail).
- Risk: beat-snapping fights the caption-sync timing `_concat_xfade` already manages — Phase 2
  must recompute caption offsets from the snapped boundaries. Called out as the primary Phase 2
  integration test.

## Testing

- **Unit:** `transitions.resolve` (known/unknown/Cut), directional alternation, `pick_auto`
  branches.
- **Render smoke (Phase 1):** 3-segment montage renders successfully for each catalog key;
  output duration ≈ requested (overlap absorbed); single-segment still copy-falls-back; `Cut`
  produces hard cuts.
- **Caption sync:** captions remain aligned after a non-default transition (regression guard on
  the existing sync behavior).
- **Phase 2:** beat timestamps detected on a known-BPM test bed; boundaries land within tolerance
  of beats; caption offsets recomputed correctly.

## Out of scope (M1)

- Batch clipping (M2), RuckTalk ingest (M3), schedule queue (M4), multi-creator profiles (M5).
- Per-cut individual transition selection (deliberately excluded — guardrail).
- Custom/animated transitions beyond ffmpeg `xfade` (e.g. GL/shader transitions) — revisit only
  if the curated set proves insufficient.

## Build order

Phase 1 (catalog → `_concat_xfade` param → film + multi wiring → handler whitelist → UI picker →
tests) ships and is verifiable on its own. Phase 2 (beats → snap → toggle → caption recompute)
follows as a second plan.
