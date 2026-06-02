---
phase: 12-variant-montage-assembly
plan: 02
subsystem: forge
tags: [ffmpeg, topic-clip, variant-strategy, caption-overlay, branding, comfyui]

# Dependency graph
requires:
  - phase: 12-01
    provides: "_cut_segment, _concat_segments, enforce_duration, _detect_has_video — the cut+concat engine topic_clip.py extends"
  - phase: film_montage
    provides: "make_branded, LOGO_PATH — reused directly, not redeclared"
provides:
  - "_safe_drawtext: apostrophe/colon/percent/backslash sanitiser for ffmpeg drawtext"
  - "overlay_captions: bottom-third caption burn on 9:16 video"
  - "_build_variant_assemblies: 5 structural variant strategies (original/reverse/hook-first/trimmed/shuffle-N)"
  - "_synthesise_visual: audio-only background via ComfyUI Cloud lazy import, black fallback"
  - "assemble_variant: full cut->concat->visual->caption->brand pipeline for one variant"
  - "render(params, out_path): top-level entry matching film_montage.render() signature"
affects: [12-03, forge-handler, multiply]

# Tech tracking
tech-stack:
  added: [textwrap, copy, random (stdlib — no new deps)]
  patterns:
    - "Lazy ComfyUI import inside _synthesise_visual — module importable in test env without rucktalk_common"
    - "PIL-banned: all image generation via run_comfyui_cloud, never inline PIL"
    - "Audio re-encode everywhere (no -c copy) — honours project decision"
    - "Structural variants use deep copy — no shared mutation between variant segment lists"

key-files:
  created: []
  modified:
    - core/forge/renderers/topic_clip.py
    - tests/forge/test_topic_clip.py

key-decisions:
  - "overlay_captions uses AAC re-encode (-c:a aac) not -c:a copy to honour no-codec-copy project rule"
  - "FONT_PATH = DejaVuSans-Bold.ttf (confirmed on 105); Hanken Grotesk not installed"
  - "Captions: single first-sentence overlay (split on '. ', cap 120 chars); per-segment timed captions deferred to Phase 13"
  - "_safe_drawtext order: backslash first, then apostrophe->U+2019, then colon, then percent"
  - "Audio-only path: ComfyUI Cloud still (lazy import), black color fallback if ComfyUI returns None"

patterns-established:
  - "render() matches film_montage.render(params, out_path) signature — uniform caller interface from plan-03 handler"
  - "_build_variant_assemblies: enforce_duration on base before deriving variants; each variant is deep copy"

requirements-completed: [CLIP-01, CLIP-02, CLIP-03]

# Metrics
duration: 3min
completed: 2026-06-02
---

# Phase 12 Plan 02: Variant Montage Assembly Summary

**Structural variant strategies (5 types), safe ffmpeg caption overlay, audio-only ComfyUI visual synthesis, and top-level render() entry point layered on top of plan-01 cut/concat engine**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-02T03:17:07Z
- **Completed:** 2026-06-02T03:20:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `_safe_drawtext`: pure-string ffmpeg drawtext sanitiser — apostrophe->U+2019, colon/percent/backslash escaped, word-wraps to 38 chars/line
- `overlay_captions`: bottom-third caption burn on already-9:16 video using DejaVuSans-Bold; passthrough on empty text
- `_build_variant_assemblies`: 5 structural strategies (original, reverse, hook-first, trimmed, shuffle-N with deterministic seed); enforce_duration applied to base first; all independent deep copies
- `_synthesise_visual`: audio-only background via lazy-imported `run_comfyui_cloud`; solid black fallback; never PIL (project rule)
- `assemble_variant`: full single-variant pipeline — cut, concat, synthesise (if audio-only), caption, extract hook audio, brand with logo
- `render(params, out_path)`: matches `film_montage.render()` signature; resolves source_id via ingest, detects has_video, auto-extracts caption from first segment if not provided
- 22 tests passing (9 new: 5 safe_drawtext + 4 variant strategy); 122/122 forge suite green

## Task Commits

1. **Task 1: Safe caption overlay + 9:16 branding** - `86b486d` (feat)
2. **Task 2: Structural variant strategies + render() + tests** - `95fc67e` (test + fix)

## Files Created/Modified

- `/home/aialfred/alfred/core/forge/renderers/topic_clip.py` — Extended with caption/branding/variant/render logic (193 → 625 lines)
- `/home/aialfred/alfred/tests/forge/test_topic_clip.py` — Added 9 new tests for sanitiser + variant strategies (240 → 356 lines)

## Decisions Made

- **overlay_captions uses AAC re-encode** — Plan specified `-c:a copy` but the pre-existing `test_cut_segment_no_copy_codec` AST scan flags any "copy" string in list literals; switched to `-c:a aac -b:a 192k` (consistent with project re-encode rule anyway)
- **FONT_PATH = DejaVuSans-Bold** — Hanken Grotesk confirmed absent on 105 at plan time; DejaVuSans-Bold confirmed present
- **Per-segment timed captions deferred to Phase 13** — documented in comment; plan-02 ships single first-sentence overlay

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] overlay_captions used `-c:a copy` which tripped the AST codec-copy guard**
- **Found during:** Task 2 (running test_cut_segment_no_copy_codec)
- **Issue:** Plan spec said "audio can be copied here" for caption overlay, but `test_cut_segment_no_copy_codec` scans ALL list literals for the bare string "copy" and fails if found
- **Fix:** Changed `-c:a copy` to `-c:a aac -b:a 192k` in overlay_captions — consistent with project's re-encode decision anyway
- **Files modified:** core/forge/renderers/topic_clip.py
- **Verification:** `test_cut_segment_no_copy_codec` passes; all 22 topic_clip tests pass
- **Committed in:** 95fc67e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Single-line fix, no scope change. Re-encode on caption pass is consistent with project's no-copy policy.

## Issues Encountered

None beyond the one auto-fixed deviation above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan-03 (handler) can call `render(params, out_path)` once per structural variant
- `_build_variant_assemblies` produces the Tier 1 distinct masters; plan-03 feeds each to `multiply.py` for Tier 2 pixel-level variants
- `assemble_variant` and `render()` are fully wired but require a real forge source_id + segments to exercise the ffmpeg paths (end-to-end tested at plan-03 checkpoint)

---
*Phase: 12-variant-montage-assembly*
*Completed: 2026-06-02*
