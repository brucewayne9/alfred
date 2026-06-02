---
phase: 12-variant-montage-assembly
plan: 01
subsystem: forge
tags: [ffmpeg, libx264, aac, topic-clip, segment-cut, concat-demuxer, sync-safety]

# Dependency graph
requires:
  - phase: 11-topic-targeted-segment-retrieval
    provides: segment dicts with start_s/end_s/source_id from topic search
provides:
  - _detect_has_video: ffprobe-based audio/video source detection
  - _cut_segment: sample-accurate re-encode cut (never -c copy)
  - _concat_segments: ffmpeg concat demuxer with re-encode output
  - enforce_duration: 10-60s duration band guard before any cutting
affects:
  - 12-02 (topic_clip variants build on this engine)
  - 12-03 (handler wires enforce_duration -> _cut_segment -> _concat_segments)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - ffprobe v:0 stream detection for audio/video routing (mirrors multiply._has_audio)
    - sample-accurate re-encode on every cut (libx264 veryfast crf23 / aac 192k 44100 2ch)
    - 9:16 scale/crop filter identical to film_montage (format=yuv420p crop=1080:1920)
    - concat demuxer (concat.txt + -f concat -safe 0) with re-encode output
    - duration guard runs before cutting, never after (enforce_duration first)

key-files:
  created:
    - core/forge/renderers/topic_clip.py
    - tests/forge/test_topic_clip.py
  modified: []

key-decisions:
  - "Re-encode on every cut (never -c copy) — keyframe seek bleeds 1-2s of prior segment without re-encode"
  - "Duration guard enforce_duration must run BEFORE cutting, not after concat"
  - "Audio-only path outputs .m4a; video path outputs .mp4 — routing via _detect_has_video"
  - "Concat uses concat demuxer (not filter_complex concat) — safe because all inputs share identical params"

patterns-established:
  - "enforce_duration -> _cut_segment -> _concat_segments pipeline order"
  - "Never mutate input segment list — enforce_duration always returns a fresh copy"

requirements-completed: [CLIP-01]

# Metrics
duration: 2min
completed: 2026-06-02
---

# Phase 12 Plan 01: Variant Montage Assembly Summary

**ffmpeg segment-cut + concat engine for topic clips: sample-accurate re-encode (no -c copy), 9:16 video or audio-only routing, 10-60s duration guard before cutting, 13 tests all passing**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-02T16:31:55Z
- **Completed:** 2026-06-02T16:34:18Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `core/forge/renderers/topic_clip.py` — full cut+concat engine (192 lines): `_detect_has_video`, `_cut_segment`, `_concat_segments`, `enforce_duration`
- `tests/forge/test_topic_clip.py` — 13 tests covering video/audio detection, segment duration accuracy (±0.3s), concat duration sum (±0.4s), codec-copy AST guard, and all enforce_duration edge cases
- 104 existing forge tests unaffected (pre-existing test_search.py chromadb import error unchanged)

## Task Commits

Each task was committed atomically:

1. **Task 1: Source detection + single-segment cut + duration guard** - `d9a5570` (feat)
2. **Task 2: Non-contiguous concat + engine tests** - `3d0b97a` (feat)

## Files Created/Modified
- `core/forge/renderers/topic_clip.py` — segment-cut/concat engine for CLIP-01
- `tests/forge/test_topic_clip.py` — 13-test suite covering all engine functions

## Decisions Made
- **No codec copy anywhere** — `-c copy` causes keyframe-seek bleed (1-2s of prior segment leaks in); every cut re-encodes to libx264/aac so trim is sample-accurate.
- **Routing on has_video** — `_detect_has_video` runs once per source; video path produces 1080x1920 mp4, audio-only produces m4a; same ffprobe pattern as `multiply._has_audio`.
- **Duration guard first** — `enforce_duration` trims the last segment's end_s before any file I/O, not after. This is the correct order per plan pitfall 6.
- **Concat demuxer over filter_complex** — because every cut is re-encoded to identical params (yuv420p/30fps/44100/2ch), the concat demuxer is safe and avoids the A/V drift that filter_complex sometimes introduces on mismatched inputs.
- **Test for no-copy via AST inspection** — string match against module docstring had a false positive (comment said "never -c copy"); switched to AST walk over list literals, which correctly checks only actual ffmpeg command arguments.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test for codec-copy guard used string search matching docstring comment**
- **Found during:** Task 2 (test run)
- **Issue:** `assert "-c copy" not in src` matched the module docstring string "never -c copy", causing a false positive failure
- **Fix:** Replaced string search with AST walk over `ast.List` string literals — only checks actual ffmpeg command list arguments, not comments
- **Files modified:** tests/forge/test_topic_clip.py
- **Verification:** `test_cut_segment_no_copy_codec` passes, correctly rejects any `"copy"` string in list literals
- **Committed in:** 3d0b97a (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test assertion)
**Impact on plan:** Test assertion tightened; stronger check (AST) than original (string grep). No scope creep.

## Issues Encountered
- `tests/forge/test_search.py` collection error (chromadb import) is pre-existing since Phase 11; confirmed by stash test. Not caused by this plan.

## Next Phase Readiness
- Engine is ready: `_detect_has_video`, `_cut_segment`, `_concat_segments`, `enforce_duration` all importable and tested
- Plan 02 can immediately build structural variants (full/highlights/soundbite) on top of this engine
- Plan 03 handler wires the Phase 11 copy-selection JSON hand-off to enforce_duration -> cut -> concat

---
*Phase: 12-variant-montage-assembly*
*Completed: 2026-06-02*
