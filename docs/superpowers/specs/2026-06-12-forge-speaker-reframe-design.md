# Forge — Speaker-Aware Reframe (Design Spec)

**Date:** 2026-06-12
**Status:** Approved (Mike: autonomous build)
**Branch:** `feat/forge-studio`

---

## Goal

When cutting a 9:16 vertical clip from interview/podcast footage, **follow the active speaker** —
crop the frame to whoever is talking, and cut to the other person when they take over. Replace
today's dumb static center-crop. Like Opus Clip / Vizard / CapCut auto-reframe.

## Current state

`topic_clip.py::_cut_segment` (61–119) and `film_montage.py::_cut_segment` (182–204) both apply a
static `scale=W:H:force_original_aspect_ratio=increase,crop=W:H` — center cover-crop, speaker-blind.

## Architecture (synthesized from research)

Three layers, each independently testable, with **graceful fallback to static center-crop** at every
failure point (single face, no faces, ASD error, missing deps → today's behavior, zero regression).

### 1. ASD provider — `core/forge/renderers/asd_provider.py`
- Vendors **LR-ASD** (https://github.com/Junhua-Liao/LR-ASD, MIT, 3.4MB weights) under `vendor/lr-asd/`.
- `detect_active_speakers(video_path) -> list[{start_s, end_s, bbox:(x,y,w,h), track_id}]`
- Subprocesses LR-ASD's `Columbia_test.py` (S3FD face detect → IoU track → ASD scoring), reads
  `tracks.pckl` + `scores.pckl`, computes per-frame highest-scoring track = active speaker bbox.
- **25fps space** (LR-ASD forces `-r 25`); `timestamp = frame/25`. Cache results per source.
- Fallback: any exception / no GPU / no faces → returns `[]` (caller center-crops).
- TalkNet-ASD is a drop-in model swap if LR-ASD struggles on crosstalk-heavy audio.

### 2. Reframe core — `core/forge/renderers/reframe.py` (PURE LOGIC + ffmpeg, the TDD heart)
Pure, unit-tested functions:
- `crop_window(bbox, W, H, target_w, target_h, headroom=0.42, zoom=0.33) -> (x,y,w,h)` — 9:16 window
  centered on the speaker with head-room, clamped to bounds, **even dims**.
- `build_segments(active_bboxes, min_dwell=1.2, switch_on=0.65, switch_off=0.40) -> [Segment]` —
  merge per-frame active-speaker into speaker-stable segments; hysteresis + min-dwell suppress
  ping-pong on crosstalk/backchannels (the #1 quality lever).
- `smooth_path(centers, alpha=0.12, dead_zone=0.10, max_pan_frac=0.02) -> centers` — EMA + dead-zone
  + velocity clamp so a moving subject stays framed without jitter.
- `reframe_segment(src, start_s, end_s, out, W, H, active_bboxes|None) -> Path` — orchestrates: if no
  bboxes → static center-crop (calls/mirrors existing `_cut_segment`); else segment-crop+concat with
  smoothed pan, then audio mux. Same codec params as `_cut_segment` (libx264/yuv420p/30fps).

### 3. Wiring — behind a flag, shared by both renderers
- `params["reframe"]`: `"speaker"` | `"center"` (default `"center"` until proven, then flip to
  `"auto"`). `topic_clip.assemble_variant` (~876) and `film_montage.render` (~299) call
  `reframe.reframe_segment(...)` when enabled, else the existing `_cut_segment`.
- API: accept `reframe` in job params. UI: a toggle in the Auto-Clips / montage panel.

## ffmpeg approach (from research)
Segment-by-segment: split timeline into speaker-stable segments; render each to 1080×1920 with
**identical encoder params** (libx264, yuv420p, fixed fps, `setsar=1`); animate crop `x,y` via
`sendcmd` for intra-segment motion (crop w/h fixed within a segment); concat via demuxer (`-c copy`);
mux original audio back. Speaker switches = segment boundaries = free hard cuts.

## Smoothing defaults (research starting points)
`alpha=0.12`, `dead_zone=0.10·w`, `min_dwell=1.2s`, `switch_on=0.65`, `switch_off=0.40`,
`headroom=0.42`, `target_face_frac=0.33`, `cmd_rate=10Hz`. Tune on real footage.

## Testing
- Unit (pure, no ffmpeg/ML): `crop_window` (centering, headroom, clamp, even dims), `build_segments`
  (merge, min-dwell suppression, hysteresis), `smooth_path` (EMA, dead-zone, velocity clamp).
- Integration: `reframe_segment` with `active_bboxes=None` → byte-for-byte matches static crop
  behavior (fallback regression guard). With bboxes on a real 2-person clip → speaker centered,
  cuts on switch (manual/visual verify on a RuckTalk or Rod Wave interview).
- ASD provider: runs on a short test video, produces non-empty windows.

## Out of scope (v1)
- 3+ person panel optimization (falls back to group-centroid or center).
- Realtime. This is batch/offline.
- Per-word lip-precise framing — segment-level speaker following is the v1 target.

## Build order
1. `reframe.py` pure core + tests (works today with center-crop fallback). 2. ASD provider
(LR-ASD env, the risky part — built/validated in parallel). 3. Wire flag into topic_clip +
film_montage + API/UI. 4. Verify end-to-end on a real interview clip.
