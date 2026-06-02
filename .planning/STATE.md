# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** v1.2 Forge Intelligent Clipping — Phase 12 executing

## Current Position

Phase: 12 — Variant Montage Assembly
Plan: 12-02 (2 of 3 complete)
Status: In Progress
Progress: [███████░░░] 65% (v1.2 — 2/4 phases done + Phase 12 at 2/3 plans; Phase 10 done; Phase 11 done)

Last activity: 2026-06-02 — 12-02 complete (_safe_drawtext, overlay_captions, _build_variant_assemblies, _synthesise_visual, assemble_variant, render(); 22 tests passing; 122/122 forge suite green)

## Performance Metrics

**v1.0 Summary:**
- 5 phases, 13 plans, 40 commits
- 45 files changed (+7,136 / -94)
- 24 days (2026-01-28 → 2026-02-21)

**v1.1 Summary:**
- 4 phases, 10 plans, 41 commits
- 17 files changed (+4,709 / -28)
- 1 day (2026-02-26)

**v1.2 Running:**
- 4 phases defined, Phase 10 complete (3/3 plans), Phase 11 in progress
- Phase 11 Plan 01: 2 tasks, 3 files, ~15 min
- Target: 14 requirements across Phases 10-13

## Accumulated Context

### Decisions

- **Reuse RuckTalk pipeline** — Whisper + bge-m3 embeddings already proven; wire to Forge ingest, don't rebuild
- **Phase 11 is the make-or-break risk** — topic-retrieval precision determines whether the whole concept is credible; validate before wiring Phase 12
- **Human-in-the-loop distribution only** — no auto-post to any platform; operator pushes to Postiz as drafts
- **YouTube is source only** — no posting destination; Meta (IG+FB) + TikTok are the targets
- **No calendar deadlines** — phases gate on capability/metrics per operating agreement
- **Single shared forge_segments ChromaDB collection** — sources scoped via source_id metadata, not one collection per source
- **Duration-only windowing** — no speaker-boundary splits; short interjections flip speaker not topic; duration guard absorbs them

- **sys.modules stub injection** — API tests stub core.forge.search via sys.modules autouse fixture to avoid chromadb dep in test env; Wave-2 test isolation leak fixed in 11-03 (0b417f4); 106/106 passing
- **Topic default threshold 0.72** — Calibrated against real RuckTalk interview (episode_5.mp3); 0.45 plan default too low; 0.70 let floor-scraper through; 0.72 cuts it cleanly — Mike-approved at precision checkpoint
- **Precision checkpoint retired** — On real RuckTalk source, "sleep and recovery" returns 3 genuine hits at 79%; "cooking recipes food preparation" returns 0 — Phase 11 make-or-break risk retired
- **Re-encode on every topic_clip cut (no -c copy)** — keyframe seek bleeds 1-2s of prior segment without re-encode; libx264 veryfast crf23 / aac 192k 44100 2ch is the safe baseline
- **enforce_duration runs before cutting** — duration guard trims last segment end_s before any ffmpeg I/O (pitfall: guard-after-concat is too late)
- **Concat demuxer over filter_complex** — all inputs share identical params post-cut; demuxer is safe and avoids A/V drift
- **overlay_captions re-encodes audio (no -c:a copy)** — consistent with project no-copy rule; AST guard in test_cut_segment_no_copy_codec scans all list literals
- **FONT_PATH = DejaVuSans-Bold.ttf** — Hanken Grotesk not installed on 105; DejaVuSans-Bold confirmed present
- **Single first-sentence caption overlay (Phase 12)** — per-segment timed word-level captions deferred to Phase 13
- **render() matches film_montage.render() signature** — both renderers callable uniformly from plan-03 handler
- **_build_variant_assemblies: deep copy per variant** — independent mutation safety; enforce_duration on base before deriving variants

### Pending Todos

- None

### Blockers/Concerns

- None

## Session Continuity

Last session: 2026-06-02
Stopped at: Phase 12 Plan 02 complete — caption/branding/variant/render pipeline implemented and tested (22/22, 122/122 forge suite)
Resume at: Phase 12 Plan 03 — handler wiring (calls render() per structural variant, feeds each to multiply.py for pixel-level Tier 2)
