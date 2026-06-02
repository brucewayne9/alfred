# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-01)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** v1.2 Forge Intelligent Clipping — Phase 11 executing

## Current Position

Phase: 11 — Topic-Targeted Segment Retrieval
Plan: 11-03 (3 of 3 complete)
Status: Phase Complete
Progress: [█████░░░░░] 50% (v1.2 — 2/4 phases; Phase 10 done; Phase 11 done)

Last activity: 2026-06-02 — 11-03 complete (Topic tab UI + precision checkpoint: threshold 0.72, real RuckTalk interview validated)

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

### Pending Todos

- None

### Blockers/Concerns

- Phase 11 retrieval precision is the highest technical risk — bge-m3 embeddings are good for semantic similarity but topic precision on long-form transcripts needs validation at plan time

## Session Continuity

Last session: 2026-06-02
Stopped at: Phase 11 Plan 03 complete — Topic tab UI + precision checkpoint approved (threshold 0.72, 106/106 tests)
Resume at: `/gsd:plan-phase 12` — Phase 12 Variant Montage Assembly (consumes Phase 11 Copy-selection JSON hand-off)
