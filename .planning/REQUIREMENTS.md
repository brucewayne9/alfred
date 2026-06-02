# Requirements: Alfred Platform — v1.2 Forge Intelligent Clipping

**Defined:** 2026-06-01
**Core Value:** Turn a long-form source into topic-targeted, multi-variant vertical clips the Mainstay social team can post — find the right moments automatically, then spin distinct on-brand uniques from them.

## v1 Requirements

Requirements for the v1.2 milestone. Each maps to exactly one roadmap phase.

### Ingest & Transcription

- [x] **INGEST-01**: Operator can upload a long-form source file up to ~12 GB into Forge for processing
- [x] **INGEST-02**: Operator can supply a source by URL instead of uploading a file
- [x] **INGEST-03**: Forge transcribes a long-form source into a timestamped transcript, processed asynchronously and surviving a service restart
- [x] **INGEST-04**: Forge attributes transcript segments to speakers so the operator can target one person's moments
- [x] **INGEST-05**: Operator can pick a long-form source already dropped in the shared Nextcloud folder (Sources/) and ingest it without uploading — the reliable path for very large files

### Topic Retrieval

- [x] **TOPIC-01**: Operator can enter a topic/theme and get back the source segments where it is discussed, ranked by relevance with in/out timestamps
- [x] **TOPIC-02**: Operator can preview the matched segments (timestamps + transcript text) before building anything
- [x] **TOPIC-03**: Operator can deselect or adjust matched segments before assembly

### Variant Assembly

- [x] **CLIP-01**: Forge cuts the matched segments out of the source into clips with audio and video in sync
- [x] **CLIP-02**: Forge generates multiple distinct variants from the matched segments (different segment order, cut points, and captions)
- [x] **CLIP-03**: Each variant is branded in the Mainstay style (logo + caption) as a 9:16 vertical
- [x] **CLIP-04**: Variants deliver to the library and can be handed to Postiz as drafts

### Operator Workflow

- [ ] **FLOW-01**: Operator runs the full pipeline as one "Intelligent Clip" format in the Forge UI (source → topic → variant count → Forge it)
- [ ] **FLOW-02**: Operator sees job progress in the render queue and is notified on completion
- [ ] **FLOW-03**: Operator can run the flow solo end-to-end without Alfred intervention

## v2 Requirements

Deferred — acknowledged but not in the v1.2 roadmap.

### Intelligence

- **INTEL-01**: Operator can connect official + burner accounts and read per-post views/likes/saves (gated on full account access; the label already has sound-tracking tools — see 2026-06-01 call)

### Generative

- **GEN-01**: Operator can generate AI vessel clips from scratch (Higgsfield engine swap) for montage b-roll (stubbed; blocked on Mainstay budget — not the priority per Dharmic)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Auto-posting to social platforms | Mike killed it on the call — API auto-post does more harm than good and platforms penalize it; human hits send |
| YouTube as a posting destination | Team has no YouTube accounts; YouTube is a clip *source* only |
| Calendar/date-based phase deadlines | Operating-agreement hard rule — gate on capability/metrics, not dates |
| Higgsfield AI engine swap (this milestone) | Not blocking per Dharmic; deferred to v2 GEN-01 pending budget |
| Intelligence scoreboard (this milestone) | Shelved per Dharmic; needs account access, comes together over time → v2 INTEL-01 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INGEST-01 | Phase 10 | Complete |
| INGEST-02 | Phase 10 | Complete |
| INGEST-03 | Phase 10 | Complete |
| INGEST-04 | Phase 10 | Complete |
| INGEST-05 | Phase 10 | Complete |
| TOPIC-01 | Phase 11 | Complete |
| TOPIC-02 | Phase 11 | Complete |
| TOPIC-03 | Phase 11 | Complete |
| CLIP-01 | Phase 12 | Complete |
| CLIP-02 | Phase 12 | Complete |
| CLIP-03 | Phase 12 | Complete |
| CLIP-04 | Phase 12 | Complete |
| FLOW-01 | Phase 13 | Pending |
| FLOW-02 | Phase 13 | Pending |
| FLOW-03 | Phase 13 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-01*
*Last updated: 2026-06-01 — traceability confirmed during roadmap creation*
