# Roadmap: Alfred Platform Stabilization & Ad Management

## Milestones

- ✅ **v1.0 Ops Ready** — Phases 1-5 (shipped 2026-02-21)
- ✅ **v1.1 Infrastructure Resilience** — Phases 6-9 (shipped 2026-02-26)
- 🔄 **v1.2 Forge Intelligent Clipping** — Phases 10-13 (in progress)

## Phases

<details>
<summary>✅ v1.0 Ops Ready (Phases 1-5) — SHIPPED 2026-02-21</summary>

- [x] Phase 1: Infrastructure Repairs (2/2 plans) — completed 2026-02-20
- [x] Phase 2: Alfred Claw Config Fixes (5/5 plans) — completed 2026-02-20
- [x] Phase 3: CRM Reliability (1/1 plan) — completed 2026-02-21
- [x] Phase 4: Google Ads Budget Control (2/2 plans) — completed 2026-02-21
- [x] Phase 5: Ad Workflow Validation & Hardening (3/3 plans) — completed 2026-02-21

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v1.1 Infrastructure Resilience (Phases 6-9) — SHIPPED 2026-02-26</summary>

- [x] Phase 6: SSH Access & Server Audit (2/2 plans) — completed 2026-02-26
- [x] Phase 7: Backup System (3/3 plans) — completed 2026-02-26
- [x] Phase 8: Recovery & Alerting (3/3 plans) — completed 2026-02-26
- [x] Phase 9: Ad Intelligence (2/2 plans) — completed 2026-02-26

Full details: `.planning/milestones/v1.1-ROADMAP.md`

</details>

### v1.2 Forge Intelligent Clipping

- [x] **Phase 10: Long-form Ingest & Transcription** — Upload or URL-ingest a source up to 12 GB and get a timestamped, speaker-attributed transcript (completed 2026-06-01)
- [ ] **Phase 11: Topic-targeted Segment Retrieval** — Query the transcript by topic and get ranked, previewable segment matches the operator can curate
- [ ] **Phase 12: Variant Montage Assembly** — Cut matched segments into multiple distinct branded 9:16 verticals and deliver them to the library and Postiz
- [ ] **Phase 13: Operator Self-serve Intelligent Clip Flow** — Wire the full pipeline into one Forge UI format Jordan can run solo end-to-end

## Phase Details

### Phase 10: Long-form Ingest & Transcription
**Goal:** The operator can get a complete, restart-safe, speaker-attributed transcript out of any long-form source — file or URL — without Alfred's involvement
**Depends on:** Nothing (builds on existing RuckTalk Whisper + bge-m3 pipeline)
**Requirements:** INGEST-01, INGEST-02, INGEST-03, INGEST-04, INGEST-05
**Success Criteria** (what must be TRUE):
  1. Operator can upload a source file up to ~12 GB via the Forge UI and see it enter the processing queue
  2. Operator can supply a YouTube (or other supported) URL in place of a file upload and processing begins from that source
  3. Operator can restart the Forge service mid-transcription and the job resumes where it left off rather than starting over
  4. The finished transcript shows each segment labelled with a speaker identifier so the operator can distinguish who is speaking
  5. Operator can pick a source already dropped in the shared Nextcloud Sources/ folder and ingest it without uploading (the reliable big-file path)
**Plans:** 3/3 plans complete
- [ ] 10-01-PLAN.md — Storage + ingest foundation (sources/transcript_segments tables, ingest.py, extract_audio)
- [ ] 10-02-PLAN.md — Streaming 12GB upload + URL ingest + shared-cloud-folder pick (INGEST-01, INGEST-02, INGEST-05)
- [ ] 10-03-PLAN.md — Checkpointed faster-whisper transcription + speaker attribution (INGEST-03, INGEST-04)

### Phase 11: Topic-targeted Segment Retrieval
**Goal:** The operator can describe a topic in plain language and see exactly which segments of the source cover it, ranked and previewable — before building anything
**Depends on:** Phase 10 (transcript + embeddings must exist)
**Requirements:** TOPIC-01, TOPIC-02, TOPIC-03
**Success Criteria** (what must be TRUE):
  1. Operator types a topic or theme into the Forge UI and receives a ranked list of matching source segments with in/out timestamps and relevance scores
  2. Operator can read the transcript text of each matched segment in the UI without leaving the page to verify it is genuinely on-topic
  3. Operator can uncheck or trim matched segments so only the ones they want go forward to assembly
**Plans:** 3 plans
- [ ] 11-01-PLAN.md — Retrieval core: windowing + bge-m3 embedding + source-scoped query, inline embed at transcription (TOPIC-01)
- [ ] 11-02-PLAN.md — Search + list-sources API endpoints with inline text preview + lazy backfill (TOPIC-01, TOPIC-02)
- [ ] 11-03-PLAN.md — Topic tab UI: pick source, query, ranked previewable curatable results + real-transcript precision check (TOPIC-01/02/03)

### Phase 12: Variant Montage Assembly
**Goal:** The operator can turn their curated segment list into multiple distinct, on-brand vertical clips ready to hand to the social team
**Depends on:** Phase 11 (curated segment list with timestamps)
**Requirements:** CLIP-01, CLIP-02, CLIP-03, CLIP-04
**Success Criteria** (what must be TRUE):
  1. Each assembled clip plays back with audio and video correctly synced to the original source
  2. Forge produces multiple variants from the same segment set — differing in segment order, cut points, or caption timing — so no two are identical
  3. Every variant is formatted as a 9:16 vertical with the Mainstay logo and styled captions applied
  4. Finished variants appear in the Forge library and the operator can push them to Postiz as drafts in one action
**Plans:** TBD

### Phase 13: Operator Self-serve Intelligent Clip Flow
**Goal:** Jordan can open Forge, run the full pipeline from source to delivered variants, and never need to ping Alfred for help
**Depends on:** Phase 12 (all pipeline stages must be functional)
**Requirements:** FLOW-01, FLOW-02, FLOW-03
**Success Criteria** (what must be TRUE):
  1. A single "Intelligent Clip" format in the Forge UI walks the operator through source → topic → variant count → Forge it in one continuous flow
  2. The running job is visible in the render queue with progress indicators, and the operator receives a notification when the job completes
  3. The operator can complete the full flow from a cold start with no guidance from Alfred, achieving a usable Postiz-ready clip set
**Plans:** TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Infrastructure Repairs | v1.0 | 2/2 | Complete | 2026-02-20 |
| 2. Alfred Claw Config Fixes | v1.0 | 5/5 | Complete | 2026-02-20 |
| 3. CRM Reliability | v1.0 | 1/1 | Complete | 2026-02-21 |
| 4. Google Ads Budget Control | v1.0 | 2/2 | Complete | 2026-02-21 |
| 5. Ad Workflow Validation & Hardening | v1.0 | 3/3 | Complete | 2026-02-21 |
| 6. SSH Access & Server Audit | v1.1 | 2/2 | Complete | 2026-02-26 |
| 7. Backup System | v1.1 | 3/3 | Complete | 2026-02-26 |
| 8. Recovery & Alerting | v1.1 | 3/3 | Complete | 2026-02-26 |
| 9. Ad Intelligence | v1.1 | 2/2 | Complete | 2026-02-26 |
| 10. Long-form Ingest & Transcription | 1/3 | Complete    | 2026-06-01 | - |
| 11. Topic-targeted Segment Retrieval | v1.2 | 0/? | Not started | - |
| 12. Variant Montage Assembly | v1.2 | 0/? | Not started | - |
| 13. Operator Self-serve Intelligent Clip Flow | v1.2 | 0/? | Not started | - |
