---
phase: 11-topic-targeted-segment-retrieval
plan: 03
subsystem: ui
tags: [forge, topic-retrieval, chromadb, bge-m3, vanilla-js, semantic-search]

# Dependency graph
requires:
  - phase: 11-02
    provides: GET /forge/sources and GET /forge/sources/{id}/search endpoints with relevance scores and inline transcript text
  - phase: 11-01
    provides: windowing + bge-m3 embeddings + source-scoped ChromaDB query
provides:
  - Topic tab in Forge UI (services/forge-web/index.html)
  - Source picker populated from /forge/sources?status=done
  - Ranked result cards: in/out timestamps, percentage relevance score, inline transcript text
  - Deselect + trim (In/Out inputs) per segment with live selection summary
  - Copy selection — JSON hand-off list to Phase 12 (start_s/end_s/text/speaker/score)
  - Precision validated on real RuckTalk interview: on-topic returns genuine matches, off-topic returns 0
  - Default threshold tuned 0.70 -> 0.72 (Mike-approved, precision checkpoint)
affects:
  - 12-variant-montage-assembly (consumes the Copy-selection JSON hand-off from this tab)
  - 13-operator-selfserve-flow (Topic tab is the retrieval step in the operator flow)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Topic tab follows same data-panel/data-tab scaffold as existing Forge tabs
    - topicState object (results + selected keyed by `${start_s}-${end_s}`) holds curate state client-side
    - Percentage score = Math.round(score*100)+'%' — never raw distance
    - navigator.clipboard for Phase-12 hand-off JSON
    - Lazy source load on first tab switch via tabTo hook

key-files:
  created: []
  modified:
    - services/forge-web/index.html

key-decisions:
  - "Default topic threshold set to 0.72 (tuned from 0.45 baseline -> 0.70 in fix commit -> 0.72 after precision checkpoint on real RuckTalk interview — Mike-approved)"
  - "Precision validated on real source: source_id 7a0b98dcbd2e4ae9981661540a1dd335 (episode_5.mp3, status=done) — 'sleep and recovery' returns 3 genuinely on-topic segments at 79%; 'cooking recipes food preparation' returns 0"
  - "Copy selection emits JSON array [{start_s, end_s, text, speaker, score}] as the Phase-12 hand-off; no clip cutting in this phase"
  - "Scores shown as percentage (79%) never as raw cosine distance"

patterns-established:
  - "Threshold tuning: calibrate against real transcripts before locking default — on-topic floor scraper borderline case drove 0.70->0.72"
  - "Precision checkpoint pattern: on-topic probe returns genuine matches, off-topic probe returns 0 — both required before approval"

requirements-completed: [TOPIC-01, TOPIC-02, TOPIC-03]

# Metrics
duration: ~45min
completed: 2026-06-02
---

# Phase 11 Plan 03: Topic-Targeted Segment Retrieval Summary

**Topic tab UI in Forge: source picker, semantic topic search with ranked previewable result cards, deselect/trim curation, and Phase-12 JSON hand-off — precision validated on real RuckTalk interview (sleep/recovery = 3 genuine hits at 79%; cooking query = 0 results)**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-06-02
- **Completed:** 2026-06-02
- **Tasks:** 3 (Tasks 1+2 code, Task 3 precision checkpoint — human-approved)
- **Files modified:** 1 (services/forge-web/index.html)

## Accomplishments

- Topic tab added to Forge UI: source picker (from /forge/sources?status=done), topic query box, advanced filters (speaker/top_k/threshold), ranked result cards
- Each result card shows in/out timestamps, percentage relevance score (never raw distance), inline transcript text, deselect checkbox, and trim In/Out inputs — completing TOPIC-01, TOPIC-02, TOPIC-03
- Copy selection emits the Phase-12 hand-off JSON (kept+trimmed [{start_s, end_s, text, speaker, score}]) to clipboard
- Precision checkpoint passed on real RuckTalk interview: "sleep and recovery" query returns 3 genuinely on-topic windows at 79% relevance; "cooking recipes food preparation" returns 0 results — the make-or-break retrieval risk is retired
- Default threshold tuned 0.70 -> 0.72 (Mike-approved) to cleanly cut a borderline off-topic floor-scraper result
- Wave-2 test isolation leak fixed (commit 0b417f4); forge test suite 106/106 passing

## Task Commits

1. **Tasks 1+2: Topic tab scaffold + search wiring** — `2d2452a` (feat)
2. **Dev fix: threshold baseline 0.45->0.70** — `425fc0d` (fix)
3. **Precision checkpoint: threshold 0.70->0.72** — `705ecd9` (feat, Mike-approved)

Prerequisite fix (Wave-2 test isolation): `0b417f4` (fix, part of 11-02 cleanup carried into this plan)

## Files Created/Modified

- `services/forge-web/index.html` — Topic tab added: nav item (sidebar + mobile), LABELS map entry, panel with source picker/query/advanced row/results container/selection summary/copy button, and all JS logic (loadTopicSources, runTopicSearch, renderTopicResults, updateTopicSelectionSummary, fmtTime)

## Decisions Made

- **Default threshold 0.72** — Calibrated against real RuckTalk interview (episode_5.mp3, source_id 7a0b98dcbd2e4ae9981661540a1dd335). Baseline 0.45 was too low; 0.70 was correct on on-topic probe but a borderline floor-scraper slipped through; 0.72 cut it cleanly. Mike explicitly approved.
- **Copy selection as Phase-12 hand-off** — No clip cutting in Phase 11. Operator curates and copies JSON; Phase 12 will consume it for ffmpeg assembly.
- **Percentage display** — All scores rendered as Math.round(score*100)+'%'. Raw cosine distance never surfaced to the operator.
- **Client-side state only** — topicState held in browser memory (no server persist); acceptable for the operator session model.

## Precision Checkpoint Evidence

Real source: `source_id=7a0b98dcbd2e4ae9981661540a1dd335` (episode_5.mp3, status=done, RuckTalk interview)

| Query | Results | Top result score | Verdict |
|-------|---------|-----------------|---------|
| "sleep and recovery" | 3 | 79% — transcript: nap/sleep/recovery/rebuild language | PASS — genuinely on-topic |
| "cooking recipes food preparation" | 0 | — | PASS — off-topic correctly rejected |

Threshold tuning trace: 0.45 (original plan value) -> 0.70 (fix, pre-checkpoint) -> 0.72 (precision checkpoint, Mike-approved, floor-scraper cut).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Threshold default too permissive (0.45)**
- **Found during:** Task 2 and precision checkpoint
- **Issue:** Plan specified threshold default 0.45 which was too low — returned off-topic segments on real transcript
- **Fix:** Raised to 0.70 (commit 425fc0d), then further to 0.72 after identifying a borderline floor-scraper (commit 705ecd9, Mike-approved)
- **Files modified:** services/forge-web/index.html
- **Verification:** On real source, off-topic query returns 0; on-topic returns 3 genuine hits
- **Committed in:** 425fc0d, 705ecd9

**2. [Rule 1 - Bug] Wave-2 test isolation leak**
- **Found during:** Pre-checkpoint suite run
- **Issue:** test_search.py stub injected on sys.modules but not on the package attribute, causing intermittent test isolation failures in Wave-2
- **Fix:** Added package-level stub injection alongside sys.modules stub (commit 0b417f4)
- **Files modified:** tests (core/forge/)
- **Verification:** 106/106 forge tests passing
- **Committed in:** 0b417f4

---

**Total deviations:** 2 auto-fixed (2 bug fixes)
**Impact on plan:** Threshold tuning directly enabled precision checkpoint to pass. Test fix unblocked clean CI. No scope creep.

## Issues Encountered

- Precision checkpoint required two threshold nudges before landing at 0.72 — both were instrumented transparently and each commit is atomic with clear rationale. No logic changes to the retrieval core (search.py unchanged).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 11 complete: all three requirements (TOPIC-01, TOPIC-02, TOPIC-03) delivered and precision-validated on real long-form speech
- Copy selection hand-off JSON format: `[{start_s, end_s, text, speaker, score}]` — Phase 12 ffmpeg assembly can consume directly
- Default threshold 0.72 locked; operator can override via the advanced row in the UI
- Phase 12 (Variant Montage Assembly) can begin — it has both the API surface (11-02) and the curated segment hand-off format (this plan)

---
*Phase: 11-topic-targeted-segment-retrieval*
*Completed: 2026-06-02*
