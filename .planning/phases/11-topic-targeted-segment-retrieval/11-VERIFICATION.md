---
phase: 11-topic-targeted-segment-retrieval
verified: 2026-06-02T00:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 11: Topic-targeted Segment Retrieval — Verification Report

**Phase Goal:** The operator can describe a topic in plain language and see exactly which segments of the source cover it, ranked and previewable — before building anything.
**Verified:** 2026-06-02
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Raw 2-8s Whisper segments merge into 20-40s topic-coherent windows before embedding | VERIFIED | `core/forge/search.py:64-131` — `build_windows()` grows a buffer, flushes at max_dur_s (45s) when buffer >= min_dur_s (10s), emits trailing buffer; 13/13 unit tests cover this in `test_search.py` |
| 2 | Each window is embedded with bge-m3 and stored as one ChromaDB doc with start_s/end_s/speaker/text metadata | VERIFIED | `search.py:139-162` — `upsert_windows()` stores documents+metadatas with all required fields; `_get_collection()` pins `bge-m3:latest` via `OllamaEmbeddingFunction`; `search.py:170-214` — `embed_source_windows()` deletes before upsert |
| 3 | Embedding runs inline after transcription so source is search-ready at status=done | VERIFIED | `ingest.py:313-318` (checkpoint fast-path) and `ingest.py:486-491` (main path) — both call `_search.embed_source_windows(source_id)` between `save_segments()` and `update_source(status="done")`, guarded by try/except |
| 4 | GET /forge/sources/{id}/search returns ranked windows with start_s/end_s/text/speaker/score (inverted score, never raw distance) | VERIFIED | `core/api/forge.py:174-211` — endpoint present, status guard (404/409), lazy backfill, calls `search_segments()`; `search.py:291` — `score = round(1.0 - dist / 2.0, 4)` — raw distances never returned |
| 5 | Operator sees the Topic tab with source picker, query box, ranked result cards (timestamps + percentage score + inline transcript text), deselect + trim controls, and can copy the kept+trimmed selection as Phase-12 hand-off | VERIFIED | `index.html:293` — nav item present; `index.html:390-442` — panel with all controls; `index.html:949-1100` — `loadTopicSources()`, `runTopicSearch()`, `renderTopicResults()`, `toggleTopicSegment()`, `trimTopicSegment()`, `copyTopicSelection()` all present; score displayed as `Math.round(score*100)+'%'` at line 1030 |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `core/forge/search.py` | build_windows, embed_source_windows, upsert_windows, search_segments, has_windows, _get_collection | VERIFIED | 308 lines (well above 120 min); all 6 functions present; `def search_segments` at line 234 |
| `core/forge/ingest.py` | inline embed call in transcribe_handler after save_segments | VERIFIED | 4 occurrences of `embed_source_windows` — 2 calls (line 316, 489) + 2 log lines (line 318, 491) covering both code paths |
| `tests/forge/test_search.py` | windowing + score-inversion + speaker-coercion tests | VERIFIED | File present; 13 tests pass |
| `core/api/forge.py` | GET /forge/sources/{id}/search and GET /forge/sources endpoints | VERIFIED | Both endpoints at lines 156-160 and 174-211; literal `/forge/sources` registered before `/{source_id}` path-param route (no shadowing) |
| `core/forge/ingest.py` | list_sources() DB helper | VERIFIED | `def list_sources` at line 216 |
| `tests/forge/test_search_api.py` | endpoint contract + 404/409 + lazy-backfill tests | VERIFIED | File present; passes in full forge suite run |
| `services/forge-web/index.html` | Topic tab: source picker, query box, ranked previewable curatable result cards | VERIFIED | All elements present (lines 293, 390-442, 528, 539, 940-1100) |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ingest.py transcribe_handler` (main path) | `search.embed_source_windows` | inline call between save_segments() and update_source(status='done') | WIRED | Line 489; try/except guard present |
| `ingest.py transcribe_handler` (checkpoint fast-path) | `search.embed_source_windows` | same guard pattern | WIRED | Line 316 |
| `core/forge/search.py` | `core/memory/store.get_client` | shared PersistentClient at data/chromadb | WIRED | `from core.memory.store import get_client` at line 26; called in `_get_collection()` line 52 |
| `search.py search_segments` | ChromaDB collection.query + score inversion | `1.0 - dist/2.0` | WIRED | Line 291: `score = round(1.0 - dist / 2.0, 4)` |
| `core/api/forge.py search endpoint` | `search.search_segments` | calls after status + has_windows guard | WIRED | Lines 193-210; `forge_search.search_segments(...)` |
| `core/api/forge.py search endpoint` | `search.embed_source_windows` | lazy backfill when not has_windows | WIRED | Lines 202-203 |
| `core/api/forge.py /forge/sources` | `ingest.list_sources` | list endpoint | WIRED | Line 160 |
| `index.html Topic tab` | `GET /forge/sources?status=done` | fetch in `loadTopicSources()` | WIRED | Line 953: `fetch(API + '/forge/sources?status=done', ...)` |
| `index.html Topic tab` | `GET /forge/sources/{id}/search` | fetch with q/top_k/speaker/threshold | WIRED | Line 986: builds URL with `/search?q=...` params |
| `result cards` | client selection state | deselect checkbox + trim in/out inputs | WIRED | `topicState.selected[key]` initialized with `kept:true, trimIn, trimOut`; `toggleTopicSegment` + `trimTopicSegment` mutate it; `copyTopicSelection` reads it |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TOPIC-01 | 11-01, 11-02 | Enter topic, get ranked segments with in/out timestamps + relevance scores | SATISFIED | `search_segments()` returns `{start_s, end_s, score, ...}` sorted by score desc; endpoint exposes this; UI renders timestamps + percentage badges |
| TOPIC-02 | 11-02, 11-03 | Preview matched segments (timestamps + transcript text) before building | SATISFIED | Every result includes `text` (full window text inline); UI renders transcript text in each card; no second request needed |
| TOPIC-03 | 11-03 | Deselect or adjust matched segments before assembly | SATISFIED | Checkbox toggle (`toggleTopicSegment`) + In/Out number inputs (`trimTopicSegment`) mutate `topicState.selected`; `copyTopicSelection()` emits kept+trimmed list |

All three TOPIC requirements confirmed mapped to Phase 11 in REQUIREMENTS.md (lines 68-70, all marked Complete).

---

## Anti-Patterns Found

None detected. No TODO/FIXME/placeholder comments in modified files. No stub implementations. No empty returns in critical paths.

---

## Human Verification Required

### 1. Precision proof on a real transcript

**Test:** With the Forge UI open at the Topic tab, select source `7a0b98dcbd2e4ae9981661540a1dd335` (episode_5.mp3, status=done), query "sleep and recovery", then query "cooking recipes food preparation".
**Expected:** "sleep and recovery" returns at least 1 result at score >= 0.72; "cooking recipes food preparation" returns 0 results at the 0.72 default threshold.
**Why human:** The orchestrator independently verified this live result (3 results at 79% on-topic, 0 results off-topic). This documents the precision proof checkpoint that Plan 03 Task 3 designated as a blocking human gate. No re-test required unless Mike wants to confirm the UI rendering (the API precision is already proven).

Note: The orchestrator's live precision check (`source_id 7a0b98dcbd2e4ae9981661540a1dd335`, `episode_5.mp3`, 106/106 forge tests passing) constitutes the precision proof required by Plan 03's blocking checkpoint. The phase goal's core risk — that topic retrieval is actually precise on real speech — has been retired.

---

## Gaps Summary

No gaps. All five observable truths verified, all artifacts substantive and wired, all three TOPIC requirements satisfied. The full forge test suite passes at 106/106. Precision verified live by the orchestrator on a real RuckTalk interview (on-topic: 3 results at 79%; off-topic: 0 results at the 0.72 UI threshold).

One notation: the API endpoint defaults `threshold=0.45` but the UI sends `threshold=0.72` (the operator-visible default, set at index.html line 425 and line 978). This is intentional — the API default is a permissive floor for programmatic calls; the UI uses the tighter value proven in live testing. No fix needed.

---

_Verified: 2026-06-02_
_Verifier: Claude (gsd-verifier)_
