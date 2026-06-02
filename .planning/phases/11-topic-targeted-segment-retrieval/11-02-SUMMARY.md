---
phase: 11-topic-targeted-segment-retrieval
plan: 02
subsystem: forge-search-api
tags: [forge, search, api, topic-retrieval, TOPIC-01, TOPIC-02]
dependency-graph:
  requires: [11-01]
  provides: [GET /forge/sources, GET /forge/sources/{id}/search, list_sources()]
  affects: [core/api/forge.py, core/forge/ingest.py]
tech-stack:
  added: []
  patterns: [sys.modules stub injection for chromadb-free API tests]
key-files:
  created: [tests/forge/test_search_api.py]
  modified: [core/api/forge.py, core/forge/ingest.py]
decisions:
  - sys.modules stub injection to isolate chromadb dep in test environment (pre-existing test_search.py already broken; kept out-of-scope)
  - GET /forge/sources registered before GET /forge/sources/{source_id} to prevent path shadowing
metrics:
  duration: 2m
  completed: 2026-06-02
  tasks: 2
  files_changed: 3
requirements: [TOPIC-01, TOPIC-02]
---

# Phase 11 Plan 02: Search API Endpoints Summary

**One-liner:** HTTP search API exposing source-scoped topic retrieval with inline transcript text, lazy Phase-10 backfill, and 7-test contract coverage.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | list_sources() + GET /forge/sources | e8874fa | core/forge/ingest.py, core/api/forge.py |
| 2 | GET /forge/sources/{id}/search + API tests | 5310528 | core/api/forge.py, tests/forge/test_search_api.py |

## What Was Built

### Task 1 — list_sources() DB helper + list endpoint
- Added `list_sources(status=None)` to `core/forge/ingest.py` — returns source rows newest-first, optionally filtered by status. Follows the same `_conn()` + `dict(row)` pattern as `get_source()`.
- Added `GET /forge/sources` to `core/api/forge.py`, registered at line 156 — before the `{source_id}` path-param route at line 162 to prevent shadowing.
- Operators call `GET /forge/sources?status=done` to populate the source picker in the UI.

### Task 2 — Search endpoint + tests
- Added `GET /forge/sources/{source_id}/search` with params: `q` (required), `top_k` (1–50, default 10), `speaker` (optional), `threshold` (0.0–1.0, default 0.45).
- 404 when source not found; 409 when source status != "done".
- Lazy backfill: calls `forge_search.embed_source_windows(source_id)` when `has_windows()` returns False — transparent for Phase-10 sources ingested before embedding existed.
- Each result carries full window `text` inline — TOPIC-02 operator preview with no second request required.
- Created `tests/forge/test_search_api.py` with 7 tests: list endpoint, status filter pass-through, 404, 409, 200 happy path (text-inline assertion), lazy-backfill trigger, no-backfill when windows exist.
- Test pattern: `sys.modules` stub injection to replace `core.forge.search` before the lazy import fires — avoids `chromadb` dependency in the test environment.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] chromadb import breaks search API tests**
- **Found during:** Task 2
- **Issue:** `core.forge.search` imports `chromadb` at module level via `core.memory.store`. First test approach using `monkeypatch.setattr` on the module failed because the module couldn't even be imported in the test environment.
- **Fix:** Inject a `types.ModuleType` stub into `sys.modules["core.forge.search"]` via an `autouse` fixture before the FastAPI app is registered. The endpoint's lazy `from core.forge import search as forge_search` then resolves to the stub.
- **Files modified:** tests/forge/test_search_api.py
- **Note:** `tests/forge/test_search.py` has the same pre-existing failure (also imports `build_windows` which pulls chromadb); it predates this plan and is out-of-scope per deviation rules. Logged to deferred items.

## Deferred Issues

- `tests/forge/test_search.py` fails to collect due to pre-existing chromadb import at module level (from Plan 01). Fix: move `from core.memory.store import get_client` inside `_get_collection()` to make `core.forge.search` importable without chromadb. Out of scope for this plan.

## Verification

```
python3 -m pytest tests/forge/test_search_api.py -v   # 7 passed
python3 -m pytest tests/forge -q --ignore=tests/forge/test_search.py  # 100 passed
grep -n '/forge/sources' core/api/forge.py
# 156: /forge/sources  (literal, before path-param)
# 162: /forge/sources/{source_id}
# 174: /forge/sources/{source_id}/search
# 213: /forge/sources/{source_id}/transcript
```

## Self-Check: PASSED

- `/home/aialfred/alfred/core/forge/ingest.py` — list_sources() added
- `/home/aialfred/alfred/core/api/forge.py` — both endpoints registered
- `/home/aialfred/alfred/tests/forge/test_search_api.py` — created, 7 tests pass
- Commits e8874fa and 5310528 exist in git log
