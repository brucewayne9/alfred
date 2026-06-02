---
phase: 11-topic-targeted-segment-retrieval
plan: 01
subsystem: forge/search
tags: [embeddings, chromadb, bge-m3, windowing, retrieval]
dependency_graph:
  requires: [core/forge/ingest.py, core/memory/store.py]
  provides: [core/forge/search.py, forge_segments ChromaDB collection]
  affects: [core/forge/ingest.py transcribe_handler]
tech_stack:
  added: []
  patterns: [delete-before-upsert idempotency, lazy import for circular-dep avoidance, cosine-distance inversion]
key_files:
  created:
    - core/forge/search.py
    - tests/forge/test_search.py
  modified:
    - core/forge/ingest.py
decisions:
  - "Single shared forge_segments collection scoped via source_id metadata — not one collection per source"
  - "Duration-only windowing (no speaker-boundary splits) — short interjections flip speaker not topic"
  - "embed_source_windows wrapped in try/except in both transcribe paths — embedding failure never aborts a completed transcription"
  - "Lazy import of core.forge.ingest inside embed_source_windows to break circular dependency"
metrics:
  duration: 15m
  completed: 2026-06-02
  tasks_completed: 2
  files_changed: 3
---

# Phase 11 Plan 01: Windowing + bge-m3 Embedding + Source-Scoped Query Summary

**One-liner:** Duration-based segment windowing (20-40s) + bge-m3 ChromaDB embedding wired inline to transcription, with cosine-distance-inverted ranked query scoped per source.

## What Was Built

### core/forge/search.py (new, 307 lines)

Complete vector-retrieval module for Forge. Exports:

- `_get_collection()` — returns the shared `forge_segments` ChromaDB collection with `OllamaEmbeddingFunction(bge-m3:latest)` and cosine distance
- `build_windows(segments, target_dur_s=30, max_dur_s=45, min_dur_s=10)` — merges adjacent segments into topic-coherent windows; dominant speaker via Counter; speaker None coerced to ""
- `upsert_windows(source_id, windows)` — ChromaDB upsert with source_id/start_s/end_s/speaker/seq_start/seq_end metadata
- `embed_source_windows(source_id)` — delete-before-upsert then embed; returns window count; lazy-imports ingest to avoid circular dep
- `has_windows(source_id)` — lightweight presence check
- `search_segments(source_id, query, top_k=10, speaker=None, score_threshold=0.45)` — source-scoped cosine query; score = round(1.0 - dist/2.0, 4); never returns raw distance

### core/forge/ingest.py (modified)

- Added `import logging` + module-level `logger = logging.getLogger(__name__)`
- Guarded `embed_source_windows` call added at checkpoint fast-path (line ~301) and main completion path (line ~474), between `save_segments()` and `update_source(status='done')` — identical pattern in both paths

### tests/forge/test_search.py (new, 6 tests)

Pure-unit tests, no ChromaDB/Ollama required:
- `test_build_windows_merges_short_segments` — 20×5s segs → 2-19 windows, contiguous seq coverage, non-empty text
- `test_window_win_id_format` — win_id matches `{source_id}_w{seq_start:04d}`
- `test_dominant_speaker_majority` — A/A/B → speaker A
- `test_dominant_speaker_none_coerced_to_empty` — None speaker → ""
- `test_score_inversion` — dist 0→1.0, dist 2→0.0, dist 1→0.5; formula verified via source inspection
- `test_empty_segments_returns_no_windows` — build_windows([]) == []

## Verification Results

```
tests/forge/test_search.py: 6 passed
tests/forge (full suite): 99 passed, 0 failures
grep -n "_search.embed_source_windows" core/forge/ingest.py: 2 call sites (lines 301, 474)
python3 -c "import core.forge.search": OK (no side effects)
build_windows smoke: 20×5s segs → 3 windows, first_dur=45.0, win_id='t_w0000', speaker='A'
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing] Added logger to ingest.py**
- **Found during:** Task 2 — the inline embed call uses `logger.exception()` but ingest.py had no logger
- **Issue:** Module had no `import logging` or `logger` defined
- **Fix:** Added `import logging` and `logger = logging.getLogger(__name__)` at module level
- **Files modified:** core/forge/ingest.py
- **Commit:** b0d5540

## Self-Check: PASSED

- core/forge/search.py: FOUND
- tests/forge/test_search.py: FOUND
- commit 8e1814e (search.py): FOUND
- commit b0d5540 (ingest.py + tests): FOUND
