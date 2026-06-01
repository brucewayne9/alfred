---
phase: 10-long-form-ingest-transcription
plan: "01"
subsystem: forge-ingest
tags: [sqlite, ingest, transcription, speaker-heuristic, audio-extraction]
dependency_graph:
  requires: []
  provides: [sources-table, transcript-segments-table, ingest-crud, checkpoint-helpers, assign-speakers, extract-audio]
  affects: [10-02-ingest-paths, 10-03-transcription-handler]
tech_stack:
  added: []
  patterns: [atomic-checkpoint-via-mkstemp-os-replace, col-whitelist-update, idempotent-migration]
key_files:
  created:
    - core/forge/ingest.py
  modified:
    - core/forge/db.py
    - core/forge/audio.py
decisions:
  - "Checkpoint write uses mkstemp + os.replace for crash-safety; never leaves a half-written JSON file"
  - "update_source col whitelist (frozenset) prevents SQL injection without parameterised DDL"
  - "No faster-whisper/ffmpeg imports in ingest.py — keeps the module import-safe for unit tests"
metrics:
  duration_minutes: 6
  completed_date: "2026-06-01"
  tasks_completed: 3
  files_changed: 3
---

# Phase 10 Plan 01: Long-form Ingest Foundation — Storage Tables, ingest.py, extract_audio

**One-liner:** SQLite migration adds `sources` + `transcript_segments` tables; `ingest.py` provides idempotent source CRUD, atomic checkpoint helpers, and the proven A/B speaker heuristic; `audio.py` gains a 16 kHz mono WAV extractor ready for faster-whisper.

---

## Tasks Completed

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | Add sources + transcript_segments to Forge DB | e0acf2c | core/forge/db.py |
| 2 | Create core/forge/ingest.py | eb52885 | core/forge/ingest.py (new) |
| 3 | Add extract_audio() to core/forge/audio.py | 2317237 | core/forge/audio.py |

---

## What Was Built

### Task 1 — DB Migration (db.py)

`init_db()` extended with two new tables in the same `executescript` block:

- **`sources`** — one stable row per long-form file or URL. Columns: `id` (uuid hex), `kind` ('upload'|'url'), `spec` (filename or URL), `file_path`, `status` (pending→extracting→transcribing→done→error), `duration_s`, `language`, `error`, `created_at`, `updated_at`. Index on `(status, created_at)`.
- **`transcript_segments`** — Whisper output keyed to `source_id`. Columns: `id` (autoincrement), `source_id` (FK → CASCADE DELETE), `seq`, `start_s`, `end_s`, `text`, `speaker`, `words` (JSON blob). Index on `(source_id, seq)`.

Both use `CREATE TABLE IF NOT EXISTS` — migration is idempotent. Verified: second `init_db()` call is a no-op on the live DB.

### Task 2 — core/forge/ingest.py (205 lines)

**Source CRUD:**
- `create_source(kind, spec, file_path) -> str` — inserts row, returns uuid hex id
- `get_source(source_id) -> dict | None`
- `update_source(source_id, **fields)` — col-whitelist guard: `{status, duration_s, language, error, file_path}` only; raises ValueError on unknown cols
- `save_segments(source_id, segments)` — DELETE + bulk INSERT; idempotent; JSON-serialises `words` list automatically

**Checkpoint helpers (INGEST-03):**
- `_checkpoint_dir()` — returns `data/forge_transcripts/` (respects `FORGE_TRANSCRIPT_DIR` env override)
- `checkpoint_path(source_id)` — `<dir>/<source_id>.json`
- `load_checkpoint(source_id)` — reads JSON; returns `{"segments":[],"complete":False}` if missing or corrupt
- `write_checkpoint(source_id, data)` — **atomic**: writes to temp file in same dir via `mkstemp`, then `os.replace` onto final path; interrupted write never corrupts checkpoint

**Speaker heuristic (INGEST-04):**
- `SHORT_SEGMENT_THRESHOLD_S = 1.2`
- `assign_speakers(segments)` — alternates A/B; inherits prior label when duration < 1.2 s; does not mutate inputs

### Task 3 — extract_audio() in audio.py

`extract_audio(src, out_path) -> Path` — full-file demux to 16 kHz mono PCM WAV via ffmpeg. Follows the existing `subprocess.run(..., check=True)` pattern. Returns `Path(out_path)`. Verified: 16000 Hz, 1 channel confirmed via ffprobe.

---

## Verification Results

```
# Task 1
$ FORGE_DB_PATH=data/forge_live.db python -c "from core.forge import db; db.init_db(); db.init_db()"
# → idempotent OK; sources + transcript_segments present in live DB

# Task 2 — CRUD + checkpoint round-trip
$ FORGE_DB_PATH=data/forge_live.db python -c "..."
pending
True

# Task 2 — speaker heuristic
$ python -c "from core.forge.ingest import assign_speakers; ..."
['A', 'B', 'B', 'A']

# Task 3 — WAV extraction
$ python -c "... extract_audio('/tmp/_t.m4a', '/tmp/_t.wav') ..."
True  # file size > 0
$ ffprobe ... /tmp/_t.wav
16000,1  # 16 kHz, 1 channel

# Full test suite
93 passed, 3 warnings in 10.42s
```

---

## Deviations from Plan

None — plan executed exactly as written.

---

## Self-Check

- [x] `core/forge/ingest.py` exists (205 lines)
- [x] `core/forge/db.py` — sources + transcript_segments in init_db
- [x] `core/forge/audio.py` — `def extract_audio` present
- [x] Commits: e0acf2c, eb52885, 2317237
- [x] 93 tests pass, 0 failures

## Self-Check: PASSED
