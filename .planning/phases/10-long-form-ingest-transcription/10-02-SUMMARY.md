---
plan: 10-02
phase: 10-long-form-ingest-transcription
status: complete
requirements: [INGEST-01, INGEST-02, INGEST-05]
completed: 2026-06-01
---

# 10-02 Summary — Ingest entry points

## What was built
Three ingest paths, all converging on one `source_id` + `ingest_transcribe` job:
- **Streaming upload** (`uploads.save_upload_stream`, 8MB chunks) — `POST /forge/uploads` no longer does `await file.read()` (the OOM blocker on 12GB files); creates an `upload` source.
- **URL ingest** — `POST /forge/ingest/url` resolves via `clips.resolve_source()` (no `SECTION_WINDOW` cap), creates a `url` source.
- **Cloud pick** — `GET /forge/ingest/cloud-sources` lists `Content/Mainstay-RodWave/Sources/`; `POST /forge/ingest/cloud` (path guarded to SOURCES_ROOT) creates a `cloud` source. Pairs with the live file-drop button.

## Key files
- `core/forge/uploads.py` — `save_upload_stream()`
- `core/forge/library.py` — `SOURCES_ROOT`, `list_source_files()`, `AUDIO_EXT`
- `core/api/forge.py` — `/forge/uploads` (streamed), `/forge/ingest/url`, `/forge/ingest/cloud-sources`, `/forge/ingest/cloud`
- `/etc/caddy/Caddyfile` — `request_body { max_size 15GB }` on both forge entry points (orchestrator-applied, validated + reloaded)

## Commits
- 42a0199 streaming upload · 21f00a9 URL endpoint · 6b685a9 cloud list+pick

## Verification (live, orchestrator)
- Upload path → `source` row + transcript (butler sample). ✓ INGEST-01
- `POST /forge/ingest/url` (zoo clip) → full download, transcript. ✓ INGEST-02
- Drop in Sources/ → `cloud-sources` lists → `/ingest/cloud` → stream-down + transcript. ✓ INGEST-05
- Caddy validated + reloaded; service active; all endpoints registered.

## Deviations
- Caddy edit + live verification done by orchestrator (the checkpoint), not the executor.
