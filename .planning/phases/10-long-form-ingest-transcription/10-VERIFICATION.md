---
phase: 10-long-form-ingest-transcription
verified: 2026-06-01T00:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 10: Long-form Ingest & Transcription — Verification Report

**Phase Goal:** The operator can get a complete, restart-safe, speaker-attributed transcript out of any long-form source (file upload up to ~12 GB, or a URL, or a pick from the shared Nextcloud Sources/ folder) without Alfred's involvement.
**Verified:** 2026-06-01
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Operator can upload a source file up to ~12 GB and see it enter the queue | VERIFIED | `uploads.save_upload_stream` streams in 8MB chunks (no whole-file `await file.read()`). `POST /forge/uploads` creates a source row + enqueues `ingest_transcribe`. Caddy `request_body { max_size 15GB }` applied. Live-verified by orchestrator. |
| 2 | Operator can supply a YouTube (or other) URL and processing begins | VERIFIED | `POST /forge/ingest/url` calls `clips.resolve_source()`, creates a `url` source, enqueues `ingest_transcribe`. No `SECTION_WINDOW`/`--download-sections` in API or handler. Live-verified by orchestrator (zoo clip). |
| 3 | Forge service can restart mid-transcription and job resumes rather than restarts | VERIFIED | `write_checkpoint` is atomic (mkstemp + os.replace, ingest.py:160-164). `transcribe_handler` checks `ckpt["complete"]` at entry and fast-paths with `resumed: True`. `check_cancel()` called per segment iteration. Checkpoint every 10 segments. Live-verified: re-enqueue after restart fast-pathed without re-transcribing. |
| 4 | Finished transcript shows each segment labelled with a speaker identifier | VERIFIED | `assign_speakers()` alternates A/B and inherits prior label on segments shorter than 1.2s. Uses `start`/`end` alias keys correctly (set alongside `start_s`/`end_s` in accumulated dict). `GET /forge/sources/{id}/transcript` returns segments with `speaker` column. Live-verified: butler sample showed speaker `A`, transcript complete. |
| 5 | Operator can pick a source from shared Nextcloud Sources/ folder and ingest it without uploading | VERIFIED | `library.SOURCES_ROOT = "Content/Mainstay-RodWave/Sources"`. `GET /forge/ingest/cloud-sources` lists files. `POST /forge/ingest/cloud` path-guarded to SOURCES_ROOT. Handler streams cloud file in 8MB chunks via `nc._webdav_url` + `nc._auth()` (no OOM-causing `nc.download_file`). Live-verified by orchestrator. |

**Score: 5/5 truths verified**

---

### Required Artifacts

| Artifact | Status | Evidence |
|----------|--------|---------|
| `core/forge/db.py` | VERIFIED | `sources` and `transcript_segments` tables present in `init_db()` executescript block. Both use `CREATE TABLE IF NOT EXISTS`. All required columns confirmed. Indexes on `(status, created_at)` and `(source_id, seq)`. |
| `core/forge/ingest.py` | VERIFIED | 476 lines. All required functions present: `create_source`, `get_source`, `update_source`, `save_segments`, `load_checkpoint`, `write_checkpoint`, `assign_speakers`, `transcribe_handler`, `get_segments`. Imports only stdlib + `core.forge.db._conn`. |
| `core/forge/uploads.py` | VERIFIED | `save_upload_stream` async function present. Chunk size is `8 * 1024 * 1024`. No whole-file read. Original `save_upload(content: bytes, ...)` kept for back-compat. |
| `core/forge/library.py` | VERIFIED | `SOURCES_ROOT = "Content/Mainstay-RodWave/Sources"`. `list_source_files()` present; filters `VIDEO_EXT | AUDIO_EXT`; degrades to `[]` on 404; calls `_ensure_folder(SOURCES_ROOT)`. |
| `core/forge/handlers.py` | VERIFIED | `_ingest_transcribe_handler` defined with lazy import. `register_default_handlers()` calls `forge_jobs.register_handler("ingest_transcribe", _ingest_transcribe_handler)`. |
| `core/api/forge.py` | VERIFIED | All 6 required endpoints present: `POST /forge/uploads` (streaming), `POST /forge/ingest/url`, `GET /forge/ingest/cloud-sources`, `POST /forge/ingest/cloud`, `GET /forge/sources/{id}`, `GET /forge/sources/{id}/transcript`. All guarded by `require_auth`. |

---

### Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|---------|
| `core/api/forge.py` upload endpoint | `core/forge/uploads.save_upload_stream` | chunked stream write | WIRED | `uid = await uploads.save_upload_stream(file, ...)` at forge.py:108. No `await file.read()` found in upload handler. |
| `core/api/forge.py` ingest endpoints | `ingest.create_source + forge_jobs.enqueue` | create source then enqueue | WIRED | All three ingest endpoints call `ingest.create_source(...)` then `_forge_jobs.enqueue("ingest_transcribe", ...)`. |
| `core/api/forge.py` url endpoint | `clips.resolve_source` | resolve URL, no SECTION_WINDOW | WIRED | `target, kind = clips.resolve_source(url)` at forge.py:151. No SECTION_WINDOW/fetch_source anywhere in API layer. |
| `core/forge/handlers.py` | `core/forge/ingest.transcribe_handler` | `register_handler("ingest_transcribe", ...)` | WIRED | handlers.py:111 registers the handler. Lazy import confirmed. |
| `core/forge/ingest.transcribe_handler` | `data/forge_transcripts/<source_id>.json` | `load_checkpoint` at start, `write_checkpoint` every 10 segments | WIRED | ingest.py:292 loads checkpoint; ingest.py:437-438 writes checkpoint per `_CHECKPOINT_INTERVAL`. Atomic via mkstemp+os.replace. |
| `core/forge/ingest.transcribe_handler` | `faster_whisper.WhisperModel medium on cuda` | lazy import, `vad_filter + word_timestamps` | WIRED | ingest.py:396-415. CUDA with CPU int8 fallback. `vad_filter=True`, `word_timestamps=True`, `beam_size=5`. |
| `core/forge/ingest.transcribe_handler` | `assign_speakers + save_segments` | label then persist | WIRED | ingest.py:445-461. `assign_speakers(accumulated)` then `save_segments(source_id, labelled_clean)`. |
| `core/forge/ingest.transcribe_handler` | Nextcloud chunked WebDAV GET | `nc._webdav_url` + `nc._auth()` streaming | WIRED | ingest.py:364-368. `stream=True`, `iter_content(8MB)`. Bug-fixed from original `NextcloudClient` reference — confirmed absent. |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| INGEST-01 | 10-02-PLAN.md | Upload long-form source up to ~12 GB | SATISFIED | Streaming upload endpoint live; 8MB chunked write; Caddy 15GB limit; live-verified multi-GB upload. |
| INGEST-02 | 10-02-PLAN.md | Supply source by URL instead of file | SATISFIED | `POST /forge/ingest/url` live; full yt-dlp download without section window; live-verified zoo clip. |
| INGEST-03 | 10-01-PLAN.md / 10-03-PLAN.md | Transcription survives service restart | SATISFIED | Atomic checkpoint every 10 segments; `complete=True` fast-path on re-enqueue; live-verified restart fast-path. |
| INGEST-04 | 10-01-PLAN.md / 10-03-PLAN.md | Transcript segments attributed to speakers | SATISFIED | `assign_speakers` A/B heuristic with short-segment inheritance; speaker column in DB + API; live-verified. |
| INGEST-05 | 10-02-PLAN.md | Pick from shared Nextcloud Sources/ folder without uploading | SATISFIED | `list_source_files()` + `POST /forge/ingest/cloud`; path-guarded to SOURCES_ROOT; chunked WebDAV download in handler; live-verified drop-in-folder flow. |

REQUIREMENTS.md traceability table confirms INGEST-01..04 marked `[x]` complete, INGEST-05 marked `[ ]` (pending) — that reflects a pre-completion snapshot. Code and live verification confirm all five are implemented.

No orphaned requirements: REQUIREMENTS.md maps no additional INGEST-* IDs to Phase 10 beyond the declared five.

---

### Anti-Patterns Found

None blocking. Observations:

- `ingest.py` assigns `"start"` and `"end"` aliases alongside `"start_s"` / `"end_s"` per segment during transcription loop, then strips them before DB persistence (`labelled_clean`). This is intentional (the speaker heuristic reads `"start"`/`"end"`) and correctly cleaned up.
- `core/forge/uploads.py` still retains the original `save_upload(content: bytes, ...)` whole-file function for back-compat with other callers. This is not used in the ingest path and poses no risk.
- `assign_speakers` reads `seg.get("end", 0.0) - seg.get("start", 0.0)` for duration — relies on aliases being present. Both keys are confirmed set in the accumulated dict at ingest.py:429-430.

---

### Human Verification (orchestrator-confirmed, no further human action needed)

The orchestrator performed live end-to-end verification during plan execution:

1. Upload path: butler sample file uploaded, source row created, transcript completed with `status=done` and speaker `A` label, checkpoint `complete: true`.
2. URL path: zoo YouTube clip submitted via `POST /forge/ingest/url`, full download without section window, transcript completed.
3. Cloud path: file dropped in `Content/Mainstay-RodWave/Sources/`, listed via `GET /forge/ingest/cloud-sources`, ingested via `POST /forge/ingest/cloud`, streamed down, transcribed.
4. Restart safety: longer source mid-transcription, service restarted, re-enqueue returned `resumed: True`, no re-transcription.

---

### Test Suite

```
93 passed, 3 warnings in 10.44s
```

All 93 existing tests pass. No regressions introduced.

---

## Summary

All five success criteria for Phase 10 are verified against the actual codebase. The three delivery modules (`ingest.py`, `uploads.py`, `library.py`) are substantive implementations — no stubs. All key wiring paths are confirmed: upload streams without OOM, URL ingest has no section window cap, cloud download uses chunked WebDAV (the bug-fixed path, not the buffering `nc.download_file`), checkpoint writes are atomic, handler is registered, transcript endpoints are wired. The 10-03 SUMMARY notes the `NextcloudClient` bug was caught and fixed during live verification — confirmed absent from the current code. Phase goal is fully achieved.

---

_Verified: 2026-06-01_
_Verifier: Claude (gsd-verifier)_
