---
plan: 10-03
phase: 10-long-form-ingest-transcription
status: complete
requirements: [INGEST-03, INGEST-04]
completed: 2026-06-01
---

# 10-03 Summary — Checkpointed transcribe handler

## What was built
The `ingest_transcribe` job handler (`core/forge/ingest.transcribe_handler`), registered in `handlers.py`:
- **Source localization** — `url` sources download full via yt-dlp (no `SECTION_WINDOW`); `cloud` sources stream down from Nextcloud in 8MB chunks; `upload` sources use the local file.
- **Transcription** — lazy faster-whisper `medium` on CUDA (`vad_filter`, `word_timestamps`), CPU int8 fallback.
- **Restart-safety (INGEST-03)** — incremental disk checkpoint every 10 segments (atomic temp+replace); handler entry fast-paths on a complete checkpoint (re-enqueue after restart resumes, does not re-transcribe).
- **Speakers (INGEST-04)** — `assign_speakers` A/B heuristic on the final segment list.
- **Read endpoints** — `GET /forge/sources/{id}` (status), `GET /forge/sources/{id}/transcript` (segments + speakers + words).

## Key files
- `core/forge/ingest.py` — `transcribe_handler`, `get_segments`
- `core/forge/handlers.py` — registers `ingest_transcribe`
- `core/api/forge.py` — source status + transcript read endpoints

## Commits
- 30ed370 handler · bc3f234 register + read endpoints · (fix) cloud download uses module-level nextcloud client

## Verification (live, orchestrator)
- Butler sample → `status=done`, accurate transcript, speaker `A`, checkpoint `complete:true`. ✓ INGEST-04 + transcription
- Re-enqueue same source → `resumed: True` fast-path, no re-transcription. ✓ INGEST-03
- Read endpoint returns segments with speaker labels. ✓
- 93 tests pass.

## Deviations
- **Bug caught + fixed in live verify:** cloud-download referenced a non-existent `NextcloudClient` class; rewritten to `nc._webdav_url` + `nc._auth()` streaming (committed). This is why live verification mattered.
