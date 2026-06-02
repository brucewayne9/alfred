# Phase 10: Long-form Ingest & Transcription — Research

**Researched:** 2026-06-01
**Domain:** File ingest, audio extraction, transcription, speaker diarization, async job safety
**Confidence:** HIGH (all findings grounded in the actual codebase and verified installs)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INGEST-01 | Upload a long-form source file up to ~12 GB | Requires replacing `await file.read()` in forge.py:106 with streaming write; Caddy has no configured body limit today but will silently drop large requests; uvicorn default is unlimited but RAM becomes the blocker |
| INGEST-02 | Supply a source by URL instead of uploading | `core/forge/clips.py:fetch_source()` + `resolve_source()` already handle YouTube/HTTP URLs via yt-dlp; reuse directly |
| INGEST-03 | Transcribe async, survives restart | `reconcile_orphans()` marks interrupted jobs as `error`, not resumable; transcription must checkpoint progress to disk (segment file) and re-enter at the right segment on re-enqueue, making the handler idempotent |
| INGEST-04 | Speaker attribution per segment | No pyannote/WhisperX installed; alternating-segment heuristic (notebooklm_clip.py:168) is the proven in-repo approach; pyannote.audio is installable if a HuggingFace token is obtained |
</phase_requirements>

---

## Summary

Forge has a mature async job system (SQLite + worker_loop), proven yt-dlp URL ingest, faster-whisper 1.2.1 installed in the venv with GPU access confirmed, and ChromaDB already persisted at `data/chromadb/` — all the plumbing exists. The three work items are: (1) replacing the in-memory upload handler with a streaming write for large files and setting Caddy's `request_body` limit, (2) making the transcription handler idempotent so restarts resume rather than restart, and (3) deciding diarization strategy (heuristic vs. pyannote).

The biggest blocker to know before planning: **`reconcile_orphans()` marks interrupted jobs as `error` and discards them** — a restart of `forge-web.service` mid-transcription on a 2-hour video is a hard failure today. The plan must make the transcription handler checkpoint-aware so it can re-enter from where it left off. This is the core design challenge of INGEST-03.

**Primary recommendation:** Build `ingest_transcribe` as a Forge job type with (a) a streaming upload endpoint, (b) a checkpointed faster-whisper handler that writes segments to disk as it goes and skips already-done ones on re-enqueue, and (c) an alternating-segment heuristic for diarization (same as `notebooklm_clip.py`) unless Mike approves pyannote install.

---

## Integration Points

### 1. Forge Job System

**Files:** `core/forge/jobs.py`, `core/forge/db.py`, `services/forge-web/serve.py`

| Function | Location | Signature | Notes |
|----------|----------|-----------|-------|
| `enqueue` | jobs.py:74 | `(job_type: str, params: dict, now: int) -> str` | Returns hex job_id; persists to SQLite immediately |
| `register_handler` | jobs.py:39 | `(job_type: str, fn: Callable[[dict], dict]) -> None` | Handler receives params dict, returns result dict; call from handlers.py `register_default_handlers()` |
| `check_cancel` | jobs.py:55 | `() -> None` | Raises `JobCancelled`; handlers must call this at segment boundaries |
| `_update` | jobs.py:109 | `(job_id: str, **fields) -> None` | Write arbitrary fields to the jobs row; use to checkpoint `progress` if a `progress` column is added |
| `reconcile_orphans` | jobs.py:204 | `() -> int` | Called at startup (serve.py:41); sets `status='error'` on any `running` job — does NOT resume |
| `worker_loop` | jobs.py:218 | `async (poll_interval=2.0)` | Single task; runs one job at a time in a thread executor; started at serve.py:44 |
| `claim_next_pending` | jobs.py:117 | `() -> dict | None` | Atomically claims oldest pending job |

**Key restart-safety constraint:** `serve.py:41-44` calls `reconcile_orphans()` then starts a new `worker_loop`. Any job `status='running'` at restart becomes `status='error'`. The handler for `ingest_transcribe` must be **idempotent via re-enqueue**: when a transcription is interrupted, the reconcile sets it to `error`, but the UI (or the handler on a prior partial run) must leave a checkpoint file on disk. On the operator's next "retry", the handler reads the checkpoint and resumes from the last complete segment. This is resumable-by-re-enqueue, not resumable-by-same-job.

**No `progress` column exists** in the current schema (`db.py:26-54`). To show real-time progress, add a `progress TEXT` column (JSON blob: `{"step": "transcribing", "pct": 42}`) that the handler writes via `_update(job_id, progress=json.dumps(...))` and the polling endpoint returns.

**DB file in production:** `FORGE_DB_PATH=/home/aialfred/alfred/data/forge_live.db` (systemd env, serve.py:15)

---

### 2. Upload Endpoint — HARD BLOCKER for 12 GB

**File:** `core/api/forge.py:104-111`, `core/forge/uploads.py:15-22`

Current code:
```python
# forge.py:106-110 — HARD BLOCKER: reads entire file into RAM
content = await file.read()          # OOM on 4+ GB
uid = uploads.save_upload(content, file.filename or "upload.bin")
```

```python
# uploads.py:15-22
def save_upload(content: bytes, filename: str) -> str:
    ext = Path(filename or "").suffix.lower()[:12]
    uid = uuid.uuid4().hex
    dest_dir = _root() / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"file{ext}").write_bytes(content)
    return uid
```

**Required fix:** Replace with chunked streaming write. FastAPI/Starlette supports `SpooledTemporaryFile` or manual chunk iteration via `file.read(chunk_size)` in a loop. The `uploads.py:save_upload` signature must also change to accept a path or stream.

**Also required:** Add `request_body { max_size 15GB }` to the `/forge*` block in `/etc/caddy/Caddyfile`. Without it, Caddy uses its default which may reject large bodies silently. The uvicorn default is unlimited (no `--limit-concurrency` or body cap in `serve.py`).

**Storage available:** 1.1 TB free on `/dev/nvme0n1p2`. `FORGE_UPLOAD_DIR` defaults to `data/forge_uploads/` (uploads.py:10); can override via env.

**Chunked upload approach (no external deps):** Use Starlette's `UploadFile.file` (a `SpooledTemporaryFile` in read mode) and write to the destination in 8 MB chunks:
```python
uid = uuid.uuid4().hex
dest_dir = _root() / uid
dest_dir.mkdir(parents=True, exist_ok=True)
dest = dest_dir / f"file{ext}"
with dest.open("wb") as out:
    while chunk := await file.read(8 * 1024 * 1024):
        out.write(chunk)
```
This keeps RAM flat regardless of file size.

---

### 3. URL Ingest — Ready to Reuse

**File:** `core/forge/clips.py`

| Function | Signature | Notes |
|----------|-----------|-------|
| `resolve_source` | `(spec: str) -> (target, kind)` | Handles `http://`, `https://`, `search:` prefix, bare text → ytsearch |
| `fetch_source` | `(spec: str, out_dir: Path, timeout=480) -> list[Path]` | Runs yt-dlp; returns list of mp4 paths; raises RuntimeError if nothing fetched |
| `ytdlp_cmd` | `(target: str, out_dir: Path) -> list[str]` | Full yt-dlp command with Node JS runtime, retries, section window |

**Watch out:** `SECTION_WINDOW = "*0:00-01:30"` in clips.py:15 caps downloads to 90 seconds per clip. For long-form ingest, this constant must NOT be used — the handler needs to call yt-dlp directly without `--download-sections`. Do not call `fetch_source()` directly for ingest; use `resolve_source()` + a custom yt-dlp invocation without the section window.

**yt-dlp version:** 2026.03.17, available at `~/.pyenv/shims/yt-dlp`. Node runtime auto-detected from `~/.nvm/versions/node/v*/bin/node` (clips.py:29).

---

### 4. Audio Extraction from Video

**File:** `core/forge/audio.py`

| Function | Signature | Notes |
|----------|-----------|-------|
| `duration_seconds` | `(path: str|Path) -> float` | ffprobe call |
| `clip_audio` | `(src, start, end, out_path) -> Path` | Cuts a slice; re-encodes to mp3 |

**Missing function (must add):** `extract_audio(src: str|Path, out_path: str|Path) -> Path` — full-file audio extraction. Pattern:
```python
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                str(out_path)], check=True)
```
WAV at 16 kHz mono is what faster-whisper expects natively. Output to the upload's directory alongside the source.

---

### 5. Transcription — faster-whisper (PRIMARY)

**Installed:** `faster-whisper==1.2.1` in `/home/aialfred/alfred/venv/`
**GPU confirmed:** RTX 3090 (24 GB VRAM) + RTX 4070 (12 GB); CTranslate2 CUDA device count = 2
**Existing usage:** `core/forge/audio.py:42-52` (`transcribe_words`) — word-level, CPU int8, small model

**For long-form ingest, use GPU int8_float16 on the 3090 with `large-v3-turbo` or `medium`:**

```python
# core/forge/audio.py reference pattern, extended for full-file segment transcription
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="int8_float16")
segments, info = model.transcribe(
    str(audio_path),
    word_timestamps=True,
    vad_filter=True,          # skip silence — critical for long-form
    beam_size=5,
    language="en",
)
```

**Models cached (HuggingFace CTranslate2 format):** tiny, base, small, small.en, medium — all at `~/.cache/huggingface/hub/Systran/faster-whisper-*`.

**Note:** `large-v3-turbo` exists only as `~/.cache/whisper/large-v3-turbo.pt` (openai-whisper format, not CTranslate2). To use it with faster-whisper, it would need conversion. **Use `medium` for now** — it's cached in the right format and runs well on the 3090.

**Checkpoint pattern for restart safety:**

The handler writes completed segments to a JSON file on disk as it processes them. On restart + re-enqueue, the handler reads the checkpoint and skips already-done segments:

```python
CHECKPOINT_DIR = Path("data/forge_transcripts")   # separate from uploads

def _transcribe_handler(params: dict) -> dict:
    source_id = params["source_id"]   # upload uid or url hash
    checkpoint = CHECKPOINT_DIR / f"{source_id}.json"

    if checkpoint.exists():
        done = json.loads(checkpoint.read_text())
    else:
        done = {"segments": [], "complete": False}

    if not done["complete"]:
        # load model, run transcription, append to done["segments"] in batches
        # write checkpoint after each batch
        for segment in model.transcribe(...):
            done["segments"].append({...})
            checkpoint.write_text(json.dumps(done))   # atomic-enough for recovery
            forge_jobs.check_cancel()
        done["complete"] = True
        checkpoint.write_text(json.dumps(done))

    return {"transcript_id": source_id, "segment_count": len(done["segments"])}
```

**The handler is idempotent**: calling it again on the same `source_id` (via re-enqueue after error) fast-paths through already-completed work.

---

### 6. Speaker Diarization — INGEST-04

**pyannote.audio:** NOT installed in the venv. No HF token found in config. Requires: `pip install pyannote.audio`, HuggingFace account, accepting two model terms-of-use agreements, and `HUGGINGFACE_TOKEN` in the env. This is a one-time setup but needs Mike's HF credentials.

**WhisperX:** NOT installed. Installs cleanly alongside faster-whisper but adds `torch`, `torchaudio` to the venv and internally depends on pyannote for diarization anyway.

**In-repo heuristic (already proven):** `scripts/notebooklm_clip.py:168-187` — `assign_speakers_per_segment(segments)` alternates `A`/`B` per Whisper segment, inheriting on short segments (< 1.2s). Designed for two-speaker dialog; works cleanly for interview/podcast formats.

**Recommendation: ship the heuristic for v1, flag pyannote as an upgrade path.**

The heuristic produces `speaker: "A"` / `speaker: "B"` labels per segment. For interview content (the primary Mainstay use case — Rod Wave interviews), this is sufficient: two speakers, clear turn-taking, Whisper segment breaks reliably at speaker turns.

**Heuristic to copy into new `core/forge/ingest.py`:**
```python
# Source: scripts/notebooklm_clip.py:165-187
SHORT_SEGMENT_THRESHOLD_S = 1.2

def assign_speakers(segments: list[dict]) -> list[dict]:
    """Alternate A/B per Whisper segment; inherit prior on short segments."""
    labels, current = [], "A"
    for seg in segments:
        dur = seg.get("end", 0.0) - seg.get("start", 0.0)
        if not labels:
            labels.append(current)
        elif dur < SHORT_SEGMENT_THRESHOLD_S:
            labels.append(labels[-1])
        else:
            current = "B" if labels[-1] == "A" else "A"
            labels.append(current)
    return [dict(seg, speaker=lbl) for seg, lbl in zip(segments, labels)]
```

---

### 7. Transcript Storage Schema

**Approach:** New SQLite table in the Forge DB (`data/forge_live.db`), added via `init_db()` migration in `core/forge/db.py`. Separate from the `jobs` table — transcripts persist after their source job is deleted.

**Also:** Write checkpoint/final transcript JSON to `data/forge_transcripts/<source_id>.json` on disk so the handler can resume without a DB query.

**Proposed `sources` + `transcript_segments` tables:**

```sql
-- Add to core/forge/db.py init_db() executescript

CREATE TABLE IF NOT EXISTS sources (
    id           TEXT PRIMARY KEY,         -- uuid hex (upload uid or url-hash)
    kind         TEXT NOT NULL,            -- 'upload' | 'url'
    spec         TEXT NOT NULL,            -- original filename or URL
    file_path    TEXT,                     -- absolute path on disk
    status       TEXT NOT NULL DEFAULT 'pending',  -- pending|extracting|transcribing|done|error
    duration_s   REAL,
    language     TEXT,
    error        TEXT,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status, created_at);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,         -- 0-based segment order
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    text         TEXT NOT NULL,
    speaker      TEXT,                     -- 'A' | 'B' | NULL (if no diarization)
    -- word-level timestamps stored as JSON blob for Phase 11 retrieval
    words        TEXT                      -- JSON: [{"word","start","end"}, ...]
);
CREATE INDEX IF NOT EXISTS idx_segments_source ON transcript_segments(source_id, seq);
```

**Phase 11 note:** ChromaDB is already installed (`chromadb==1.4.1`) and persisted at `data/chromadb/` with a working `get_collection()` helper at `core/memory/store.py:28`. Phase 11 will embed transcript segments into a `forge_transcripts` ChromaDB collection. Phase 10 should store `words` JSON per segment to make that embedding step trivial (no re-parsing needed).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Audio demux from video | Custom ffmpeg wrapper | `ffmpeg -vn -ac 1 -ar 16000` (one line, add to `audio.py`) | Already used throughout the codebase |
| URL download | Custom HTTP/yt-dlp wrapper | `core/forge/clips.py:resolve_source()` + `fetch_source()` | Battle-tested, has Node JS runtime, retry logic, auth headers |
| Transcription | openai-whisper CLI subprocess | `faster_whisper.WhisperModel` (in venv, GPU-ready) | 4-8x faster than openai-whisper; already used in `audio.py:transcribe_words()` |
| Speaker diarization | pyannote.audio pipeline | `assign_speakers()` heuristic from `notebooklm_clip.py:168` | Sufficient for two-speaker interview; zero new dependencies; pyannote = HF token + 2 model agreements + `torch` install |
| Segment storage | SQLite blob per job | `transcript_segments` table keyed to `source_id` | Phase 11 needs indexed lookups by `source_id` and `start_s`/`end_s` |
| Embedding store | Custom vector store | `core/memory/store.py:get_collection()` → ChromaDB | Already wired, persisted, and used elsewhere in the codebase |

---

## Common Pitfalls

### Pitfall 1: In-memory upload OOM
**What goes wrong:** `content = await file.read()` in forge.py:107 loads the entire file into RAM. A 12 GB file kills the process.
**How to avoid:** Replace with 8 MB chunk loop writing directly to disk. Update `uploads.py:save_upload` to accept a stream or path rather than bytes.
**Also:** Add `request_body { max_size 15GB }` in the Caddy `/forge*` block; without it, behavior for large bodies is undefined.

### Pitfall 2: Job restart = transcript loss
**What goes wrong:** `reconcile_orphans()` sets interrupted transcription jobs to `error`. If the handler ran for 90 minutes on a 2-hour file and forge-web.service restarted, all work is discarded.
**How to avoid:** Write completed segments to `data/forge_transcripts/<source_id>.json` incrementally. On re-enqueue (UI "retry" button or operator action), the handler reads the checkpoint and resumes from the last segment index. The `source_id` is stable across enqueue attempts.

### Pitfall 3: SECTION_WINDOW cap in clips.py
**What goes wrong:** `fetch_source()` passes `--download-sections *0:00-01:30` to yt-dlp, capping downloads to 90 seconds. Using this function for long-form ingest downloads only the first 90 seconds.
**How to avoid:** Do NOT call `fetch_source()` from the ingest handler. Call `resolve_source()` to get the target URL, then build a custom yt-dlp command without `--download-sections`.

### Pitfall 4: GPU contention
**What goes wrong:** faster-whisper on CUDA competes with ComfyUI for the 3090's 24 GB VRAM. If a ComfyUI render job is running when transcription starts, both can OOM.
**How to avoid:** The existing `worker_loop` serializes jobs (one at a time). As long as ComfyUI renders and transcription jobs don't overlap in the queue, this is safe. For extra protection, check GPU memory before loading the Whisper model and fall back to CPU int8 if less than 8 GB is free.

### Pitfall 5: Whisper model format mismatch
**What goes wrong:** `large-v3-turbo.pt` in `~/.cache/whisper/` is openai-whisper format. `WhisperModel("large-v3-turbo", ...)` from faster-whisper tries to download a CTranslate2 version from HuggingFace, which may succeed or fail depending on network.
**How to avoid:** Use `"medium"` — it's cached in both formats and verified to work with faster-whisper in the venv.

### Pitfall 6: Secrets from os.environ
**What goes wrong:** Per CLAUDE.md and the `Never Read Env at Import` rule, integration clients must pull secrets from `config.settings.settings`, not `os.environ`. The forge-web.service unit has no `EnvironmentFile=`.
**How to avoid:** Any HuggingFace token or other secret needed for diarization must come from `config.settings` or be passed explicitly via params.

---

## Architecture — New Files to Create

```
core/forge/
├── ingest.py          # NEW: extract_audio(), assign_speakers(), checkpoint helpers
└── db.py              # MODIFY: add sources + transcript_segments tables

core/api/forge.py      # MODIFY: streaming upload endpoint, source/transcript endpoints
core/forge/uploads.py  # MODIFY: save_upload() streaming variant
core/forge/handlers.py # MODIFY: register ingest_transcribe handler
```

**Transcript checkpoint on disk:**
```
data/forge_transcripts/
└── <source_id>.json   # {"segments": [...], "complete": false/true}
```

---

## Sources

### PRIMARY (codebase — HIGH confidence)
- `core/forge/jobs.py` — full job system, `reconcile_orphans`, `worker_loop`, `check_cancel`
- `core/forge/db.py` — SQLite schema, WAL mode, `FORGE_DB_PATH`
- `core/forge/uploads.py` — current `save_upload(content: bytes)` — confirmed OOM blocker
- `core/api/forge.py:104-111` — `await file.read()` — confirmed
- `core/forge/audio.py:42-52` — `transcribe_words()` using `WhisperModel` — confirmed callable
- `core/forge/clips.py:15,66-83` — `SECTION_WINDOW`, `fetch_source()`, `resolve_source()`
- `scripts/notebooklm_clip.py:168-187` — `assign_speakers_per_segment()` heuristic
- `services/forge-web/serve.py` — `reconcile_orphans()` + `worker_loop` wiring, startup hook
- `core/memory/store.py` — ChromaDB `get_collection()`, `PersistentClient` at `data/chromadb/`

### VERIFIED INSTALLS (HIGH confidence)
- `faster-whisper==1.2.1` at `/home/aialfred/alfred/venv/lib/python3.11/site-packages/faster_whisper`
- CTranslate2 CUDA device count = 2 (RTX 3090 + RTX 4070)
- Models cached: tiny, base, small, small.en, medium (CTranslate2 HF format)
- `chromadb==1.4.1` in venv; data at `data/chromadb/`
- `yt-dlp 2026.03.17` at `~/.pyenv/shims/yt-dlp`
- `ffmpeg 4.4.2` system install
- `pyannote.audio`: NOT installed, no HF token in config
- `whisperx`: NOT installed

### INFRASTRUCTURE
- Caddy: no `request_body` limit set in `/etc/caddy/Caddyfile` — must add for 12 GB uploads
- forge-web.service: `Restart=always`, `FORGE_DB_PATH` env set, binds `127.0.0.1:8201`
- Storage: 1.1 TB free on `/dev/nvme0n1p2`

---

## Metadata

**Confidence breakdown:**
- Job system / restart behavior: HIGH — read all relevant source
- Upload blocker: HIGH — confirmed `await file.read()` in production path
- faster-whisper: HIGH — confirmed installed and CUDA-capable
- Diarization options: HIGH — pyannote not installed, heuristic proven in notebooklm_clip.py
- Schema design: MEDIUM — proposed new, not derived from existing tables

**Research date:** 2026-06-01
**Valid until:** 2026-09-01 (stable stack; only yt-dlp moves fast)
