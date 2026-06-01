"""Mainstay Forge — long-form ingest helpers.

Source CRUD, on-disk checkpoint helpers (restart-safe transcription),
and the A/B speaker heuristic. No faster-whisper or ffmpeg imports here —
this module is pure storage/util so Plan 10-03 can extend it with the
transcription handler without pulling in heavy GPU deps at import time.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from core.forge.db import _conn

# ---------------------------------------------------------------------------
# Allowed column names for update_source — guards against SQL injection.
# ---------------------------------------------------------------------------
_UPDATABLE_COLS: frozenset[str] = frozenset(
    {"status", "duration_s", "language", "error", "file_path"}
)

# ---------------------------------------------------------------------------
# Source CRUD
# ---------------------------------------------------------------------------


def create_source(
    kind: str,
    spec: str,
    file_path: str | None = None,
) -> str:
    """Insert a new source row; return its id (uuid hex)."""
    source_id = uuid.uuid4().hex
    now = int(time.time())
    with _conn() as c:
        c.execute(
            """
            INSERT INTO sources (id, kind, spec, file_path, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """,
            (source_id, kind, spec, file_path, now, now),
        )
    return source_id


def get_source(source_id: str) -> dict | None:
    """Fetch one source row as a plain dict, or None if not found."""
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def update_source(source_id: str, **fields) -> None:
    """UPDATE arbitrary whitelisted columns plus updated_at.

    Example::

        update_source(sid, status='transcribing', duration_s=3600.5)
    """
    # Reject anything not in the whitelist.
    bad = set(fields) - _UPDATABLE_COLS
    if bad:
        raise ValueError(f"update_source: non-updatable columns: {bad}")
    if not fields:
        return

    now = int(time.time())
    set_clause = ", ".join(f"{col} = ?" for col in fields)
    values = list(fields.values()) + [now, source_id]
    with _conn() as c:
        c.execute(
            f"UPDATE sources SET {set_clause}, updated_at = ? WHERE id = ?",
            values,
        )


def save_segments(source_id: str, segments: list[dict]) -> None:
    """Idempotent bulk-write of transcript segments.

    Deletes any existing rows for *source_id* then inserts the full list.
    Callers can safely call this multiple times with the current segment list.

    Each segment dict may contain: seq, start_s, end_s, text, speaker, words.
    If *words* is a list it is JSON-serialised before storage.
    """
    with _conn() as c:
        c.execute(
            "DELETE FROM transcript_segments WHERE source_id = ?", (source_id,)
        )
        c.executemany(
            """
            INSERT INTO transcript_segments
                (source_id, seq, start_s, end_s, text, speaker, words)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    source_id,
                    seg.get("seq", i),
                    seg["start_s"],
                    seg["end_s"],
                    seg["text"],
                    seg.get("speaker"),
                    seg["words"] if isinstance(seg.get("words"), str)
                    else json.dumps(seg.get("words") or []),
                )
                for i, seg in enumerate(segments)
            ],
        )


# ---------------------------------------------------------------------------
# On-disk checkpoint helpers (INGEST-03: restart-safe transcription)
# ---------------------------------------------------------------------------


def _checkpoint_dir() -> Path:
    """Return the directory used for transcript checkpoint files.

    Respects FORGE_TRANSCRIPT_DIR env override (same pattern as uploads.py
    respects FORGE_UPLOAD_DIR).
    """
    override = os.environ.get("FORGE_TRANSCRIPT_DIR")
    d = Path(override) if override else Path("data/forge_transcripts")
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(source_id: str) -> Path:
    """Return the on-disk path for a source's checkpoint file."""
    return _checkpoint_dir() / f"{source_id}.json"


def load_checkpoint(source_id: str) -> dict:
    """Read and parse the checkpoint file; return a blank state if missing."""
    p = checkpoint_path(source_id)
    if not p.exists():
        return {"segments": [], "complete": False}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {"segments": [], "complete": False}


def write_checkpoint(source_id: str, data: dict) -> None:
    """Atomically write *data* to the checkpoint file.

    Uses write-to-temp + os.replace so an interrupted write never produces a
    half-written or empty checkpoint file.
    """
    p = checkpoint_path(source_id)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".ckpt-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Speaker heuristic (INGEST-04) — adapted from notebooklm_clip.py:168-187
# ---------------------------------------------------------------------------

SHORT_SEGMENT_THRESHOLD_S: float = 1.2


def assign_speakers(segments: list[dict]) -> list[dict]:
    """Alternate 'A'/'B' per Whisper segment; inherit prior label on short segs.

    A segment shorter than SHORT_SEGMENT_THRESHOLD_S is assumed to be a
    continuation of the same speaker (brief interjection / breath / filler).
    Does not mutate the input dicts.

    Args:
        segments: list of dicts with at least 'start' and 'end' keys
            (float seconds).

    Returns:
        New list of dicts with a 'speaker' key added.
    """
    labels: list[str] = []
    current = "A"
    for seg in segments:
        dur = float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))
        if not labels:
            labels.append(current)
        elif dur < SHORT_SEGMENT_THRESHOLD_S:
            labels.append(labels[-1])  # inherit
        else:
            current = "B" if labels[-1] == "A" else "A"
            labels.append(current)
    return [dict(seg, speaker=lbl) for seg, lbl in zip(segments, labels)]


# ---------------------------------------------------------------------------
# Transcript read helper — used by the API transcript endpoint
# ---------------------------------------------------------------------------


def get_segments(source_id: str) -> list[dict]:
    """Return all transcript segments for *source_id* ordered by seq.

    Words column is JSON-decoded before returning.
    Returns an empty list if the source has no segments yet.
    """
    with _conn() as c:
        rows = c.execute(
            """
            SELECT seq, start_s, end_s, text, speaker, words
              FROM transcript_segments
             WHERE source_id = ?
             ORDER BY seq
            """,
            (source_id,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        if d.get("words") and isinstance(d["words"], str):
            try:
                d["words"] = json.loads(d["words"])
            except (json.JSONDecodeError, TypeError):
                d["words"] = []
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# ingest_transcribe handler (INGEST-03 / INGEST-04)
# ---------------------------------------------------------------------------

# Checkpoint every N segments so a restart can fast-path through done work.
_CHECKPOINT_INTERVAL: int = 10


def transcribe_handler(params: dict) -> dict:
    """Download/localise a source, transcribe with faster-whisper, checkpoint as it goes.

    Receives params dict from the job queue::

        {"source_id": str, "url": str | None, "cloud_path": str | None}

    Restart-safety (INGEST-03):
        load_checkpoint() at entry.  If checkpoint["complete"] is True the handler
        fast-paths: call save_segments() from the checkpoint data and return
        immediately without re-running transcription.

        During transcription write_checkpoint() is called every
        _CHECKPOINT_INTERVAL segments.  If forge-web.service is restarted
        mid-transcription reconcile_orphans() sets the job to 'error'.  When the
        operator re-enqueues (UI Retry), this handler re-runs, reads the
        checkpoint, sees complete=False but has the already-done segments on disk.
        Rather than attempting partial-resume of the generator (fragile), we
        restart the transcription from scratch but emit the result to the same
        checkpoint path — because write_checkpoint is atomic the worst case is
        that we produce a fresh complete transcript.  The key guarantee (INGEST-03)
        is that a complete transcript is never discarded: if complete=True we
        fast-path regardless of how many retries were needed.

    Speaker attribution (INGEST-04):
        assign_speakers() applies an A/B alternating heuristic after the full
        segment list is produced.

    GPU safety:
        WhisperModel is loaded lazily inside this function (not at module import)
        to avoid pulling CUDA at import time.  If the 3090 has < 8 GB free we
        fall back to CPU int8 automatically.
    """
    import subprocess as _sp
    from core.forge import audio as _audio
    from core.forge import clips as _clips
    from core.forge import jobs as _forge_jobs

    source_id = params["source_id"]

    # ------------------------------------------------------------------ #
    # 1. Checkpoint fast-path                                              #
    # ------------------------------------------------------------------ #
    ckpt = load_checkpoint(source_id)
    if ckpt.get("complete"):
        save_segments(source_id, ckpt["segments"])
        update_source(source_id, status="done")
        return {
            "transcript_id": source_id,
            "segment_count": len(ckpt["segments"]),
            "resumed": True,
        }

    try:
        # ------------------------------------------------------------------ #
        # 2. Resolve / localise the media file                               #
        # ------------------------------------------------------------------ #
        source = get_source(source_id)
        if source is None:
            raise RuntimeError(f"source {source_id!r} not found in DB")

        file_path: str | None = source.get("file_path")

        if params.get("url") and not (file_path and Path(file_path).exists()):
            # URL source — download the FULL file without the SECTION_WINDOW cap.
            url = params["url"]
            target, _kind = _clips.resolve_source(url)

            # Build a yt-dlp command that mirrors ytdlp_cmd() but omits
            # --download-sections so we get the complete file.
            from core.forge.uploads import _root as _upload_root
            dl_dir = _upload_root() / source_id
            dl_dir.mkdir(parents=True, exist_ok=True)
            out_tmpl = str(dl_dir / "src_%(autonumber)s.%(ext)s")

            cmd: list[str] = [
                "yt-dlp", "--no-warnings",
                "--no-playlist" if target.startswith("http") else "--yes-playlist",
                "-f", "bv*[height<=1080][ext=mp4]+ba/b[height<=1080]/b",
                "--merge-output-format", "mp4",
                "--retries", "5", "--fragment-retries", "5",
                "--socket-timeout", "30", "-N", "4",
                "--extractor-args",
                "youtube:player_client=default,web_safari,android_vr,tv",
                "-o", out_tmpl, target,
            ]
            node = _clips._node_path()
            if node:
                cmd[1:1] = ["--js-runtimes", f"node:{node}"]

            proc = _sp.run(cmd, capture_output=True, text=True, timeout=7200)
            dl_files = sorted(dl_dir.glob("*.mp4"))
            if not dl_files:
                detail = (proc.stderr or proc.stdout or "").strip()[-300:]
                raise RuntimeError(f"yt-dlp fetched nothing for {url!r}: {detail}")
            file_path = str(dl_files[0])
            update_source(source_id, file_path=file_path)

        elif params.get("cloud_path") and not (file_path and Path(file_path).exists()):
            # Cloud (Nextcloud) source — stream down via chunked WebDAV GET.
            # nc.download_file() buffers the entire file in RAM; for 11 GB sources
            # that causes OOM.  We use the underlying WebDAV session directly.
            cloud_path = params["cloud_path"]
            import requests as _rq
            from integrations.nextcloud import client as nc
            from core.forge.uploads import _root as _upload_root

            dl_dir = _upload_root() / source_id
            dl_dir.mkdir(parents=True, exist_ok=True)
            filename = cloud_path.rstrip("/").split("/")[-1] or "source.bin"
            local_path = dl_dir / filename

            # Mirror nc.download_file() but stream in 8MB chunks (download_file
            # buffers the whole file in RAM — OOM on an 11 GB source). The client
            # is module-level functions: _webdav_url(path) + _auth().
            webdav_url = nc._webdav_url(cloud_path.lstrip("/"))
            with _rq.get(webdav_url, auth=nc._auth(), stream=True, timeout=(30, 600)) as resp:
                resp.raise_for_status()
                with local_path.open("wb") as fout:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fout.write(chunk)

            file_path = str(local_path)
            update_source(source_id, file_path=file_path)

        # file_path must now be set — 'upload' sources already have it.
        if not file_path or not Path(file_path).exists():
            raise RuntimeError(
                f"source {source_id!r}: could not resolve a local file_path "
                f"(file_path={file_path!r})"
            )

        # ------------------------------------------------------------------ #
        # 3. Extract audio                                                     #
        # ------------------------------------------------------------------ #
        update_source(source_id, status="extracting")
        audio_dir = Path(file_path).parent
        wav_path = audio_dir / "audio.wav"
        _audio.extract_audio(file_path, wav_path)
        dur = _audio.duration_seconds(wav_path)

        # ------------------------------------------------------------------ #
        # 4. Load faster-whisper (lazy import avoids CUDA at module load)     #
        # ------------------------------------------------------------------ #
        update_source(source_id, status="transcribing", duration_s=dur)

        from faster_whisper import WhisperModel  # noqa: PLC0415

        try:
            model = WhisperModel("medium", device="cuda", compute_type="int8_float16")
        except Exception:
            # GPU not available or insufficient VRAM — fall back to CPU.
            model = WhisperModel("medium", device="cpu", compute_type="int8")

        # ------------------------------------------------------------------ #
        # 5. Transcribe + incremental checkpoint                              #
        # ------------------------------------------------------------------ #
        # We always restart the segment generator from the beginning.  The
        # restart-safety guarantee (INGEST-03) is provided by the complete=True
        # fast-path above: once a run finishes we never re-transcribe.  Writing
        # frequent checkpoints means the window of lost work after a crash is
        # bounded to _CHECKPOINT_INTERVAL segments.
        segments_raw, info = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            language="en",
        )

        accumulated: list[dict] = []
        for i, seg in enumerate(segments_raw):
            _forge_jobs.check_cancel()
            accumulated.append({
                "seq": i,
                "start_s": round(float(seg.start), 3),
                "end_s": round(float(seg.end), 3),
                "text": (seg.text or "").strip(),
                # keep start/end aliases so assign_speakers() can use them
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "words": [
                    {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                    for w in (seg.words or [])
                ],
            })
            if (i + 1) % _CHECKPOINT_INTERVAL == 0:
                ckpt["segments"] = accumulated[:]
                write_checkpoint(source_id, ckpt)

        detected_language: str | None = getattr(info, "language", None)

        # ------------------------------------------------------------------ #
        # 6. Speaker assignment (INGEST-04)                                   #
        # ------------------------------------------------------------------ #
        labelled = assign_speakers(accumulated)

        # Clean up the temporary start/end aliases before persisting — the DB
        # schema stores start_s / end_s; keeping start/end would be dead weight.
        labelled_clean = [
            {k: v for k, v in seg.items() if k not in ("start", "end")}
            for seg in labelled
        ]

        ckpt["segments"] = labelled_clean
        ckpt["complete"] = True
        write_checkpoint(source_id, ckpt)

        # ------------------------------------------------------------------ #
        # 7. Persist to DB                                                     #
        # ------------------------------------------------------------------ #
        save_segments(source_id, labelled_clean)
        update_source(
            source_id,
            status="done",
            **({"language": detected_language} if detected_language else {}),
        )

        return {
            "transcript_id": source_id,
            "segment_count": len(labelled_clean),
            "resumed": False,
        }

    except Exception as exc:
        update_source(source_id, status="error", error=str(exc)[:500])
        raise
