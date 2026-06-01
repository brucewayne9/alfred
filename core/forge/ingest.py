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
