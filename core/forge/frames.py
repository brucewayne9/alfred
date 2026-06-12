"""Mainstay Forge — single-frame thumbnails for clip previews.

Auto-Clips shows a still from the source at each candidate's moment so the team
can eyeball a clip before committing the (slow) render. Frames are extracted
with ffmpeg on first request and cached to disk keyed by source + timestamp, so
repeat loads are instant and a re-score never re-extracts.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _frames_root() -> Path:
    override = os.environ.get("FORGE_FRAMES_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "data" / "forge_frames"


def clamp_t(t: float, duration: float | None) -> float:
    """Clamp a requested timestamp into [0, duration] (or just >=0 if unknown)."""
    t = max(0.0, float(t))
    if duration and duration > 0:
        # Pull back a hair from the very end so ffmpeg still lands on a frame.
        t = min(t, float(duration) - 0.2)
    return max(0.0, round(t, 1))


def frame_cache_path(source_id: str, t: float) -> Path:
    """Deterministic cache path for a source's frame at time *t*.

    Keyed by deciseconds so 12.34 and 12.3 resolve to the same cached frame.
    """
    ds = int(round(float(t) * 10))
    return _frames_root() / source_id / f"{ds}.jpg"


def extract_frame(file_path: str, t: float, out_path: Path) -> bool:
    """Extract one JPEG frame at *t* seconds from *file_path* into *out_path*.

    Fast seek (-ss before -i). Returns True on success. No-op (True) if the
    frame is already cached.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-ss", f"{max(0.0, float(t)):.3f}",
        "-i", file_path, "-frames:v", "1", "-q:v", "3",
        "-vf", "scale=540:-2", str(out_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
