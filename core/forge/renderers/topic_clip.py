"""
topic_clip.py — Segment-cutting and concatenation engine for CLIP-01.

Cuts individual transcript segments out of an ingested audio or video source
with sample-accurate, sync-safe re-encoding (never -c copy), concatenates
non-contiguous segments into one uniform file, and enforces the 10–60s
duration band BEFORE any cutting.

Public API
----------
    _detect_has_video(src)          -> bool
    _cut_segment(src, start_s, end_s, out_path, has_video) -> Path
    _concat_segments(seg_paths, out_path, has_video, work_dir) -> Path
    enforce_duration(segments, min_s, max_s) -> list[dict]
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _detect_has_video(src: str | Path) -> bool:
    """Return True if *src* contains at least one video stream.

    Mirrors the exact subprocess pattern used by multiply._has_audio() —
    same ffprobe invocation shape, capture_output, text=True.
    Audio-only .mp3/.m4a return False; .mp4/.mov with a video track return True.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _cut_segment(
    src: str | Path,
    start_s: float,
    end_s: float,
    out_path: str | Path,
    has_video: bool,
) -> Path:
    """Cut ONE segment from *src* spanning [start_s, end_s).

    ALWAYS re-encodes — never uses -c copy.  Copy-mode uses keyframe seek and
    bleeds 1–2 s of the previous segment into speech (sync hazard).

    Seek is placed BEFORE -i (fast seek) then -t for duration.

    Video path  → 1080x1920 9:16, yuv420p, 30 fps, libx264/aac, out .mp4
    Audio-only  → aac 192k/44100 Hz stereo, no video, out .m4a

    Raises RuntimeError (last 500 chars of stderr) on non-zero exit.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_s - start_s

    if has_video:
        # Reuse EXACT scale/crop filter string from film_montage._cut_segment
        # so 9:16 framing matches all other Forge formats.
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start_s), "-i", str(src),
            "-t", str(duration),
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,fps=30,format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ]
        if out_path.suffix not in (".mp4", ".mov"):
            out_path = out_path.with_suffix(".mp4")
    else:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start_s), "-i", str(src),
            "-t", str(duration),
            "-vn",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ]
        if out_path.suffix not in (".m4a", ".mp4", ".aac"):
            out_path = out_path.with_suffix(".m4a")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"_cut_segment failed [{src} {start_s}–{end_s}s]: {proc.stderr[-500:]}"
        )
    return out_path


def _concat_segments(
    seg_paths: list[Path],
    out_path: Path,
    has_video: bool,
    work_dir: Path,
) -> Path:
    """Concatenate already-cut, uniform segments using the ffmpeg concat demuxer.

    Every input segment was re-encoded to identical params in _cut_segment
    (yuv420p/30fps/aac 44100/2ch), so the demuxer is safe and avoids A/V drift.

    Replicates the concat pattern from film_montage.render (lines ~163-173):
      concat.txt with `file '{abspath}'` lines + -f concat -safe 0.

    Re-encodes the concat output (libx264/aac) as film_montage does.
    Raises RuntimeError with stderr tail on failure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    concat_txt = work_dir / "concat.txt"
    concat_txt.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in seg_paths)
    )

    if has_video:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c:v", "libx264", "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-vn",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"_concat_segments failed: {proc.stderr[-500:]}"
        )
    return out_path


def enforce_duration(
    segments: list[dict],
    min_s: float = 10.0,
    max_s: float = 60.0,
) -> list[dict]:
    """Enforce the 10–60 s duration band on a list of segment dicts.

    Each dict must have 'start_s' and 'end_s' float keys.

    Rules:
      - total < min_s  → raise ValueError
      - total > max_s  → return a copy with the LAST segment's end_s trimmed
                         down by the overage (floored at start_s + 1.0 s)
      - otherwise      → return a shallow copy of the list unchanged

    Never mutates the input list.  Must be called BEFORE any cutting.
    """
    total = sum(s["end_s"] - s["start_s"] for s in segments)

    if total < min_s:
        raise ValueError(
            f"total segment duration {total:.1f}s < {min_s}s minimum"
        )

    if total <= max_s:
        return [dict(s) for s in segments]

    # Trim the last segment's end_s by the overage.
    overage = total - max_s
    result = [dict(s) for s in segments]
    last = result[-1]
    new_end = last["end_s"] - overage
    # Floor: segment must be at least 1 s long.
    new_end = max(last["start_s"] + 1.0, new_end)
    last["end_s"] = new_end
    return result
