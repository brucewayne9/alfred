"""
topic_clip.py — Segment-cutting, concatenation, caption, branding, and variant engine.

Cuts individual transcript segments out of an ingested audio or video source
with sample-accurate, sync-safe re-encoding (never -c copy), concatenates
non-contiguous segments into one uniform file, enforces the 10–60s duration
band BEFORE any cutting, applies safe caption overlays, builds structurally
distinct variant assemblies (Tier 1), and produces branded 9:16 masters.

Public API
----------
    _detect_has_video(src)                              -> bool
    _cut_segment(src, start_s, end_s, out_path, has_video) -> Path
    _concat_segments(seg_paths, out_path, has_video, work_dir) -> Path
    enforce_duration(segments, min_s, max_s)            -> list[dict]
    _safe_drawtext(text)                                -> str
    overlay_captions(body_path, caption_text, out_path) -> Path
    _build_variant_assemblies(segments, variant_count)  -> list[dict]
    _synthesise_visual(caption_seed, audio_path, out_path, work_dir) -> Path
    assemble_variant(source_path, has_video, variant, caption_text,
                     out_path, work_dir)                -> Path
    render(params, out_path)                            -> Path
"""
from __future__ import annotations

import copy
import random
import subprocess
import textwrap
from pathlib import Path

# Reuse branding from film_montage — do NOT redeclare LOGO_PATH or make_branded.
from core.forge.renderers.film_montage import make_branded, LOGO_PATH  # noqa: F401

# DejaVuSans-Bold is confirmed present on 105; Hanken Grotesk is not installed
# (fc-list | grep -i hanken returns nothing at plan time).
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


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


# ---------------------------------------------------------------------------
# Caption sanitiser — safe for ffmpeg drawtext (CLIP-03)
# ---------------------------------------------------------------------------


def _safe_drawtext(text: str) -> str:
    """Sanitise *text* for use inside an ffmpeg drawtext filter value.

    ffmpeg drawtext parses the text string with its own escaping rules.
    Characters that crash or corrupt the filter:
      - straight apostrophe / single-quote  ( ' )  → replaced with U+2019 RIGHT SINGLE QUOTATION MARK
      - backslash  ( \\ )                          → escaped to \\\\
      - colon      ( : )                           → escaped to \\:
      - percent    ( % )                           → escaped to \\%

    The sanitised string is also word-wrapped to ≤38 chars/line using
    ``textwrap.fill`` so long captions don't overflow the frame width.
    Newlines in the wrapped output are honoured verbatim by drawtext.

    This is a pure string function — no subprocess, no ffmpeg.  Easy to unit test.
    """
    # Order matters: backslash must be escaped first to avoid double-escaping.
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "’")   # U+2019 RIGHT SINGLE QUOTATION MARK
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    text = textwrap.fill(text, width=38)
    return text


def overlay_captions(
    body_path: Path,
    caption_text: str,
    out_path: Path,
    font_size: int = 52,
) -> Path:
    """Burn a bottom-third caption onto an already-9:16 video.

    If *caption_text* is empty or whitespace the body is copied to *out_path*
    unchanged (no drawtext pass).  Raises RuntimeError with the ffmpeg stderr
    tail on non-zero exit.

    NOTE: Per-segment timed word-level captions are deferred to Phase 13.
    For a multi-segment clip use the first sentence of the first segment's
    text as the single overlay line (split on ". ", cap 120 chars); the caller
    is responsible for passing that pre-extracted caption_text.
    """
    body_path = Path(body_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not caption_text or not caption_text.strip():
        # No caption — plain re-encode copy (ensures uniform codec parameters).
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(body_path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"overlay_captions (passthrough) failed: {proc.stderr[-500:]}"
            )
        return out_path

    safe = _safe_drawtext(caption_text)
    drawtext_filter = (
        f"drawtext=fontfile={FONT_PATH}"
        f":fontsize={font_size}"
        ":fontcolor=white"
        ":bordercolor=black"
        ":borderw=3"
        ":x=(w-text_w)/2"
        ":y=h-text_h-120"
        f":text='{safe}'"
    )
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(body_path),
            "-vf", drawtext_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"overlay_captions failed: {proc.stderr[-500:]}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Structural variant strategies — Tier 1 (CLIP-02)
# ---------------------------------------------------------------------------


def _build_variant_assemblies(
    segments: list[dict],
    variant_count: int,
) -> list[dict]:
    """Return *variant_count* structurally distinct segment orderings.

    enforce_duration() is applied to the BASE set first.  Each variant is an
    independent deep copy — mutating one never bleeds into another.

    Strategies (in order):
      V0  "original"    — curated order as-is
      V1  "reverse"     — segments reversed
      V2  "hook-first"  — highest-score segment moved to index 0
      V3  "trimmed"     — each segment nudged +0.5s start / -0.5s end,
                          re-enforced, windows floored at 1 s
      V4+ "shuffle-{i}" — deterministic shuffle via random.Random(i)

    If variant_count == 1, only V0 is returned.
    variant_count is clamped to [1, 10].
    """
    variant_count = max(1, min(10, variant_count))

    # Apply duration guard to the canonical order before deriving variants.
    base = enforce_duration(segments)

    def _deep(segs: list[dict]) -> list[dict]:
        return copy.deepcopy(segs)

    variants: list[dict] = []

    # V0 — original
    variants.append({"label": "original", "segments": _deep(base)})
    if variant_count == 1:
        return variants

    # V1 — reverse
    variants.append({"label": "reverse", "segments": list(reversed(_deep(base)))})
    if variant_count <= 2:
        return variants[:variant_count]

    # V2 — hook-first (highest score first)
    hf = _deep(base)
    if len(hf) > 1:
        best_idx = max(range(len(hf)), key=lambda i: hf[i].get("score", 0.0))
        hf.insert(0, hf.pop(best_idx))
    variants.append({"label": "hook-first", "segments": hf})
    if variant_count <= 3:
        return variants[:variant_count]

    # V3 — trimmed (nudge each segment window, re-enforce, floor at 1 s)
    trimmed = _deep(base)
    for seg in trimmed:
        window = seg["end_s"] - seg["start_s"]
        if window > 2.0:
            seg["start_s"] = seg["start_s"] + 0.5
            seg["end_s"] = seg["end_s"] - 0.5
        # Floor: each segment must be at least 1 s.
        if seg["end_s"] - seg["start_s"] < 1.0:
            seg["end_s"] = seg["start_s"] + 1.0
    try:
        trimmed = enforce_duration(trimmed)
    except ValueError:
        # If trimming pushed total below min, fall back to original order.
        trimmed = _deep(base)
    variants.append({"label": "trimmed", "segments": trimmed})
    if variant_count <= 4:
        return variants[:variant_count]

    # V4+ — deterministic shuffles
    shuffle_idx = 0
    while len(variants) < variant_count:
        shuffled = _deep(base)
        rng = random.Random(shuffle_idx)
        rng.shuffle(shuffled)
        variants.append({"label": f"shuffle-{shuffle_idx}", "segments": shuffled})
        shuffle_idx += 1

    return variants[:variant_count]


# ---------------------------------------------------------------------------
# Audio-only visual synthesis (CLIP-01 audio-only path)
# ---------------------------------------------------------------------------


def _synthesise_visual(
    caption_seed: str,
    audio_path: Path,
    out_path: Path,
    work_dir: Path,
) -> Path:
    """Build a 9:16 video from a generated still + the concatenated audio.

    For audio-only sources that have no video track, this produces a 1080x1920
    background still via ComfyUI Cloud (research Option B) and loops it under
    the audio.

    ComfyUI is lazy-imported inside this function to keep the module importable
    in test environments without the ComfyUI / rucktalk_common dependency.
    NEVER inline PIL — project hard rule. If run_comfyui_cloud returns None/falsey,
    fall back to a solid black background (color=black ffmpeg source).

    Raises RuntimeError with stderr tail on failure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path)

    # Lazy import — never at module level (keeps imports safe in test env)
    try:
        from scripts.rucktalk_common import run_comfyui_cloud  # noqa: PLC0415
        img = run_comfyui_cloud(caption_seed, width=1080, height=1920)
    except Exception:
        img = None

    if img and Path(img).exists():
        img_input = ["-loop", "1", "-i", str(img)]
    else:
        # Solid black background via ffmpeg lavfi color source.
        img_input = [
            "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30",
        ]

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *img_input,
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,"
               "crop=1080:1920,fps=30,format=yuv420p",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"_synthesise_visual failed: {proc.stderr[-500:]}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Full single-variant assembly pipeline
# ---------------------------------------------------------------------------


def assemble_variant(
    source_path: str | Path,
    has_video: bool,
    variant: dict,
    caption_text: str,
    out_path: str | Path,
    work_dir: str | Path,
) -> Path:
    """Assemble one structural variant into a branded 9:16 master.

    Pipeline:
      1. Cut each segment from source_path into work_dir.
      2. Concat all cut segments -> body (video) OR audio-only track.
      3. If audio-only: _synthesise_visual to get a picture for the body.
      4. overlay_captions on the body.
      5. Extract concat audio as hook file.
      6. make_branded(captioned, hook_audio, "", out_path) — logo + final mux.

    Returns the final branded master path.
    """
    source_path = Path(source_path)
    out_path = Path(out_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    segments = variant["segments"]
    label = variant.get("label", "v")

    # 1. Cut each segment.
    seg_paths: list[Path] = []
    ext = ".mp4" if has_video else ".m4a"
    for i, seg in enumerate(segments):
        sp = _cut_segment(
            source_path,
            seg["start_s"],
            seg["end_s"],
            work_dir / f"seg_{label}_{i:03d}{ext}",
            has_video=has_video,
        )
        seg_paths.append(sp)

    # 2. Concat all cuts -> body.
    concat_ext = ".mp4" if has_video else ".m4a"
    concat_out = work_dir / f"concat_{label}{concat_ext}"
    body = _concat_segments(seg_paths, concat_out, has_video=has_video, work_dir=work_dir / f"cwork_{label}")

    # 3. Audio-only → synthesise a visual background.
    if not has_video:
        visual_out = work_dir / f"visual_{label}.mp4"
        body = _synthesise_visual(
            caption_text or label,
            body,
            visual_out,
            work_dir,
        )

    # 4. Overlay captions.
    captioned = work_dir / f"captioned_{label}.mp4"
    overlay_captions(body, caption_text, captioned)

    # 5. Extract concatenated audio as hook file for make_branded mux.
    hook_audio = work_dir / f"hook_{label}.m4a"
    _extract_audio(captioned, hook_audio)

    # 6. Brand: logo overlay + hook audio mux.
    make_branded(captioned, hook_audio, "", out_path)
    return out_path


def _extract_audio(video_path: Path, out_path: Path) -> Path:
    """Extract the audio track of *video_path* to *out_path* as AAC .m4a.

    Used internally to produce the hook_audio file for make_branded.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-vn",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"_extract_audio failed: {proc.stderr[-500:]}"
        )
    return out_path


# ---------------------------------------------------------------------------
# Top-level render() entry point — matches film_montage.render() signature
# ---------------------------------------------------------------------------


def render(params: dict, out_path: str | Path) -> Path:
    """Assemble one structural variant into a branded 9:16 topic clip master.

    Matches the ``film_montage.render(params, out_path)`` signature so both
    renderers can be called uniformly from the plan-03 handler.

    Required params keys:
      source_id     — forge ingest source ID
      segments      — list of segment dicts with start_s / end_s / text / score

    Optional params keys:
      variant_index — which structural variant to build (default 0 = original)
      caption       — caption text override; if absent, the first sentence of
                      the first segment's text is used (split on ". ", cap 120 chars)

    The plan-03 handler calls render() once per structural variant; the per-variant
    pixel-level multiply loop lives in plan-03.

    Returns the final branded master Path.
    """
    import tempfile
    import shutil

    from core.forge import ingest

    source_id = params["source_id"]
    segments = params["segments"]
    variant_index = int(params.get("variant_index", 0))
    caption = params.get("caption", "")

    # Resolve source file path.
    src_row = ingest.get_source(source_id)
    if src_row is None:
        raise RuntimeError(f"source not found: {source_id!r}")
    source_path = Path(src_row["file_path"])
    if not source_path.exists():
        raise RuntimeError(f"source file missing on disk: {source_path}")

    has_video = _detect_has_video(source_path)

    # Derive caption from first segment if not provided.
    if not caption and segments:
        raw_text = segments[0].get("text", "")
        first_sentence = raw_text.split(". ")[0][:120]
        caption = first_sentence

    # Build variant assemblies and select by index.
    assemblies = _build_variant_assemblies(segments, variant_index + 1)
    variant = assemblies[variant_index]

    work = Path(tempfile.mkdtemp(prefix="forge_clip_"))
    try:
        return assemble_variant(
            source_path=source_path,
            has_video=has_video,
            variant=variant,
            caption_text=caption,
            out_path=out_path,
            work_dir=work,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
