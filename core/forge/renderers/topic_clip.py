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
    w: int = 1080,
    h: int = 1920,
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
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},fps=30,format=yuv420p",
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

    # Trim segments from the tail until total <= max_s.
    # A single pass trimming only the last segment fails when the overage is
    # larger than the last segment's duration (the floor at start_s+1s leaves
    # the total still above max_s). Iterate from last to first.
    result = [dict(s) for s in segments]
    for idx in range(len(result) - 1, -1, -1):
        current_total = sum(s["end_s"] - s["start_s"] for s in result)
        if current_total <= max_s:
            break
        overage = current_total - max_s
        seg = result[idx]
        new_end = seg["end_s"] - overage
        # Floor: each segment must be at least 1 s long.
        new_end = max(seg["start_s"] + 1.0, new_end)
        seg["end_s"] = new_end
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
# Timed rolling captions — follow the speaker's words (CLIP-03, phase-13 pull-forward)
# ---------------------------------------------------------------------------


def _format_ass_time(t: float) -> str:
    """Format *t* seconds as an ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Escape + wrap *text* for an ASS Dialogue line.

    ASS uses ``\\N`` for hard line breaks and treats ``{ }`` as override blocks,
    so curly braces are neutralised.  Lines are wrapped to ≤24 chars (≤2 lines)
    so a phrase reads as a tight lower-third, not a full-width paragraph.
    """
    text = " ".join((text or "").split())            # collapse whitespace
    text = text.replace("{", "(").replace("}", ")")   # kill override blocks
    text = text.replace("\\", "")                      # drop stray backslashes
    wrapped = textwrap.wrap(text, width=24)[:2]        # ≤2 lines
    return "\\N".join(wrapped)


def _build_caption_events(
    variant_segments: list[dict],
    fine_segments: list[dict],
    min_event_s: float = 0.3,
) -> list[tuple[float, float, str]]:
    """Map fine (≈2 s) transcript phrases onto the OUTPUT timeline.

    The assembled clip is the variant's window segments concatenated in order.
    For each window we pull the fine transcript phrases that overlap it and
    rebase their timings to clip-local output time (accounting for window
    reorder/trim).  The result is a rolling caption track that follows speech.

    Falls back to one window-spanning event when a window has no fine phrases.
    Returns ``[(out_start, out_end, text), ...]`` in output order.
    """
    events: list[tuple[float, float, str]] = []
    offset = 0.0
    for w in variant_segments:
        w_start = float(w["start_s"])
        w_end = float(w["end_s"])
        w_dur = max(0.0, w_end - w_start)
        subs = [
            f for f in fine_segments
            if float(f["end_s"]) > w_start and float(f["start_s"]) < w_end
            and (f.get("text") or "").strip()
        ]
        if subs:
            for f in subs:
                ls = max(0.0, float(f["start_s"]) - w_start)
                le = min(w_dur, float(f["end_s"]) - w_start)
                if le - ls < min_event_s:
                    continue
                events.append((offset + ls, offset + le, (f.get("text") or "").strip()))
        elif (w.get("text") or "").strip():
            events.append((offset, offset + w_dur, (w.get("text") or "").strip()))
        offset += w_dur
    return events


# Caption style presets — pick via params["caption_style"].
#   clean   — phrase-by-phrase white bold lower-third (default; reads anywhere)
#   bold    — big ALL-CAPS white, heavier outline (punchy / meme look)
#   karaoke — word-by-word gold highlight that fills as he speaks (TikTok look)
# ASS colours are &HAABBGGRR.  Gold = RGB(255,215,0) -> &H0000D7FF.
CAPTION_STYLES = {
    "clean":   {"font_size": 56, "primary": "&H00FFFFFF", "secondary": "&H00FFFFFF",
                "outline": 4, "upper": False, "karaoke": False},
    "bold":    {"font_size": 82, "primary": "&H00FFFFFF", "secondary": "&H00FFFFFF",
                "outline": 6, "upper": True,  "karaoke": False},
    "karaoke": {"font_size": 70, "primary": "&H0000D7FF", "secondary": "&H00FFFFFF",
                "outline": 5, "upper": True,  "karaoke": True},
}
DEFAULT_CAPTION_STYLE = "clean"

# Position presets — pick via params["caption_position"].  (alignment, marginV)
CAPTION_POSITIONS = {
    "lower":  (2, 300),   # bottom-centre, 300px up (default lower-third)
    "center": (5, 0),     # screen middle
    "upper":  (8, 220),   # top-centre, 220px down
}
DEFAULT_CAPTION_POSITION = "lower"

# Font picker — only families confirmed installed via fc-list on 105.  libass
# resolves by family NAME; unknown names fall back to DejaVu Sans harmlessly.
CAPTION_FONTS = {
    "default": "DejaVu Sans",
    "anton": "Anton",
    "bebas": "Bebas Neue",
    "archivo": "Archivo Black",
    "bungee": "Bungee",
    "blackops": "Black Ops One",
    "dmsans": "DM Sans",
}
DEFAULT_CAPTION_FONT = "default"

# Colour picker — ASS colours are &HAABBGGRR (note: BGR, not RGB).
CAPTION_COLORS = {
    "white": "&H00FFFFFF",
    "gold":  "&H0000D7FF",   # RGB(255,215,0)
    "pink":  "&H009314FF",   # RGB(255,20,147) hot pink
    "lime":  "&H0000FF00",   # RGB(0,255,0)
    "cyan":  "&H00FFFF00",   # RGB(0,255,255)
    "red":   "&H000000FF",   # RGB(255,0,0)
}
DEFAULT_CAPTION_COLOR = None   # None → use the style preset's own primary colour


def _resolve_font(font: str | None) -> str:
    """Map a font key (or family name) to an installed family; default DejaVu Sans."""
    if not font:
        return CAPTION_FONTS[DEFAULT_CAPTION_FONT]
    key = font.lower()
    if key in CAPTION_FONTS:
        return CAPTION_FONTS[key]
    # Already a family name (allow direct passthrough of any installed family).
    return font


def _resolve_color(color: str | None) -> str | None:
    if not color:
        return None
    return CAPTION_COLORS.get(color.lower(), None)


def _style_block(
    style_cfg: dict,
    alignment: int,
    margin_v: int,
    font: str = "DejaVu Sans",
    primary: str | None = None,
    scale: float = 1.0,
) -> str:
    """Build the ASS [V4+ Styles] block for a caption style + position.

    *font* is a family name; *primary* overrides the preset text/highlight colour.
    *scale* proportionally resizes font + bottom margin for non-vertical aspects.
    """
    prim = primary or style_cfg["primary"]
    fs = max(18, int(round(style_cfg["font_size"] * scale)))
    mv = max(20, int(round(margin_v * scale)))
    return (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font},{fs},{prim},"
        f"{style_cfg['secondary']},&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,"
        f"{style_cfg['outline']},1,{alignment},80,80,{mv},1\n\n"
    )


def _ass_header(w: int = 1080, h: int = 1920) -> str:
    """ASS [Script Info] header. PlayRes MUST match output dims or libass
    stretches the whole caption canvas to fit (distorting square/landscape)."""
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\n"
        f"PlayResY: {h}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
    )


def _write_ass_file(
    events: list[tuple[float, float, str]],
    out_path: Path,
    font_size: int = 56,
    style: str = DEFAULT_CAPTION_STYLE,
    position: str = DEFAULT_CAPTION_POSITION,
    font: str | None = None,
    color: str | None = None,
    w: int = 1080,
    h: int = 1920,
) -> Path:
    """Write a phrase-level styled ASS file (clean / bold).  Karaoke is separate."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(CAPTION_STYLES.get(style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE]))
    cfg["font_size"] = font_size or cfg["font_size"]
    align, margin_v = CAPTION_POSITIONS.get(position, CAPTION_POSITIONS[DEFAULT_CAPTION_POSITION])
    sc = max(0.62, min(1.15, h / 1920))
    lines = [
        _ass_header(w, h),
        _style_block(cfg, align, margin_v, font=_resolve_font(font), primary=_resolve_color(color), scale=sc),
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    ]
    for (s, e, text) in events:
        if e <= s:
            continue
        if cfg["upper"]:
            text = text.upper()
        lines.append(
            f"Dialogue: 0,{_format_ass_time(s)},{_format_ass_time(e)},Cap,,0,0,0,,{_ass_escape(text)}\n"
        )
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def _group_karaoke_lines(
    words: list[dict],
    max_words: int = 5,
    gap_break: float = 0.6,
) -> list[list[dict]]:
    """Group word-timing dicts ({'word','start','end'}) into short caption lines.

    New line on a >gap_break pause or once a line reaches max_words.
    """
    lines: list[list[dict]] = []
    cur: list[dict] = []
    prev_end = None
    for w in words:
        if not (w.get("word") or "").strip():
            continue
        gap = (w["start"] - prev_end) if prev_end is not None else 0.0
        if cur and (gap > gap_break or len(cur) >= max_words):
            lines.append(cur)
            cur = []
        cur.append(w)
        prev_end = w["end"]
    if cur:
        lines.append(cur)
    return lines


def _write_karaoke_ass(
    words: list[dict],
    out_path: Path,
    style: str = "karaoke",
    position: str = DEFAULT_CAPTION_POSITION,
    font: str | None = None,
    color: str | None = None,
    w: int = 1080,
    h: int = 1920,
) -> Path:
    """Write a word-by-word karaoke ASS file.

    *words* must be in OUTPUT (clip-local) time.  Each grouped line becomes one
    Dialogue using ASS ``\\kf`` karaoke tags so libass sweeps the highlight
    colour across each word in sync with speech (TikTok-style fill).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = CAPTION_STYLES.get(style, CAPTION_STYLES["karaoke"])
    align, margin_v = CAPTION_POSITIONS.get(position, CAPTION_POSITIONS[DEFAULT_CAPTION_POSITION])
    sc = max(0.62, min(1.15, h / 1920))
    out = [
        _ass_header(w, h),
        _style_block(cfg, align, margin_v, font=_resolve_font(font), primary=_resolve_color(color), scale=sc),
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    ]
    for line in _group_karaoke_lines(words):
        if not line:
            continue
        start = float(line[0]["start"])
        end = float(line[-1]["end"])
        if end <= start:
            continue
        parts = []
        for w in line:
            dur_cs = max(1, int(round((float(w["end"]) - float(w["start"])) * 100)))
            tok = _ass_escape(w["word"].upper() if cfg["upper"] else w["word"]).replace("\\N", " ")
            parts.append(f"{{\\kf{dur_cs}}}{tok} ")
        out.append(
            f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Cap,,0,0,0,,{''.join(parts).rstrip()}\n"
        )
    out_path.write_text("".join(out), encoding="utf-8")
    return out_path


def _burn_ass(body_path: Path, ass_path: Path, out_path: Path) -> Path:
    """Burn an ASS subtitle file onto *body_path* via libass (single re-encode pass)."""
    esc = str(ass_path).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(body_path),
            "-vf", f"ass='{esc}'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"_burn_ass failed: {proc.stderr[-500:]}")
    return out_path


def overlay_timed_captions(
    body_path: Path,
    events: list[tuple[float, float, str]],
    out_path: Path,
    work_dir: Path,
    font_size: int = 56,
    style: str = DEFAULT_CAPTION_STYLE,
    position: str = DEFAULT_CAPTION_POSITION,
    words: list[dict] | None = None,
    font: str | None = None,
    color: str | None = None,
    w: int = 1080,
    h: int = 1920,
) -> Path:
    """Burn a rolling, speech-synced caption track onto an already-9:16 video.

    *style* selects a CAPTION_STYLES preset; *position* a CAPTION_POSITIONS one;
    *font* / *color* override the family and primary (text/highlight) colour.
    For the ``karaoke`` style, *words* (output-time word timings) drives a
    word-by-word highlight; if absent it falls back to phrase events.
    If *events* is empty the body is re-encoded through unchanged.
    """
    body_path = Path(body_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = CAPTION_STYLES.get(style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
    if cfg["karaoke"] and words:
        ass_path = _write_karaoke_ass(
            words, work_dir / "captions.ass", style=style, position=position,
            font=font, color=color, w=w, h=h,
        )
    elif events:
        ass_path = _write_ass_file(
            events, work_dir / "captions.ass", font_size=font_size, style=style,
            position=position, font=font, color=color, w=w, h=h,
        )
    else:
        return overlay_captions(body_path, "", out_path)

    return _burn_ass(body_path, ass_path, out_path)


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
    w: int = 1080,
    h: int = 1920,
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
        img = run_comfyui_cloud(caption_seed, width=w, height=h)
    except Exception:
        img = None

    if img and Path(img).exists():
        img_input = ["-loop", "1", "-i", str(img)]
    else:
        # Solid black background via ffmpeg lavfi color source.
        img_input = [
            "-f", "lavfi", "-i", f"color=black:s={w}x{h}:r=30",
        ]

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *img_input,
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,"
               f"crop={w}:{h},fps=30,format=yuv420p",
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
    fine_segments: list[dict] | None = None,
    caption_style: str = DEFAULT_CAPTION_STYLE,
    caption_position: str = DEFAULT_CAPTION_POSITION,
    caption_font: str | None = None,
    caption_color: str | None = None,
    w: int = 1080,
    h: int = 1920,
) -> Path:
    """Assemble one structural variant into a branded master at w x h.

    Pipeline:
      1. Cut each segment from source_path into work_dir.
      2. Concat all cut segments -> body (video) OR audio-only track.
      3. If audio-only: _synthesise_visual to get a picture for the body.
      4. Captions: if *fine_segments* are supplied, burn a rolling speech-synced
         caption track (follows the words); otherwise fall back to the single
         static *caption_text* line.
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
            w=w, h=h,
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
            w=w, h=h,
        )

    # 4. Captions — style-aware rolling track.
    #    karaoke: word-by-word highlight (transcribe the assembled body for
    #    output-time word timings); clean/bold: phrase events from fine segments.
    captioned = work_dir / f"captioned_{label}.mp4"
    cfg = CAPTION_STYLES.get(caption_style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
    words = None
    if cfg["karaoke"]:
        try:
            from core.forge import audio
            words = audio.transcribe_words(body)
        except Exception:  # noqa: BLE001 — fall back to phrase events
            words = None
    events = _build_caption_events(segments, fine_segments) if fine_segments else []
    if (cfg["karaoke"] and words) or events:
        overlay_timed_captions(
            body, events, captioned, work_dir / f"caps_{label}",
            style=caption_style, position=caption_position, words=words,
            font=caption_font, color=caption_color, w=w, h=h,
        )
    else:
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

    from core.forge import ingest, sizes

    W, H, _tag = sizes.resolve(params.get("aspect"))
    source_id = params["source_id"]
    segments = params["segments"]
    variant_index = int(params.get("variant_index", 0))
    caption = params.get("caption", "")
    caption_style = params.get("caption_style", DEFAULT_CAPTION_STYLE)
    caption_position = params.get("caption_position", DEFAULT_CAPTION_POSITION)
    caption_font = params.get("caption_font")
    caption_color = params.get("caption_color")

    # Resolve source file path.
    src_row = ingest.get_source(source_id)
    if src_row is None:
        raise RuntimeError(f"source not found: {source_id!r}")
    source_path = Path(src_row["file_path"])
    if not source_path.exists():
        raise RuntimeError(f"source file missing on disk: {source_path}")

    has_video = _detect_has_video(source_path)

    # Fine-grained transcript phrases for rolling speech-synced captions.
    # Falls back to a single static caption if unavailable.
    try:
        fine_segments = ingest.get_segments(source_id)
    except Exception:  # noqa: BLE001
        fine_segments = []

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
            fine_segments=fine_segments,
            caption_style=caption_style,
            caption_position=caption_position,
            caption_font=caption_font,
            caption_color=caption_color,
            w=W, h=H,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
