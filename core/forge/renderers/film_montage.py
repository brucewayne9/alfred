"""Film-montage renderer: hook + borrowed/generated clips -> branded 9:16 montage.

Optional word-synced lyric captions: when params carry a valid ``caption_style``
(see core.forge.caption_styles), the silent montage body is run through a
Remotion caption-overlay pass (CaptionStudioRig) before logo + audio mux. The
hook audio is transcribed to word timings, so captions follow the sung lyric.
"""
from __future__ import annotations
import json
import math
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from core.forge import audio, clips, uploads, caption_styles
from core.forge.renderers.kinetic_lyric import build_karaoke_lines, _node_env, REMOTION

LOGO_PATH = Path("/home/aialfred/remotion/public/mainstay-logo.png")

# Swift dissolve between montage segments (seconds). Hides the hard-cut "jump"
# between clips sampled from different moments. Kept short so it reads as a clean
# transition, not a slow blend. Set transition="cut" in params to disable.
TRANSITION_FADE = 0.18


def _concat_xfade(seg_paths: list[Path], out: Path, fade: float = TRANSITION_FADE) -> Path:
    """Concat segments with a quick crossfade dissolve at every boundary.

    Chains ffmpeg ``xfade`` filters. Each transition overlaps two clips by
    ``fade`` seconds, so the running composite duration is tracked to place every
    offset correctly. Segments are cut ``fade`` seconds long (see render) so the
    overlap is absorbed and total duration ≈ the sum of requested seconds — the
    lyric captions stay synced. Falls back to a copy for a single segment.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(seg_paths) == 1:
        shutil.copyfile(seg_paths[0], out)
        return out

    durs = []
    for p in seg_paths:
        try:
            durs.append(audio.duration_seconds(p))
        except Exception:
            durs.append(fade * 2)

    inputs: list[str] = []
    for p in seg_paths:
        inputs += ["-i", str(p)]

    filt, prev, acc = [], "0:v", durs[0]
    for i in range(1, len(seg_paths)):
        off = max(0.0, acc - fade)
        label = f"vx{i}"
        filt.append(
            f"[{prev}][{i}:v]xfade=transition=fade:duration={fade}:"
            f"offset={off:.3f}[{label}]")
        prev, acc = label, acc + durs[i] - fade

    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", ";".join(filt),
         "-map", f"[{prev}]", "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-an", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"xfade concat failed: {proc.stderr[-500:]}")
    return out


def _caption_overlay(body: Path, lines: list, style_spec: dict, work: Path,
                     w: int = 1080, h: int = 1920) -> Path:
    """Burn word-synced captions over the silent montage via Remotion CaptionStudioRig.

    Returns a new silent mp4 (captions baked, no audio/logo yet — make_branded
    adds those). The montage video is staged into Remotion's public/ under a
    unique name and always cleaned up.
    """
    pub_name = f"forge_mtgcap_{uuid.uuid4().hex}.mp4"
    pub_path = REMOTION / "public" / pub_name
    shutil.copyfile(body, pub_path)
    try:
        props = {"brand": "mainstay", "bgClip": pub_name, "lines": lines,
                 "style": style_spec, "scrim": True, "width": w, "height": h}
        props_path = work / "capprops.json"
        props_path.write_text(json.dumps(props))
        out = work / "captioned.mp4"
        rr = subprocess.run(
            ["npx", "remotion", "render", "src/index.ts", "CaptionStudioRig",
             str(out), f"--props={props_path}"],
            cwd=str(REMOTION), capture_output=True, text=True, timeout=1800,
            env=_node_env())
        if rr.returncode != 0 or not out.exists():
            raise RuntimeError(f"caption overlay render failed: {rr.stderr[-800:]}")
        return out
    finally:
        try:
            pub_path.unlink()
        except OSError:
            pass


def plan_segments(clip_count: int, hook_seconds: float, seg_seconds: float = 2.5) -> list[dict]:
    """Round-robin segments covering hook_seconds; last one trimmed to land exactly."""
    if clip_count <= 0:
        raise ValueError("no clips to montage")
    n = max(1, math.ceil(hook_seconds / seg_seconds))
    segs, used = [], 0.0
    for i in range(n):
        remaining = hook_seconds - used
        secs = min(seg_seconds, remaining)
        if secs <= 0.01:
            break
        segs.append({"clip_index": i % clip_count, "seconds": round(secs, 3)})
        used += secs
    return segs


def assign_offsets(segs: list[dict], durations: list[float]) -> list[dict]:
    """Spread each segment's start offset across its source clip's timeline.

    When several segments reuse the same clip (e.g. a single long video), they
    must sample *different* moments — otherwise the montage loops one slice.
    For a clip used k times, its uses walk evenly from 0 to (duration - seconds).
    Returns a new list; each seg gains an ``offset`` (seconds into its source).
    """
    # Group the positions of segments per clip, preserving order.
    uses: dict[int, list[int]] = {}
    for i, s in enumerate(segs):
        uses.setdefault(s["clip_index"], []).append(i)

    out = [dict(s) for s in segs]
    for clip_index, positions in uses.items():
        dur = durations[clip_index] if clip_index < len(durations) else 0.0
        for j, pos in enumerate(positions):
            seconds = out[pos]["seconds"]
            playable = max(0.0, dur - seconds)
            k = len(positions)
            frac = 0.0 if k <= 1 else j / (k - 1)
            out[pos]["offset"] = round(playable * frac, 3)
    return out


def _cut_segment(src: str | Path, seconds: float, out: str | Path,
                 offset: float | None = None, w: int = 1080, h: int = 1920) -> Path:
    """Cut `seconds` starting at `offset` (default: deterministic 10%-in, cap 1s);
    cover-crop to w x h @30, no audio."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if offset is not None:
        off = max(0.0, offset)
    else:
        try:
            off = min(1.0, max(0.0, audio.duration_seconds(src) * 0.1))
        except Exception:
            off = 0.0
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(off), "-i", str(src),
         "-t", str(seconds), "-an",
         "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},fps=30,format=yuv420p",
         "-c:v", "libx264", "-preset", "veryfast", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"_cut_segment failed for {src}: {proc.stderr[-500:]}")
    return out


def make_branded(body: str | Path, hook: str | Path, caption: str, out_path: str | Path) -> Path:
    """Overlay the Mainstay logo bottom-right and mux the hook audio with explicit stream mapping.

    Caption drawtext is intentionally skipped (apostrophes/quotes break ffmpeg drawtext and
    would risk the whole render). If the logo is absent, fall back to a plain audio mux.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if LOGO_PATH.exists():
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(body), "-i", str(LOGO_PATH), "-i", str(hook),
            "-filter_complex",
            "[1:v]scale=150:-1[lg];[0:v][lg]overlay=W-w-40:H-h-60:format=auto[v]",
            "-map", "[v]", "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(body), "-i", str(hook),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"make_branded failed: {proc.stderr[-500:]}")
    return out_path


def render(params: dict, out_path: str | Path) -> Path:
    """Assemble a branded 9:16 montage: hook audio + ingested/generated clips."""
    # 1. Resolve hook audio (before mkdtemp so a bad audio path fails cheap).
    if params.get("audio_upload_id"):
        src = uploads.get_upload_path(params["audio_upload_id"])
        if src is None:
            raise RuntimeError(f"audio upload not found: {params['audio_upload_id']}")
    elif params.get("audio_path"):
        src = Path(params["audio_path"])
    else:
        raise RuntimeError("no audio provided")
    src = Path(src)
    if not src.exists():
        raise RuntimeError(f"audio source missing on disk: {src}")

    from core.forge import sizes
    W, H, _tag = sizes.resolve(params.get("aspect"))

    work = Path(tempfile.mkdtemp(prefix="forge_mtg_"))
    try:
        if params.get("clip_start") is not None and params.get("clip_end") is not None:
            hook = audio.clip_audio(src, params["clip_start"], params["clip_end"], work / "hook.mp3")
        else:
            hook = src
        hook_seconds = audio.duration_seconds(hook)

        # 2. Gather raw clips.
        raw: list[Path] = []
        for s in params.get("sources") or []:
            raw.extend(clips.fetch_source(s, work / "raw"))
        gen_prompt = (params.get("generate_prompt") or params.get("caption")
                      or "cinematic moody emotional b-roll, low-key lighting, film grain, vertical")
        video_source = params.get("video_source", "higgsfield")
        # If the user uploaded a still, use it as the Kling start frame. No such
        # param exists today, but resolve it if present so the wiring is ready;
        # otherwise generate_clip auto-synthesises a frame for Higgsfield/Kling.
        start_frame = None
        sample_id = params.get("base_image_upload_id")
        if not sample_id:
            samples = params.get("samples") or params.get("sample_upload_ids")
            if isinstance(samples, (list, tuple)) and samples:
                sample_id = samples[0]
        if sample_id:
            sp = uploads.get_upload_path(sample_id)
            if sp is not None and Path(sp).exists():
                start_frame = sp
        for _ in range(int(params.get("generate") or 0)):
            raw.append(clips.generate_clip(
                gen_prompt, work / "raw",
                source=video_source, start_frame=start_frame))
        if not raw:
            raise RuntimeError("no clips — provide sources or set generate>0")

        # 3-4. Plan + spread offsets (so reused clips sample distinct moments) + cut.
        segs = plan_segments(len(raw), hook_seconds)
        durations = []
        for p in raw:
            try:
                durations.append(audio.duration_seconds(p))
            except Exception:
                durations.append(0.0)
        segs = assign_offsets(segs, durations)
        # Quick dissolve by default; "cut" gives a hard splice (already clean).
        transition = (params.get("transition") or "fade").lower()
        fade = TRANSITION_FADE if transition != "cut" and len(segs) > 1 else 0.0
        seg_paths: list[Path] = []
        for i, seg in enumerate(segs):
            # Pad each segment by `fade` so the crossfade overlap is absorbed and
            # the body length stays matched to the hook audio.
            seg_paths.append(
                _cut_segment(raw[seg["clip_index"]], seg["seconds"] + fade,
                             work / "segments" / f"seg_{i:03d}.mp4",
                             offset=seg.get("offset"), w=W, h=H))

        # 5. Stitch: crossfade dissolve (default) or clean hard concat.
        body = work / "body.mp4"
        if fade > 0.0:
            _concat_xfade(seg_paths, body, fade=fade)
        else:
            concat_txt = work / "concat.txt"
            concat_txt.write_text(
                "".join(f"file '{p.resolve()}'\n" for p in seg_paths))
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(concat_txt), "-c:v", "libx264", "-preset", "veryfast",
                 "-pix_fmt", "yuv420p", "-an", str(body)],
                capture_output=True, text=True)
            if proc.returncode != 0 or not body.exists():
                raise RuntimeError(f"concat failed: {proc.stderr[-500:]}")

        # 6. Optional word-synced lyric captions (Remotion overlay pass).
        style_id = params.get("caption_style")
        if style_id and style_id not in ("none", "off") and caption_styles.is_valid(style_id):
            words = audio.transcribe_words(hook)
            lines = build_karaoke_lines(words, max_words=4, uppercase=False) if words else []
            if lines:
                body = _caption_overlay(body, lines, caption_styles.resolve(style_id), work, w=W, h=H)

        # 7. Brand + mux + guard.
        make_branded(body, hook, params.get("caption", ""), out_path)
        audio.assert_audible(out_path)
        return Path(out_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
