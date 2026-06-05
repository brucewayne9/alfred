"""Kinetic-lyric renderer: audio hook -> word-timed lyric vertical (Remotion KineticTypeRig)."""
from __future__ import annotations
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
REMOTION = Path("/home/aialfred/remotion")


def _node_env() -> dict:
    """Env for the Remotion subprocess with a modern Node on PATH.

    Under systemd the service PATH has no nvm, so `npx` falls back to the
    system Node (v12), which Remotion can't run. Prepend the newest nvm
    Node >= 18 so the renderer works regardless of ambient PATH.
    """
    import os
    env = dict(os.environ)
    candidates = []
    nvm = Path("/home/aialfred/.nvm/versions/node")
    if nvm.is_dir():
        for d in nvm.iterdir():
            m = re.match(r"v(\d+)\.", d.name)
            if m and int(m.group(1)) >= 18 and (d / "bin" / "node").exists():
                candidates.append((int(m.group(1)), d / "bin"))
    if candidates:
        best = max(candidates)[1]
        env["PATH"] = f"{best}:{env.get('PATH', '')}"
    return env


def _is_marker(t: str) -> bool:
    return re.fullmatch(r"2[0-9],?", t.strip()) is not None


def build_karaoke_lines(words: list[dict], fps: int = 30, max_words: int = 7,
                        gap_break: float = 0.7, uppercase: bool = True) -> list[list[dict]]:
    """Group word-timing dicts into karaoke lines for the caption rigs.

    New line when: the next word is an age marker (21..29), a >gap_break pause,
    or the current line hit max_words. Trailing punct stripped. `uppercase`
    controls case baking — KineticTypeRig wants UPPER baked in; the Film Montage
    caption gallery passes uppercase=False so each style's own `upper` token
    (CSS text-transform) decides case per-preset.
    """
    def fr(t: float) -> int:
        return max(0, round(float(t) * fps))

    lines: list[list[dict]] = []
    cur: list[dict] = []
    prev_end = None
    for w in words:
        gap = (w["start"] - prev_end) if prev_end is not None else 0.0
        if cur and (_is_marker(w["word"]) or gap > gap_break or len(cur) >= max_words):
            lines.append(cur); cur = []
        text = w["word"].rstrip(",.?!").strip()
        if uppercase:
            text = text.upper()
        if text:
            cur.append({"text": text, "startFrame": fr(w["start"]), "endFrame": fr(w["end"])})
        prev_end = w["end"]
    if cur:
        lines.append(cur)
    return [ln for ln in lines if ln]


def _vessel_prompt(caption: str, override: str | None = None) -> str:
    """Moody cinematic 9:16 vessel/scene prompt for the kinetic background.

    Describes a SCENE (not a portrait) so the kinetic lyrics sit over atmosphere.
    """
    if override:
        return override
    caption = (caption or "").strip()
    base = (
        "cinematic melancholic moody scene, low-key teal and amber lighting, "
        "deep shadows, film grain, 35mm, lonely introspective night mood, "
        "vertical 9:16, atmospheric haze, no text, no watermark"
    )
    return f"{base}, themed around: {caption}" if caption else base


def render(params: dict, out_path: str | Path) -> Path:
    """Render a word-timed kinetic-lyric vertical to `out_path` (mp4). Raises on failure.

    Pipeline: resolve hook audio -> whisper word timings -> karaoke lines ->
    ComfyUI-Cloud vessel still -> ken-burns motion bg -> Remotion KineticTypeRig
    (silent) -> guarded mux of the hook audio -> assert_audible.
    """
    from core.forge import audio, uploads, caption_styles, sizes

    W, H, _tag = sizes.resolve(params.get("aspect"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="forge_kin_render_"))

    # 1. Resolve source audio
    uid = params.get("audio_upload_id")
    if uid:
        src = uploads.get_upload_path(uid)
        if src is None:
            raise RuntimeError(f"no upload found for audio_upload_id={uid}")
    elif params.get("audio_path"):
        src = Path(params["audio_path"])
    else:
        raise RuntimeError("no audio provided")
    src = Path(src)
    if not src.exists():
        raise RuntimeError(f"source audio missing on disk: {src}")

    # 2. Hook clip
    clip_start = params.get("clip_start")
    clip_end = params.get("clip_end")
    if clip_start is not None and clip_end is not None:
        hook = audio.clip_audio(src, float(clip_start), float(clip_end), work / "hook.mp3")
    else:
        hook = src

    # 3. Transcribe -> karaoke lines
    words = audio.transcribe_words(hook)
    if not words:
        raise RuntimeError("transcription produced no words")
    # Caption style picks how the lyric animates (shared gallery across formats).
    # 'none'/'off' = clean vessel, no captions. Default mimics the classic gold
    # karaoke sweep this format always had.
    style_id = params.get("caption_style")
    captions_on = style_id not in ("none", "off")
    lines = build_karaoke_lines(words, max_words=5, uppercase=False) if captions_on else []
    style_spec = caption_styles.resolve(
        style_id if (style_id and caption_styles.is_valid(style_id)) else "karaoke_gold")

    # 4. Vessel image via the image dispatcher (ComfyUI Cloud, or Higgsfield when
    #    params opt in). Same lazy-import pattern as leak_graphic.
    from core.forge import image_generation  # lazy: heavy import

    img = image_generation.render_image(
        _vessel_prompt(params.get("caption", ""), params.get("vessel_prompt")),
        W, H,
        source=params.get("image_source", "higgsfield"),
        model=params.get("image_model"),
        aspect=params.get("aspect"),
    )
    if not img:
        raise RuntimeError("vessel image generation returned no image")
    img = Path(img)
    if not img.exists():
        raise RuntimeError(f"vessel image missing on disk: {img}")

    try:
        # 5. Ken-burns the still into a 9:16 motion mp4
        secs = math.ceil(audio.duration_seconds(hook)) + 1
        vessel_mp4 = work / "vessel.mp4"
        kb = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30",
             "-i", str(img), "-t", str(secs),
             "-vf", (f"scale={int(W*1.5)}:{int(H*1.5)}:force_original_aspect_ratio=increase,"
                     f"crop={int(W*1.5)}:{int(H*1.5)},"
                     f"zoompan=z='min(1.0+0.00010*in,1.12)':x='iw/2-(iw/zoom/2)':"
                     f"y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps=30"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-an", str(vessel_mp4)],
            capture_output=True, text=True)
        if kb.returncode != 0 or not vessel_mp4.exists():
            raise RuntimeError(f"ken-burns ffmpeg failed: {kb.stderr[-500:]}")

        # 6. Copy vessel into Remotion public dir under a unique name (always cleaned up)
        pub_name = f"forge_vessel_{uuid.uuid4().hex}.mp4"
        pub_path = REMOTION / "public" / pub_name
        try:
            shutil.copyfile(vessel_mp4, pub_path)

            # 7. Props — shared CaptionStudioRig (same engine as Film Montage),
            #    so all 37 gallery styles work here too.
            props = {
                "brand": "mainstay",
                "bgClip": pub_name,
                "lines": lines,
                "style": style_spec,
                "scrim": True,
                "width": W,
                "height": H,
            }
            props_path = work / "props.json"
            props_path.write_text(json.dumps(props))

            # 8. Render silent video via Remotion
            silent = work / "silent.mp4"
            rr = subprocess.run(
                ["npx", "remotion", "render", "src/index.ts", "CaptionStudioRig",
                 str(silent), f"--props={props_path}"],
                cwd=str(REMOTION), capture_output=True, text=True, timeout=1200,
                env=_node_env())
            if rr.returncode != 0 or not silent.exists():
                raise RuntimeError(f"remotion render failed: {rr.stderr[-800:]}")

            # 9. Mux hook audio with EXPLICIT stream mapping (the critical fix)
            mux = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(hook),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                 "-b:a", "192k", "-shortest", str(out_path)],
                capture_output=True, text=True)
            if mux.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"mux ffmpeg failed: {mux.stderr[-500:]}")

            # 10. Guard: hook audio must be present
            audio.assert_audible(out_path)
        finally:
            try:
                pub_path.unlink()
            except OSError:
                pass
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # 11.
    return Path(out_path)
