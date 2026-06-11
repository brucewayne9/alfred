"""Multi-clip hand-pick montage renderer.

The operator picks exact moments from one or more ingested sources (via the
Topic tab), orders them, and Forge stitches those precise cuts together into a
single branded 9:16 vertical — captions and all.  Unlike film_montage (which
auto-chooses ~2.5 s beats), every cut here is a deliberate operator pick.

Reuses the topic_clip engine: each pick is cut from ITS OWN source (video kept
in sync, audio-only picks get a synthesised visual), normalised to the same
1080x1920 / 30 fps / aac params, then concatenated, captioned, and branded.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from core.forge import ingest
from core.forge.renderers.film_montage import make_branded  # noqa: F401  (branding reuse)
from core.forge.renderers.topic_clip import (
    CAPTION_STYLES,
    DEFAULT_CAPTION_POSITION,
    DEFAULT_CAPTION_STYLE,
    _build_caption_events,
    _concat_segments,
    _cut_segment,
    _detect_has_video,
    _extract_audio,
    _synthesise_visual,
    enforce_duration,
    overlay_captions,
    overlay_timed_captions,
)


def _resolve_source_path(source_id: str) -> Path:
    """Resolve an ingest source_id to its on-disk media path, or raise."""
    row = ingest.get_source(source_id)
    if row is None:
        raise RuntimeError(f"source not found: {source_id!r}")
    p = Path(row["file_path"]) if row.get("file_path") else None
    if p is None or not p.exists():
        raise RuntimeError(f"source file missing on disk for {source_id!r}: {p}")
    return p


def _resolve_bed(params: dict) -> Path | None:
    """Resolve an optional song-bed audio source (upload id or path), or None."""
    if params.get("bed_audio_upload_id"):
        from core.forge import uploads
        p = uploads.get_upload_path(params["bed_audio_upload_id"])
        if p is None:
            raise RuntimeError(f"bed audio upload not found: {params['bed_audio_upload_id']}")
        return Path(p)
    if params.get("bed_audio_path"):
        p = Path(params["bed_audio_path"])
        if not p.exists():
            raise RuntimeError(f"bed audio path missing: {p}")
        return p
    return None


def _prepare_bed(bed_src: Path, target_seconds: float, out_path: Path,
                 start: float = 0.0, end: float | None = None) -> Path:
    """Loop+trim *bed_src* to exactly *target_seconds* as 44.1k stereo AAC.

    *start*/*end* (seconds) let the operator pick which slice of the song to
    use: the chosen region is extracted first, then looped to cover the full
    montage even if that slice is shorter; the trim caps it so make_branded's
    -shortest lands on the video.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = Path(bed_src)
    start = max(0.0, float(start or 0.0))

    # If a region was picked, extract [start, end] (or start→EOF) before looping.
    if start > 0.0 or end is not None:
        region = out_path.parent / "bed_region.m4a"
        args = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src)]
        if end is not None and float(end) > start + 0.2:
            args += ["-t", f"{float(end) - start:.3f}"]
        args += ["-vn", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(region)]
        rp = subprocess.run(args, capture_output=True, text=True)
        if rp.returncode == 0 and region.exists() and region.stat().st_size > 0:
            src = region

    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-stream_loop", "-1", "-i", str(src),
            "-t", f"{max(0.5, float(target_seconds)):.3f}",
            "-vn", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"_prepare_bed failed: {proc.stderr[-500:]}")
    return out_path


def _song_caption_events(bed_path: Path, max_words: int = 6):
    """Transcribe a song bed slice into karaoke lyric line events.

    Returns ``([(start_s, end_s, text), ...], words)`` on the montage timeline
    (the bed plays from t=0 in replace mode, so the slice-local word timings map
    straight onto the output), or ``([], None)`` if no lyrics were detected.
    """
    try:
        from core.forge import audio
        from core.forge.renderers.kinetic_lyric import build_karaoke_lines
        words = audio.transcribe_words(bed_path)
    except Exception:  # noqa: BLE001
        return [], None
    if not words:
        return [], None
    fps = 30.0
    events: list[tuple[float, float, str]] = []
    for ln in build_karaoke_lines(words, max_words=max_words, uppercase=False):
        if not ln:
            continue
        s = min(w["startFrame"] for w in ln) / fps
        e = max(w["endFrame"] for w in ln) / fps
        txt = " ".join(w["text"] for w in ln).strip()
        if txt and e > s:
            events.append((s, e, txt))
    return events, words


def _duck_mix(
    voice_path: Path,
    bed_path: Path,
    out_path: Path,
    music_level: float = 0.32,
) -> Path:
    """Mix *voice* full + *bed* music ducked underneath (sidechain compression).

    The voice (interview) stays up front at full level; the music is lowered to
    *music_level* and then auto-ducked further whenever the voice is present, so
    it swells gently in the gaps and drops under speech.  Output 44.1k stereo AAC.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # [0]=voice  [1]=music.  Lower music, sidechain-duck it against the voice,
    # then sum with the full voice (normalize=0 keeps the voice at full level).
    filter_complex = (
        "[1:a]volume={lvl}[bg];"
        "[bg][0:a]sidechaincompress=threshold=0.03:ratio=8:attack=5:release=300[duck];"
        "[duck][0:a]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]"
    ).format(lvl=music_level)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(voice_path), "-i", str(bed_path),
            "-filter_complex", filter_complex,
            "-map", "[mix]",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            str(out_path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"_duck_mix failed: {proc.stderr[-500:]}")
    return out_path


def build_multiclip_events(
    picks: list[dict],
    fine_by_source: dict[str, list[dict]],
) -> list[tuple[float, float, str]]:
    """Build output-time caption events across an ordered list of multi-source picks.

    Each pick's own source's fine transcript phrases are pulled for that pick's
    window and rebased onto the running output timeline, so captions follow the
    words across clips from different sources.
    """
    events: list[tuple[float, float, str]] = []
    offset = 0.0
    for pk in picks:
        window = [{
            "start_s": float(pk["start_s"]),
            "end_s": float(pk["end_s"]),
            "text": pk.get("text", "") or "",
        }]
        fine = fine_by_source.get(pk["source_id"]) or []
        for (s, e, t) in _build_caption_events(window, fine):
            events.append((offset + s, offset + e, t))
        offset += max(0.0, float(pk["end_s"]) - float(pk["start_s"]))
    return events


def render(params: dict, out_path: str | Path) -> Path:
    """Assemble an ordered multi-source pick list into a branded 9:16 montage.

    Required params:
      picks — ordered list of {source_id, start_s, end_s, text?}

    Optional params:
      caption_style    — clean | bold | karaoke (default clean)
      caption_position — lower | center | upper  (default lower)

    Honours the 10–60 s duration band (trims the tail).  Returns the master path.
    """
    from core.forge import sizes
    W, H, _tag = sizes.resolve(params.get("aspect"))
    out_path = Path(out_path)
    picks = params.get("picks") or []
    if not picks:
        raise RuntimeError("no picks — provide at least one segment")
    caption_style = params.get("caption_style", DEFAULT_CAPTION_STYLE)
    caption_position = params.get("caption_position", DEFAULT_CAPTION_POSITION)
    caption_font = params.get("caption_font")
    caption_color = params.get("caption_color")

    # Clamp total to the 10–60 s band (enforce_duration trims the last pick's end).
    picks = enforce_duration(picks)

    work = Path(tempfile.mkdtemp(prefix="forge_multimtg_"))
    try:
        # 1. Cut each pick from its own source -> uniform 1080x1920 mp4.
        seg_paths: list[Path] = []
        for i, pk in enumerate(picks):
            src = _resolve_source_path(pk["source_id"])
            has_video = _detect_has_video(src)
            if has_video:
                sp = _cut_segment(
                    src, float(pk["start_s"]), float(pk["end_s"]),
                    work / f"seg_{i:03d}.mp4", has_video=True, w=W, h=H,
                )
            else:
                # Audio-only pick: cut audio, synthesise a visual for it.
                a = _cut_segment(
                    src, float(pk["start_s"]), float(pk["end_s"]),
                    work / f"seg_{i:03d}.m4a", has_video=False, w=W, h=H,
                )
                sp = _synthesise_visual(
                    pk.get("text", "") or "clip", a, work / f"seg_{i:03d}.mp4", work, w=W, h=H,
                )
            seg_paths.append(sp)

        # 2. Concat all picks (all uniform) -> body.
        body = _concat_segments(
            seg_paths, work / "body.mp4", has_video=True, work_dir=work / "cwork",
        )

        # 3-4. Captions + audio.
        from core.forge import audio
        captioned = work / "captioned.mp4"
        cfg = CAPTION_STYLES.get(caption_style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
        bed_src = _resolve_bed(params)
        body_seconds = audio.duration_seconds(body)

        if (params.get("caption_source") or "").lower() == "song" and bed_src is not None:
            # Music-video montage: the song bed IS the audio (always replace) and
            # the caption source — transcribe the chosen song slice into karaoke
            # lyric lines and burn them, synced to the music, over the visuals.
            bed = _prepare_bed(bed_src, body_seconds, work / "bed.m4a",
                               start=params.get("bed_start") or 0.0,
                               end=params.get("bed_end"))
            lyric_events, lyric_words = _song_caption_events(bed)
            if lyric_events:
                overlay_timed_captions(
                    body, lyric_events, captioned, work / "caps",
                    style=caption_style,  # honour the operator's chosen lyric style
                    position=caption_position, words=lyric_words,
                    font=caption_font, color=caption_color, w=W, h=H,
                )
            else:
                import shutil as _sh
                _sh.copy(body, captioned)  # no lyrics detected — leave uncaptioned
            hook_audio = bed
        else:
            # Interview / b-roll montage: captions follow the picks' transcript
            # (or a typed caption); bed is optional and ducks under the voice.
            words = None
            if cfg["karaoke"]:
                try:
                    words = audio.transcribe_words(body)
                except Exception:  # noqa: BLE001
                    words = None
            fine_by_source = {
                sid: ingest.get_segments(sid)
                for sid in {p["source_id"] for p in picks}
            }
            events = build_multiclip_events(picks, fine_by_source)
            manual_caption = (params.get("caption_text") or "").strip()
            if manual_caption:
                overlay_timed_captions(
                    body, [(0.0, max(1.0, body_seconds), manual_caption[:200])],
                    captioned, work / "caps",
                    style=caption_style, position=caption_position, words=None,
                    font=caption_font, color=caption_color, w=W, h=H,
                )
            elif (cfg["karaoke"] and words) or events:
                overlay_timed_captions(
                    body, events, captioned, work / "caps",
                    style=caption_style, position=caption_position, words=words,
                    font=caption_font, color=caption_color, w=W, h=H,
                )
            else:
                overlay_captions(body, (picks[0].get("text") or "")[:120], captioned)

            bed_mode = (params.get("bed_mode") or "duck").lower()
            voice_audio = work / "voice.m4a"
            _extract_audio(captioned, voice_audio)
            if bed_src is not None:
                bed = _prepare_bed(bed_src, audio.duration_seconds(captioned), work / "bed.m4a",
                                   start=params.get("bed_start") or 0.0,
                                   end=params.get("bed_end"))
                hook_audio = bed if bed_mode == "replace" else _duck_mix(voice_audio, bed, work / "mixed.m4a")
            else:
                hook_audio = voice_audio

        # 5. Brand: logo overlay + audio mux.
        make_branded(captioned, hook_audio, "", out_path)
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
