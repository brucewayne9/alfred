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
    out_path = Path(out_path)
    picks = params.get("picks") or []
    if not picks:
        raise RuntimeError("no picks — provide at least one segment")
    caption_style = params.get("caption_style", DEFAULT_CAPTION_STYLE)
    caption_position = params.get("caption_position", DEFAULT_CAPTION_POSITION)

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
                    work / f"seg_{i:03d}.mp4", has_video=True,
                )
            else:
                # Audio-only pick: cut audio, synthesise a 9:16 visual for it.
                a = _cut_segment(
                    src, float(pk["start_s"]), float(pk["end_s"]),
                    work / f"seg_{i:03d}.m4a", has_video=False,
                )
                sp = _synthesise_visual(
                    pk.get("text", "") or "clip", a, work / f"seg_{i:03d}.mp4", work,
                )
            seg_paths.append(sp)

        # 2. Concat all picks (all uniform) -> body.
        body = _concat_segments(
            seg_paths, work / "body.mp4", has_video=True, work_dir=work / "cwork",
        )

        # 3. Captions — style-aware, following the words across sources.
        captioned = work / "captioned.mp4"
        cfg = CAPTION_STYLES.get(caption_style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
        words = None
        if cfg["karaoke"]:
            try:
                from core.forge import audio
                words = audio.transcribe_words(body)
            except Exception:  # noqa: BLE001
                words = None
        fine_by_source = {
            sid: ingest.get_segments(sid)
            for sid in {p["source_id"] for p in picks}
        }
        events = build_multiclip_events(picks, fine_by_source)
        if (cfg["karaoke"] and words) or events:
            overlay_timed_captions(
                body, events, captioned, work / "caps",
                style=caption_style, position=caption_position, words=words,
            )
        else:
            overlay_captions(body, (picks[0].get("text") or "")[:120], captioned)

        # 4. Brand: logo overlay + montage audio mux.
        hook_audio = work / "hook.m4a"
        _extract_audio(captioned, hook_audio)
        make_branded(captioned, hook_audio, "", out_path)
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
