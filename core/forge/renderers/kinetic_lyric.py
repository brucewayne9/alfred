"""Kinetic-lyric renderer: audio hook -> word-timed lyric vertical (Remotion KineticTypeRig)."""
from __future__ import annotations
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
REMOTION = Path("/home/aialfred/remotion")


def _is_marker(t: str) -> bool:
    return re.fullmatch(r"2[0-9],?", t.strip()) is not None


def build_karaoke_lines(words: list[dict], fps: int = 30, max_words: int = 7,
                        gap_break: float = 0.7) -> list[list[dict]]:
    """Group word-timing dicts into karaoke lines for KineticTypeRig.

    New line when: the next word is an age marker (21..29), a >gap_break pause,
    or the current line hit max_words. Words are uppercased, trailing punct stripped.
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
        text = w["word"].upper().rstrip(",.?!").strip()
        if text:
            cur.append({"text": text, "startFrame": fr(w["start"]), "endFrame": fr(w["end"])})
        prev_end = w["end"]
    if cur:
        lines.append(cur)
    return [ln for ln in lines if ln]
