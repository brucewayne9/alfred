"""Forge output sizes — the three social aspect ratios offered across formats.

One source of truth so the UI picker, the renderers, and the Remotion props all
agree. Default is vertical (9:16) — unchanged from before this was added.
"""
from __future__ import annotations

# id -> (width, height, label, filename tag)
ASPECTS: dict[str, tuple[int, int, str, str]] = {
    "9x16": (1080, 1920, "Vertical · 9:16", "9x16"),   # TikTok / Reels / Shorts
    "1x1":  (1080, 1080, "Square · 1:1",    "1x1"),     # Feed posts
    "16x9": (1920, 1080, "Landscape · 16:9", "16x9"),   # YouTube / X / landscape
}

DEFAULT_ASPECT = "9x16"


def resolve(aspect: str | None) -> tuple[int, int, str]:
    """Return (width, height, tag) for an aspect id, falling back to vertical."""
    w, h, _label, tag = ASPECTS.get(aspect or "", ASPECTS[DEFAULT_ASPECT])
    return w, h, tag


def is_valid(aspect: str | None) -> bool:
    return (aspect or "") in ASPECTS


def list_sizes() -> list[dict]:
    """Picker payload for the UI."""
    return [{"id": k, "label": v[2], "w": v[0], "h": v[1]} for k, v in ASPECTS.items()]
