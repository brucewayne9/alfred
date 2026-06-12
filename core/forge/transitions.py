"""Forge montage transition catalog (M1 Phase 1).

Single source of truth mapping tasteful, named transitions to ffmpeg ``xfade``
types. Instagram-Edits-style restraint: a curated shortlist, one style per
montage, with an "Auto" default that never reaches for the flashy options.

Consumed by the montage renderers (``film_montage``, ``multi_montage``) and the
forge-web picker. ffmpeg ``xfade`` shares one ``duration``/``offset`` mechanic
across every transition type, so the renderers only need the resolved ``xfade``
name — the heavy lifting already exists in ``_concat_xfade``.
"""
from __future__ import annotations

# Quick blend, not a slow wipe — hides the hard-cut jump without drawing attention.
DEFAULT_DURATION = 0.18

# key -> (label, xfade type or None for a hard cut, directional, duration)
# ``directional`` types alternate left/right per boundary for variety within one style.
_ENTRIES: list[dict] = [
    {"key": "cut", "label": "Cut", "xfade": None, "directional": False, "duration": 0.0},
    {"key": "dissolve", "label": "Dissolve", "xfade": "fade", "directional": False, "duration": DEFAULT_DURATION},
    {"key": "whip", "label": "Whip", "xfade": "slideleft", "directional": True, "duration": DEFAULT_DURATION},
    {"key": "wipe", "label": "Wipe", "xfade": "wipeleft", "directional": True, "duration": DEFAULT_DURATION},
    {"key": "zoom", "label": "Zoom", "xfade": "zoomin", "directional": False, "duration": DEFAULT_DURATION},
    {"key": "flash", "label": "Flash", "xfade": "fadewhite", "directional": False, "duration": 0.12},
    {"key": "blur", "label": "Blur", "xfade": "smoothleft", "directional": True, "duration": DEFAULT_DURATION},
    {"key": "glitch", "label": "Glitch", "xfade": "pixelize", "directional": False, "duration": DEFAULT_DURATION},
]

_BY_KEY = {e["key"]: e for e in _ENTRIES}

# Default fallback for unknown/auto keys — the safe, tasteful baseline.
_FALLBACK = "dissolve"

# Auto must stay tasteful: these are deliberate opt-ins, never auto-selected.
_FLASHY = {"flash", "glitch"}


def menu() -> list[dict]:
    """Picker entries for the UI — Auto first, then the curated catalog."""
    auto = {"key": "auto", "label": "Auto", "xfade": None, "directional": False, "duration": DEFAULT_DURATION}
    return [auto, *(_BY_KEY[e["key"]] for e in _ENTRIES)]


def resolve(key: str) -> dict:
    """Resolve a concrete transition key to its spec.

    Unknown keys (and "auto", which must be resolved via :func:`pick_auto` first)
    fall back to the tasteful default.
    """
    return dict(_BY_KEY.get(key, _BY_KEY[_FALLBACK]))


def pick_auto(params: dict) -> str:
    """Choose a transition for "Auto" — conservative and never flashy."""
    if params.get("energy") == "high":
        return "whip"
    return "dissolve"


def render_spec(params: dict) -> dict:
    """Resolve a montage's ``params`` into what the renderer needs.

    Handles "auto" (via :func:`pick_auto`), "cut" (hard-cut path), concrete keys,
    and the default. Returns the catalog spec plus a ``hard_cut`` flag.
    """
    key = (params.get("transition") or _FALLBACK).lower()
    if key == "auto":
        key = pick_auto(params)
    spec = resolve(key)
    spec["hard_cut"] = spec["xfade"] is None
    return spec
