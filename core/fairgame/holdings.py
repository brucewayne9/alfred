"""Fans First seat-map holdings overlay.

Derives, deterministically from the OPEN venue geometry, which sections Rod
holds and whether each is available or sold -- so the seat map can paint the
arena green (ours, available) / red (ours, sold) / grey (not ours). This is a
DEMO/visual layer decoupled from the pricing inventory in events.py; when real
held-seat data lands it replaces section_status(). No randomness or clock use,
so the same section always paints the same colour across requests.
"""
from __future__ import annotations

from . import seatmap

# Rod holds the premium block: floor/court + the lower bowl. Upper bowl is grey.
_HELD_TIERS = ("floor", "lower")


def classify_tier(name: str, ga: bool) -> str:
    """Map a geometry section name to a tier: floor | lower | upper."""
    if ga:
        return "floor"
    # Names starting with a non-digit are court/letter codes (C1, C4W, FLOOR) -> floor
    if name and not name[0].isdigit():
        return "floor"
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        return "floor"
    num = int(digits)
    if num < 200:
        return "lower"
    return "upper"


def _stable_hash(name: str) -> int:
    """Deterministic small int from a section name (no randomness)."""
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def section_status(name: str, ga: bool) -> str:
    """available | sold | not_ours for one section. Deterministic + stable."""
    if classify_tier(name, ga) not in _HELD_TIERS:
        return "not_ours"
    # ~1 in 6 held sections shown as sold, deterministically.
    return "sold" if _stable_hash(name) % 6 == 0 else "available"


def overlay(show_id: str) -> dict | None:
    """Per-section status map for a show, or None if no geometry is ingested."""
    ov = seatmap.overview(show_id)
    if ov is None:
        return None
    sections = {}
    available = sold = 0
    for s in ov.get("sections", []):
        st = section_status(s["name"], s.get("ga", False))
        sections[s["name"]] = st
        if st == "available":
            available += 1
        elif st == "sold":
            sold += 1
    return {"sections": sections, "held": available + sold,
            "available": available, "sold": sold}
