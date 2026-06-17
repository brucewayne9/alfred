"""DLD waitlist priority (Klaviyo seam).

M1: read a local seed file (one email per line, lowercased). Returns False if
the file is absent. M5 swaps the body for a live Klaviyo pull
(list V75mRt 'DLD Waitlist', account XYKnGf) — the `is_priority` signature is
stable across that swap so callers never change.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _file() -> Path:
    override = os.environ.get("FAIRGAME_WAITLIST_FILE")
    if override:
        return Path(override)
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "mainstay"
        / "fairgame"
        / "waitlist_emails.txt"
    )


@lru_cache(maxsize=1)
def _load() -> frozenset:
    p = _file()
    if not p.exists():
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in p.read_text().splitlines()
        if line.strip()
    )


def is_priority(email: str) -> bool:
    _load.cache_clear()  # cheap file; keep fresh in M1
    return (email or "").strip().lower() in _load()
