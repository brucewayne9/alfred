"""Forge Remix — expand a job into N distinct-looking param-sets (varied vessel mood).

Remix changes the *look* (vessel/mood) per output; the stealth multiplier then makes
each look algorithmically-unique. Remix = variety across pages; variations = anti-suppression.
"""
from __future__ import annotations

REMIX_MOODS = [
    "rain-streaked window at night",
    "neon-lit city street, wet reflections",
    "golden-hour rooftop, hazy light",
    "dim empty room, single lamp",
    "car interior at night, passing streetlights",
    "foggy forest at dawn",
    "rooftop under heavy storm clouds",
    "lonely highway at dusk",
]


def build_remixes(params: dict, n: int) -> list[dict]:
    """Return N param-sets, each a copy of `params` with a distinct vessel_prompt.

    n<=1 returns a single untouched copy (no remix). The caption/lyric text is never
    altered — only the vessel mood (the backdrop look) changes per remix.
    """
    n = max(1, int(n or 1))
    if n == 1:
        return [dict(params)]
    base = (params.get("caption") or "").strip()
    out: list[dict] = []
    for i in range(n):
        p = dict(params)
        mood = REMIX_MOODS[i % len(REMIX_MOODS)]
        p["vessel_prompt"] = f"{mood}, cinematic moody, {base}".strip(", ")
        p["remix_index"] = i
        out.append(p)
    return out
