"""Pick algorithm — variety + dedup + freshness scoring.

Inputs:
  candidates: list of dicts with keys
              {id, name, color_family, material_class, style_class, days_in_stock}
  history:   list of past pick records for the same email — each a dict with
             color_tags (list[str]) and style_tags (list[str])
  rng:       random.Random — exposed for deterministic tests

Output: list of 5 candidate dicts (no semantic ordering).

Algorithm: weighted sampling without replacement. Each candidate's weight
is base × dedup_penalty × freshness_factor. After sampling, we check the
variety constraint (no duplicate color_family + material_class pair); if
the constraint isn't met, retry up to MAX_RETRIES times. If we still
can't satisfy variety, return the best-effort sample (catalog might be
homogeneous enough that strict variety is impossible).
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List, Optional

from core.jewelry.bracelet_box.tags import BUNDLE_SIZE, InsufficientStock

MAX_RETRIES = 10


def _dedup_penalty(candidate: dict, history_color_counts: Counter,
                   history_style_counts: Counter) -> float:
    """Multiplier in (0, 1] — closer to 1 means less penalty.

    A candidate whose color_family appeared 0 times in history → multiplier 1.0
    A candidate whose color appeared 3+ times → multiplier 0.4
    Style class has a milder penalty (smaller swing).
    """
    color_seen = history_color_counts.get(candidate['color_family'], 0)
    style_seen = history_style_counts.get(candidate.get('style_class', ''), 0)
    color_factor = max(0.4, 1.0 - 0.2 * color_seen)
    style_factor = max(0.6, 1.0 - 0.1 * style_seen)
    return color_factor * style_factor


def _freshness_factor(candidate: dict) -> float:
    """Older inventory gets a small nudge upward.

    Capped at 1.3x for very old inventory so freshness doesn't dominate.
    """
    days = candidate.get('days_in_stock', 7)
    return min(1.3, 0.9 + 0.02 * days)


def _has_variety(picks: List[dict]) -> bool:
    """No two picks share BOTH color_family AND material_class."""
    pairs = Counter((p['color_family'], p['material_class']) for p in picks)
    return max(pairs.values()) == 1


def pick_five(candidates: List[dict], history: List[dict],
              rng: Optional[random.Random] = None) -> List[dict]:
    """Pick 5 candidates with variety + dedup + freshness scoring.

    Raises InsufficientStock if fewer than BUNDLE_SIZE candidates supplied.
    Returns all candidates if exactly BUNDLE_SIZE supplied (no scoring needed).
    """
    if rng is None:
        rng = random.Random()

    if len(candidates) < BUNDLE_SIZE:
        raise InsufficientStock(
            f"need at least {BUNDLE_SIZE} eligible candidates, got {len(candidates)}"
        )

    if len(candidates) == BUNDLE_SIZE:
        return list(candidates)

    color_counts: Counter = Counter()
    style_counts: Counter = Counter()
    for h in history:
        color_counts.update(h.get('color_tags', []))
        style_counts.update(h.get('style_tags', []))

    weights = [
        _dedup_penalty(c, color_counts, style_counts) * _freshness_factor(c)
        for c in candidates
    ]

    # Up to MAX_RETRIES attempts to satisfy the variety constraint; otherwise
    # accept best effort (catalog may be too homogeneous).
    best_attempt: List[dict] = []
    for _ in range(MAX_RETRIES):
        picks = _weighted_sample(candidates, weights, BUNDLE_SIZE, rng)
        if _has_variety(picks):
            return picks
        best_attempt = picks
    return best_attempt


def _weighted_sample(items: List[dict], weights: List[float], k: int,
                     rng: random.Random) -> List[dict]:
    """Sample k items without replacement, weighted."""
    items = list(items)
    weights = list(weights)
    chosen: List[dict] = []
    for _ in range(k):
        total = sum(weights)
        if total <= 0:
            # All-zero weights — fall back to uniform random.
            i = rng.randrange(len(items))
        else:
            r = rng.random() * total
            cum = 0.0
            i = 0
            for idx, w in enumerate(weights):
                cum += w
                if cum >= r:
                    i = idx
                    break
        chosen.append(items[i])
        items.pop(i)
        weights.pop(i)
    return chosen
