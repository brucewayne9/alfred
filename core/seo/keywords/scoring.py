"""Keyword opportunity scoring.

Opportunity = how likely this keyword is to drive value if we rank for it,
adjusted for how hard it is to rank. Higher = attack first.

Components:
- volume_factor: log-scaled search volume (so 100 vs 10k isn't a 100x weight)
- intent_weight: transactional > commercial > informational > navigational
- difficulty_penalty: shrinks score as difficulty grows
- rank_proximity: bonus if we're already ranking somewhere in top 50
                  (cheaper to push #20 → #5 than to break in cold)
"""
from __future__ import annotations

import math
from typing import Optional

INTENT_WEIGHT = {
    "transactional": 1.0,
    "commercial":    0.85,
    "informational": 0.55,
    "navigational":  0.30,
    None:            0.50,  # unknown intent — middle weight
    "":              0.50,
}


def score_keyword(
    volume: Optional[int],
    difficulty: Optional[int],
    intent: Optional[str],
    current_rank: Optional[float] = None,
) -> float:
    """Return a 0-100 opportunity score.

    Higher = better to attack. Returns 0.0 for keywords with no volume.
    """
    if not volume or volume <= 0:
        return 0.0

    # Volume factor: log10(volume+1) normalized to ~0–1 in the 10–100k range.
    volume_factor = min(math.log10(volume + 1) / 5.0, 1.0)

    # Intent weight
    intent_w = INTENT_WEIGHT.get((intent or "").lower(), 0.5)

    # Difficulty penalty (higher diff → smaller multiplier)
    # diff=0 → 1.0, diff=50 → 0.5, diff=100 → 0.1
    diff = difficulty if isinstance(difficulty, (int, float)) else 50
    diff_factor = max(0.1, 1.0 - (diff / 110.0))

    # Rank proximity bonus
    rank_factor = 1.0
    if isinstance(current_rank, (int, float)) and current_rank > 0:
        if current_rank <= 10:
            rank_factor = 1.4   # already on page 1, easy to push
        elif current_rank <= 20:
            rank_factor = 1.25  # page 2 — high-value pushes
        elif current_rank <= 50:
            rank_factor = 1.10  # page 3-5 — moderate boost
        elif current_rank <= 100:
            rank_factor = 1.05

    raw = volume_factor * intent_w * diff_factor * rank_factor * 100.0
    return round(raw, 2)
