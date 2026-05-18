"""Top rank movers — compares each keyword's most-recent capture to its
prior capture and surfaces the biggest gainers/losers.

A keyword that newly entered the top-100 (NULL → integer) shows up with
positive delta = (101 - new_position). A keyword that fell out of top-100
(integer → NULL) shows up with negative delta = -(101 - prior_position).
This biases toward keywords that crossed the visibility threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from core.seo.db import SessionLocal


@dataclass
class Mover:
    keyword: str
    current_rank: Optional[int]   # most-recent position (None = not in top-100)
    prior_rank: Optional[int]
    delta: int                    # prior - current (positive = improvement)
    target_url: Optional[str]


def top_movers(site_id: int, limit: int = 5) -> tuple[list[Mover], list[Mover]]:
    """Return (gainers, losers). Each capped at `limit` rows."""
    with SessionLocal() as s:
        rows = s.execute(text("""
            WITH ranked AS (
                SELECT
                    query,
                    position,
                    captured_at,
                    ROW_NUMBER() OVER (PARTITION BY query ORDER BY captured_at DESC) AS rn
                FROM seo_rankings_daily
                WHERE site_id = :sid
            ),
            paired AS (
                SELECT
                    r1.query,
                    r1.position AS curr,
                    r2.position AS prior
                FROM ranked r1
                LEFT JOIN ranked r2 ON r2.query = r1.query AND r2.rn = 2
                WHERE r1.rn = 1
            )
            SELECT
                p.query,
                p.curr,
                p.prior,
                k.target_url
            FROM paired p
            LEFT JOIN seo_keywords k ON k.site_id = :sid AND k.keyword = p.query
            WHERE NOT (p.curr IS NULL AND p.prior IS NULL)
        """), {"sid": site_id}).all()

    movers: list[Mover] = []
    for r in rows:
        curr = int(r.curr) if r.curr is not None else None
        prior = int(r.prior) if r.prior is not None else None

        # Compute delta with a virtual 101 sentinel for missing-from-top-100.
        # That ranks "entered top-100" as a meaningful gainer event.
        sentinel = 101
        curr_eff = curr if curr is not None else sentinel
        prior_eff = prior if prior is not None else sentinel
        delta = prior_eff - curr_eff
        if delta == 0:
            continue
        movers.append(Mover(
            keyword=r.query,
            current_rank=curr,
            prior_rank=prior,
            delta=delta,
            target_url=r.target_url,
        ))

    gainers = sorted([m for m in movers if m.delta > 0], key=lambda m: -m.delta)[:limit]
    losers = sorted([m for m in movers if m.delta < 0], key=lambda m: m.delta)[:limit]
    return gainers, losers
