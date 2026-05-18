"""API-cost rollups for the Spend screen.

Three views:
  - by purpose (rank_tracker, site_audit, keyword_discovery, content_writer …)
  - by site (cross-tenant who's costing what)
  - daily series (last 30 days, for the bar/line chart)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from core.seo.db import SessionLocal


@dataclass
class SpendRow:
    label: str
    call_count: int
    total_usd: float


@dataclass
class DailySpend:
    day: dt.date
    total_usd: float


@dataclass
class SpendBreakdown:
    total_30d_usd: float
    total_mtd_usd: float
    by_purpose: list[SpendRow]
    by_site: list[SpendRow]      # label = site display_name (or "—" for unassigned)
    daily_30d: list[DailySpend]


def spend_breakdown() -> SpendBreakdown:
    today = dt.date.today()
    cutoff_30d = today - dt.timedelta(days=30)
    month_start = today.replace(day=1)

    with SessionLocal() as s:
        total_30d = s.execute(text("""
            SELECT COALESCE(SUM(cost_usd), 0) FROM seo_api_costs
            WHERE called_at::date >= :c
        """), {"c": cutoff_30d}).scalar() or 0.0

        total_mtd = s.execute(text("""
            SELECT COALESCE(SUM(cost_usd), 0) FROM seo_api_costs
            WHERE called_at::date >= :c
        """), {"c": month_start}).scalar() or 0.0

        by_purpose_rows = s.execute(text("""
            SELECT COALESCE(purpose, '—') AS label,
                   COUNT(*) AS n,
                   COALESCE(SUM(cost_usd), 0) AS total
            FROM seo_api_costs
            WHERE called_at::date >= :c
            GROUP BY purpose
            ORDER BY total DESC
        """), {"c": cutoff_30d}).all()

        by_site_rows = s.execute(text("""
            SELECT COALESCE(s.display_name, '— unassigned —') AS label,
                   COUNT(*) AS n,
                   COALESCE(SUM(c.cost_usd), 0) AS total
            FROM seo_api_costs c
            LEFT JOIN seo_sites s ON s.id = c.site_id
            WHERE c.called_at::date >= :cutoff
            GROUP BY s.display_name
            ORDER BY total DESC
        """), {"cutoff": cutoff_30d}).all()

        daily_rows = s.execute(text("""
            SELECT called_at::date AS day,
                   COALESCE(SUM(cost_usd), 0) AS total
            FROM seo_api_costs
            WHERE called_at::date >= :c
            GROUP BY called_at::date
            ORDER BY called_at::date
        """), {"c": cutoff_30d}).all()

    return SpendBreakdown(
        total_30d_usd=float(total_30d),
        total_mtd_usd=float(total_mtd),
        by_purpose=[SpendRow(r.label, int(r.n), float(r.total)) for r in by_purpose_rows],
        by_site=[SpendRow(r.label, int(r.n), float(r.total)) for r in by_site_rows],
        daily_30d=[DailySpend(r.day, float(r.total)) for r in daily_rows],
    )
