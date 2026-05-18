"""Per-site KPI aggregation. Pulls from seo_keywords (current_rank),
seo_audit_issues (open issue count), seo_decided (content shipped),
seo_api_costs (spend MTD), seo_pending (drafts awaiting approval).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text

from core.seo.db import SessionLocal


@dataclass
class SiteKpis:
    site_id: int
    keyword_count: int
    ranked_count: int               # in top-100
    top10_count: int
    top3_count: int
    avg_rank: Optional[float]       # NULL-safe avg of current_rank
    open_issues: int                # audit issues with fixed_at IS NULL
    warning_issues: int
    info_issues: int
    error_issues: int
    pending_drafts: int
    content_shipped_total: int      # all-time approved
    content_shipped_30d: int
    spend_mtd_usd: float
    last_rank_capture: Optional[dt.date]


def site_kpis(site_id: int) -> SiteKpis:
    today = dt.date.today()
    month_start = today.replace(day=1)
    cutoff_30d = today - dt.timedelta(days=30)

    with SessionLocal() as s:
        # Keyword + rank stats — pull current_rank stats
        kw = s.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(current_rank) AS ranked,
                COUNT(*) FILTER (WHERE current_rank <= 10) AS top10,
                COUNT(*) FILTER (WHERE current_rank <= 3) AS top3,
                AVG(current_rank) AS avg_rank,
                MAX(rank_checked_at)::date AS last_capture
            FROM seo_keywords
            WHERE site_id = :sid AND status = 'active'
        """), {"sid": site_id}).first()

        # Open audit issues by severity
        sev = s.execute(text("""
            SELECT severity, COUNT(*) AS n
            FROM seo_audit_issues
            WHERE site_id = :sid AND fixed_at IS NULL
            GROUP BY severity
        """), {"sid": site_id}).all()
        sev_map = {r.severity: r.n for r in sev}
        open_issues = sum(sev_map.values())

        # Pending drafts
        pending = s.execute(text("""
            SELECT COUNT(*) FROM seo_pending
            WHERE site_id = :sid AND status = 'pending'
        """), {"sid": site_id}).scalar() or 0

        # Content shipped (approved & published)
        shipped_total = s.execute(text("""
            SELECT COUNT(*) FROM seo_decided
            WHERE site_id = :sid AND outcome IN ('approved', 'published')
        """), {"sid": site_id}).scalar() or 0
        shipped_30d = s.execute(text("""
            SELECT COUNT(*) FROM seo_decided
            WHERE site_id = :sid AND outcome IN ('approved', 'published')
              AND decided_at >= :cutoff
        """), {"sid": site_id, "cutoff": cutoff_30d}).scalar() or 0

        # Spend MTD (calendar month)
        spend_mtd = s.execute(text("""
            SELECT COALESCE(SUM(cost_usd), 0) FROM seo_api_costs
            WHERE site_id = :sid AND called_at::date >= :ms
        """), {"sid": site_id, "ms": month_start}).scalar() or 0.0

    return SiteKpis(
        site_id=site_id,
        keyword_count=int(kw.total or 0),
        ranked_count=int(kw.ranked or 0),
        top10_count=int(kw.top10 or 0),
        top3_count=int(kw.top3 or 0),
        avg_rank=float(kw.avg_rank) if kw.avg_rank is not None else None,
        open_issues=open_issues,
        warning_issues=int(sev_map.get("warning", 0)),
        info_issues=int(sev_map.get("info", 0)),
        error_issues=int(sev_map.get("error", 0)),
        pending_drafts=int(pending),
        content_shipped_total=int(shipped_total),
        content_shipped_30d=int(shipped_30d),
        spend_mtd_usd=float(spend_mtd),
        last_rank_capture=kw.last_capture,
    )
