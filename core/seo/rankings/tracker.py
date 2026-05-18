"""Rank tracker — for each active keyword on a site, pull Google's SERP and
find where the site ranks (if at all within the top 100 organic results).

Storage:
  - seo_rankings_daily: one row per (site_id, keyword, capture date) — the
    time-series record. Unique constraint enforces idempotency: re-running
    on the same day overwrites the existing row.
  - seo_keywords: current_rank, current_rank_url, rank_source, rank_checked_at
    refreshed so the dashboard's "current" view stays cheap (single query).

Cost: DataForSEO `/serp/google/organic/live/regular` is ~$0.0007 per call.
25-keyword Roen run ≈ $0.018 per weekly pull (~$1/yr). Logged via SeoApiCost.

Domain matching: a SERP `domain` field of `www.roenhandmade.com` or
`roenhandmade.com` both match a site whose registered domain is either
form. We strip the leading `www.` on both sides before comparing.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.db import SessionLocal
from core.seo.models import SeoApiCost, SeoKeyword, SeoRankingDaily, SeoSite
from integrations.dataforseo.client import DataForSEOClient

log = logging.getLogger("seo.rankings")

DEFAULT_LOCATION_CODE = 2840   # USA
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_DEPTH = 100             # top 100 organic results
RANK_SOURCE = "dataforseo"


@dataclass
class RankResult:
    keyword: str
    position: Optional[int]      # rank_absolute (1-100), or None if not found
    found_url: Optional[str]     # exact URL that ranks, or None
    prior_position: Optional[int] = None
    delta: Optional[int] = None  # prior - current (positive = improvement)


@dataclass
class SiteRankReport:
    site_slug: str
    captured_at: dt.date
    results: list[RankResult] = field(default_factory=list)
    cost_usd: float = 0.0
    failures: list[dict] = field(default_factory=list)

    @property
    def ranked_count(self) -> int:
        return sum(1 for r in self.results if r.position is not None)

    @property
    def top10_count(self) -> int:
        return sum(1 for r in self.results if r.position and r.position <= 10)

    @property
    def top3_count(self) -> int:
        return sum(1 for r in self.results if r.position and r.position <= 3)


def _normalize_domain(d: str) -> str:
    """Lowercase + strip leading www. for matching."""
    d = (d or "").lower().strip()
    return d[4:] if d.startswith("www.") else d


def _find_site_rank(serp_results: list[dict], site_domain: str) -> tuple[Optional[int], Optional[str]]:
    """Scan organic SERP results for first match on site_domain.

    Returns (rank_absolute, url) or (None, None) if not in results.
    """
    target = _normalize_domain(site_domain)
    for item in serp_results:
        if _normalize_domain(item.get("domain", "")) == target:
            return item.get("rank_absolute"), item.get("url")
    return None, None


def _load_prior_position(session, site_id: int, keyword: str, today: dt.date) -> Optional[int]:
    """Most-recent prior position for keyword (excluding today). None if no history."""
    row = session.execute(
        select(SeoRankingDaily.position)
        .where(
            SeoRankingDaily.site_id == site_id,
            SeoRankingDaily.query == keyword,
            SeoRankingDaily.captured_at < today,
        )
        .order_by(SeoRankingDaily.captured_at.desc())
        .limit(1)
    ).scalar()
    return int(row) if row is not None else None


def _persist_capture(
    session, site_id: int, keyword: str, position: Optional[int], captured_at: dt.date
) -> None:
    """Upsert a row into seo_rankings_daily.

    Re-running the tracker on the same day overwrites the position
    rather than failing on the unique constraint.
    """
    stmt = pg_insert(SeoRankingDaily).values(
        site_id=site_id,
        query=keyword,
        position=position,
        captured_at=captured_at,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["site_id", "query", "captured_at"],
        set_={"position": position},
    )
    session.execute(stmt)


def _refresh_current_rank(
    session,
    site_id: int,
    keyword: str,
    position: Optional[int],
    found_url: Optional[str],
    checked_at: dt.datetime,
) -> None:
    """Update the cached current_rank fields on seo_keywords."""
    row = session.execute(
        select(SeoKeyword).where(
            SeoKeyword.site_id == site_id,
            SeoKeyword.keyword == keyword,
        )
    ).scalar_one_or_none()
    if not row:
        return
    row.current_rank = position
    row.current_rank_url = found_url
    row.rank_source = RANK_SOURCE
    row.rank_checked_at = checked_at


def _log_cost(session, *, cost_usd: float, site_id: int, keyword: str) -> None:
    """Append a per-call row to seo_api_costs."""
    if cost_usd <= 0:
        return
    session.add(SeoApiCost(
        api_name="dataforseo",
        endpoint="/serp/google/organic/live/regular",
        cost_usd=cost_usd,
        site_id=site_id,
        purpose="rank_tracker",
        meta={"keyword": keyword},
    ))


def track_site(
    site_slug: str,
    *,
    limit: Optional[int] = None,
    dry_run: bool = False,
    location_code: int = DEFAULT_LOCATION_CODE,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    depth: int = DEFAULT_DEPTH,
    client: Optional[DataForSEOClient] = None,
) -> SiteRankReport:
    """Track today's rank for every active keyword belonging to a site.

    Args:
        site_slug: which site (e.g. "roen")
        limit: cap on number of keywords (None = all active)
        dry_run: skip DB writes (still calls the SERP API)
        client: inject a DataForSEOClient (for tests). None = build one.
    """
    if client is None:
        client = DataForSEOClient()

    today = dt.date.today()
    now = dt.datetime.now(dt.timezone.utc)

    with SessionLocal() as session:
        site = session.execute(
            select(SeoSite).where(SeoSite.slug == site_slug)
        ).scalar_one_or_none()
        if site is None:
            raise ValueError(f"unknown site slug: {site_slug!r}")
        site_id = site.id
        site_domain = site.domain

        kw_query = select(SeoKeyword).where(
            SeoKeyword.site_id == site_id,
            SeoKeyword.status == "active",
        ).order_by(
            SeoKeyword.priority.asc().nullslast(),
            SeoKeyword.search_volume.desc().nullslast(),
        )
        if limit:
            kw_query = kw_query.limit(limit)
        keywords = session.execute(kw_query).scalars().all()
        keyword_strings = [k.keyword for k in keywords]

    log.info("tracking %d keywords for site=%s domain=%s", len(keyword_strings), site_slug, site_domain)
    report = SiteRankReport(site_slug=site_slug, captured_at=today)
    cost_before = client.total_cost_usd

    # We open a fresh session per keyword so a single failure doesn't
    # poison a long batch's transaction. Volumes are tiny (25 keywords),
    # the overhead is negligible.
    for keyword in keyword_strings:
        try:
            serp = client.serp_organic(
                keyword,
                location_code=location_code,
                language_code=language_code,
                depth=depth,
            )
            position, found_url = _find_site_rank(serp["results"], site_domain)
        except Exception as e:  # network, API, parse — log + continue
            log.exception("serp pull failed for keyword=%r", keyword)
            report.failures.append({"keyword": keyword, "error": f"{type(e).__name__}: {e}"})
            continue

        # Per-call cost: total_cost_usd advances inside the client; diff it.
        call_cost = client.total_cost_usd - cost_before
        cost_before = client.total_cost_usd

        if dry_run:
            with SessionLocal() as s:
                prior = _load_prior_position(s, site_id, keyword, today)
        else:
            with SessionLocal() as s:
                prior = _load_prior_position(s, site_id, keyword, today)
                _persist_capture(s, site_id, keyword, position, today)
                _refresh_current_rank(s, site_id, keyword, position, found_url, now)
                _log_cost(s, cost_usd=call_cost, site_id=site_id, keyword=keyword)
                s.commit()

        delta = (prior - position) if (prior is not None and position is not None) else None
        report.results.append(RankResult(
            keyword=keyword,
            position=position,
            found_url=found_url,
            prior_position=prior,
            delta=delta,
        ))

    report.cost_usd = client.total_cost_usd
    log.info(
        "rank tracker done: %d/%d ranked  top10=%d  top3=%d  spend=$%.4f  failures=%d",
        report.ranked_count, len(report.results),
        report.top10_count, report.top3_count,
        report.cost_usd, len(report.failures),
    )
    return report
