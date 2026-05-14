# core/seo/ingest/ga4.py
"""GA4 daily sync — pulls organic sessions + conversions per page."""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.api_clients.ga4_client import get_client, run_page_organic_report
from core.seo.db import SessionLocal
from core.seo.models import SeoPage, SeoSite

logger = logging.getLogger(__name__)


def _build_client():
    return get_client()


def _absolute_url(site: SeoSite, path: str) -> str:
    if path.startswith("http"):
        return path
    return f"https://{site.domain.rstrip('/')}{path if path.startswith('/') else '/' + path}"


def sync_site_for_date(site_id: int, date: dt.date) -> int:
    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        if not site:
            raise ValueError(f"site_id {site_id} not found")
        if not site.ga4_property_id:
            logger.warning("site %s has no ga4_property_id; skipping", site.slug)
            return 0
    client = _build_client()
    payload = run_page_organic_report(client, site.ga4_property_id, date)
    rows = payload.get("rows", []) or []

    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        written = 0
        for row in rows:
            path = row["dimensionValues"][0]["value"]
            sessions = int(row["metricValues"][0]["value"])
            conversions = int(row["metricValues"][1]["value"])
            url = _absolute_url(site, path)
            stmt = pg_insert(SeoPage).values(
                site_id=site_id,
                url=url,
                page_type=None,
                organic_sessions=sessions,
                conversions=conversions,
            ).on_conflict_do_update(
                index_elements=["site_id", "url"],
                set_=dict(
                    organic_sessions=pg_insert(SeoPage).excluded.organic_sessions,
                    conversions=pg_insert(SeoPage).excluded.conversions,
                    last_seen_at=dt.datetime.utcnow(),
                ),
            )
            s.execute(stmt)
            written += 1
        s.commit()
    logger.info("GA4 sync site_id=%s date=%s rows=%d", site_id, date, written)
    return written


def sync_all_sites_for_date(date: dt.date) -> dict[str, int]:
    from core.seo.sites.registry import list_sites
    out: dict[str, int] = {}
    for site in list_sites():
        if not site.ga4_property_id:
            continue
        try:
            out[site.slug] = sync_site_for_date(site.id, date)
        except Exception:
            logger.exception("GA4 sync failed for %s", site.slug)
            out[site.slug] = -1
    return out
