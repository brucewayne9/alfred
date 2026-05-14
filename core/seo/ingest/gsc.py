# core/seo/ingest/gsc.py
"""GSC daily sync — pulls per-query data for one site for one date."""
from __future__ import annotations

import datetime as dt
import logging
from decimal import Decimal

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.api_clients.gsc_client import get_client, query_analytics
from core.seo.db import SessionLocal
from core.seo.models import SeoQuery, SeoSite

logger = logging.getLogger(__name__)


def _build_client():
    # Indirection so tests can patch.
    return get_client()


def sync_site_for_date(site_id: int, date: dt.date) -> int:
    """Pull GSC data for one site on one date. Returns row count written."""
    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        if not site:
            raise ValueError(f"site_id {site_id} not found")
        if not site.gsc_property:
            logger.warning("site %s has no gsc_property; skipping", site.slug)
            return 0

    client = _build_client()
    payload = query_analytics(client, site.gsc_property, date, date, row_limit=5000)
    rows = payload.get("rows", []) or []

    with SessionLocal() as s:
        written = 0
        for row in rows:
            keys = row.get("keys") or []
            if not keys:
                continue
            stmt = pg_insert(SeoQuery).values(
                site_id=site_id,
                query=keys[0],
                position=Decimal(str(row.get("position", 0))),
                impressions=int(row.get("impressions", 0)),
                clicks=int(row.get("clicks", 0)),
                ctr=Decimal(str(row.get("ctr", 0))),
                captured_at=date,
            ).on_conflict_do_update(
                index_elements=["site_id", "query", "captured_at"],
                set_=dict(
                    position=pg_insert(SeoQuery).excluded.position,
                    impressions=pg_insert(SeoQuery).excluded.impressions,
                    clicks=pg_insert(SeoQuery).excluded.clicks,
                    ctr=pg_insert(SeoQuery).excluded.ctr,
                ),
            )
            s.execute(stmt)
            written += 1
        s.commit()
    logger.info("GSC sync site_id=%s date=%s rows=%d", site_id, date, written)
    return written


def sync_all_sites_for_date(date: dt.date) -> dict[str, int]:
    """Sync every active site that has a gsc_property. Returns slug→row_count."""
    from core.seo.sites.registry import list_sites
    out: dict[str, int] = {}
    for site in list_sites():
        if not site.gsc_property:
            continue
        try:
            out[site.slug] = sync_site_for_date(site.id, date)
        except Exception:
            logger.exception("GSC sync failed for %s", site.slug)
            out[site.slug] = -1
    return out
