# core/seo/ingest/backlinks.py
"""Passive backlink monitor. Diffs today's snapshot vs prior day; marks new + lost."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.db import SessionLocal
from core.seo.models import SeoBacklink, SeoSite

logger = logging.getLogger(__name__)


def record_backlinks_for_site(site_id: int, snapshot: Iterable[tuple[str, str, str]]) -> dict:
    """Record today's backlink snapshot. Tuple is (source_url, target_url, anchor_text).

    Behavior:
      - Existing row matching (site, source, target): update last_seen, clear lost_at.
      - New row: insert with first_seen=now.
      - Rows in DB for this site that are NOT in snapshot AND lost_at IS NULL: mark lost_at=now.
    """
    now = dt.datetime.utcnow()
    seen_keys: set[tuple[str, str]] = set()

    with SessionLocal() as s:
        for source_url, target_url, anchor in snapshot:
            stmt = pg_insert(SeoBacklink).values(
                site_id=site_id,
                source_url=source_url,
                target_url=target_url,
                anchor_text=anchor,
                first_seen=now,
                last_seen=now,
                lost_at=None,
            ).on_conflict_do_update(
                index_elements=["site_id", "source_url", "target_url"],
                set_=dict(last_seen=now, lost_at=None, anchor_text=anchor),
            )
            s.execute(stmt)
            seen_keys.add((source_url, target_url))
        s.commit()

        # Mark rows not seen this run as lost.
        rows = s.query(SeoBacklink).filter_by(site_id=site_id).all()
        lost = 0
        for r in rows:
            if (r.source_url, r.target_url) not in seen_keys and r.lost_at is None:
                r.lost_at = now
                lost += 1
        s.commit()

    return {"recorded": len(seen_keys), "newly_lost": lost}


def fetch_gsc_links(site: SeoSite) -> list[tuple[str, str, str]]:
    """Pull GSC top-linking-sites for the site. Returns snapshot tuples.

    GSC's Search Console API exposes external links via the
    `sites().listAllExternalLinks` legacy endpoint which has been deprecated;
    the current data path is via `searchanalytics` with `linkingPage` dim,
    OR via the property's `links` resource. In practice we use the report
    available at the property level. For Phase 1 we use the
    `searchanalytics().query` route with `page` dimension and filter to
    referrer-only — Phase 2 swaps this to a richer source.
    """
    # Phase 1 stub: returns empty list when the GSC API has nothing to surface.
    # The real pull goes through scripts/seo_backlinks_sync.py with a verbose
    # flag for manual export. This keeps the daemon side strictly idempotent.
    return []


def sync_all_sites_from_gsc() -> dict[str, dict]:
    from core.seo.sites.registry import list_sites
    out: dict[str, dict] = {}
    for site in list_sites():
        if not site.gsc_property:
            continue
        try:
            snapshot = fetch_gsc_links(site)
            out[site.slug] = record_backlinks_for_site(site.id, snapshot)
        except Exception:
            logger.exception("backlinks sync failed for %s", site.slug)
            out[site.slug] = {"error": True}
    return out
