# core/seo/ingest/cwv.py
"""Core Web Vitals daily sync via PageSpeed Insights API.

For each site, pulls CWV for the top-20 pages by recent organic_sessions.
"""
from __future__ import annotations

import datetime as dt
import logging
from decimal import Decimal

import requests
from sqlalchemy import desc
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.seo.api_clients.psi_client import ENDPOINT
from core.seo.db import SessionLocal
from core.seo.models import SeoPage, SeoSite

logger = logging.getLogger(__name__)


def _parse_metrics(payload: dict) -> dict:
    """Extract LCP (ms), CLS (decimal), INP (ms) from PSI loadingExperience."""
    le = payload.get("loadingExperience") or {}
    metrics = le.get("metrics") or {}
    lcp = metrics.get("LARGEST_CONTENTFUL_PAINT_MS", {}).get("percentile")
    cls_raw = metrics.get("CUMULATIVE_LAYOUT_SHIFT_SCORE", {}).get("percentile")
    inp = metrics.get("INTERACTION_TO_NEXT_PAINT", {}).get("percentile")
    return {
        "lcp_ms": int(lcp) if lcp is not None else None,
        # PSI returns CLS as int * 100 — i.e. 8 means 0.08.
        "cls":    (Decimal(cls_raw) / Decimal(100)) if cls_raw is not None else None,
        "inp_ms": int(inp) if inp is not None else None,
    }


def sync_url(site_id: int, url: str) -> dict:
    from config.settings import settings
    api_key = settings.seo_psi_api_key
    if not api_key:
        raise RuntimeError("SEO_PSI_API_KEY not set in config/.env")
    params = {"url": url, "key": api_key, "strategy": "mobile", "category": "PERFORMANCE"}
    resp = requests.get(ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    parsed = _parse_metrics(resp.json())
    with SessionLocal() as s:
        stmt = pg_insert(SeoPage).values(
            site_id=site_id,
            url=url,
            cwv_lcp_ms=parsed["lcp_ms"],
            cwv_cls=parsed["cls"],
            cwv_inp_ms=parsed["inp_ms"],
            last_seen_at=dt.datetime.utcnow(),
        ).on_conflict_do_update(
            index_elements=["site_id", "url"],
            set_=dict(
                cwv_lcp_ms=pg_insert(SeoPage).excluded.cwv_lcp_ms,
                cwv_cls=pg_insert(SeoPage).excluded.cwv_cls,
                cwv_inp_ms=pg_insert(SeoPage).excluded.cwv_inp_ms,
                last_seen_at=dt.datetime.utcnow(),
            ),
        )
        s.execute(stmt)
        s.commit()
    return parsed


def sync_top_pages_for_site(site_id: int, limit: int = 20) -> int:
    """Pick the top N pages by organic_sessions, sync CWV for each."""
    with SessionLocal() as s:
        top = s.query(SeoPage).filter_by(site_id=site_id).order_by(desc(SeoPage.organic_sessions)).limit(limit).all()
        urls = [p.url for p in top]
    # If we have no traffic data yet (Plan 1 day 1), seed with the homepage.
    if not urls:
        with SessionLocal() as s:
            site = s.get(SeoSite, site_id)
            if not site:
                return 0
            urls = [f"https://{site.domain.rstrip('/')}/"]
    count = 0
    for u in urls:
        try:
            sync_url(site_id, u)
            count += 1
        except Exception:
            logger.exception("PSI sync failed for %s", u)
    return count


def sync_all_sites(limit_per_site: int = 20) -> dict[str, int]:
    from core.seo.sites.registry import list_sites
    return {s.slug: sync_top_pages_for_site(s.id, limit_per_site) for s in list_sites()}
