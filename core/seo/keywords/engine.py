"""Keyword Engine — discover, enrich, score, and persist keywords for a site.

Flow for a single site:

  1. Load site + brand profile.
  2. Build seed list = profile.target_keywords (primary + local + long_tail).
  3. Expand each seed via DataForSEO Labs keyword_suggestions
     (gives keyword + volume + difficulty + intent in one call).
  4. Dedupe across seeds; filter junk (other languages, brand names, low-signal).
  5. For candidates missing volume from Labs, batch-fill via Keywords Data
     search_volume (cost-amortized across up to 1000 keywords per call).
  6. Fetch WP URL inventory for the site (posts + pages + products).
  7. For each candidate, pick best existing target URL (or mark "create new").
  8. Score each candidate's opportunity.
  9. Take top N (default 25), persist to seo_keywords, log spend to seo_api_costs.
 10. Return a KeywordRun summary.

POC scope: Roen only. No GSC merge (Roen GSC has no data yet — 2 days old).
We'll add GSC current-rank merging in v2 when there's data.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select

from core.seo.db import SessionLocal
from core.seo.keywords.scoring import score_keyword
from core.seo.keywords.targeting import pick_target_url
from core.seo.keywords.url_inventory import SiteUrl, fetch_wp_url_inventory
from core.seo.models import SeoApiCost, SeoKeyword, SeoSite
from core.seo.sites.profile import load_profile
from integrations.dataforseo.client import DataForSEOClient

log = logging.getLogger(__name__)

# Per-seed suggestion cap. DataForSEO Labs charges per request, so smaller
# limits don't save money — we only cap to keep the working set manageable.
SUGGESTIONS_PER_SEED = 75

# Hard cap on seeds we expand. Each seed = 1 DFS Labs call (~$0.01).
MAX_SEEDS = 12

# Default punch-list size we hand to Mike.
DEFAULT_MAX_KEYWORDS = 25


@dataclass
class KeywordRun:
    """Summary of one Keyword Engine run."""
    site_id: int
    site_slug: str
    seeds_used: list[str]
    candidates_total: int
    candidates_kept: int
    final_keywords: int
    dfs_cost_usd: float
    started_at: float
    finished_at: float
    keyword_ids: list[int] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return self.finished_at - self.started_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "site_slug": self.site_slug,
            "seeds_used": self.seeds_used,
            "candidates_total": self.candidates_total,
            "candidates_kept": self.candidates_kept,
            "final_keywords": self.final_keywords,
            "dfs_cost_usd": round(self.dfs_cost_usd, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "keyword_ids": self.keyword_ids,
        }


def _seed_list(profile) -> list[str]:
    """Build a deduped seed list from a BrandProfile, ordered primary→long_tail."""
    tk = profile.target_keywords or {}
    primary = list(tk.get("primary") or [])
    local = list(tk.get("local") or [])
    long_tail = list(tk.get("long_tail") or [])
    seen: set[str] = set()
    out: list[str] = []
    for kw in primary + long_tail + local:
        k = (kw or "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(kw.strip())
    return out[:MAX_SEEDS]


# Junk-filter rules
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_DIGIT_HEAVY_RE = re.compile(r"^\d+(\s+\d+)*$")


def _is_junk(keyword: str, profile_never_say: list[str]) -> bool:
    """Quick filters to drop obviously-bad candidates before scoring."""
    if not keyword:
        return True
    kw = keyword.strip().lower()
    if len(kw) < 3:
        return True
    if _NON_ASCII_RE.search(kw):
        return True  # non-English / unicode (DFS sometimes returns these even with en)
    if _DIGIT_HEAVY_RE.match(kw):
        return True
    words = kw.split()
    if len(words) > 8:
        return True  # too long-tail, not actionable
    # never-say words from brand profile (e.g. competitor names)
    for banned in (profile_never_say or []):
        if banned and banned.lower() in kw:
            return True
    return False


def _log_api_cost(
    session,
    *,
    api_name: str,
    endpoint: str,
    cost_usd: float,
    site_id: int,
    purpose: str,
    meta: Optional[dict] = None,
) -> None:
    """Append a row to seo_api_costs. Caller commits."""
    row = SeoApiCost(
        api_name=api_name,
        endpoint=endpoint,
        cost_usd=cost_usd,
        site_id=site_id,
        purpose=purpose,
        meta=meta or {},
    )
    session.add(row)


def discover_keywords_for_site(
    site_id: int,
    *,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    client: Optional[DataForSEOClient] = None,
    location_code: int = 2840,   # United States
    language_code: str = "en",
) -> KeywordRun:
    """Run the full Keyword Engine pipeline for one site.

    Returns a KeywordRun summary. Writes SeoKeyword + SeoApiCost rows.
    """
    started = time.time()
    log.info("keyword_engine: starting site_id=%d", site_id)

    # 1. Load site + profile -------------------------------------------------
    with SessionLocal() as s:
        site = s.execute(select(SeoSite).where(SeoSite.id == site_id)).scalar_one()
        site_slug = site.slug
        wp_rest_url = site.wp_rest_url
    profile = load_profile(site_slug)
    seeds = _seed_list(profile)
    log.info("keyword_engine: site=%s seeds=%d -> %s", site_slug, len(seeds), seeds)

    # 2. DataForSEO client ---------------------------------------------------
    dfs = client or DataForSEOClient()
    spend_before = dfs.total_cost_usd

    # 3. Expand seeds via Labs (one call per seed) ---------------------------
    candidates: dict[str, dict] = {}
    for seed in seeds:
        try:
            log.info("keyword_engine: expanding seed %r (limit=%d)", seed, SUGGESTIONS_PER_SEED)
            sugs = dfs.keyword_suggestions(
                seed_keyword=seed,
                location_code=location_code,
                language_code=language_code,
                limit=SUGGESTIONS_PER_SEED,
            )
        except Exception as e:
            log.exception("keyword_engine: seed %r failed: %s", seed, e)
            continue
        # Also keep the seed itself as a candidate
        if seed.lower() not in candidates:
            candidates[seed.lower()] = {
                "keyword": seed,
                "search_volume": None,
                "keyword_difficulty": None,
                "cpc": None,
                "competition": None,
                "competition_level": None,
                "search_intent": None,
                "source": "seed",
            }
        for item in sugs:
            kw = (item.get("keyword") or "").strip()
            if not kw:
                continue
            k = kw.lower()
            if k not in candidates:
                item["source"] = "labs"
                candidates[k] = item
            else:
                # Merge richer data from Labs onto a seed-originated row
                existing = candidates[k]
                for field_name in ("search_volume", "keyword_difficulty", "cpc",
                                   "competition", "competition_level", "search_intent"):
                    if existing.get(field_name) in (None, "") and item.get(field_name) not in (None, ""):
                        existing[field_name] = item[field_name]

    total_candidates = len(candidates)
    log.info("keyword_engine: %d unique candidate keywords from %d seeds",
             total_candidates, len(seeds))

    # 4. Junk-filter ---------------------------------------------------------
    never_say = list(profile.never_say or [])
    filtered: list[dict] = []
    for c in candidates.values():
        if _is_junk(c["keyword"], never_say):
            continue
        filtered.append(c)
    log.info("keyword_engine: kept %d after junk filter", len(filtered))

    # 5. Backfill missing volumes via Keywords Data (cheap; batched ≤1000) ---
    needs_volume = [c for c in filtered if c.get("search_volume") in (None,)]
    if needs_volume:
        kws = [c["keyword"] for c in needs_volume][:1000]
        log.info("keyword_engine: backfilling search volume for %d keywords", len(kws))
        try:
            vol_results = dfs.keyword_search_volume(
                keywords=kws,
                location_code=location_code,
                language_code=language_code,
            )
            by_kw = {(r.get("keyword") or "").lower(): r for r in vol_results}
            for c in needs_volume:
                m = by_kw.get(c["keyword"].lower())
                if not m:
                    continue
                if c.get("search_volume") is None and m.get("search_volume") is not None:
                    c["search_volume"] = m.get("search_volume")
                for k in ("cpc", "competition", "competition_level"):
                    if c.get(k) in (None, "") and m.get(k) is not None:
                        c[k] = m.get(k)
        except Exception as e:
            log.exception("keyword_engine: search_volume backfill failed: %s", e)

    # 6. Score ---------------------------------------------------------------
    for c in filtered:
        c["opportunity_score"] = score_keyword(
            volume=c.get("search_volume"),
            difficulty=c.get("keyword_difficulty"),
            intent=c.get("search_intent"),
            current_rank=None,  # POC: no GSC data yet
        )

    # 7. Sort + cap ----------------------------------------------------------
    filtered.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
    top = filtered[:max_keywords]
    log.info("keyword_engine: top %d candidates kept (best score=%.1f)",
             len(top), top[0]["opportunity_score"] if top else 0)

    # 8. URL inventory + targeting ------------------------------------------
    try:
        urls: list[SiteUrl] = fetch_wp_url_inventory(wp_rest_url)
        log.info("keyword_engine: fetched %d URLs from %s", len(urls), wp_rest_url)
    except Exception as e:
        log.exception("keyword_engine: URL inventory failed: %s", e)
        urls = []

    for c in top:
        c["target_url"] = pick_target_url(c["keyword"], urls)

    # 9. Persist -------------------------------------------------------------
    keyword_ids: list[int] = []
    seed_cost = max(0.0, dfs.total_cost_usd - spend_before)

    with SessionLocal() as s:
        for c in top:
            existing = s.execute(
                select(SeoKeyword).where(
                    SeoKeyword.site_id == site_id,
                    SeoKeyword.keyword == c["keyword"],
                )
            ).scalar_one_or_none()

            payload = {
                "search_volume": c.get("search_volume"),
                "keyword_difficulty": c.get("keyword_difficulty"),
                "cpc": c.get("cpc"),
                "competition": c.get("competition"),
                "competition_level": c.get("competition_level"),
                "search_intent": c.get("search_intent"),
                "target_url": c.get("target_url"),
                "meta_payload": {
                    "opportunity_score": c.get("opportunity_score"),
                    "source": c.get("source") or "labs",
                    "discovered_via": "dfs_labs_keyword_suggestions",
                },
            }
            if existing:
                for k, v in payload.items():
                    setattr(existing, k, v)
                s.flush()
                keyword_ids.append(existing.id)
            else:
                row = SeoKeyword(
                    site_id=site_id,
                    keyword=c["keyword"],
                    status="active",
                    priority=2,
                    **payload,
                )
                s.add(row)
                s.flush()
                keyword_ids.append(row.id)

        # Log cost (single roll-up — sum across all calls this run)
        _log_api_cost(
            s,
            api_name="dataforseo",
            endpoint="keyword_engine_run",
            cost_usd=seed_cost,
            site_id=site_id,
            purpose="keyword_discovery",
            meta={
                "seeds": seeds,
                "candidates_total": total_candidates,
                "candidates_kept": len(filtered),
                "final_keywords": len(top),
            },
        )
        s.commit()

    finished = time.time()
    log.info("keyword_engine: DONE site=%s final=%d cost=$%.4f elapsed=%.1fs",
             site_slug, len(keyword_ids), seed_cost, finished - started)

    return KeywordRun(
        site_id=site_id,
        site_slug=site_slug,
        seeds_used=seeds,
        candidates_total=total_candidates,
        candidates_kept=len(filtered),
        final_keywords=len(top),
        dfs_cost_usd=seed_cost,
        started_at=started,
        finished_at=finished,
        keyword_ids=keyword_ids,
    )
