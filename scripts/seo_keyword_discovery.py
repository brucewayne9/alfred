"""CLI: run keyword discovery for a site.

Usage:
  PYTHONPATH=. venv/bin/python scripts/seo_keyword_discovery.py --slug roen [--max 25]

Writes results to seo_keywords; logs spend to seo_api_costs.
Prints a summary table to stdout for quick inspection.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("seo_keyword_discovery")

from sqlalchemy import select

from core.seo.db import SessionLocal
from core.seo.keywords import discover_keywords_for_site
from core.seo.models import SeoKeyword, SeoSite


def _print_top(site_id: int, n: int = 25) -> None:
    with SessionLocal() as s:
        rows = s.execute(
            select(SeoKeyword)
            .where(SeoKeyword.site_id == site_id)
            .order_by(SeoKeyword.id.desc())
            .limit(200)
        ).scalars().all()

    # Sort by opportunity score from meta_payload (most recent rows first)
    def opp(r):
        return (r.meta_payload or {}).get("opportunity_score") or 0.0

    rows.sort(key=opp, reverse=True)
    rows = rows[:n]

    print("\n┌─────┬──────┬───────┬─────┬──────┬──────────────────────────────────────────────┬────────────────────────────────────────────────────┐")
    print(f"│ {'Opp':>3} │ {'Vol':>4} │ {'KD':>5} │ {'Int':<3} │ {'Comp':<4} │ {'Keyword':<44} │ {'Target URL':<50} │")
    print("├─────┼──────┼───────┼─────┼──────┼──────────────────────────────────────────────┼────────────────────────────────────────────────────┤")
    for r in rows:
        meta = r.meta_payload or {}
        score = meta.get("opportunity_score") or 0
        vol = r.search_volume or 0
        kd = r.keyword_difficulty if r.keyword_difficulty is not None else "—"
        intent = (r.search_intent or "—")[:3]
        comp = (r.competition_level or "—")[:4]
        kw = (r.keyword or "")[:44]
        url = (r.target_url or "(create new)")[:50]
        print(f"│ {score:>3.0f} │ {vol:>4} │ {str(kd):>5} │ {intent:<3} │ {comp:<4} │ {kw:<44} │ {url:<50} │")
    print("└─────┴──────┴───────┴─────┴──────┴──────────────────────────────────────────────┴────────────────────────────────────────────────────┘")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, help="Site slug, e.g. roen")
    parser.add_argument("--max", type=int, default=25, help="Number of keywords to keep (default 25)")
    parser.add_argument("--no-print", action="store_true", help="Suppress the summary table")
    args = parser.parse_args()

    with SessionLocal() as s:
        site = s.execute(select(SeoSite).where(SeoSite.slug == args.slug)).scalar_one_or_none()
    if not site:
        print(f"ERROR: no site with slug={args.slug!r}", file=sys.stderr)
        return 2

    print(f"\n=== Keyword Discovery — {site.display_name} ({site.domain}) ===")
    run = discover_keywords_for_site(site_id=site.id, max_keywords=args.max)
    print(f"\nDONE in {run.elapsed_seconds:.1f}s — "
          f"{run.candidates_total} candidates → {run.candidates_kept} kept → {run.final_keywords} final")
    print(f"DataForSEO spend this run: ${run.dfs_cost_usd:.4f}")

    if not args.no_print:
        _print_top(site.id, n=args.max)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
