#!/usr/bin/env python3
"""CLI: run site audit for a slug.

Usage:
    seo_audit_site.py --slug roen [--max-pages 100] [--with-alt-backfill]

Wires core.seo.audit.runner.run_site_audit. The DataForSEO On-Page audit
is a blocking ~3-5 min call wall-clock; the script prints progress so
Mike isn't staring at a silent terminal.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from core.seo.audit.runner import run_site_audit  # noqa: E402
from core.seo.sites.registry import get_site_by_slug  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run a full SEO site audit for one registered site.",
    )
    p.add_argument("--slug", required=True, help="seo_sites.slug (e.g. 'roen')")
    p.add_argument("--max-pages", type=int, default=100,
                   help="DataForSEO max_crawl_pages (default 100)")
    p.add_argument("--with-alt-backfill", action="store_true",
                   help="Generate alt-text suggestions for up to 10 missing-alt images")
    p.add_argument("--alt-cap", type=int, default=10,
                   help="Max images to alt-text per run (default 10)")
    args = p.parse_args()

    site = get_site_by_slug(args.slug)
    if not site:
        print(f"error: no site registered with slug={args.slug!r}", file=sys.stderr)
        return 2

    print(f"running site audit: slug={site.slug} domain={site.domain} max_pages={args.max_pages}")
    run = run_site_audit(
        site_id=site.id,
        max_pages=args.max_pages,
        with_alt_backfill=args.with_alt_backfill,
        alt_backfill_cap=args.alt_cap,
    )

    print(json.dumps({
        "run_id": run.run_id,
        "site_id": run.site_id,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat(),
        "pages_crawled": run.pages_crawled,
        "issues_detected": run.issues_detected,
        "issues_new": run.issues_new,
        "issues_resolved": run.issues_resolved,
        "issues_still_open": run.issues_still_open,
        "dfs_cost_usd": run.dfs_cost_usd,
        "alt_texts_generated": run.alt_texts_generated,
        "errors": run.errors,
    }, indent=2))
    return 0 if not run.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
