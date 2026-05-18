"""Weekly Rank Tracker — pulls Google SERPs for every active keyword on a
given site and writes results to seo_rankings_daily.

Default site: roen (the POC). Pass --site to track a different slug.

Cron entry (host TZ = America/New_York, so this fires at 9 AM ET):
    0 9 * * 0  /home/aialfred/alfred/venv/bin/python -m scripts.seo_rank_tracker --site roen \
        >> /home/aialfred/alfred/data/seo/rank_tracker.log 2>&1

Sunday 9 AM ET = quiet window, after Saturday CRM auto-jobs, before the
Monday morning brief reads the dashboard.
"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("rank_tracker_runner")

from core.seo.rankings import track_site


def _fmt_pos(p):
    return f"{p:>3}" if p is not None else "  —"


def _fmt_delta(d):
    if d is None:
        return "    "
    if d > 0:
        return f"+{d:>2}↑"
    if d < 0:
        return f"{d:>3}↓"
    return "  ="


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="roen", help="seo_sites.slug (default: roen)")
    ap.add_argument("--limit", type=int, default=None, help="cap keywords (default: all active)")
    ap.add_argument("--dry-run", action="store_true", help="skip DB writes")
    args = ap.parse_args()

    report = track_site(args.site, limit=args.limit, dry_run=args.dry_run)

    print()
    print("=" * 90)
    print(f"RANK TRACKER — {report.site_slug} — {report.captured_at}  (dry_run={args.dry_run})")
    print("=" * 90)
    print(f"  pulled:  {len(report.results)}")
    print(f"  ranked:  {report.ranked_count}  (top10={report.top10_count}, top3={report.top3_count})")
    print(f"  spend:   ${report.cost_usd:.4f}")
    print(f"  fails:   {len(report.failures)}")
    print()
    print(f"  {'POS':>4}  {'Δ':>5}  {'PRIOR':>5}  KEYWORD  ->  URL")
    print("-" * 90)
    # Sort: ranked first (best first), then unranked alphabetical.
    def _sort_key(r):
        return (r.position is None, r.position if r.position is not None else 999, r.keyword)
    for r in sorted(report.results, key=_sort_key):
        url = (r.found_url or "")[-55:] if r.found_url else ""
        prior = _fmt_pos(r.prior_position)
        print(f"  {_fmt_pos(r.position)}  {_fmt_delta(r.delta):>5}  {prior:>5}  {r.keyword:<35}  {url}")

    if report.failures:
        print()
        print("FAILURES:")
        for f in report.failures:
            print(f"  {f['keyword']}: {f['error']}")

    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
