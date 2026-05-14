#!/usr/bin/env python3
"""CLI: GSC sync. Defaults to yesterday's data for all active sites."""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.gsc import sync_all_sites_for_date


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD, defaults to yesterday UTC")
    args = p.parse_args()
    date = dt.date.fromisoformat(args.date) if args.date else (dt.datetime.utcnow().date() - dt.timedelta(days=1))
    result = sync_all_sites_for_date(date)
    for slug, n in result.items():
        print(f"  {slug}: {n} rows" if n >= 0 else f"  {slug}: FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
