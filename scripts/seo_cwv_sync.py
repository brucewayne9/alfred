#!/usr/bin/env python3
"""CLI: CWV sync. Pulls top-N pages per site through PageSpeed Insights."""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.cwv import sync_all_sites


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    for slug, n in sync_all_sites(args.limit).items():
        print(f"  {slug}: {n} URLs scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
