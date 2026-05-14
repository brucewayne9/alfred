#!/usr/bin/env python3
"""CLI: backlinks sync (Layer 1 — passive monitor)."""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from core.seo.ingest.backlinks import sync_all_sites_from_gsc


def main() -> int:
    p = argparse.ArgumentParser()
    p.parse_args()
    for slug, result in sync_all_sites_from_gsc().items():
        print(f"  {slug}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
