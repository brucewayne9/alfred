#!/usr/bin/env python3
"""Cron wrapper for core.audit.booking_sync.

Add to crontab on the same cadence as the email monitor:
    */10 * * * * /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/booking_sync_cron.py >> /home/aialfred/alfred/data/booking_sync.log 2>&1

Until `EMAIL_PASS_MJOHNSON_GW` is set in config/.env, this exits cleanly with a
"skipped" status so the cron doesn't generate errors.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

from core.audit.booking_sync import run_once  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

if __name__ == "__main__":
    summary = run_once()
    logging.info(f"booking_sync: {summary}")
