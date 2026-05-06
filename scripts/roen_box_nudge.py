#!/usr/bin/env python3
"""
Daily reminder: ping Sarah on Telegram if there are bracelet-box pick
sessions pending for more than 24 hours. Run from cron at 9am ET.

Exit codes:
    0 — ran cleanly (whether or not a message was sent)
    1 — error (missing config, Telegram failure, etc.)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, "/home/aialfred/alfred")

import requests

from config.settings import Settings
from core.jewelry.bracelet_box import db as box_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("box-nudge")

PENDING_THRESHOLD_SECONDS = 24 * 3600


def _load_env() -> dict:
    """Same minimal .env reader the bot uses — Settings doesn't expose
    TELEGRAM_BOT_ROENHANDMADE_TOKEN or ROEN_INTAKE_ALLOWED_CHAT_IDS."""
    env: dict[str, str] = {}
    path = Path("/home/aialfred/alfred/config/.env")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip("'").strip('"')
    return env


def main() -> int:
    env = _load_env()
    token = env.get("TELEGRAM_BOT_ROENHANDMADE_TOKEN", "").strip()
    if not token:
        log.error("TELEGRAM_BOT_ROENHANDMADE_TOKEN missing from config/.env")
        return 1

    allowed = [
        int(x.strip())
        for x in env.get("ROEN_INTAKE_ALLOWED_CHAT_IDS", "").split(",")
        if x.strip().isdigit()
    ]
    if not allowed:
        log.error("ROEN_INTAKE_ALLOWED_CHAT_IDS missing or empty")
        return 1
    sarah_chat_id = allowed[0]

    pending = box_db.list_pending(older_than_seconds=PENDING_THRESHOLD_SECONDS)
    if not pending:
        log.info("no pending picks older than 24h — nothing to nudge")
        return 0

    n = len(pending)
    text = f"📦 {n} bracelet-box pick{'s' if n > 1 else ''} waiting on you."
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": sarah_chat_id, "text": text},
            timeout=15,
        )
        r.raise_for_status()
        log.info("nudge sent: %s", text)
        return 0
    except requests.RequestException:
        log.exception("nudge sendMessage failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
