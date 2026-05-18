"""Daily check for stale drafts in the SEO approval queue.

If any seo_pending row has been sitting >5 days, fire a Telegram nudge to
Mike. Single message per run, even if multiple drafts are stale.

Cron (host TZ = America/New_York):
  0 8 * * * /home/aialfred/alfred/venv/bin/python -m scripts.seo_queue_stale_check \
    >> /home/aialfred/alfred/data/seo/blog_engine.log 2>&1
"""
from __future__ import annotations

import datetime as dt
import logging
import sys

import httpx

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("seo.queue_stale_check")

from config.settings import settings
from core.seo.db import SessionLocal
from sqlalchemy import text

STALE_DAYS = 5
MIKE_CHAT_ID = "7582976864"


def main() -> int:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=STALE_DAYS)
    with SessionLocal() as s:
        rows = s.execute(text("""
            SELECT p.id, p.title, p.content_type, p.created_at,
                   s.display_name AS site_name, s.slug AS site_slug
            FROM seo_pending p
            JOIN seo_sites s ON s.id = p.site_id
            WHERE p.status = 'pending' AND p.created_at < :cutoff
            ORDER BY p.created_at ASC
        """), {"cutoff": cutoff}).all()

    if not rows:
        log.info("no stale drafts (cutoff %d days)", STALE_DAYS)
        return 0

    log.info("found %d stale drafts", len(rows))
    lines = [f"⏳ *SEO queue has {len(rows)} draft{'' if len(rows) == 1 else 's'} stale (>{STALE_DAYS}d)*"]
    for r in rows[:8]:
        age_d = (dt.datetime.now(dt.timezone.utc) - r.created_at).days
        title = (r.title or "Untitled")[:60]
        lines.append(f"\n• _{r.site_name}_ · {r.content_type} · {age_d}d old\n  {title}")
    if len(rows) > 8:
        lines.append(f"\n+ {len(rows) - 8} more…")
    lines.append("\n→ https://aialfred.groundrushcloud.com/admin/seo/pending")

    token = getattr(settings, "telegram_bot_token", "") or ""
    if not token:
        log.warning("telegram skipped — TELEGRAM_BOT_TOKEN missing")
        return 0
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": MIKE_CHAT_ID,
                "text": "\n".join(lines),
                "disable_web_page_preview": True,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        r.raise_for_status()
    except Exception:
        log.exception("telegram send failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
