#!/usr/bin/env python3
"""RuckTalk Toolkit — daily 8 AM ET pulse to Mike.

Queries the Lucius-built nervous-system toolkit's SQLite DB on 111 over
SSH (no public API needed) and sends a tight Telegram summary of:
  • New signups in the last 24h
  • Set-password completion rate
  • Daily active users
  • Top app + total minutes logged
  • Total verified users

Cron: 0 8 * * *  (server America/New_York)
DB:   /var/www/html/toolkit/data/toolkit.db inside lucius-lab-wp on 111
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from typing import Any

import requests

sys.path.insert(0, "/home/aialfred/alfred")
from config.settings import settings  # noqa: E402

MIKE_CHAT_ID = "7582976864"
SSH_HOST = "server-111"
CONTAINER = "lucius-lab-wp"
DB_PATH = "/var/www/html/toolkit/data/toolkit.db"


def _bot_token() -> str:
    return getattr(settings, "telegram_bot_token", "") or ""


def _run_php(php: str) -> Any:
    """Run a PHP snippet inside the lucius-lab-wp container that opens
    the toolkit DB and prints a JSON-encoded result on the last line.
    Returns the parsed object or None on any failure.
    """
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        SSH_HOST,
        f"timeout 30s docker exec {CONTAINER} php -r {php!r}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    # Parse the LAST JSON line in stdout (PHP may emit warnings before).
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _fetch_stats() -> dict:
    php = r"""
$pdo = new PDO('sqlite:""" + DB_PATH + r"""');
$pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
$now = time();
$day = $now - 86400;
$out = [];
$out['total_users']      = (int)$pdo->query('SELECT COUNT(*) FROM users')->fetchColumn();
$out['verified_users']   = (int)$pdo->query('SELECT COUNT(*) FROM users WHERE verified=1')->fetchColumn();
$out['pending_password'] = (int)$pdo->query('SELECT COUNT(*) FROM users WHERE reset_token IS NOT NULL')->fetchColumn();
$out['signups_24h']      = (int)$pdo->query("SELECT COUNT(*) FROM users WHERE created_at >= $day")->fetchColumn();

$rows = $pdo->query("SELECT email, display_name, created_at, reset_token, last_login FROM users WHERE created_at >= $day ORDER BY created_at DESC")->fetchAll(PDO::FETCH_ASSOC);
$out['new_signups'] = $rows;

$out['dau'] = (int)$pdo->query("SELECT COUNT(DISTINCT user_id) FROM analytics_logins WHERE at >= $day")->fetchColumn();

$st = $pdo->query("SELECT app_name, COUNT(*) as sess, SUM(duration_sec) as secs FROM analytics_sessions WHERE started_at >= $day GROUP BY app_name ORDER BY sess DESC LIMIT 1");
$top = $st->fetch(PDO::FETCH_ASSOC);
$out['top_app']     = $top ? $top['app_name'] : null;
$out['top_app_sessions'] = $top ? (int)$top['sess'] : 0;
$out['top_app_minutes']  = $top ? (int)round((int)$top['secs'] / 60) : 0;

$tot = $pdo->query("SELECT COUNT(*) as sess, SUM(duration_sec) as secs FROM analytics_sessions WHERE started_at >= $day")->fetch(PDO::FETCH_ASSOC);
$out['total_sessions_24h'] = (int)$tot['sess'];
$out['total_minutes_24h']  = (int)round(((int)$tot['secs']) / 60);

echo "\n" . json_encode($out);
"""
    res = _run_php(php)
    return res or {}


def _format_message(s: dict) -> str:
    now_et = datetime.now().strftime("%a %b %d")
    if not s:
        return (f"🧰 *RuckTalk Toolkit* — {now_et}\n\n"
                "Stats fetch failed. Check `scripts/rucktalk_toolkit_daily_pulse.py`.")

    lines = [f"🧰 *RuckTalk Toolkit* — {now_et}"]

    if s.get("signups_24h", 0) == 0 and s.get("dau", 0) == 0:
        lines.append("\n_Quiet 24h — no new signups, no active users._")
    else:
        if s.get("signups_24h"):
            lines.append(f"\n*New signups (24h):* {s['signups_24h']}")
            for r in (s.get("new_signups") or [])[:5]:
                name = r.get("display_name") or "—"
                em = r.get("email") or ""
                done = "✅ set password" if not r.get("reset_token") else "⏳ pending password"
                lines.append(f"  · {name} — {em} ({done})")
        else:
            lines.append("\n*New signups (24h):* 0")

        lines.append("")
        lines.append(f"*Active (24h):* {s.get('dau', 0)} users · {s.get('total_sessions_24h', 0)} sessions · {s.get('total_minutes_24h', 0)} min")
        if s.get("top_app"):
            lines.append(f"*Top app:* {s['top_app']} — {s['top_app_sessions']} sessions / {s['top_app_minutes']} min")

    lines.append("")
    lines.append(f"*Roster:* {s.get('verified_users', 0)} verified · {s.get('pending_password', 0)} pending password · {s.get('total_users', 0)} total")
    lines.append("\nDashboard: https://tech.groundrushlabs.com/toolkit/admin/")
    return "\n".join(lines)


def _send_telegram(text: str) -> bool:
    token = _bot_token()
    if not token:
        print("ERROR: telegram bot token missing", file=sys.stderr)
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={
        "chat_id": MIKE_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }, timeout=15)
    if r.status_code != 200:
        print(f"telegram send failed: {r.status_code} {r.text}", file=sys.stderr)
        return False
    return True


def main() -> int:
    s = _fetch_stats()
    msg = _format_message(s)
    return 0 if _send_telegram(msg) else 1


if __name__ == "__main__":
    sys.exit(main())
