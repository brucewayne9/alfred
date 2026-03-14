#!/usr/bin/env python3
"""Alfred Evening Task Ping — asks Mike at 6 PM what needs doing tomorrow.

Flow:
1. At 6 PM, sends Telegram message asking "What do we need to get done tomorrow?"
2. Mike replies with a list (natural language)
3. Claw picks up the reply and saves to daily_tasks.json
4. Morning brief includes those tasks as a checklist

This script handles:
- The 6 PM outbound ping
- Checking for replies and parsing tasks
- Saving to daily_tasks.json for morning_brief.py

Cron:
  0 18 * * * cd /home/aialfred/alfred && /home/aialfred/.pyenv/versions/3.11.11/bin/python3 scripts/evening_task_ping.py ping >> /tmp/evening_ping.log 2>&1
  */5 18-23 * * * cd /home/aialfred/alfred && /home/aialfred/.pyenv/versions/3.11.11/bin/python3 scripts/evening_task_ping.py check >> /tmp/evening_ping.log 2>&1
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evening_ping")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = "7582976864"
TASK_FILE = Path("/home/aialfred/alfred/data/daily_tasks.json")
PING_STATE = Path("/home/aialfred/alfred/data/evening_ping_state.json")
ALFRED_API = "http://localhost:8400"


def send_telegram(text, reply_markup=None):
    """Send message to Mike via Telegram."""
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json=payload,
        timeout=10,
    )
    return resp.json()


def get_recent_messages(limit=10):
    """Get recent messages from Telegram to check for Mike's reply."""
    resp = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
        params={"offset": -limit, "limit": limit},
        timeout=10,
    )
    data = resp.json()
    return data.get("result", [])


def load_ping_state():
    if PING_STATE.exists():
        return json.loads(PING_STATE.read_text())
    return {}


def save_ping_state(state):
    PING_STATE.write_text(json.dumps(state, indent=2))


def load_tasks():
    if TASK_FILE.exists():
        return json.loads(TASK_FILE.read_text())
    return {}


def save_tasks(data):
    TASK_FILE.write_text(json.dumps(data, indent=2))


def push_tasks_to_webapp(task_texts: list[str], target_date: str):
    """Push tasks to the Alfred Labs task app via /api/tasks/bulk."""
    try:
        session = requests.Session()
        # Auto-login from localhost to get JWT cookie
        resp = session.get(f"{ALFRED_API}/auth/auto", timeout=5)
        if not resp.json().get("auto_login"):
            log.error("Tasks webapp: auto-login failed")
            return False

        # Bulk create tasks
        resp = session.post(
            f"{ALFRED_API}/api/tasks/bulk",
            json={"texts": task_texts, "target_date": target_date, "source": "evening_ping"},
            timeout=10,
        )
        if resp.status_code == 200:
            count = resp.json().get("count", 0)
            log.info(f"Tasks webapp: pushed {count} tasks for {target_date}")
            return True
        else:
            log.error(f"Tasks webapp: bulk create failed ({resp.status_code}): {resp.text[:200]}")
            return False
    except Exception as e:
        log.error(f"Tasks webapp: push failed: {e}")
        return False


def parse_tasks_from_text(text):
    """Parse a natural language task list into structured tasks."""
    tasks = []
    lines = text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common list prefixes
        line = re.sub(r"^[-•*]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = line.strip()

        if line and len(line) > 2:
            tasks.append({"text": line, "done": False})

    # If no newlines found, try splitting by commas or "and"
    if not tasks and text.strip():
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", text.strip())
        for part in parts:
            part = part.strip().rstrip(".")
            if part and len(part) > 2:
                tasks.append({"text": part, "done": False})

    return tasks


def cmd_ping():
    """Send the evening ping to Mike."""
    state = load_ping_state()
    today = datetime.now().strftime("%Y-%m-%d")

    # Don't ping twice in one day
    if state.get("last_ping_date") == today:
        log.info("Already pinged today, skipping")
        return

    result = send_telegram(
        "🌙 <b>Evening Check-in</b>\n\n"
        "Hey Mike — what do we need to get done tomorrow?\n\n"
        "Just reply with your list and I'll have it ready in your morning brief. "
        "Can be bullet points, numbered, or just free text.\n\n"
        "<i>Example: Follow up with Nike, review sponsorship deck, update LoovaCast playlist, call accountant</i>"
    )

    if result.get("ok"):
        state["last_ping_date"] = today
        state["ping_message_id"] = result["result"]["message_id"]
        state["ping_timestamp"] = datetime.now(timezone.utc).isoformat()
        state["tasks_received"] = False
        save_ping_state(state)
        log.info("Evening ping sent")
    else:
        log.error(f"Failed to send ping: {result}")


def cmd_check():
    """Check for Mike's reply with tomorrow's tasks."""
    state = load_ping_state()
    today = datetime.now().strftime("%Y-%m-%d")

    # Only check if we pinged today and haven't received tasks yet
    if state.get("last_ping_date") != today or state.get("tasks_received"):
        return

    ping_ts = state.get("ping_timestamp", "")
    if not ping_ts:
        return

    ping_time = datetime.fromisoformat(ping_ts)

    # Get recent Telegram updates
    updates = get_recent_messages(20)

    for update in updates:
        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        msg_date = datetime.fromtimestamp(msg.get("date", 0), tz=timezone.utc)
        text = msg.get("text", "")

        # Check if it's from Mike's chat, after the ping, and has content
        if chat_id == TELEGRAM_CHAT_ID and msg_date > ping_time and text and len(text) > 3:
            # Skip if it looks like a command or unrelated
            if text.startswith("/") or text.startswith("!"):
                continue

            # Parse tasks
            tasks = parse_tasks_from_text(text)

            if tasks:
                # Save tasks for tomorrow
                tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                task_data = load_tasks()
                task_data[tomorrow] = tasks
                # Also set as "tasks" key for easy access
                task_data["tasks"] = tasks
                task_data["set_date"] = today
                task_data["for_date"] = tomorrow
                save_tasks(task_data)

                # Push to Alfred Labs task app
                push_tasks_to_webapp([t["text"] for t in tasks], tomorrow)

                # Confirm to Mike
                task_list = "\n".join([f"  ✓ {t['text']}" for t in tasks])
                send_telegram(
                    f"✅ <b>Got it!</b> {len(tasks)} tasks locked in for tomorrow:\n\n"
                    f"{task_list}\n\n"
                    f"I'll have these in your morning brief. Sleep well! 🌙"
                )

                state["tasks_received"] = True
                save_ping_state(state)

                log.info(f"Received {len(tasks)} tasks for tomorrow")
                return

    log.info("No task reply found yet")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: evening_task_ping.py [ping|check]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ping":
        cmd_ping()
    elif cmd == "check":
        cmd_check()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
