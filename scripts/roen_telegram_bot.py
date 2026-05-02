"""
Roen Handmade intake bot — Telegram polling loop.

Sarah sends photos + a price; bot acks, runs the pipeline in a worker
thread, replies with the draft URL.

Allowlisted to a small set of chat IDs (Sarah, Mike). Anyone else is ignored.

Run as a systemd service. Polls Telegram getUpdates with long offsets so we
don't lose messages across restarts.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Set

import requests

# Make /home/aialfred/alfred importable when run via systemd
sys.path.insert(0, "/home/aialfred/alfred")

from core.jewelry import db, pipeline  # noqa: E402

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("roen-bot")

CONFIG_PATH = Path("/home/aialfred/alfred/config/.env")
UPLOAD_DIR = Path("/home/aialfred/alfred/data/roen/uploads")
OFFSET_PATH = Path("/home/aialfred/alfred/data/roen/last_offset.txt")
LOG_FILE = Path("/home/aialfred/alfred/data/roen/bot.log")


# ----------------------- config loading -----------------------

def load_env() -> dict:
    env = {}
    for line in CONFIG_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip("'").strip('"')
    return env


_env = load_env()
TOKEN = _env.get("TELEGRAM_BOT_ROENHANDMADE_TOKEN", "").strip()
if not TOKEN:
    print("FATAL: TELEGRAM_BOT_ROENHANDMADE_TOKEN missing from config/.env", file=sys.stderr)
    sys.exit(2)

ALLOWED_CHAT_IDS: Set[int] = {
    int(x.strip())
    for x in _env.get("ROEN_INTAKE_ALLOWED_CHAT_IDS", "").split(",")
    if x.strip().isdigit()
}
if not ALLOWED_CHAT_IDS:
    print("FATAL: ROEN_INTAKE_ALLOWED_CHAT_IDS empty", file=sys.stderr)
    sys.exit(2)

API = f"https://api.telegram.org/bot{TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{TOKEN}"


# ----------------------- file logger (in addition to stderr/journald) -----------------------

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(file_handler)


# ----------------------- Telegram helpers -----------------------

def tg(method: str, **params):
    """Call a Telegram Bot API method, return parsed result or raise."""
    r = requests.post(f"{API}/{method}", json=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"telegram {method} failed: {j}")
    return j["result"]


def send_message(chat_id: int, text: str, reply_to: Optional[int] = None) -> None:
    try:
        params = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_to is not None:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        tg("sendMessage", **params)
    except Exception:
        logger.exception("send_message to %s failed", chat_id)


def download_photo(file_id: str, dest_path: Path) -> None:
    """Resolve a Telegram file_id to a downloaded file."""
    info = tg("getFile", file_id=file_id)
    rel = info["file_path"]
    url = f"{FILE_API}/{rel}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest_path.open("wb") as f:
            shutil.copyfileobj(r.raw, f)


# ----------------------- intake helpers -----------------------

PRICE_PATTERN = re.compile(r"\$?\s*(\d{1,5})(?:[.,](\d{2}))?")


def parse_price_cents(text: str) -> Optional[int]:
    """Return price in cents, or None if no clear price in the text."""
    if not text:
        return None
    m = PRICE_PATTERN.search(text)
    if not m:
        return None
    dollars = int(m.group(1))
    cents = int(m.group(2)) if m.group(2) else 0
    return dollars * 100 + cents


def get_or_open_intake(chat_id: int, user_handle: Optional[str]) -> int:
    row = db.get_active_intake(chat_id)
    if row is not None:
        return int(row["id"])
    return db.open_intake(chat_id, user_handle)


# ----------------------- worker -----------------------

executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="roen-pipeline")


def kick_off_pipeline(intake_id: int, chat_id: int) -> None:
    """Submit the pipeline to the worker pool and wire progress back to Telegram."""
    def progress(msg: str):
        send_message(chat_id, msg)

    def run():
        try:
            result = pipeline.process_intake(intake_id, progress_callback=progress)
            send_message(
                chat_id,
                (
                    f"Draft is ready, sir.\n\n"
                    f"<b>{result['name']}</b>\n"
                    f"SKU: <code>{result['sku']}</code>\n\n"
                    f"Preview: {result['preview_url']}\n"
                    f"Edit:    {result['edit_url']}\n\n"
                    f"It's saved as a draft. Reply <b>publish</b> to make it live."
                ),
            )
        except Exception as e:
            logger.exception("pipeline failed for intake %d", intake_id)
            db.set_status(intake_id, "error", error=str(e))
            send_message(
                chat_id,
                f"Something went wrong while processing that piece: {e}\n\n"
                f"Mike's been notified — try again or send fresh photos."
            )

    executor.submit(run)


# ----------------------- update handlers -----------------------

def handle_photo(msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    user = msg.get("from", {})
    user_handle = user.get("username") or str(user.get("id", ""))

    photos = msg.get("photo") or []
    if not photos:
        return

    # Telegram sends each photo at multiple resolutions. The last entry is the largest.
    largest = photos[-1]
    file_id = largest["file_id"]

    intake_id = get_or_open_intake(chat_id, user_handle)
    intake_dir = UPLOAD_DIR / f"intake_{intake_id}"
    intake_dir.mkdir(parents=True, exist_ok=True)
    local = intake_dir / f"photo_{int(time.time()*1000)}.jpg"
    try:
        download_photo(file_id, local)
    except Exception:
        logger.exception("download_photo failed for %s", file_id)
        send_message(chat_id, "Couldn't download that photo — try again?")
        return
    db.add_photo(intake_id, str(local), file_id)

    # If the photo arrived with a caption containing a price, capture it.
    caption = (msg.get("caption") or "").strip()
    price_cents = parse_price_cents(caption)
    if price_cents:
        db.set_price(intake_id, price_cents, caption)
        _react_and_maybe_kick(chat_id, intake_id, msg.get("message_id"))
    else:
        # Quiet ack — the eventual price message will trigger processing.
        # Don't spam reactions on every photo.
        intake = db.get_intake(intake_id)
        photo_count = len(_photos_of(intake))
        if photo_count == 1:
            send_message(
                chat_id,
                "Got the first photo, sir. Send the rest plus the price (e.g. $45) and I'll get to work.",
                reply_to=msg.get("message_id"),
            )


def handle_text(msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()
    if not text:
        return

    lower = text.lower()

    if lower in ("/start", "/help"):
        _send_help(chat_id, msg.get("message_id"))
        return

    if lower in ("/status", "status"):
        _send_status(chat_id, msg.get("message_id"))
        return

    if lower in ("/cancel", "cancel"):
        row = db.get_active_intake(chat_id)
        if row is not None:
            db.set_status(int(row["id"]), "cancelled")
            send_message(chat_id, "Cancelled the current intake. Send fresh photos when you're ready.")
        else:
            send_message(chat_id, "Nothing to cancel — no active intake.")
        return

    if lower in ("publish", "/publish"):
        _publish_last(chat_id, msg.get("message_id"))
        return

    # Default: try to parse a price.
    price_cents = parse_price_cents(text)
    if price_cents is None:
        send_message(
            chat_id,
            "I didn't catch a price in that. Send something like <b>$45</b> with your photos.",
            reply_to=msg.get("message_id"),
        )
        return

    row = db.get_active_intake(chat_id)
    if row is None:
        send_message(
            chat_id,
            f"Got the price (${price_cents/100:.2f}) but I don't have any photos yet — send the photos first.",
            reply_to=msg.get("message_id"),
        )
        return

    intake_id = int(row["id"])
    db.set_price(intake_id, price_cents, text)
    _react_and_maybe_kick(chat_id, intake_id, msg.get("message_id"))


def _react_and_maybe_kick(chat_id: int, intake_id: int, reply_to: Optional[int]) -> None:
    intake = db.get_intake(intake_id)
    photos = _photos_of(intake)
    price_cents = intake["price_cents"]
    if not photos or not price_cents:
        return
    n = len(photos)
    send_message(
        chat_id,
        f"Got {n} photo{'s' if n != 1 else ''} at ${price_cents/100:.2f}. Working on it...",
        reply_to=reply_to,
    )
    kick_off_pipeline(intake_id, chat_id)


def _photos_of(intake) -> List[dict]:
    if intake is None:
        return []
    import json as _json
    return _json.loads(intake["photos_json"])


def _send_help(chat_id: int, reply_to: Optional[int]) -> None:
    msg = (
        "<b>Roen Handmade intake</b>\n\n"
        "Send 1 to 5 photos of a finished piece, plus the price (e.g. <code>$45</code>). "
        "I'll write the listing and create a draft on the website.\n\n"
        "<b>Commands</b>\n"
        "  <code>/status</code>  — show your last intake\n"
        "  <code>/cancel</code>  — drop the in-progress intake\n"
        "  <code>publish</code>  — publish the last draft (Mike only, for now)"
    )
    send_message(chat_id, msg, reply_to=reply_to)


def _send_status(chat_id: int, reply_to: Optional[int]) -> None:
    row = db.get_active_intake(chat_id, max_age_seconds=86400)
    if row is None:
        send_message(chat_id, "No active intake. Send photos to start one.", reply_to=reply_to)
        return
    photos = _photos_of(row)
    price = row["price_cents"]
    msg = (
        f"Intake #{row['id']}: <b>{row['status']}</b>\n"
        f"Photos: {len(photos)}\n"
        f"Price: {'$%.2f' % (price/100) if price else 'not set'}\n"
    )
    if row["woocommerce_post_id"]:
        from core.jewelry.woocommerce import preview_url, admin_edit_url
        msg += (
            f"Draft post: {row['woocommerce_post_id']}\n"
            f"Preview: {preview_url(row['woocommerce_post_id'])}\n"
            f"Edit: {admin_edit_url(row['woocommerce_post_id'])}"
        )
    send_message(chat_id, msg, reply_to=reply_to)


def _publish_last(chat_id: int, reply_to: Optional[int]) -> None:
    """Mike-only: publish the most recent draft for this chat."""
    if chat_id != 7582976864:
        send_message(chat_id, "Only Mike can publish drafts for now.", reply_to=reply_to)
        return
    row = db.get_active_intake(chat_id, max_age_seconds=86400)
    if row is None or not row["woocommerce_post_id"]:
        send_message(chat_id, "No recent draft to publish.", reply_to=reply_to)
        return
    from core.jewelry.woocommerce import _ssh_docker_wp
    rc, out, err = _ssh_docker_wp(
        ["post", "update", str(row["woocommerce_post_id"]), "--post_status=publish"],
        timeout=30,
    )
    if rc == 0:
        send_message(chat_id, f"Published post {row['woocommerce_post_id']}.", reply_to=reply_to)
    else:
        send_message(chat_id, f"Publish failed: {err.strip()}", reply_to=reply_to)


# ----------------------- main loop -----------------------

def load_offset() -> int:
    if OFFSET_PATH.exists():
        try:
            return int(OFFSET_PATH.read_text().strip())
        except Exception:
            pass
    return 0


def save_offset(offset: int) -> None:
    OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_PATH.write_text(str(offset))


_running = True


def _sigterm(*_):
    global _running
    logger.info("SIGTERM received, shutting down")
    _running = False


def main() -> None:
    db.init()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    me = tg("getMe")
    logger.info("bot online as @%s (id=%s)", me["username"], me["id"])
    logger.info("allowed chat_ids: %s", ALLOWED_CHAT_IDS)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    offset = load_offset()

    while _running:
        try:
            updates = tg("getUpdates", offset=offset, timeout=25, allowed_updates=["message"])
        except Exception:
            logger.exception("getUpdates failed; sleeping")
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            save_offset(offset)
            msg = update.get("message")
            if not msg:
                continue
            chat_id = msg["chat"]["id"]
            if chat_id not in ALLOWED_CHAT_IDS:
                logger.warning("ignoring message from unallowed chat %s (text=%r)", chat_id, msg.get("text"))
                continue
            try:
                if "photo" in msg:
                    handle_photo(msg)
                elif "text" in msg:
                    handle_text(msg)
            except Exception:
                logger.exception("handler crashed for message %s", msg.get("message_id"))

    logger.info("bot loop exited")


if __name__ == "__main__":
    main()
