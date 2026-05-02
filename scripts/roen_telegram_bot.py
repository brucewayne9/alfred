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

from core.jewelry import db, pipeline, edit as edit_mod  # noqa: E402
from core.jewelry.woocommerce import preview_url, admin_edit_url  # noqa: E402

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

executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="roen-pipeline")

# --- album coalescing state ---
# Telegram delivers albums as N separate messages with the same media_group_id.
# We map group_id -> intake_id so all photos land in one intake, and we use a
# debounced timer so the pipeline only fires once the album has gone quiet.
_group_intakes: dict = {}            # media_group_id -> intake_id
_group_intakes_lock = threading.Lock()
_album_timers: dict = {}             # media_group_id -> threading.Timer
_album_lock = threading.Lock()
_kicked: set = set()                 # intake_ids already submitted to the pipeline
_kicked_lock = threading.Lock()
ALBUM_QUIET_SECONDS = 3.0


def _intake_for_album(chat_id: int, user_handle: Optional[str], media_group_id: str) -> int:
    """Return the intake_id for this album, creating it on first photo."""
    with _group_intakes_lock:
        existing = _group_intakes.get(media_group_id)
        if existing is not None:
            row = db.get_intake(existing)
            if row and row["status"] == "received":
                return existing
            _group_intakes.pop(media_group_id, None)
        intake_id = db.open_intake(chat_id, user_handle)
        _group_intakes[media_group_id] = intake_id
        return intake_id


def _schedule_album_kick(media_group_id: str, chat_id: int, intake_id: int) -> None:
    """Debounce: every photo arrival resets a 3-second quiet timer; when it fires we kick off."""
    with _album_lock:
        existing = _album_timers.pop(media_group_id, None)
        if existing:
            existing.cancel()
        timer = threading.Timer(
            ALBUM_QUIET_SECONDS,
            _album_timer_fire,
            args=(media_group_id, chat_id, intake_id),
        )
        timer.daemon = True
        _album_timers[media_group_id] = timer
        timer.start()


def _album_timer_fire(media_group_id: str, chat_id: int, intake_id: int) -> None:
    with _album_lock:
        _album_timers.pop(media_group_id, None)
    intake = db.get_intake(intake_id)
    if intake is None or intake["status"] != "received":
        return
    photos = _photos_of(intake)
    if not photos:
        return
    if not intake["price_cents"]:
        send_message(
            chat_id,
            f"Got {len(photos)} photos. What's the price for that piece? (e.g. <b>$45</b>)",
        )
        return
    _react_and_maybe_kick(chat_id, intake_id, reply_to=None)


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

    media_group_id = msg.get("media_group_id")  # set when the photo is part of an album

    if media_group_id:
        intake_id = _intake_for_album(chat_id, user_handle, media_group_id)
    else:
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

    caption = (msg.get("caption") or "").strip()
    price_cents = parse_price_cents(caption)
    if price_cents:
        db.set_price(intake_id, price_cents, caption)

    if media_group_id:
        # Defer to the debounce timer so all album siblings land first.
        _schedule_album_kick(media_group_id, chat_id, intake_id)
        return

    if price_cents:
        _react_and_maybe_kick(chat_id, intake_id, msg.get("message_id"))
    else:
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

    if lower in ("/today", "today"):
        _send_today(chat_id, msg.get("message_id"))
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

    # If this is a quote-reply to a bot draft message, treat as an edit.
    reply_to_msg = msg.get("reply_to_message") or {}
    quoted_text = (reply_to_msg.get("text") or "") + " " + (reply_to_msg.get("caption") or "")
    quoted_post_id = edit_mod.extract_post_id(quoted_text)
    if quoted_post_id is not None:
        _handle_edit(chat_id, quoted_post_id, text, msg.get("message_id"))
        return

    # Default: try to parse a price.
    price_cents = parse_price_cents(text)
    if price_cents is None:
        send_message(
            chat_id,
            "I didn't catch a price in that. Send something like <b>$45</b> with your photos, "
            "or quote-reply a draft message to edit it.",
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


def _handle_edit(chat_id: int, post_id: int, text: str, reply_to: Optional[int]) -> None:
    """Quote-reply edit: parse intent, apply, reply with refreshed preview link."""
    send_message(chat_id, "Working on that edit...", reply_to=reply_to)
    try:
        intent = edit_mod.parse_intent(text)
    except Exception:
        logger.exception("intent parse failed")
        send_message(chat_id, "Couldn't parse that edit — try again or be more specific.", reply_to=reply_to)
        return

    if intent["action"] == "unknown":
        send_message(
            chat_id,
            (
                "I didn't catch that edit. Try one of:\n"
                "  <code>change name to ...</code>\n"
                "  <code>price is $55</code>\n"
                "  <code>rewrite description, more poetic</code>\n"
                "  <code>delete</code>  /  <code>publish</code>"
            ),
            reply_to=reply_to,
        )
        return

    try:
        result = edit_mod.apply_edit(post_id, intent)
    except Exception as e:
        logger.exception("apply_edit failed for post %s", post_id)
        send_message(chat_id, f"Edit failed: {e}", reply_to=reply_to)
        return

    if not result["ok"]:
        send_message(chat_id, result["msg"], reply_to=reply_to)
        return

    if intent["action"] == "delete":
        send_message(chat_id, result["msg"], reply_to=reply_to)
        return

    send_message(
        chat_id,
        f"{result['msg']}\n\nPreview: {preview_url(post_id)}",
        reply_to=reply_to,
    )


def _react_and_maybe_kick(chat_id: int, intake_id: int, reply_to: Optional[int]) -> None:
    intake = db.get_intake(intake_id)
    photos = _photos_of(intake)
    price_cents = intake["price_cents"]
    if not photos or not price_cents:
        return
    with _kicked_lock:
        if intake_id in _kicked:
            return
        _kicked.add(intake_id)
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
        "Send photos of a finished piece (an album works — all the photos go to one product), "
        "plus the price (e.g. <code>$45</code>). I'll write the listing and create a draft on the website.\n\n"
        "<b>Bulk</b>: send several albums in a row, each becomes its own product.\n\n"
        "<b>Edit a draft</b>: quote-reply the <i>Draft is ready</i> message and tell me what to change. "
        "Examples:\n"
        "  <code>change name to Sterling Cuff</code>\n"
        "  <code>price is $55</code>\n"
        "  <code>rewrite description, more poetic</code>\n"
        "  <code>delete</code>  /  <code>publish</code>\n\n"
        "<b>Commands</b>\n"
        "  <code>/today</code>   — drafts created today\n"
        "  <code>/status</code>  — show your last intake\n"
        "  <code>/cancel</code>  — drop the in-progress intake\n"
        "  <code>publish</code>  — publish the last draft (Mike only, for now)"
    )
    send_message(chat_id, msg, reply_to=reply_to)


def _send_today(chat_id: int, reply_to: Optional[int]) -> None:
    """List every draft created today with preview links."""
    import datetime as _dt
    today_midnight = int(_dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    rows = list(db.list_since(chat_id, today_midnight))
    if not rows:
        send_message(chat_id, "Nothing today yet, sir.", reply_to=reply_to)
        return
    lines = [f"<b>Today's drafts</b> ({len(rows)})", ""]
    for r in rows:
        if r["woocommerce_post_id"]:
            name = r["seo_title"] or "(unnamed)"
            price = f"${r['price_cents']/100:.2f}" if r["price_cents"] else "?"
            lines.append(
                f"#{r['woocommerce_post_id']}  <b>{name}</b>  {price}  [{r['status']}]\n"
                f"  {preview_url(r['woocommerce_post_id'])}"
            )
        else:
            n_photos = len(_photos_of(r))
            lines.append(f"#intake-{r['id']}  ({n_photos} photos)  [{r['status']}]")
    send_message(chat_id, "\n".join(lines), reply_to=reply_to)


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
    row = db.latest_intake_with_post(chat_id, max_age_seconds=86400)
    if row is None:
        send_message(chat_id, "No recent draft to publish.", reply_to=reply_to)
        return
    post_id = int(row["woocommerce_post_id"])
    try:
        from core.jewelry.woocommerce import publish_product
        publish_product(post_id)
        send_message(chat_id, f"Published draft #{post_id}.\n{preview_url(post_id)}", reply_to=reply_to)
    except Exception as e:
        send_message(chat_id, f"Publish failed: {e}", reply_to=reply_to)


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
