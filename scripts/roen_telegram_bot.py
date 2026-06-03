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
from core.jewelry import orders as roen_orders  # noqa: E402
from core.jewelry import coupons as roen_coupons  # noqa: E402
from core.jewelry.woocommerce import preview_url, admin_edit_url  # noqa: E402
from core.jewelry.bracelet_box import handlers as box_handlers  # noqa: E402
from core.jewelry.bracelet_box import wc_orders as box_wc  # noqa: E402
from integrations.paypal_tracking import (  # noqa: E402
    detect_carrier,
    get_client_from_env,
    normalize_tracking,
    PayPalTrackingError,
)

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

# Ordered list preserving env-var declaration order (set loses order).
# First entry is Sarah (per CLAUDE.md); used by the bracelet-box poller.
ROEN_ALLOWED_LIST = [
    int(x.strip())
    for x in _env.get("ROEN_INTAKE_ALLOWED_CHAT_IDS", "").split(",")
    if x.strip().isdigit()
]
SARAH_CHAT_ID: int = ROEN_ALLOWED_LIST[0] if ROEN_ALLOWED_LIST else 0

# Mike's Telegram chat_id — also used for the admin-only batch publish/delete
# gates further down. Centralized here for the address helper too.
MIKE_CHAT_ID: int = 7582976864


def _address_suffix(chat_id: int) -> str:
    """Return a tail-of-sentence personal address (', sir' / ', Sarah' / '').
    Sarah gets her first name (no 'ma'am' — feels off for a maker), Mike keeps
    the butler 'sir', any unknown chat gets a clean drop."""
    if chat_id == MIKE_CHAT_ID:
        return ", sir"
    if chat_id == SARAH_CHAT_ID:
        return ", Sarah"
    return ""


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


def send_message(chat_id: int, text: str, reply_to: Optional[int] = None, reply_markup: Optional[dict] = None) -> Optional[dict]:
    try:
        params = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_to is not None:
            params["reply_to_message_id"] = reply_to
            params["allow_sending_without_reply"] = True
        if reply_markup is not None:
            params["reply_markup"] = reply_markup
        return tg("sendMessage", **params)
    except Exception:
        logger.exception("send_message to %s failed", chat_id)
        return None


def draft_keyboard(post_id: int) -> dict:
    """Three-button inline keyboard rendered under every 'Draft is ready' message."""
    return {
        "inline_keyboard": [[
            {"text": "✅ Publish", "callback_data": f"pub:{post_id}"},
            {"text": "✏️ Edit",    "callback_data": f"edt:{post_id}"},
            {"text": "🗑️ Delete",  "callback_data": f"del:{post_id}"},
        ]]
    }


def remove_keyboard(chat_id: int, message_id: int) -> None:
    """Strip the inline keyboard off a previously-sent message (after the user acted on it)."""
    try:
        tg("editMessageReplyMarkup", chat_id=chat_id, message_id=message_id, reply_markup={"inline_keyboard": []})
    except Exception:
        logger.exception("remove_keyboard failed for %s/%s", chat_id, message_id)


def answer_callback(callback_id: str, text: str = "") -> None:
    """Acknowledge a callback so Telegram stops the spinner on the user's button."""
    try:
        tg("answerCallbackQuery", callback_query_id=callback_id, text=text)
    except Exception:
        logger.exception("answer_callback failed")


def _enhance_photo(path: Path) -> None:
    """Subtle auto-tune for phone-shot product photos: gentle brightness /
    contrast / saturation / sharpness bumps so dim shots pop a little
    without looking processed. White backgrounds stay white (Sarah shoots
    on a neutral surface — autocontrast was tinting it green, so we don't
    use it). Runs in-place; any failure keeps the original file."""
    try:
        from PIL import Image, ImageEnhance
    except Exception:
        logger.exception("Pillow import failed — skipping enhance")
        return
    try:
        with Image.open(path) as src:
            img = src.convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(1.08)        # +8% brightness
        img = ImageEnhance.Contrast(img).enhance(1.15)          # +15% contrast
        img = ImageEnhance.Color(img).enhance(1.12)             # +12% saturation
        img = ImageEnhance.Sharpness(img).enhance(1.20)         # +20% sharpness
        img.save(path, "JPEG", quality=90, optimize=True)
        logger.info("enhanced photo %s", path.name)
    except Exception:
        logger.exception("enhance failed for %s — keeping original", path)


def download_photo(file_id: str, dest_path: Path) -> None:
    """Resolve a Telegram file_id to a downloaded file, then auto-enhance."""
    info = tg("getFile", file_id=file_id)
    rel = info["file_path"]
    url = f"{FILE_API}/{rel}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with dest_path.open("wb") as f:
            shutil.copyfileobj(r.raw, f)
    _enhance_photo(dest_path)


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

# pending edit state: chat_id -> (post_id, expires_at_epoch)
# Set when user taps the Edit button; consumed by the next text message they send.
_pending_edit: dict = {}
_pending_edit_lock = threading.Lock()
PENDING_EDIT_TTL = 300  # 5 minutes

# pending ship state: chat_id -> (order_id, expires_at_epoch)
# Set when user taps Ship & Complete; consumed by the tracking number they paste.
_pending_ship: dict = {}
_pending_ship_lock = threading.Lock()
PENDING_SHIP_TTL = 600  # 10 minutes — longer than edit since users may grab the label first


# pending coupon state: chat_id -> (CouponSpec, expires_at_epoch)
# Set when Sarah starts a coupon without naming it (or picks a duplicate name);
# consumed by the next text she sends, which becomes the code name.
_pending_coupon: dict = {}
_pending_coupon_lock = threading.Lock()
PENDING_COUPON_TTL = 300  # 5 minutes


def set_pending_coupon(chat_id: int, spec) -> None:
    with _pending_coupon_lock:
        _pending_coupon[chat_id] = (spec, int(time.time()) + PENDING_COUPON_TTL)


def pop_pending_coupon(chat_id: int):
    """Return the pending CouponSpec for this chat, else None. Clears on read."""
    with _pending_coupon_lock:
        entry = _pending_coupon.get(chat_id)
        if not entry:
            return None
        spec, expires = entry
        if time.time() > expires:
            _pending_coupon.pop(chat_id, None)
            return None
        _pending_coupon.pop(chat_id, None)
        return spec


def peek_pending_coupon(chat_id: int):
    """Return the pending CouponSpec WITHOUT consuming, else None."""
    with _pending_coupon_lock:
        entry = _pending_coupon.get(chat_id)
        if not entry:
            return None
        spec, expires = entry
        if time.time() > expires:
            _pending_coupon.pop(chat_id, None)
            return None
        return spec


def set_pending_edit(chat_id: int, post_id: int) -> None:
    with _pending_edit_lock:
        _pending_edit[chat_id] = (post_id, int(time.time()) + PENDING_EDIT_TTL)


def pop_pending_edit(chat_id: int) -> Optional[int]:
    """Return post_id if there's a fresh pending edit for this chat, else None. Clears on read."""
    with _pending_edit_lock:
        entry = _pending_edit.get(chat_id)
        if not entry:
            return None
        post_id, expires = entry
        if time.time() > expires:
            _pending_edit.pop(chat_id, None)
            return None
        _pending_edit.pop(chat_id, None)
        return post_id


def set_pending_ship(chat_id: int, order_id: int) -> None:
    with _pending_ship_lock:
        _pending_ship[chat_id] = (order_id, int(time.time()) + PENDING_SHIP_TTL)


def pop_pending_ship(chat_id: int) -> Optional[int]:
    """Return order_id if there's a fresh pending ship for this chat, else None. Clears on read."""
    with _pending_ship_lock:
        entry = _pending_ship.get(chat_id)
        if not entry:
            return None
        order_id, expires = entry
        if time.time() > expires:
            _pending_ship.pop(chat_id, None)
            return None
        _pending_ship.pop(chat_id, None)
        return order_id


def peek_pending_ship(chat_id: int) -> Optional[int]:
    """Return order_id if pending, WITHOUT consuming. Used to decide if next text is tracking."""
    with _pending_ship_lock:
        entry = _pending_ship.get(chat_id)
        if not entry:
            return None
        order_id, expires = entry
        if time.time() > expires:
            _pending_ship.pop(chat_id, None)
            return None
        return order_id


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
                    f"Draft is ready{_address_suffix(chat_id)}.\n\n"
                    f"<b>{result['name']}</b>\n"
                    f"SKU: <code>{result['sku']}</code>\n\n"
                    f"Preview: {result['preview_url']}\n"
                    f"Edit:    {result['edit_url']}\n\n"
                    f"Tap a button below or quote-reply to edit."
                ),
                reply_markup=draft_keyboard(result["post_id"]),
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
                f"Got the first photo{_address_suffix(chat_id)}. Send the rest plus the price (e.g. $45) and I'll get to work.",
                reply_to=msg.get("message_id"),
            )


def handle_text(msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()
    if not text:
        return

    # Bracelet-box approval flow: 'swap N' replies and quote-replied note rewrites
    # are consumed here BEFORE the existing intake flow gets a shot.
    try:
        if box_handlers.handle_swap_message(chat_id, text, send_message_fn=_send_message_box):
            return
        if box_handlers.handle_note_edit_message(chat_id, text, send_message_fn=_send_message_box):
            return
    except Exception:
        logger.exception("box text handler failed for %r", text)

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

    if lower in ("/orders", "orders", "pending orders", "what orders", "what pending orders"):
        _send_orders_list(chat_id, msg.get("message_id"))
        return

    # Coupon flow: "create a coupon ...", "my coupons", or a name reply to a
    # pending coupon. Handled here before the generic price-parse fallback.
    if handle_coupon_text(chat_id, text, msg.get("message_id")):
        return

    # If the user has a pending Local Delivery, the next non-command text is the delivery note.
    pending_local = _peek_pending_local(chat_id)
    if pending_local is not None and not lower.startswith("/"):
        _pop_pending_local(chat_id)  # consume
        _complete_local_flow(chat_id, pending_local, text, msg.get("message_id"))
        return

    # If the user has a pending Ship & Complete, the next non-command text is the tracking number
    # (or the word "local" / "hand delivery" to flip to the no-carrier path).
    pending_order = peek_pending_ship(chat_id)
    if pending_order is not None and not lower.startswith("/"):
        pop_pending_ship(chat_id)  # consume
        if lower in _LOCAL_KEYWORDS or any(lower.startswith(k + " ") for k in _LOCAL_KEYWORDS):
            _ask_for_local_note(chat_id, pending_order)
            return
        _complete_ship_flow(chat_id, pending_order, text, msg.get("message_id"))
        return

    if lower in ("/cancel", "cancel"):
        # Cancel any pending order action first (Ship & Complete or Local Delivery),
        # then fall back to pending Edit, then to active intake.
        if pop_pending_ship(chat_id) is not None:
            send_message(chat_id, "Ship & Complete cancelled.")
            return
        if _pop_pending_local(chat_id) is not None:
            send_message(chat_id, "Local Delivery cancelled.")
            return
        if pop_pending_coupon(chat_id) is not None:
            send_message(chat_id, "Coupon cancelled.")
            return
        if pop_pending_edit(chat_id) is not None:
            send_message(chat_id, "Edit cancelled.")
            return
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

    # Admin batch commands: "delete 188 194" / "publish 188 194" — explicit post IDs.
    if lower.startswith("delete ") or lower.startswith("/delete "):
        _admin_batch(chat_id, "delete", lower.split()[1:], msg.get("message_id"))
        return
    if lower.startswith("publish ") or lower.startswith("/publish "):
        _admin_batch(chat_id, "publish", lower.split()[1:], msg.get("message_id"))
        return

    # Pending Edit (set when the user tapped the ✏️ Edit button) — consume it now.
    pending_post_id = pop_pending_edit(chat_id)
    if pending_post_id is not None:
        _handle_edit(chat_id, pending_post_id, text, msg.get("message_id"))
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


def handle_callback_query(query: dict) -> None:
    """Inline-keyboard taps land here. callback_data format: 'pub:<id>' / 'edt:<id>' / 'del:<id>'."""
    cb_id = query.get("id")
    data = query.get("data", "")
    msg = query.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    msg_id = msg.get("message_id")
    if chat_id not in ALLOWED_CHAT_IDS:
        answer_callback(cb_id, "Not allowed.")
        return

    # Bracelet-box callback?
    if data.startswith("bx:"):
        try:
            box_handlers.handle_callback(
                data, chat_id,
                send_message_fn=_send_message_box,
                send_document_fn=_send_document_box,
            )
            answer_callback(cb_id)
        except Exception:
            logger.exception("box callback failed: %s", data)
            answer_callback(cb_id, "Box action failed.")
        return

    # Order management callback? `ord:ship:<id>` or `ord:detail:<id>` or `ord:cancel:<id>`
    if data.startswith("ord:"):
        try:
            _handle_order_callback(data, chat_id, cb_id, msg_id)
        except Exception:
            logger.exception("order callback failed: %s", data)
            answer_callback(cb_id, "Order action failed.")
        return

    # Coupon callback? `cpn:del:<code>`
    if data.startswith("cpn:"):
        try:
            _handle_coupon_callback(data, chat_id, cb_id, msg_id)
        except Exception:
            logger.exception("coupon callback failed: %s", data)
            answer_callback(cb_id, "Coupon action failed.")
        return

    if ":" not in data:
        answer_callback(cb_id, "Bad action.")
        return
    action, _, raw_id = data.partition(":")
    try:
        post_id = int(raw_id)
    except ValueError:
        answer_callback(cb_id, "Bad post id.")
        return

    if action == "pub":
        try:
            from core.jewelry.woocommerce import publish_product
            publish_product(post_id)
            answer_callback(cb_id, "Published.")
            remove_keyboard(chat_id, msg_id)
            send_message(chat_id, f"Published draft #{post_id}.\n{preview_url(post_id)}")
        except Exception as e:
            logger.exception("publish via button failed for %s", post_id)
            answer_callback(cb_id, "Publish failed.")
            send_message(chat_id, f"Publish failed for #{post_id}: {e}")
            return
        # Fire-and-best-effort: sync to Meta catalog + enqueue FB Page draft.
        # Failures here do NOT roll back the WC publish.
        try:
            from core.jewelry.meta_publish_hook import on_product_published
            result = on_product_published(post_id)
            if result.get("errors"):
                logger.warning("meta hook partial for %s: %s", post_id, result["errors"])
            else:
                logger.info("meta hook OK for %s: %s", post_id, result)
            if result.get("fb_draft_id"):
                send_message(
                    chat_id,
                    f"📨 FB Page draft queued — review at https://aialfred.groundrushcloud.com/admin/roen/social-pending",
                )
        except Exception:
            logger.exception("meta hook crashed for %s — WC publish stands", post_id)
        return

    if action == "del":
        try:
            from core.jewelry.woocommerce import trash_product
            trash_product(post_id)
            row = db.find_intake_by_post(post_id)
            if row:
                db.set_status(int(row["id"]), "deleted")
            answer_callback(cb_id, "Deleted.")
            remove_keyboard(chat_id, msg_id)
            send_message(chat_id, f"Draft #{post_id} moved to trash.")
        except Exception as e:
            logger.exception("delete via button failed for %s", post_id)
            answer_callback(cb_id, "Delete failed.")
            send_message(chat_id, f"Delete failed for #{post_id}: {e}")
        return

    if action == "edt":
        set_pending_edit(chat_id, post_id)
        answer_callback(cb_id, "Tell me what to change.")
        send_message(
            chat_id,
            (
                f"What should I change about draft #{post_id}?\n\n"
                "Examples:\n"
                "  <code>change name to Sterling Cuff</code>\n"
                "  <code>price is $55</code>\n"
                "  <code>rewrite description, more poetic</code>\n\n"
                "(Or send <code>/cancel</code> to drop this edit.)"
            ),
        )
        return

    answer_callback(cb_id, "Unknown action.")


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
        "<b>Edit a draft</b>: tap ✏️ <b>Edit</b> under the draft message and tell me what to change, "
        "or quote-reply the message with the change. Examples:\n"
        "  <code>change name to Sterling Cuff</code>\n"
        "  <code>price is $55</code>\n"
        "  <code>rewrite description, more poetic</code>\n\n"
        "<b>Publish or delete</b>: tap ✅ Publish or 🗑️ Delete under the draft.\n\n"
        "<b>Orders</b>: <code>/orders</code> lists pending shipments. Tap 📦 Ship & Complete, paste the "
        "USPS tracking #, and I'll save it on the order, push it to PayPal, mark the order completed, "
        "and email the customer.\n\n"
        "<b>Coupons</b> (friends & family): just tell me what you want —\n"
        "  <code>create a coupon 20% off Brittany</code>\n"
        "  <code>create a coupon $15 off Mom</code>\n"
        "  <code>create a coupon free shipping Kelly</code>\n"
        "  <code>create a free coupon Sarah</code>\n"
        "Codes are reusable by default; add <code>one time</code> for single-use or "
        "<code>good for a month</code> to set an expiry. <code>my coupons</code> lists them with a 🗑️ Delete button.\n\n"
        "<b>Commands</b>\n"
        "  <code>/orders</code>         — pending orders to ship\n"
        "  <code>/coupons</code>        — your coupons (with delete)\n"
        "  <code>/today</code>          — drafts created today\n"
        "  <code>/status</code>         — show your last intake\n"
        "  <code>/cancel</code>         — drop in-progress intake or pending edit\n"
        "  <code>publish</code>         — publish most recent draft (Mike)\n"
        "  <code>publish 188 194</code> — batch-publish by post ID (Mike)\n"
        "  <code>delete 188 194</code>  — batch-delete by post ID (Mike)"
    )
    send_message(chat_id, msg, reply_to=reply_to)


# ----------------------- /orders flow -----------------------

# USPS 20-22 digits, UPS 1Z+16, FedEx 12/15/20 digits, Intl USPS LL\d{9}LL.
# Loose validation — PayPal's API does strict validation, we just reject obvious garbage.
_TRACKING_RE = re.compile(r"^[A-Za-z0-9]{8,30}$")


def _render_order_line(o: roen_orders.OrderSummary) -> str:
    waited = ""
    if o.date_created:
        try:
            created = roen_orders._parse_iso(o.date_created)
            now = time.time()
            secs = int(now - created.timestamp())
            if secs < 3600:
                waited = f"{secs // 60} min ago"
            elif secs < 86400:
                waited = f"{secs // 3600}h ago"
            else:
                waited = f"{secs // 86400}d ago"
        except Exception:
            pass
    items_str = ", ".join(f"{i.name}" for i in o.items[:3])
    if len(o.items) > 3:
        items_str += f" +{len(o.items)-3} more"
    badge = "🟡" if o.status == "processing" else "⏸"
    return (
        f"{badge} <b>#{o.id}</b> — {o.customer_name}\n"
        f"   {items_str}\n"
        f"   <b>${o.total}</b> • {waited or o.status}"
    )


def _send_orders_list(chat_id: int, reply_to: Optional[int]) -> None:
    try:
        pending = roen_orders.list_pending_orders()
    except Exception as e:
        logger.exception("orders list failed")
        send_message(chat_id, f"Couldn't fetch orders{_address_suffix(chat_id)}: {e}", reply_to=reply_to)
        return

    if not pending:
        send_message(chat_id, f"No pending orders{_address_suffix(chat_id)}. Inbox zero.", reply_to=reply_to)
        return

    header = f"<b>📋 {len(pending)} pending order{'s' if len(pending) != 1 else ''}</b>\n\n"
    # One message per order so each gets its own keyboard.
    send_message(chat_id, header + _render_order_line(pending[0]),
                 reply_to=reply_to,
                 reply_markup=_order_keyboard(pending[0].id))
    for o in pending[1:]:
        send_message(chat_id, _render_order_line(o), reply_markup=_order_keyboard(o.id))


def _order_keyboard(order_id: int) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "📦 Ship & Complete", "callback_data": f"ord:ship:{order_id}"},
                {"text": "👁 Details", "callback_data": f"ord:detail:{order_id}"},
            ],
            [
                {"text": "🏠 Local Delivery", "callback_data": f"ord:local:{order_id}"},
            ],
        ]
    }


# Words that, when pasted in place of a tracking number, mean "no carrier — hand-delivered"
_LOCAL_KEYWORDS = {"local", "local delivery", "hand", "hand delivery", "hand-delivery",
                   "in person", "in-person", "delivered", "pickup", "pick up", "pick-up"}


def _send_order_details(chat_id: int, order_id: int) -> None:
    try:
        o = roen_orders.get_order_details(order_id)
    except Exception as e:
        send_message(chat_id, f"Couldn't load order #{order_id}: {e}")
        return

    lines = [
        f"<b>Order #{o.id}</b> — {o.status}",
        f"<b>${o.total}</b> {o.currency} • {o.payment_method_title or o.payment_method}",
        "",
        f"<b>Customer:</b> {o.customer_name}",
        f"<b>Email:</b> <code>{o.customer_email}</code>",
    ]
    if o.shipping_address_lines:
        lines += ["", "<b>Ship to:</b>"] + [f"  {ln}" for ln in o.shipping_address_lines]
    if o.items:
        lines += ["", "<b>Items:</b>"]
        for i in o.items:
            lines.append(f"  • {i.name} ×{i.quantity} (${i.total})")
    if o.transaction_id:
        lines += ["", f"<b>PayPal txn:</b> <code>{o.transaction_id}</code>"]
    else:
        lines += ["", "⚠️ <i>No PayPal transaction id on this order — tracking will save locally but won't push to PayPal.</i>"]
    send_message(chat_id, "\n".join(lines), reply_markup=_order_keyboard(o.id))


def _ask_for_tracking(chat_id: int, order_id: int) -> None:
    set_pending_ship(chat_id, order_id)
    send_message(
        chat_id,
        (
            f"📦 Order <b>#{order_id}</b> — paste the tracking number.\n\n"
            f"USPS by default. Prefix with <code>UPS </code> or <code>FEDEX </code> if it's another carrier.\n\n"
            f"Examples:\n"
            f"  <code>9405511206213098765432</code>\n"
            f"  <code>UPS 1Z9Y34670344585291</code>\n\n"
            f"Hand-delivered? Type <code>local</code> instead.\n\n"
            f"<code>/cancel</code> to abort."
        ),
    )


# Per-chat state for the local-delivery flow: chat_id -> (order_id, expires_at_epoch)
# Separate from _pending_ship so a Ship & Complete in-progress doesn't get confused
# with a Local Delivery in-progress.
_pending_local: dict = {}
_pending_local_lock = threading.Lock()
PENDING_LOCAL_TTL = 600  # 10 minutes


def _ask_for_local_note(chat_id: int, order_id: int) -> None:
    with _pending_local_lock:
        _pending_local[chat_id] = (order_id, int(time.time()) + PENDING_LOCAL_TTL)
    send_message(
        chat_id,
        (
            f"🏠 Order <b>#{order_id}</b> — local delivery.\n\n"
            f"Send a brief delivery note. Examples:\n"
            f"  <code>Delivered to Kelly in person 5/19</code>\n"
            f"  <code>Handed off at Sarah's market booth, 5/19 noon</code>\n\n"
            f"<i>⚠️ PayPal seller protection requires carrier tracking or signed proof. "
            f"For your records, snap a quick photo of the handoff — keep it on your phone, "
            f"don't send it anywhere. You'll only need it if there's ever a dispute.</i>\n\n"
            f"<code>/cancel</code> to abort."
        ),
    )


def _pop_pending_local(chat_id: int) -> Optional[int]:
    with _pending_local_lock:
        entry = _pending_local.get(chat_id)
        if not entry:
            return None
        order_id, expires = entry
        if time.time() > expires:
            _pending_local.pop(chat_id, None)
            return None
        _pending_local.pop(chat_id, None)
        return order_id


def _peek_pending_local(chat_id: int) -> Optional[int]:
    with _pending_local_lock:
        entry = _pending_local.get(chat_id)
        if not entry:
            return None
        order_id, expires = entry
        if time.time() > expires:
            _pending_local.pop(chat_id, None)
            return None
        return order_id


def _complete_local_flow(chat_id: int, order_id: int, note: str, reply_to: Optional[int]) -> None:
    """Local hand-delivery: save the note, complete the order, skip PayPal."""
    note = note.strip()
    if not note or len(note) < 5:
        send_message(
            chat_id,
            f"Need a slightly longer note{_address_suffix(chat_id)} — at least who and when. Try again or <code>/cancel</code>.",
            reply_to=reply_to,
        )
        # restore state
        with _pending_local_lock:
            _pending_local[chat_id] = (order_id, int(time.time()) + PENDING_LOCAL_TTL)
        return

    full_note = f"Local hand-delivery (no carrier tracking). {note}"
    try:
        # Save as a customer-visible WC order note via wc-cli
        roen_orders._wp([
            "wc", "shop_order_note", "create",
            "--user=1",
            f"--order_id={order_id}",
            f"--note={full_note}",
            "--customer_note=true",
            "--porcelain",
        ], timeout=20)
        # Also stamp our own meta for searchability
        roen_orders._wp([
            "post", "meta", "update", str(order_id),
            "_roen_local_delivery_note", note,
        ], timeout=15)
        roen_orders._wp([
            "post", "meta", "update", str(order_id),
            "_roen_tracking_carrier", "LOCAL",
        ], timeout=15)
    except Exception as e:
        logger.exception("save local note failed for %s", order_id)
        send_message(chat_id, f"❌ Couldn't save the local-delivery note on #{order_id}: {e}", reply_to=reply_to)
        return

    try:
        roen_orders.complete_order(order_id)
    except Exception as e:
        logger.exception("complete_order failed for %s", order_id)
        send_message(chat_id, f"⚠️ Saved the note but couldn't flip #{order_id} to completed: {e}", reply_to=reply_to)
        return

    send_message(
        chat_id,
        (
            f"🏠 <b>Order #{order_id} closed as local delivery</b>\n"
            f"Note: <i>{note}</i>\n\n"
            f"✅ Order marked completed — customer emailed\n"
            f"⏭ PayPal: <i>skipped</i> (no carrier — seller protection N/A for hand-delivery)"
        ),
        reply_to=reply_to,
    )


def _complete_ship_flow(chat_id: int, order_id: int, raw_text: str, reply_to: Optional[int]) -> None:
    """User pasted tracking text. Parse → save WC → push PayPal → complete order."""
    text = raw_text.strip()
    # Optional carrier prefix: "UPS 1Z..." / "FEDEX 1234..."
    carrier_override: Optional[str] = None
    tracking_part = text
    parts = text.split(None, 1)
    if len(parts) == 2 and parts[0].upper() in ("USPS", "UPS", "FEDEX", "FED-EX", "FEDX", "DHL"):
        carrier_override = parts[0].upper().replace("FED-EX", "FEDEX").replace("FEDX", "FEDEX")
        tracking_part = parts[1]

    tracking = normalize_tracking(tracking_part)
    if not _TRACKING_RE.match(tracking):
        send_message(
            chat_id,
            f"That doesn't look like a tracking number{_address_suffix(chat_id)}. Try again or <code>/cancel</code>.",
            reply_to=reply_to,
        )
        set_pending_ship(chat_id, order_id)  # restore state for retry
        return

    carrier = carrier_override or detect_carrier(tracking)

    # Save to WC + add customer-visible note
    try:
        roen_orders.save_tracking(order_id, tracking, carrier)
    except Exception as e:
        logger.exception("save_tracking failed for %s", order_id)
        send_message(chat_id, f"❌ Couldn't save tracking on order #{order_id}: {e}", reply_to=reply_to)
        return

    # Push to PayPal (best-effort — if it fails, WC state still stands)
    paypal_msg = ""
    txn_id = roen_orders.get_paypal_transaction_id(order_id)
    if txn_id:
        try:
            client = get_client_from_env(_env)
            client.add_tracker(txn_id, tracking, carrier=carrier, notify_buyer=True, status="SHIPPED")
            paypal_msg = "✅ PayPal notified (buyer protection active)"
        except PayPalTrackingError as e:
            logger.warning("PayPal tracker push failed for %s: %s", order_id, e)
            paypal_msg = f"⚠️ PayPal push failed: {e}"
        except Exception as e:
            logger.exception("PayPal tracker push crashed for %s", order_id)
            paypal_msg = f"⚠️ PayPal push crashed: {e}"
    else:
        paypal_msg = "⚠️ No PayPal txn id on order — skipped PayPal push"

    # Mark WC order completed (fires customer email)
    try:
        roen_orders.complete_order(order_id)
        wc_msg = "✅ Order marked completed — customer emailed"
    except Exception as e:
        logger.exception("complete_order failed for %s", order_id)
        send_message(
            chat_id,
            f"⚠️ Saved tracking but couldn't flip order #{order_id} to completed: {e}",
            reply_to=reply_to,
        )
        return

    summary = (
        f"📦 <b>Order #{order_id} shipped</b>\n"
        f"Carrier: <b>{carrier}</b>\n"
        f"Tracking: <code>{tracking}</code>\n\n"
        f"{wc_msg}\n"
        f"{paypal_msg}"
    )
    send_message(chat_id, summary, reply_to=reply_to)


def _handle_order_callback(data: str, chat_id: int, cb_id: str, msg_id: Optional[int]) -> None:
    parts = data.split(":")
    if len(parts) != 3:
        answer_callback(cb_id, "Bad order action.")
        return
    _, action, raw_id = parts
    try:
        order_id = int(raw_id)
    except ValueError:
        answer_callback(cb_id, "Bad order id.")
        return

    if action == "ship":
        answer_callback(cb_id, "Send tracking #")
        _ask_for_tracking(chat_id, order_id)
        return

    if action == "local":
        answer_callback(cb_id, "Send delivery note")
        _ask_for_local_note(chat_id, order_id)
        return

    if action == "detail":
        answer_callback(cb_id)
        _send_order_details(chat_id, order_id)
        return

    answer_callback(cb_id, "Unknown order action.")


# Telegram sendMessage hard-caps text at 4096 chars. Stay safely under that
# so HTML entities and reply markup don't tip us over.
TELEGRAM_TEXT_LIMIT = 3800


def _format_intake_line(r) -> str:
    if r["woocommerce_post_id"]:
        name = r["seo_title"] or "(unnamed)"
        price = f"${r['price_cents']/100:.2f}" if r["price_cents"] else "?"
        return (
            f"#{r['woocommerce_post_id']}  <b>{name}</b>  {price}  [{r['status']}]\n"
            f"  {preview_url(r['woocommerce_post_id'])}"
        )
    n_photos = len(_photos_of(r))
    return f"#intake-{r['id']}  ({n_photos} photos)  [{r['status']}]"


def _send_chunked(chat_id: int, header: str, lines: List[str], reply_to: Optional[int] = None) -> None:
    """Send `header` plus `lines`, splitting into multiple messages so each stays under Telegram's limit."""
    chunks: List[str] = []
    current = header
    for line in lines:
        candidate = current + "\n" + line if current else line
        if len(candidate) > TELEGRAM_TEXT_LIMIT:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    for i, chunk in enumerate(chunks):
        send_message(chat_id, chunk, reply_to=reply_to if i == 0 else None)


def _send_today(chat_id: int, reply_to: Optional[int]) -> None:
    """List every draft created today with preview links."""
    import datetime as _dt
    today_midnight = int(_dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    rows = list(db.list_since(chat_id, today_midnight))
    if not rows:
        send_message(chat_id, f"Nothing today yet{_address_suffix(chat_id)}.", reply_to=reply_to)
        return
    header = f"<b>Today's drafts</b> ({len(rows)})\n"
    lines = [_format_intake_line(r) for r in rows]
    _send_chunked(chat_id, header, lines, reply_to=reply_to)


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


def _admin_batch(chat_id: int, op: str, raw_ids: List[str], reply_to: Optional[int]) -> None:
    """Apply publish/delete to a list of explicit post IDs. Mike-only."""
    if chat_id != 7582976864:
        send_message(chat_id, "Only Mike can run batch commands.", reply_to=reply_to)
        return
    ids: List[int] = []
    for r in raw_ids:
        try:
            ids.append(int(r))
        except ValueError:
            pass
    if not ids:
        send_message(chat_id, f"No post IDs found. Try: <code>{op} 188 194</code>", reply_to=reply_to)
        return

    from core.jewelry.woocommerce import publish_product, trash_product
    results = []
    for pid in ids:
        try:
            if op == "publish":
                publish_product(pid)
                results.append(f"✅ #{pid} published")
            else:
                trash_product(pid)
                row = db.find_intake_by_post(pid)
                if row:
                    db.set_status(int(row["id"]), "deleted")
                results.append(f"🗑️ #{pid} deleted")
        except Exception as e:
            results.append(f"❌ #{pid} failed: {e}")
    send_message(chat_id, "\n".join(results), reply_to=reply_to)


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
        return
    try:
        from core.jewelry.meta_publish_hook import on_product_published
        result = on_product_published(post_id)
        if result.get("fb_draft_id"):
            send_message(
                chat_id,
                "📨 FB Page draft queued — review at https://aialfred.groundrushcloud.com/admin/roen/social-pending",
                reply_to=reply_to,
            )
    except Exception:
        logger.exception("meta hook crashed for /publish %s — WC publish stands", post_id)


# ----------------------- coupon flow -----------------------

_COUPON_LIST_TRIGGERS = {
    "my coupons", "/coupons", "coupons", "list coupons", "show coupons",
    "list my coupons", "show my coupons",
}


def _is_coupon_create(lower: str) -> bool:
    """True if the text is a request to create a coupon."""
    if lower.startswith("/coupon") and not lower.startswith("/coupons"):
        return True
    return "coupon" in lower and re.search(r"\b(create|make|new|add|generate|give)\b", lower) is not None


def _discount_phrase(discount_type: str, amount: float, free_shipping: bool) -> str:
    """Human description of a discount, e.g. '20% off' / '$15 off' / 'free (100% off)'."""
    def trim(a: float) -> str:
        return str(int(a)) if a == int(a) else f"{a:.2f}"
    if free_shipping and amount == 0:
        return "free shipping"
    parts = []
    if discount_type == "percent":
        parts.append("free (100% off)" if amount == 100 else f"{trim(amount)}% off")
    else:
        parts.append(f"${trim(amount)} off")
    if free_shipping:
        parts.append("+ free shipping")
    return " ".join(parts)


def _format_coupon_confirmation(chat_id: int, spec) -> str:
    usage = "single use" if spec.usage_limit == 1 else "unlimited use"
    expiry = f"expires {spec.expiry_date}" if spec.expiry_date else "no expiry"
    return (
        f"✅ <b>Coupon created{_address_suffix(chat_id)}</b>\n"
        f"Code: <code>{spec.code}</code>\n"
        f"{_discount_phrase(spec.discount_type, spec.amount, spec.free_shipping)} · {usage} · {expiry}\n\n"
        f"Share it with whoever you like."
    )


def _coupon_keyboard(code: str) -> dict:
    return {"inline_keyboard": [[{"text": "🗑️ Delete", "callback_data": f"cpn:del:{code}"}]]}


def _create_and_confirm(chat_id: int, spec, reply_to: Optional[int]) -> None:
    """Uniqueness-check, create in WooCommerce, confirm. On dup name, re-prompt."""
    try:
        if roen_coupons.coupon_exists(spec.code):
            spec.code = None
            set_pending_coupon(chat_id, spec)
            send_message(
                chat_id,
                "There's already a coupon with that name. Send a different name "
                "(or <code>/cancel</code>).",
                reply_to=reply_to,
            )
            return
        roen_coupons.create_coupon(spec)
    except Exception as e:
        logger.exception("coupon create failed")
        send_message(chat_id, f"Couldn't create that coupon{_address_suffix(chat_id)}: {e}", reply_to=reply_to)
        return
    send_message(chat_id, _format_coupon_confirmation(chat_id, spec),
                 reply_to=reply_to, reply_markup=_coupon_keyboard(spec.code))


def _start_coupon_create(chat_id: int, text: str, reply_to: Optional[int]) -> None:
    spec = roen_coupons.parse_coupon_request(text)
    if spec is None:
        send_message(
            chat_id,
            ("What discount would you like? Examples:\n"
             "  <code>create a coupon 20% off Brittany</code>\n"
             "  <code>create a coupon $15 off Mom</code>\n"
             "  <code>create a coupon free shipping Kelly</code>\n"
             "  <code>create a free coupon Sarah</code>\n\n"
             "Add <code>one time</code> for single-use, or "
             "<code>good for a month</code> to set an expiry."),
            reply_to=reply_to,
        )
        return
    if spec.code is None:
        set_pending_coupon(chat_id, spec)
        send_message(
            chat_id,
            ("What do you want to call it? (e.g. <code>Brittany</code>)\n"
             f"That'll be {_discount_phrase(spec.discount_type, spec.amount, spec.free_shipping)}."),
            reply_to=reply_to,
        )
        return
    _create_and_confirm(chat_id, spec, reply_to)


def _finish_coupon_with_name(chat_id: int, spec, name_text: str, reply_to: Optional[int]) -> None:
    code = roen_coupons.normalize_code(name_text)
    if not code:
        set_pending_coupon(chat_id, spec)  # restore — keep waiting for a usable name
        send_message(
            chat_id,
            "I need a name for the code — letters or numbers, like <code>Brittany</code>.",
            reply_to=reply_to,
        )
        return
    spec.code = code
    _create_and_confirm(chat_id, spec, reply_to)


def _send_coupons_list(chat_id: int, reply_to: Optional[int]) -> None:
    try:
        rows = roen_coupons.list_coupons()
    except Exception as e:
        logger.exception("coupon list failed")
        send_message(chat_id, f"Couldn't fetch coupons{_address_suffix(chat_id)}: {e}", reply_to=reply_to)
        return
    if not rows:
        send_message(chat_id, f"No coupons yet{_address_suffix(chat_id)}. Make one with "
                              "<code>create a coupon 20% off Brittany</code>.", reply_to=reply_to)
        return

    header = f"<b>🎟️ Your coupons ({len(rows)})</b>"
    send_message(chat_id, header, reply_to=reply_to)
    for r in rows:
        # WooCommerce stores codes lowercase; display/act on the uppercase form
        # to match how the code was created and shared (checkout is case-insensitive).
        send_message(chat_id, _format_coupon_row(r), reply_markup=_coupon_keyboard(r.code.upper()))


def _format_coupon_row(r) -> str:
    try:
        amount = float(r.amount)
    except (TypeError, ValueError):
        amount = 0.0
    desc = _discount_phrase(r.discount_type, amount, r.free_shipping)
    if r.usage_limit == 1:
        uses = "single use"
    elif r.usage_limit:
        uses = f"{r.usage_limit} uses"
    else:
        uses = "unlimited"
    exp = f"expires {r.date_expires}" if r.date_expires else "no expiry"
    used = f" · used {r.usage_count}×" if r.usage_count else ""
    return f"<b>{r.code.upper()}</b> — {desc} · {uses} · {exp}{used}"


def _handle_coupon_callback(data: str, chat_id: int, cb_id: str, msg_id: Optional[int]) -> None:
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[1] != "del":
        answer_callback(cb_id, "Bad coupon action.")
        return
    code = parts[2]
    try:
        deleted = roen_coupons.delete_coupon(code)
    except Exception as e:
        logger.exception("coupon delete failed for %s", code)
        answer_callback(cb_id, "Delete failed.")
        send_message(chat_id, f"Couldn't delete <b>{code}</b>: {e}")
        return
    if deleted:
        answer_callback(cb_id, "Deleted.")
        remove_keyboard(chat_id, msg_id)
        send_message(chat_id, f"Deleted coupon <b>{code}</b> ✅\nIt'll stop working at checkout right away.")
    else:
        answer_callback(cb_id, "Not found.")
        remove_keyboard(chat_id, msg_id)
        send_message(chat_id, f"Coupon <b>{code}</b> wasn't found — already gone?")


def handle_coupon_text(chat_id: int, text: str, reply_to: Optional[int]) -> bool:
    """Handle coupon create/list/name-reply. Returns True if it consumed the message."""
    lower = text.lower().strip()

    if lower in _COUPON_LIST_TRIGGERS:
        _send_coupons_list(chat_id, reply_to)
        return True

    if _is_coupon_create(lower):
        _start_coupon_create(chat_id, text, reply_to)
        return True

    # Name reply to a pending coupon (any non-command text while one is pending).
    if peek_pending_coupon(chat_id) is not None and not lower.startswith("/"):
        spec = pop_pending_coupon(chat_id)
        if spec is not None:
            _finish_coupon_with_name(chat_id, spec, text, reply_to)
            return True
    return False


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


# ----------------------- bracelet-box poller -----------------------

BOX_POLL_INTERVAL_SECONDS = 60


def _send_message_box(chat_id: int, text: str, reply_markup=None):
    """Adapter for box_handlers — wraps the existing send_message."""
    return send_message(chat_id, text, reply_markup=reply_markup)


def _send_media_group_box(chat_id: int, media: list):
    """Send a media group (list of {type, media, caption}) via the bot API."""
    try:
        tg("sendMediaGroup", chat_id=chat_id, media=media)
    except Exception:
        logger.exception("sendMediaGroup to %s failed", chat_id)


def _send_document_box(chat_id: int, file_path, caption: str = ""):
    """Upload a local file as a Telegram document."""
    try:
        with open(file_path, "rb") as fp:
            r = requests.post(
                f"{API}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": fp},
                timeout=60,
            )
            r.raise_for_status()
    except Exception:
        logger.exception("sendDocument to %s failed", chat_id)


def _box_poll_once():
    """One pass: fetch new orders, fan out pick sessions per qty, advance cursor."""
    cursor = box_wc.load_cursor()
    last_seen = cursor

    if not SARAH_CHAT_ID:
        return

    for item in box_wc.iter_new_box_line_items(after_id=cursor):
        for bundle_index in range(1, item['quantity'] + 1):
            try:
                box_handlers.open_pick_session(
                    order_id=item['order_id'],
                    line_item_id=item['line_item_id'],
                    bundle_index=bundle_index,
                    customer_email=item['customer_email'],
                    customer_first_name=item['customer_first_name'],
                    sarah_chat_id=SARAH_CHAT_ID,
                    send_message_fn=_send_message_box,
                    send_media_group_fn=_send_media_group_box,
                )
            except Exception:
                logger.exception(
                    "open_pick_session failed for order %d bundle %d",
                    item['order_id'], bundle_index,
                )
        last_seen = max(last_seen, item['order_id'])

    if last_seen > cursor:
        box_wc.save_cursor(box_wc.CURSOR_PATH, last_seen)


def _box_poll_loop():
    """Daemon thread: poll WC every BOX_POLL_INTERVAL_SECONDS."""
    while _running:
        try:
            _box_poll_once()
        except Exception:
            logger.exception("box poll iteration crashed")
        # Sleep in small slices so we exit promptly on shutdown
        for _ in range(BOX_POLL_INTERVAL_SECONDS):
            if not _running:
                return
            time.sleep(1)


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

    threading.Thread(target=_box_poll_loop, daemon=True, name="box-poll").start()
    logger.info("bracelet-box poll thread started (%ds interval)", BOX_POLL_INTERVAL_SECONDS)

    offset = load_offset()

    while _running:
        try:
            updates = tg("getUpdates", offset=offset, timeout=25, allowed_updates=["message", "callback_query"])
        except Exception:
            logger.exception("getUpdates failed; sleeping")
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            save_offset(offset)

            cbq = update.get("callback_query")
            if cbq:
                try:
                    handle_callback_query(cbq)
                except Exception:
                    logger.exception("callback handler crashed")
                continue

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
