"""Telegram callback router for the bracelet-box approval flow.

Inline button callback_data convention:
    bx:approve:{pick_id}     — Sarah taps ✅
    bx:swap:{pick_id}        — Sarah taps ✏️ (followed by "swap N" message)
    bx:editnote:{pick_id}    — Sarah taps 📝 (followed by quote-reply rewrite)
    bx:reroll:{pick_id}      — Sarah taps 🔄

Public API:
    open_pick_session(...) — runs picker + note gen, creates DB row, pings Sarah
    handle_callback(callback_data, chat_id, ...) — routes inline-button taps
    handle_swap_message(chat_id, text, ...) — consumes "swap N" replies
    handle_note_edit_message(chat_id, text, ...) — consumes note rewrites
    keyboard(pick_id) — builds the inline keyboard for a pick

The bot wires up Telegram send/receive helpers via callables passed in.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from core.jewelry.bracelet_box import db as box_db
from core.jewelry.bracelet_box import picker, note_writer, card_pdf, wc_orders
from core.jewelry.bracelet_box.tags import InsufficientStock

log = logging.getLogger(__name__)

# Pending free-form-reply tracker: chat_id → (pick_id, mode, expires_at_epoch)
# mode is one of "swap" or "editnote".
_pending: dict[int, tuple[int, str, float]] = {}
_pending_lock = threading.Lock()
PENDING_TTL = 600  # 10 minutes; reset on each new prompt


def _set_pending(chat_id: int, pick_id: int, mode: str) -> None:
    with _pending_lock:
        _pending[chat_id] = (pick_id, mode, time.time() + PENDING_TTL)


def _pop_pending(chat_id: int) -> Optional[tuple[int, str]]:
    """Return (pick_id, mode) and clear, or None if no pending or expired."""
    with _pending_lock:
        entry = _pending.get(chat_id)
        if not entry:
            return None
        pick_id, mode, exp = entry
        if exp < time.time():
            _pending.pop(chat_id, None)
            return None
        _pending.pop(chat_id, None)
        return (pick_id, mode)


def _peek_pending(chat_id: int) -> Optional[tuple[int, str]]:
    """Read pending without consuming. Returns None if no pending or expired."""
    with _pending_lock:
        entry = _pending.get(chat_id)
        if not entry:
            return None
        pick_id, mode, exp = entry
        if exp < time.time():
            _pending.pop(chat_id, None)
            return None
        return (pick_id, mode)


def keyboard(pick_id: int) -> dict:
    """Inline keyboard for a pick session in awaiting_sarah state."""
    return {"inline_keyboard": [[
        {"text": "✅ Approve",   "callback_data": f"bx:approve:{pick_id}"},
        {"text": "✏️ Swap one",  "callback_data": f"bx:swap:{pick_id}"},
        {"text": "📝 Edit note", "callback_data": f"bx:editnote:{pick_id}"},
        {"text": "🔄 Reroll",    "callback_data": f"bx:reroll:{pick_id}"},
    ]]}


def _piece_names_from_skus(skus: list[int],
                           candidates_cache: Optional[list[dict]] = None
                          ) -> list[str]:
    """Resolve product ids to names. Uses a candidates list if supplied,
    otherwise falls back to per-SKU lookup via wc_orders (slower)."""
    if candidates_cache:
        by_id = {c['id']: c['name'] for c in candidates_cache}
        return [by_id.get(s, f"#{s}") for s in skus]
    # Fallback: per-product lookup
    import requests
    out = []
    for sku_id in skus:
        try:
            r = requests.get(
                f"{wc_orders.WC_BASE}/products/{sku_id}",
                auth=wc_orders._auth(),
                timeout=15,
            )
            r.raise_for_status()
            out.append(r.json().get('name', f'#{sku_id}'))
        except Exception:
            out.append(f'#{sku_id}')
    return out


def open_pick_session(
    *,
    order_id: int,
    line_item_id: int,
    bundle_index: int,
    customer_email: str,
    customer_first_name: Optional[str],
    sarah_chat_id: int,
    send_message_fn: Callable,
    send_media_group_fn: Callable,
) -> int:
    """Run the picker, persist a pick row, and ping Sarah on Telegram.

    Returns the pick_id, or 0 if insufficient stock (Sarah is alerted instead).

    `send_message_fn(chat_id, text, reply_markup=None)` sends a Telegram message.
    `send_media_group_fn(chat_id, media)` sends a media group (list of photo dicts).
    """
    candidates = wc_orders.fetch_in_stock_bracelets()
    history = box_db.history_for_email(customer_email)
    history_dicts = [{
        'color_tags': json.loads(h['color_tags'] or '[]'),
        'style_tags': json.loads(h['style_tags'] or '[]'),
    } for h in history]
    past_notes = [h['note_text'] for h in history]

    try:
        picks = picker.pick_five(candidates, history_dicts, rng=random.Random())
    except InsufficientStock:
        log.error("insufficient stock for box pick — order %d", order_id)
        send_message_fn(
            sarah_chat_id,
            f"⚠ Order {order_id} came in but stock dipped below 5 — "
            f"please review manually."
        )
        return 0

    note = note_writer.generate(
        picks=picks,
        first_name=customer_first_name,
        past_notes=past_notes,
        order_count=len(history) + 1,
    )

    pick_id = box_db.create_pick(
        order_id=order_id, line_item_id=line_item_id, bundle_index=bundle_index,
        customer_email=customer_email,
        customer_first_name=customer_first_name,
        picked_skus=[p['id'] for p in picks],
        color_tags=[p['color_family'] for p in picks],
        style_tags=[p['style_class'] for p in picks],
        note_text=note,
    )
    box_db.set_status(pick_id, 'awaiting_sarah')

    # Send media group (5 thumbnails)
    media = [
        {'type': 'photo', 'media': p['image_url'],
         'caption': f"{i+1}. {p['name']}"}
        for i, p in enumerate(picks) if p.get('image_url')
    ]
    if media:
        send_media_group_fn(sarah_chat_id, media)

    send_message_fn(
        sarah_chat_id,
        f"📦 Pick #{pick_id} for order {order_id} (box {bundle_index})\n\n"
        f"Draft note:\n\n{note or '[note generation failed — please /redo or write your own]'}",
        reply_markup=keyboard(pick_id),
    )
    return pick_id


def handle_callback(
    callback_data: str,
    chat_id: int,
    *,
    send_message_fn: Callable,
    send_document_fn: Callable,
) -> bool:
    """Route a bx: inline-button tap. Returns True if it was handled."""
    parts = callback_data.split(":")
    if len(parts) != 3 or parts[0] != "bx":
        return False
    action = parts[1]
    try:
        pick_id = int(parts[2])
    except ValueError:
        return False

    pick = box_db.get_pick(pick_id)
    if not pick:
        send_message_fn(chat_id, f"pick #{pick_id} not found")
        return True

    if action == "approve":
        _approve(pick, chat_id,
                 send_message_fn=send_message_fn,
                 send_document_fn=send_document_fn)
    elif action == "swap":
        _set_pending(chat_id, pick_id, "swap")
        send_message_fn(chat_id, "reply with `swap 3` (or any slot 1-5)")
    elif action == "editnote":
        _set_pending(chat_id, pick_id, "editnote")
        send_message_fn(chat_id, "quote-reply with the rewritten note")
    elif action == "reroll":
        _reroll(pick, chat_id, send_message_fn=send_message_fn)
    else:
        send_message_fn(chat_id, f"unknown action: {action}")
    return True


def _approve(pick, chat_id: int, *,
             send_message_fn: Callable, send_document_fn: Callable) -> None:
    skus = json.loads(pick['picked_skus'])
    if not wc_orders.reserve_skus(skus):
        send_message_fn(chat_id, "⚠ stock changed — re-rolling")
        _reroll(pick, chat_id, send_message_fn=send_message_fn)
        return
    box_db.set_status(pick['id'], 'approved', approved_at=int(time.time()))

    pdf_bytes = card_pdf.render(
        recipient=pick['customer_first_name'],
        note_body=pick['note_text'],
        piece_names=_piece_names_from_skus(skus),
        signoff="with care, roen",
    )
    pdf_path = Path(f"/tmp/roen-card-{pick['id']}.pdf")
    pdf_path.write_bytes(pdf_bytes)
    send_document_fn(chat_id, pdf_path, caption="tap to print")


def _reroll(pick, chat_id: int, *, send_message_fn: Callable) -> None:
    """Re-run the picker for an existing pick row. New picks + new note."""
    candidates = wc_orders.fetch_in_stock_bracelets()
    history = box_db.history_for_email(pick['customer_email'])
    history_dicts = [{
        'color_tags': json.loads(h['color_tags'] or '[]'),
        'style_tags': json.loads(h['style_tags'] or '[]'),
    } for h in history]
    try:
        new_picks = picker.pick_five(candidates, history_dicts,
                                      rng=random.Random())
    except InsufficientStock:
        send_message_fn(chat_id, "⚠ not enough stock to reroll — investigate")
        return
    new_note = note_writer.generate(
        picks=new_picks,
        first_name=pick['customer_first_name'],
        past_notes=[h['note_text'] for h in history],
        order_count=len(history) + 1,
    )
    box_db.update_picks(
        pick['id'],
        picked_skus=[p['id'] for p in new_picks],
        color_tags=[p['color_family'] for p in new_picks],
        style_tags=[p['style_class'] for p in new_picks],
    )
    box_db.update_note(pick['id'], new_note)
    send_message_fn(
        chat_id,
        f"new draft for pick #{pick['id']}:\n\n{new_note or '[note generation failed]'}",
        reply_markup=keyboard(pick['id']),
    )


def handle_swap_message(chat_id: int, text: str, *,
                         send_message_fn: Callable) -> bool:
    """Consume a 'swap N' reply if Sarah is in swap mode. Returns True if handled."""
    pending = _peek_pending(chat_id)
    if not pending or pending[1] != "swap":
        return False
    parts = text.strip().lower().split()
    if len(parts) != 2 or parts[0] != "swap" or not parts[1].isdigit():
        send_message_fn(chat_id, "format: `swap 3` (slot 1-5)")
        return True  # consumed (we held the pending state)
    slot = int(parts[1])
    if not 1 <= slot <= 5:
        send_message_fn(chat_id, "slot must be 1-5")
        return True

    # Now consume the pending state
    _pop_pending(chat_id)
    pick_id = pending[0]
    pick = box_db.get_pick(pick_id)
    if not pick:
        send_message_fn(chat_id, f"pick #{pick_id} no longer exists")
        return True

    skus = json.loads(pick['picked_skus'])
    candidates = wc_orders.fetch_in_stock_bracelets()
    not_picked = [c for c in candidates if c['id'] not in skus]
    if not not_picked:
        send_message_fn(chat_id, "no other in-stock bracelets to swap to")
        return True
    new = random.choice(not_picked)
    skus[slot - 1] = new['id']
    by_id = {c['id']: c for c in candidates}
    box_db.update_picks(
        pick_id,
        picked_skus=skus,
        color_tags=[by_id[s]['color_family'] for s in skus if s in by_id],
        style_tags=[by_id[s]['style_class']  for s in skus if s in by_id],
    )
    send_message_fn(
        chat_id,
        f"slot {slot} → {new['name']}\n\nApprove or swap another?",
        reply_markup=keyboard(pick_id),
    )
    return True


def handle_note_edit_message(chat_id: int, text: str, *,
                              send_message_fn: Callable) -> bool:
    """Consume a quote-reply rewritten note. Returns True if handled."""
    pending = _peek_pending(chat_id)
    if not pending or pending[1] != "editnote":
        return False
    _pop_pending(chat_id)
    pick_id = pending[0]
    new_text = text.strip()
    if not new_text:
        send_message_fn(chat_id, "empty note rejected")
        return True
    box_db.update_note(pick_id, new_text)
    send_message_fn(
        chat_id,
        f"note updated. preview:\n\n{new_text}",
        reply_markup=keyboard(pick_id),
    )
    return True
