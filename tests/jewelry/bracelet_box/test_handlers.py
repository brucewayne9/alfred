"""Tests for core.jewelry.bracelet_box.handlers — Telegram callback router."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import pytest

# We patch the modules before importing handlers to avoid real DB/WC calls.
import core.jewelry.bracelet_box.handlers as handlers
from core.jewelry.bracelet_box.tags import InsufficientStock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_pending():
    """Clear the module-level _pending dict between every test."""
    handlers._pending.clear()
    yield
    handlers._pending.clear()


def _make_pick(pick_id=1, order_id=100, customer_email="sarah@example.com",
               customer_first_name="Sarah", note_text="A lovely note",
               picked_skus=None, color_tags=None, style_tags=None, status="awaiting_sarah"):
    """Return a dict that behaves like a db row."""
    return {
        "id": pick_id,
        "order_id": order_id,
        "line_item_id": 10,
        "bundle_index": 1,
        "customer_email": customer_email,
        "customer_first_name": customer_first_name,
        "note_text": note_text,
        "picked_skus": json.dumps(picked_skus or [10, 20, 30, 40, 50]),
        "color_tags": json.dumps(color_tags or ["red", "blue", "green", "pink", "gold"]),
        "style_tags": json.dumps(style_tags or ["a", "b", "c", "d", "e"]),
        "status": status,
    }


def _make_candidates(n=10):
    return [
        {
            "id": i,
            "name": f"Bracelet {i}",
            "color_family": f"color{i}",
            "style_class": f"style{i}",
            "image_url": f"https://example.com/img{i}.jpg",
        }
        for i in range(1, n + 1)
    ]


def _make_picks_5(candidates):
    """Return 5 candidate dicts (first 5 from candidates)."""
    return candidates[:5]


# ---------------------------------------------------------------------------
# 1. open_pick_session happy path
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.wc_orders")
@patch("core.jewelry.bracelet_box.handlers.box_db")
@patch("core.jewelry.bracelet_box.handlers.note_writer")
@patch("core.jewelry.bracelet_box.handlers.picker")
def test_open_pick_session_happy(mock_picker, mock_note, mock_db, mock_wc):
    candidates = _make_candidates()
    picks = _make_picks_5(candidates)
    mock_wc.fetch_in_stock_bracelets.return_value = candidates
    mock_picker.pick_five.return_value = picks
    mock_note.generate.return_value = "Lovely note text"
    mock_db.history_for_email.return_value = []
    mock_db.create_pick.return_value = 7
    mock_db.set_status = MagicMock()

    send_msg = MagicMock()
    send_media = MagicMock()

    result = handlers.open_pick_session(
        order_id=100,
        line_item_id=10,
        bundle_index=1,
        customer_email="sarah@example.com",
        customer_first_name="Sarah",
        sarah_chat_id=999,
        send_message_fn=send_msg,
        send_media_group_fn=send_media,
    )

    assert result == 7
    mock_db.create_pick.assert_called_once()
    mock_db.set_status.assert_called_once_with(7, "awaiting_sarah")
    send_media.assert_called_once_with(999, mock_media_group_arg := send_media.call_args[0][1])
    assert len(mock_media_group_arg) == 5
    send_msg.assert_called_once()
    msg_text = send_msg.call_args[0][1]
    assert "Pick #7" in msg_text
    assert "Lovely note text" in msg_text


# ---------------------------------------------------------------------------
# 2. open_pick_session — InsufficientStock
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.wc_orders")
@patch("core.jewelry.bracelet_box.handlers.box_db")
@patch("core.jewelry.bracelet_box.handlers.note_writer")
@patch("core.jewelry.bracelet_box.handlers.picker")
def test_open_pick_session_insufficient_stock(mock_picker, mock_note, mock_db, mock_wc):
    mock_wc.fetch_in_stock_bracelets.return_value = _make_candidates(3)
    mock_picker.pick_five.side_effect = InsufficientStock("not enough")
    mock_db.history_for_email.return_value = []

    send_msg = MagicMock()
    send_media = MagicMock()

    result = handlers.open_pick_session(
        order_id=101,
        line_item_id=11,
        bundle_index=1,
        customer_email="sarah@example.com",
        customer_first_name="Sarah",
        sarah_chat_id=999,
        send_message_fn=send_msg,
        send_media_group_fn=send_media,
    )

    assert result == 0
    mock_db.create_pick.assert_not_called()
    send_media.assert_not_called()
    send_msg.assert_called_once()
    assert "101" in send_msg.call_args[0][1]  # order_id in warning


# ---------------------------------------------------------------------------
# 3. handle_callback — bx:approve happy path
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.wc_orders")
@patch("core.jewelry.bracelet_box.handlers.box_db")
@patch("core.jewelry.bracelet_box.handlers.card_pdf")
def test_handle_callback_approve_happy(mock_pdf, mock_db, mock_wc, tmp_path):
    pick = _make_pick()
    mock_db.get_pick.return_value = pick
    mock_wc.reserve_skus.return_value = True
    mock_db.set_status = MagicMock()
    mock_pdf.render.return_value = b"%PDF-1.4 fake"

    send_msg = MagicMock()
    send_doc = MagicMock()

    # Patch /tmp write so we don't depend on filesystem details
    with patch.object(Path, "write_bytes"):
        handled = handlers.handle_callback(
            "bx:approve:1", chat_id=999,
            send_message_fn=send_msg, send_document_fn=send_doc,
        )

    assert handled is True
    mock_wc.reserve_skus.assert_called_once()
    mock_db.set_status.assert_called_once_with(1, "approved", approved_at=pytest.approx(time.time(), abs=5))
    mock_pdf.render.assert_called_once()
    send_doc.assert_called_once()


# ---------------------------------------------------------------------------
# 4. handle_callback — bx:approve when reserve_skus fails → reroll
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.wc_orders")
@patch("core.jewelry.bracelet_box.handlers.box_db")
@patch("core.jewelry.bracelet_box.handlers.picker")
@patch("core.jewelry.bracelet_box.handlers.note_writer")
def test_handle_callback_approve_reserve_fails_triggers_reroll(
    mock_note, mock_picker, mock_db, mock_wc
):
    pick = _make_pick()
    candidates = _make_candidates()
    mock_db.get_pick.return_value = pick
    mock_db.history_for_email.return_value = []
    mock_wc.reserve_skus.return_value = False
    mock_wc.fetch_in_stock_bracelets.return_value = candidates
    mock_picker.pick_five.return_value = _make_picks_5(candidates)
    mock_note.generate.return_value = "Rerolled note"
    mock_db.update_picks = MagicMock()
    mock_db.update_note = MagicMock()

    send_msg = MagicMock()
    send_doc = MagicMock()

    handled = handlers.handle_callback(
        "bx:approve:1", chat_id=999,
        send_message_fn=send_msg, send_document_fn=send_doc,
    )

    assert handled is True
    send_doc.assert_not_called()
    mock_db.update_picks.assert_called_once()
    # send_message called with new keyboard
    send_msg.assert_called()
    msg_text = send_msg.call_args[0][1]
    assert "Rerolled note" in msg_text


# ---------------------------------------------------------------------------
# 5. handle_callback — bx:swap sets pending
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_callback_swap_sets_pending(mock_db):
    mock_db.get_pick.return_value = _make_pick()

    send_msg = MagicMock()
    handled = handlers.handle_callback(
        "bx:swap:1", chat_id=999,
        send_message_fn=send_msg, send_document_fn=MagicMock(),
    )

    assert handled is True
    pending = handlers._peek_pending(999)
    assert pending is not None
    assert pending[0] == 1
    assert pending[1] == "swap"
    send_msg.assert_called_once()


# ---------------------------------------------------------------------------
# 6. handle_callback — bx:editnote sets pending
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_callback_editnote_sets_pending(mock_db):
    mock_db.get_pick.return_value = _make_pick()

    send_msg = MagicMock()
    handled = handlers.handle_callback(
        "bx:editnote:1", chat_id=999,
        send_message_fn=send_msg, send_document_fn=MagicMock(),
    )

    assert handled is True
    pending = handlers._peek_pending(999)
    assert pending is not None
    assert pending[1] == "editnote"
    send_msg.assert_called_once()


# ---------------------------------------------------------------------------
# 7. handle_callback — bx:reroll re-runs picker
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.wc_orders")
@patch("core.jewelry.bracelet_box.handlers.box_db")
@patch("core.jewelry.bracelet_box.handlers.picker")
@patch("core.jewelry.bracelet_box.handlers.note_writer")
def test_handle_callback_reroll(mock_note, mock_picker, mock_db, mock_wc):
    pick = _make_pick()
    candidates = _make_candidates()
    mock_db.get_pick.return_value = pick
    mock_db.history_for_email.return_value = []
    mock_wc.fetch_in_stock_bracelets.return_value = candidates
    mock_picker.pick_five.return_value = _make_picks_5(candidates)
    mock_note.generate.return_value = "Fresh note"
    mock_db.update_picks = MagicMock()
    mock_db.update_note = MagicMock()

    send_msg = MagicMock()

    handled = handlers.handle_callback(
        "bx:reroll:1", chat_id=999,
        send_message_fn=send_msg, send_document_fn=MagicMock(),
    )

    assert handled is True
    mock_picker.pick_five.assert_called_once()
    mock_db.update_picks.assert_called_once()
    mock_db.update_note.assert_called_once_with(1, "Fresh note")
    send_msg.assert_called_once()
    msg_text = send_msg.call_args[0][1]
    assert "Fresh note" in msg_text


# ---------------------------------------------------------------------------
# 8. handle_callback — non-bx prefix returns False
# ---------------------------------------------------------------------------

def test_handle_callback_non_bx_returns_false():
    handled = handlers.handle_callback(
        "something:else:here", chat_id=999,
        send_message_fn=MagicMock(), send_document_fn=MagicMock(),
    )
    assert handled is False


# ---------------------------------------------------------------------------
# 9. handle_callback — unknown bx action
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_callback_unknown_action(mock_db):
    mock_db.get_pick.return_value = _make_pick()
    send_msg = MagicMock()

    handled = handlers.handle_callback(
        "bx:explode:1", chat_id=999,
        send_message_fn=send_msg, send_document_fn=MagicMock(),
    )

    assert handled is True
    send_msg.assert_called_once()
    assert "unknown action" in send_msg.call_args[0][1]


# ---------------------------------------------------------------------------
# 10. handle_callback — non-existent pick_id
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_callback_pick_not_found(mock_db):
    mock_db.get_pick.return_value = None
    send_msg = MagicMock()

    handled = handlers.handle_callback(
        "bx:approve:99", chat_id=999,
        send_message_fn=send_msg, send_document_fn=MagicMock(),
    )

    assert handled is True
    send_msg.assert_called_once()
    assert "99" in send_msg.call_args[0][1]


# ---------------------------------------------------------------------------
# 11. handle_swap_message happy path
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.wc_orders")
@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_swap_message_happy(mock_db, mock_wc):
    # Pre-set pending state
    handlers._set_pending(999, 1, "swap")
    pick = _make_pick()  # picked_skus = [10, 20, 30, 40, 50]
    candidates = _make_candidates(10)  # ids 1-10; none overlap with [10] but 10 is in there
    # Make candidates that don't overlap with skus 10,20,30,40,50
    candidates_clean = [c for c in candidates if c["id"] not in [10, 20, 30, 40, 50]]
    candidates_clean.append({"id": 3, "name": "Bracelet 3", "color_family": "color3",
                              "style_class": "style3", "image_url": "url3"})
    mock_db.get_pick.return_value = pick
    mock_wc.fetch_in_stock_bracelets.return_value = candidates_clean
    mock_db.update_picks = MagicMock()

    send_msg = MagicMock()
    handled = handlers.handle_swap_message(999, "swap 3", send_message_fn=send_msg)

    assert handled is True
    mock_db.update_picks.assert_called_once()
    send_msg.assert_called_once()
    # pending should now be cleared
    assert handlers._peek_pending(999) is None


# ---------------------------------------------------------------------------
# 12. handle_swap_message — invalid format: pending preserved
# ---------------------------------------------------------------------------

def test_handle_swap_message_invalid_format_preserves_pending():
    handlers._set_pending(999, 1, "swap")

    send_msg = MagicMock()
    handled = handlers.handle_swap_message(999, "not valid format", send_message_fn=send_msg)

    assert handled is True
    send_msg.assert_called_once()
    # Pending NOT consumed
    assert handlers._peek_pending(999) is not None


# ---------------------------------------------------------------------------
# 13. handle_swap_message — no pending → returns False
# ---------------------------------------------------------------------------

def test_handle_swap_message_no_pending_returns_false():
    send_msg = MagicMock()
    handled = handlers.handle_swap_message(999, "swap 3", send_message_fn=send_msg)
    assert handled is False
    send_msg.assert_not_called()


# ---------------------------------------------------------------------------
# 14. handle_note_edit_message happy path
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_note_edit_message_happy(mock_db):
    handlers._set_pending(999, 1, "editnote")
    mock_db.update_note = MagicMock()

    send_msg = MagicMock()
    handled = handlers.handle_note_edit_message(
        999, "This is my new beautiful note.", send_message_fn=send_msg
    )

    assert handled is True
    mock_db.update_note.assert_called_once_with(1, "This is my new beautiful note.")
    send_msg.assert_called_once()
    assert "new beautiful note" in send_msg.call_args[0][1]
    assert handlers._peek_pending(999) is None


# ---------------------------------------------------------------------------
# 15. handle_note_edit_message — empty text rejected
# ---------------------------------------------------------------------------

@patch("core.jewelry.bracelet_box.handlers.box_db")
def test_handle_note_edit_message_empty_rejected(mock_db):
    handlers._set_pending(999, 1, "editnote")
    mock_db.update_note = MagicMock()

    send_msg = MagicMock()
    handled = handlers.handle_note_edit_message(999, "   ", send_message_fn=send_msg)

    assert handled is True
    mock_db.update_note.assert_not_called()
    send_msg.assert_called_once()
    assert "rejected" in send_msg.call_args[0][1]


# ---------------------------------------------------------------------------
# 16. handle_swap_message ignored when pending is editnote
# ---------------------------------------------------------------------------

def test_handle_swap_message_ignored_when_pending_is_editnote():
    handlers._set_pending(999, 1, "editnote")

    send_msg = MagicMock()
    handled = handlers.handle_swap_message(999, "swap 3", send_message_fn=send_msg)

    assert handled is False
    send_msg.assert_not_called()
    # Pending still there (editnote handler should consume it)
    assert handlers._peek_pending(999) is not None


# ---------------------------------------------------------------------------
# Bonus: keyboard() structure
# ---------------------------------------------------------------------------

def test_keyboard_structure():
    kb = handlers.keyboard(42)
    assert "inline_keyboard" in kb
    buttons = kb["inline_keyboard"][0]
    assert len(buttons) == 4
    cb_data = [b["callback_data"] for b in buttons]
    assert "bx:approve:42" in cb_data
    assert "bx:swap:42" in cb_data
    assert "bx:editnote:42" in cb_data
    assert "bx:reroll:42" in cb_data
