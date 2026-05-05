"""CRUD for the roen_bracelet_box_picks table.

Lives alongside core.jewelry.db (same SQLite file). Schema bootstrap is
handled by core.jewelry.db.init() — this module assumes the table exists.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

DB_PATH = Path("/home/aialfred/alfred/data/jewelry.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.row_factory = sqlite3.Row
    return c


def create_pick(
    order_id: int,
    line_item_id: int,
    bundle_index: int,
    customer_email: str,
    customer_first_name: Optional[str],
    picked_skus: List[int],
    color_tags: List[str],
    style_tags: List[str],
    note_text: str,
) -> int:
    """Insert a new pick session row in 'suggested' state. Returns the row id."""
    now = int(time.time())
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO roen_bracelet_box_picks
               (order_id, line_item_id, bundle_index, customer_email,
                customer_first_name, picked_skus, color_tags, style_tags,
                note_text, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'suggested', ?)""",
            (order_id, line_item_id, bundle_index,
             customer_email.lower().strip(), customer_first_name,
             json.dumps(picked_skus), json.dumps(color_tags),
             json.dumps(style_tags), note_text, now),
        )
        return cur.lastrowid


def get_pick(pick_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM roen_bracelet_box_picks WHERE id = ?", (pick_id,)
        ).fetchone()


def set_status(pick_id: int, status: str,
               approved_at: Optional[int] = None,
               shipped_at: Optional[int] = None) -> None:
    fields = ["status = ?"]
    args: list = [status]
    if approved_at is not None:
        fields.append("approved_at = ?")
        args.append(approved_at)
    if shipped_at is not None:
        fields.append("shipped_at = ?")
        args.append(shipped_at)
    args.append(pick_id)
    with _conn() as c:
        c.execute(
            f"UPDATE roen_bracelet_box_picks SET {', '.join(fields)} WHERE id = ?",
            tuple(args),
        )


def update_picks(
    pick_id: int,
    picked_skus: List[int],
    color_tags: List[str],
    style_tags: List[str],
) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE roen_bracelet_box_picks
               SET picked_skus = ?, color_tags = ?, style_tags = ?
               WHERE id = ?""",
            (json.dumps(picked_skus), json.dumps(color_tags),
             json.dumps(style_tags), pick_id),
        )


def update_note(pick_id: int, note_text: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE roen_bracelet_box_picks SET note_text = ? WHERE id = ?",
            (note_text, pick_id),
        )


def history_for_email(email: str, limit: int = 5) -> List[sqlite3.Row]:
    """Return prior shipped/approved picks for an email, newest first."""
    with _conn() as c:
        return list(c.execute(
            """SELECT * FROM roen_bracelet_box_picks
               WHERE customer_email = ? AND status IN ('approved','shipped')
               ORDER BY created_at DESC, id DESC LIMIT ?""",
            (email.lower().strip(), limit),
        ))


def list_pending(older_than_seconds: int = 0) -> List[sqlite3.Row]:
    """Return pick sessions still in suggested/awaiting_sarah, optionally
    only those older than N seconds (for the daily nudge)."""
    cutoff = int(time.time()) - older_than_seconds
    with _conn() as c:
        return list(c.execute(
            """SELECT * FROM roen_bracelet_box_picks
               WHERE status IN ('suggested','awaiting_sarah')
                 AND created_at <= ?
               ORDER BY created_at""",
            (cutoff,),
        ))
