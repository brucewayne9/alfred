"""
Roen jewelry intake state — SQLite, single file.

Each intake is one Telegram session: a few photos + a price + the resulting
WooCommerce product. The state machine progresses linearly:

    received -> describing -> drafting -> done | error

We never delete rows — everything is auditable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

DB_PATH = Path("/home/aialfred/alfred/data/jewelry.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS intakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_handle TEXT,
    status TEXT NOT NULL DEFAULT 'received',
    photos_json TEXT NOT NULL DEFAULT '[]',
    price_cents INTEGER,
    raw_caption TEXT,
    description TEXT,
    seo_title TEXT,
    sku TEXT,
    short_description TEXT,
    long_description TEXT,
    woocommerce_post_id INTEGER,
    error TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intakes_chat ON intakes (chat_id, status);
CREATE INDEX IF NOT EXISTS idx_intakes_status ON intakes (status, updated_at);

CREATE TABLE IF NOT EXISTS roen_bracelet_box_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    line_item_id INTEGER NOT NULL,
    bundle_index INTEGER NOT NULL,
    customer_email TEXT NOT NULL,
    customer_first_name TEXT,
    picked_skus TEXT NOT NULL,
    color_tags TEXT,
    style_tags TEXT,
    note_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'suggested',
    created_at INTEGER NOT NULL,
    approved_at INTEGER,
    shipped_at INTEGER,
    UNIQUE(order_id, line_item_id, bundle_index)
);
CREATE INDEX IF NOT EXISTS idx_box_email_status ON roen_bracelet_box_picks (customer_email, status);
CREATE INDEX IF NOT EXISTS idx_box_status_created ON roen_bracelet_box_picks (status, created_at);
"""


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.row_factory = sqlite3.Row
    return c


def init() -> None:
    with _conn() as c:
        c.executescript(SCHEMA)


def open_intake(chat_id: int, user_handle: Optional[str] = None) -> int:
    """Create a new pending intake. Returns intake id."""
    now = int(time.time())
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO intakes (chat_id, user_handle, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (chat_id, user_handle, "received", now, now),
        )
        return cur.lastrowid


def get_active_intake(chat_id: int, max_age_seconds: int = 600) -> Optional[sqlite3.Row]:
    """Return the most recent in-progress intake for this chat, if within window."""
    cutoff = int(time.time()) - max_age_seconds
    with _conn() as c:
        return c.execute(
            "SELECT * FROM intakes WHERE chat_id=? AND status IN ('received','describing','drafting') AND updated_at >= ? ORDER BY id DESC LIMIT 1",
            (chat_id, cutoff),
        ).fetchone()


def add_photo(intake_id: int, file_path: str, telegram_file_id: str) -> None:
    with _conn() as c:
        row = c.execute("SELECT photos_json FROM intakes WHERE id=?", (intake_id,)).fetchone()
        photos = json.loads(row["photos_json"]) if row else []
        photos.append({"path": file_path, "telegram_file_id": telegram_file_id, "added_at": int(time.time())})
        c.execute(
            "UPDATE intakes SET photos_json=?, updated_at=? WHERE id=?",
            (json.dumps(photos), int(time.time()), intake_id),
        )


def set_price(intake_id: int, price_cents: int, raw_caption: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE intakes SET price_cents=?, raw_caption=COALESCE(raw_caption,'') || ?, updated_at=? WHERE id=?",
            (price_cents, raw_caption + "\n", int(time.time()), intake_id),
        )


def set_status(intake_id: int, status: str, error: Optional[str] = None) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE intakes SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, int(time.time()), intake_id),
        )


def set_description(intake_id: int, description: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE intakes SET description=?, updated_at=? WHERE id=?",
            (description, int(time.time()), intake_id),
        )


def set_copy(
    intake_id: int,
    seo_title: str,
    sku: str,
    short_description: str,
    long_description: str,
) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE intakes SET seo_title=?, sku=?, short_description=?, long_description=?, updated_at=? WHERE id=?",
            (seo_title, sku, short_description, long_description, int(time.time()), intake_id),
        )


def set_woocommerce_id(intake_id: int, post_id: int) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE intakes SET woocommerce_post_id=?, updated_at=? WHERE id=?",
            (post_id, int(time.time()), intake_id),
        )


def get_intake(intake_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM intakes WHERE id=?", (intake_id,)).fetchone()


def list_recent(limit: int = 20) -> Iterable[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM intakes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def list_since(chat_id: int, since_epoch: int) -> Iterable[sqlite3.Row]:
    """Intakes for a chat with created_at >= since_epoch, newest first."""
    with _conn() as c:
        return c.execute(
            "SELECT * FROM intakes WHERE chat_id=? AND created_at >= ? ORDER BY id DESC",
            (chat_id, since_epoch),
        ).fetchall()


def latest_intake_with_post(chat_id: int, max_age_seconds: int = 86400) -> Optional[sqlite3.Row]:
    """Most recent intake for this chat that has a WooCommerce post — regardless of status."""
    cutoff = int(time.time()) - max_age_seconds
    with _conn() as c:
        return c.execute(
            "SELECT * FROM intakes WHERE chat_id=? AND woocommerce_post_id IS NOT NULL AND updated_at >= ? ORDER BY id DESC LIMIT 1",
            (chat_id, cutoff),
        ).fetchone()


def find_intake_by_post(post_id: int) -> Optional[sqlite3.Row]:
    """Look up an intake by its WooCommerce post_id."""
    with _conn() as c:
        return c.execute(
            "SELECT * FROM intakes WHERE woocommerce_post_id=? LIMIT 1",
            (post_id,),
        ).fetchone()
