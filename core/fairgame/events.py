"""Fair Game events + inventory.

Seeds Rod's 35-date "Don't Look Down" tour into the `shows` table from the
canonical tour file (`data/mainstay/tour/arena_folder_links.json`) and manages
per-show section inventory. Money is integer cents everywhere — face prices are
stored as cents and inventory math is done on `qty_available`.

Schema is owned by the Foundation phase (core/fairgame/db.py); this module only
reads/writes the `shows` and `inventory` tables, it never alters them. Show ids
are deterministic (`show_<idx>`) so seeding is idempotent and re-runnable.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from . import db

# Demo section template: (name, face price in cents, qty). Realistic Rod Wave
# arena pricing — $150 floor / $95 lower bowl / $55 upper, modest demo qty.
DEMO_SECTIONS = (
    ("Floor", 15000, 200),
    ("Lower", 9500, 400),
    ("Upper", 5500, 600),
)


def _tour_file() -> Path:
    override = os.environ.get("FAIRGAME_TOUR_FILE")
    if override:
        return Path(override)
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "mainstay" / "tour" / "arena_folder_links.json"
    )


def _show_id(idx) -> str:
    return f"show_{idx}"


def seed_shows() -> int:
    """Upsert each arena from the tour JSON into `shows`. Idempotent.

    Keyed on a deterministic `show_<idx>` id, so re-running refreshes
    city/venue/date without creating duplicates and preserves created_at.
    Returns the number of shows in the file.
    """
    rows = json.loads(_tour_file().read_text())
    now = int(time.time())
    with db.connect() as c:
        for r in rows:
            sid = _show_id(r["idx"])
            existing = c.execute("SELECT id FROM shows WHERE id=?", (sid,)).fetchone()
            if existing:
                c.execute(
                    "UPDATE shows SET idx=?,city=?,venue=?,show_date=? WHERE id=?",
                    (r["idx"], r.get("city"), r.get("venue"), r.get("dates"), sid),
                )
            else:
                c.execute(
                    "INSERT INTO shows(id,idx,city,venue,show_date,status,created_at) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (sid, r["idx"], r.get("city"), r.get("venue"),
                     r.get("dates"), "on_sale", now),
                )
    return len(rows)


def list_shows() -> list:
    with db.connect() as c:
        rows = c.execute("SELECT * FROM shows ORDER BY idx").fetchall()
    return [dict(r) for r in rows]


def get_show(show_id: str):
    with db.connect() as c:
        row = c.execute("SELECT * FROM shows WHERE id=?", (show_id,)).fetchone()
    return dict(row) if row else None


def add_inventory(show_id: str, section: str, qty: int, face_price_cents: int) -> dict:
    """Create an inventory row for a show section. qty_available starts at qty."""
    iid = "inv_" + uuid.uuid4().hex[:12]
    now = int(time.time())
    with db.connect() as c:
        c.execute(
            "INSERT INTO inventory(id,show_id,section,qty_total,qty_available,"
            "face_price_cents,created_at) VALUES(?,?,?,?,?,?,?)",
            (iid, show_id, section, qty, qty, face_price_cents, now),
        )
        row = c.execute("SELECT * FROM inventory WHERE id=?", (iid,)).fetchone()
    return dict(row)


def get_inventory(show_id: str) -> list:
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM inventory WHERE show_id=? ORDER BY face_price_cents DESC",
            (show_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def section_face_cents(show_id: str, section: str):
    """Original primary face price (cents) for a show's section, or None.

    The anti-scalper cap is computed off the TRUE face, not a seller-declared
    one, so resale callers look the face up here instead of trusting the client.
    Returns None when the show has no inventory row for that section (e.g. in
    unit tests that don't seed inventory — callers decide how to handle that).
    """
    with db.connect() as c:
        row = c.execute(
            "SELECT face_price_cents FROM inventory WHERE show_id=? AND section=? "
            "ORDER BY created_at ASC LIMIT 1",
            (show_id, section),
        ).fetchone()
    return int(row["face_price_cents"]) if row else None


def remaining(show_id: str) -> int:
    """Total tickets still available across all sections for a show."""
    with db.connect() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(qty_available), 0) AS n FROM inventory WHERE show_id=?",
            (show_id,),
        ).fetchone()
    return int(row["n"])


def update_inventory(inv_id, *, face_price_cents=None, qty_available=None, qty_total=None):
    """Patch an inventory row's price/quantities. Returns the row or None."""
    sets, vals = [], []
    for col, v in (("face_price_cents", face_price_cents),
                   ("qty_available", qty_available), ("qty_total", qty_total)):
        if v is not None:
            sets.append(f"{col}=?"); vals.append(int(v))
    with db.connect() as c:
        if sets:
            vals.append(inv_id)
            cur = c.execute(f"UPDATE inventory SET {', '.join(sets)} WHERE id=?", vals)
            if cur.rowcount == 0:
                return None
        row = c.execute("SELECT * FROM inventory WHERE id=?", (inv_id,)).fetchone()
    return dict(row) if row else None


def seed_demo_inventory() -> int:
    """Give every seeded show the 3 standard sections (idempotent per section).

    Skips a section if a row with that name already exists for the show, so
    re-running does not stack duplicate inventory. Returns rows created.
    """
    created = 0
    for show in list_shows():
        existing = {row["section"] for row in get_inventory(show["id"])}
        for name, price, qty in DEMO_SECTIONS:
            if name in existing:
                continue
            add_inventory(show["id"], name, qty, price)
            created += 1
    return created
