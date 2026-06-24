"""Fair Game presale access engine — the gate.

Runs the access waves that decide WHO can buy and HOW MANY before tickets
go on general sale. A fan must be verified; an access wave for the show must
be open; priority-only waves admit only DLD-waitlist (priority) fans; and the
requested qty is bounded by both the wave's per-fan cap and the show's
remaining inventory.

A successful grant writes an `access_grants` row and decrements live
`inventory.qty_available` (cheapest sections first, so the priciest seats are
held longest) — both inside a single transaction. Schema is owned by the
Foundation phase (core/fairgame/db.py); this module only reads/writes the
`access_waves`, `access_grants`, and `inventory` tables.
"""
from __future__ import annotations

import time
import uuid

from . import db, events, identity


class AccessError(Exception):
    """Raised when a presale access request cannot be granted."""


def _now(now=None) -> int:
    return int(now) if now is not None else int(time.time())


def create_wave(
    show_id: str,
    name: str,
    starts_at: int,
    ends_at: int,
    priority_only: bool = False,
    max_qty_per_fan: int = 4,
) -> dict:
    """Create a presale access wave for a show. Returns the wave row as a dict."""
    wid = "wave_" + uuid.uuid4().hex[:12]
    now = int(time.time())
    with db.connect() as c:
        c.execute(
            "INSERT INTO access_waves(id,show_id,name,starts_at,ends_at,"
            "priority_only,max_qty_per_fan,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (wid, show_id, name, int(starts_at), int(ends_at),
             1 if priority_only else 0, int(max_qty_per_fan), now),
        )
        row = c.execute("SELECT * FROM access_waves WHERE id=?", (wid,)).fetchone()
    return dict(row)


def open_waves(show_id: str, now=None) -> list:
    """Waves for a show that are currently open: starts_at <= now <= ends_at."""
    t = _now(now)
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM access_waves WHERE show_id=? AND starts_at<=? AND ends_at>=? "
            "ORDER BY starts_at",
            (show_id, t, t),
        ).fetchall()
    return [dict(r) for r in rows]


def _pick_wave(show_id: str, fan: dict, now) -> dict:
    """Return the most appropriate open wave the fan qualifies for, else raise."""
    waves = open_waves(show_id, now)
    if not waves:
        raise AccessError("no open access wave for this show")
    eligible = [w for w in waves if not w["priority_only"] or fan["priority"] == 1]
    if not eligible:
        raise AccessError("this access wave is priority-only")
    # Prefer the wave with the largest per-fan allowance.
    return max(eligible, key=lambda w: w["max_qty_per_fan"])


def can_purchase(fan_id: str, show_id: str, qty: int, now=None) -> bool:
    """True if grant_access would succeed for these args, without writing anything."""
    fan = identity.get_fan(fan_id)
    if not fan or fan["status"] != "verified":
        return False
    if qty < 1:
        return False
    try:
        wave = _pick_wave(show_id, fan, now)
    except AccessError:
        return False
    if qty > wave["max_qty_per_fan"]:
        return False
    if qty > events.remaining(show_id):
        return False
    return True


def grant_access(fan_id: str, show_id: str, qty: int, tm_email=None, final_sale_ack=False, now=None) -> dict:
    """Grant presale access, write the grant, and decrement inventory.

    Raises AccessError with a clear message if the fan is unverified, no wave
    is open, the wave is priority-only and the fan is not priority, the qty
    exceeds the per-fan cap, or there is not enough remaining inventory.
    """
    fan = identity.get_fan(fan_id)
    if not fan:
        raise AccessError("fan not found")
    if fan["status"] != "verified":
        raise AccessError("fan is not verified")
    if qty < 1:
        raise AccessError("qty must be at least 1")

    wave = _pick_wave(show_id, fan, now)
    if qty > wave["max_qty_per_fan"]:
        raise AccessError(
            f"qty {qty} exceeds wave limit of {wave['max_qty_per_fan']} per fan"
        )
    if qty > events.remaining(show_id):
        raise AccessError("not enough tickets remaining")

    gid = "grant_" + uuid.uuid4().hex[:12]
    created = _now(now)
    remaining_to_take = qty
    with db.connect() as c:
        # Decrement cheapest sections first (hold the priciest seats longest).
        sections = c.execute(
            "SELECT id, qty_available FROM inventory WHERE show_id=? AND qty_available>0 "
            "ORDER BY face_price_cents ASC",
            (show_id,),
        ).fetchall()
        for sec in sections:
            if remaining_to_take <= 0:
                break
            take = min(remaining_to_take, sec["qty_available"])
            c.execute(
                "UPDATE inventory SET qty_available=qty_available-? WHERE id=?",
                (take, sec["id"]),
            )
            remaining_to_take -= take
        if remaining_to_take > 0:
            # Inventory changed between the check and the write — abort cleanly.
            raise AccessError("not enough tickets remaining")
        c.execute(
            "INSERT INTO access_grants(id,fan_id,show_id,wave_id,qty,tm_email,final_sale_ack,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (gid, fan_id, show_id, wave["id"], qty, tm_email, 1 if final_sale_ack else 0, created),
        )
        row = c.execute("SELECT * FROM access_grants WHERE id=?", (gid,)).fetchone()
    return dict(row)
