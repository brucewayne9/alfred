"""Fair Game tour-admin queries — read-only rollups + broker flagging.

Powers the Tour Admin Console (Section 5.5 of the design spec): the operator
view over the whole marketplace — how many fans signed up and verified, how many
got priority off the DLD waitlist, how much inventory is left across the tour,
how many seats are listed for capped resale, where every order sits in the
escrow state machine, and how much Rod has actually earned in platform fees on
completed (released) resales.

Everything here is integer cents and reads straight off the tables owned by the
Foundation phase (core/fairgame/db.py); this module never alters the schema. The
only write is ``flag_broker`` which sets a fan's ``status`` to ``'flagged'`` so a
suspected scalper can be reviewed/excluded without deleting their record.
"""
from __future__ import annotations

from . import db


def _row_to_dict(row):
    return dict(row) if row else None


def stats() -> dict:
    """One-shot dashboard rollup across the whole marketplace.

    Returns:
        fans_total              -- every fan record
        fans_verified           -- fans with status='verified'
        fans_priority           -- fans flagged priority (DLD waitlist seed)
        shows                   -- seeded tour dates
        inventory_remaining     -- SUM(qty_available) across all inventory
        listings_active         -- listings with status='active'
        orders_by_state         -- {state: count} over all orders
        gross_platform_fees_cents
                                -- SUM(listings.platform_fee_cents) for every
                                   order that reached state='released' (Rod's $5
                                   cut, only on resales that actually settled)
    """
    with db.connect() as c:
        fans_total = c.execute("SELECT COUNT(*) AS n FROM fans").fetchone()["n"]
        fans_verified = c.execute(
            "SELECT COUNT(*) AS n FROM fans WHERE status='verified'"
        ).fetchone()["n"]
        fans_priority = c.execute(
            "SELECT COUNT(*) AS n FROM fans WHERE priority=1"
        ).fetchone()["n"]
        shows = c.execute("SELECT COUNT(*) AS n FROM shows").fetchone()["n"]
        inventory_remaining = c.execute(
            "SELECT COALESCE(SUM(qty_available), 0) AS n FROM inventory"
        ).fetchone()["n"]
        listings_active = c.execute(
            "SELECT COUNT(*) AS n FROM listings WHERE status='active'"
        ).fetchone()["n"]

        orders_by_state = {
            r["state"]: r["n"]
            for r in c.execute(
                "SELECT state, COUNT(*) AS n FROM orders GROUP BY state"
            ).fetchall()
        }

        # Rod's earned fees: platform_fee on the listing behind every order that
        # actually settled (released). Pending/paid/refunded earn nothing.
        gross_platform_fees_cents = c.execute(
            "SELECT COALESCE(SUM(l.platform_fee_cents), 0) AS n "
            "FROM orders o JOIN listings l ON l.id = o.listing_id "
            "WHERE o.state='released'"
        ).fetchone()["n"]

    return {
        "fans_total": int(fans_total),
        "fans_verified": int(fans_verified),
        "fans_priority": int(fans_priority),
        "shows": int(shows),
        "inventory_remaining": int(inventory_remaining),
        "listings_active": int(listings_active),
        "orders_by_state": orders_by_state,
        "gross_platform_fees_cents": int(gross_platform_fees_cents),
    }


def list_fans(limit: int = 100) -> list:
    """Most-recent fans first, capped at ``limit``. Returns rows as dicts."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM fans ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def list_orders(limit: int = 100) -> list:
    """Most-recent orders first, capped at ``limit``. Returns rows as dicts."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM orders ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def flag_broker(fan_id: str) -> dict | None:
    """Mark a suspected scalper's fan record ``status='flagged'`` for review.

    Non-destructive (the record is kept, just flagged). Returns the updated fan
    row as a dict, or None if no such fan exists.
    """
    with db.connect() as c:
        c.execute("UPDATE fans SET status='flagged' WHERE id=?", (fan_id,))
        row = c.execute("SELECT * FROM fans WHERE id=?", (fan_id,)).fetchone()
    return _row_to_dict(row)
