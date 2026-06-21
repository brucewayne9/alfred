"""Fair Game capped resale exchange — listings.

A verified fan who can't attend lists their seat here instead of on SeatGeek.
Resale is HARD-CAPPED by a fixed markup (Rod's call 2026-06-16): the buyer pays
face plus a flat $15, the seller is always made whole on face and keeps $10, and
Rod earns a light $5 per resale. Tickets stay fully transferable — the only
freedom removed is the freedom to gouge. Money is integer cents everywhere.

Schema is owned by the Foundation phase (core/fairgame/db.py); this module only
reads/writes the `listings` table, it never alters it.
"""
from __future__ import annotations

import time
import uuid

from . import db, events

# Fixed resale cap split (cents). Buyer pays face + 1500; seller is made whole on
# face and pockets $10; Rod takes $5. Never scales — flat per ticket in v1.
SELLER_MARKUP_CENTS = 1000   # $10 to the seller, on top of face
PLATFORM_FEE_CENTS = 500     # $5 to Rod


def _row_to_dict(row):
    return dict(row) if row else None


def quote(face_price_cents: int) -> dict:
    """Pure cap math for a given face price (no DB write).

    Returns the three derived amounts so callers (and the checkout flow) share
    one source of truth for the split.
    """
    face = int(face_price_cents)
    return {
        "face_price_cents": face,
        "seller_proceeds_cents": face + SELLER_MARKUP_CENTS,
        "platform_fee_cents": PLATFORM_FEE_CENTS,
        "buyer_total_cents": face + SELLER_MARKUP_CENTS + PLATFORM_FEE_CENTS,
    }


def create_listing(
    seller_fan_id: str,
    show_id: str,
    section: str,
    face_price_cents: int,
) -> dict:
    """List a seat for capped resale and return the stored listing row.

    The split is fixed (see SELLER_MARKUP_CENTS / PLATFORM_FEE_CENTS): the seller
    is made whole on face plus $10, Rod takes $5, the buyer pays face + $15. New
    listings start `status='active'`.

    Anti-gouge: the declared ``face_price_cents`` may never EXCEED the section's
    true primary face (looked up from inventory). A scalper can't claim a $300
    "face" on a $60 seat and have the cap computed off the lie — the listing is
    rejected. A seller may declare a lower face (selling under face is fan-
    friendly and allowed). Shows with no inventory row for the section (e.g. not
    yet seeded) skip the cap check since there is no true face to enforce against.
    """
    if int(face_price_cents) < 0:
        raise ValueError("face_price_cents must be non-negative")
    true_face = events.section_face_cents(show_id, section)
    if true_face is not None and int(face_price_cents) > true_face:
        raise ValueError(
            f"face_price_cents {int(face_price_cents)} exceeds the section's "
            f"primary face of {true_face} (resale cannot gouge above face)"
        )
    q = quote(face_price_cents)
    lid = "lst_" + uuid.uuid4().hex[:12]
    now = int(time.time())
    with db.connect() as c:
        c.execute(
            "INSERT INTO listings(id,show_id,seller_fan_id,section,face_price_cents,"
            "seller_proceeds_cents,platform_fee_cents,buyer_total_cents,status,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                lid, show_id, seller_fan_id, section, q["face_price_cents"],
                q["seller_proceeds_cents"], q["platform_fee_cents"],
                q["buyer_total_cents"], "active", now,
            ),
        )
        row = c.execute("SELECT * FROM listings WHERE id=?", (lid,)).fetchone()
    return dict(row)


def get_listing(listing_id: str):
    with db.connect() as c:
        row = c.execute("SELECT * FROM listings WHERE id=?", (listing_id,)).fetchone()
    return _row_to_dict(row)


def list_active(show_id: str) -> list:
    """All active listings for a show, cheapest face first (best fan deal up top)."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM listings WHERE show_id=? AND status='active' "
            "ORDER BY face_price_cents ASC, created_at ASC",
            (show_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_by_seller(seller_fan_id: str) -> list:
    """A seller's own listings (any status), newest first — powers "My Listings".

    Joined to the show so the UI can label each row by city/date without a second
    round trip.
    """
    with db.connect() as c:
        rows = c.execute(
            "SELECT l.*, s.city AS show_city, s.venue AS show_venue, "
            "s.show_date AS show_date "
            "FROM listings l LEFT JOIN shows s ON s.id = l.show_id "
            "WHERE l.seller_fan_id=? ORDER BY l.created_at DESC",
            (seller_fan_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def cancel_listing(listing_id: str):
    """Mark a listing cancelled (idempotent). Returns the updated row or None."""
    with db.connect() as c:
        c.execute(
            "UPDATE listings SET status='cancelled' WHERE id=?",
            (listing_id,),
        )
        row = c.execute("SELECT * FROM listings WHERE id=?", (listing_id,)).fetchone()
    return _row_to_dict(row)
