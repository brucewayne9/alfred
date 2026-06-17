"""Fair Game — resale escrow state machine (orders).

This is the orchestrator that ties the three resale primitives together so a
verified buyer can safely take over a verified seller's seat:

    listings        -- the capped offer (face + $15, seller +$10 / Rod +$5)
    stripe_connect  -- the money: HOLD the buyer's funds, then release or refund
    tm_transfer     -- the ticket: moves on Ticketmaster's rails (simulated in v1)

The money and the ticket never touch. Fair Game holds the buyer's funds the
instant they pay, the seat moves over TM's transfer system, and only once that
transfer is CONFIRMED does Fair Game release the seller's proceeds (face + $10)
and keep Rod's $5. If the transfer never completes, the held funds are refunded
in full and the seller gets nothing — the fraud wall.

Money is integer cents everywhere; the buyer always pays the listing's
``buyer_total_cents`` (the locked cap), never a free-typed amount.

Escrow state machine (orders.state, owned end-to-end here):

    create_order        --> 'paid'      (funds held, listing marked 'sold')
    confirm_transfer     --> 'released'  ('paid' only; TM transfer + payout)
    fail_transfer        --> 'refunded'  ('paid' only; refund buyer, reopen listing)

``orders.state`` is one column shared with stripe_connect, whose payout/refund
guards only accept its own ``'held'`` vocabulary. We expose ``'paid'`` once the
buyer's funds are held; confirm/fail therefore briefly re-assert ``'held'`` right
before invoking the stripe transition (so its guard passes) and stripe then sets
the terminal ``'released'`` / ``'refunded'`` state — keeping stripe_connect
untouched while our public vocabulary stays ``paid -> released | refunded``.

Any other transition (e.g. confirming an already-refunded order, or operating on
a missing order) raises ``OrderError``. Confirm/fail are idempotent on an order
already in their terminal state.

Schema is owned by the Foundation phase (core/fairgame/db.py); this module only
reads/writes the ``orders`` and ``listings`` tables, it never alters them.
"""
from __future__ import annotations

import time
import uuid

from . import db, listings, stripe_connect, tm_transfer


class OrderError(Exception):
    """Raised on an illegal escrow transition or an invalid order/listing."""


def _row_to_dict(row):
    return dict(row) if row else None


def _get_order_row(c, order_id: str):
    return c.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()


def create_order(buyer_fan_id: str, listing_id: str) -> dict:
    """Buyer claims a listing — hold their funds and lock the seat.

    Validates the listing is ``active``, prices the order at the listing's locked
    ``buyer_total_cents`` (face + $15 — never a caller-supplied amount), holds the
    buyer's funds via Stripe (``create_held_payment``), advances the order to
    ``'paid'`` (funds held in escrow), and marks the listing ``'sold'`` so no one
    else can buy it. Returns the stored order row.

    Raises ``OrderError`` if the buyer is missing, the listing doesn't exist, or
    the listing is not active (already sold/cancelled).
    """
    if not buyer_fan_id:
        raise OrderError("buyer_fan_id required")
    if not listing_id:
        raise OrderError("listing_id required")

    listing = listings.get_listing(listing_id)
    if listing is None:
        raise OrderError("no such listing")
    if listing["status"] != "active":
        raise OrderError(f"listing not available (status '{listing['status']}')")

    amount = int(listing["buyer_total_cents"])
    order_id = "ord_" + uuid.uuid4().hex[:12]

    # Hold the buyer's money (sim by default; real Stripe behind FAIRGAME_STRIPE_KEY).
    # create_held_payment creates the order row in state 'held' and records the ref.
    stripe_connect.create_held_payment(
        order_id,
        amount,
        listing_id=listing_id,
        buyer_fan_id=buyer_fan_id,
    )

    now = int(time.time())
    with db.connect() as c:
        # Funds are held -> 'paid' (our escrow vocabulary), and the seat is taken.
        c.execute(
            "UPDATE orders SET state='paid', updated_at=? WHERE id=?",
            (now, order_id),
        )
        c.execute(
            "UPDATE listings SET status='sold' WHERE id=?",
            (listing_id,),
        )
        out = _get_order_row(c, order_id)
    return dict(out)


def confirm_transfer(order_id: str) -> dict:
    """Transfer confirmed on TM rails — release the seller's proceeds.

    Only a ``'paid'`` order can be confirmed. Initiates + confirms the (simulated)
    Ticketmaster transfer, records its ref on the order, releases the held funds
    to the seller (``release_to_seller``), and advances the order to ``'released'``.
    Idempotent on an order already ``'released'``.

    Raises ``OrderError`` if the order doesn't exist or is in any state other than
    ``'paid'`` / ``'released'`` (e.g. a refunded order can never be released).
    """
    if not order_id:
        raise OrderError("order_id required")
    with db.connect() as c:
        row = _get_order_row(c, order_id)
    if row is None:
        raise OrderError("no such order")
    if row["state"] == "released":
        return dict(row)
    if row["state"] != "paid":
        raise OrderError(f"cannot confirm order in state '{row['state']}'")

    # Move the ticket on TM's rails (simulated): initiate then confirm.
    tm_transfer.initiate(order_id)
    transfer = tm_transfer.confirm(order_id)

    # Re-assert stripe's 'held' vocabulary so its guard accepts the payout, then
    # release the escrowed funds to the seller (keeps Rod's $5). release_to_seller
    # sets state -> 'released'.
    now = int(time.time())
    with db.connect() as c:
        c.execute("UPDATE orders SET state='held' WHERE id=?", (order_id,))
    stripe_connect.release_to_seller(order_id)

    with db.connect() as c:
        c.execute(
            "UPDATE orders SET transfer_ref=?, updated_at=? WHERE id=?",
            (transfer["id"], now, order_id),
        )
        out = _get_order_row(c, order_id)
    return dict(out)


def fail_transfer(order_id: str) -> dict:
    """Transfer failed — refund the buyer and reopen the listing (the fraud wall).

    Only a ``'paid'`` order can be failed. Refunds the held funds to the buyer
    (``refund`` — seller gets nothing), advances the order to ``'refunded'``, and
    flips the listing back to ``'active'`` so the seat can be sold again.
    Idempotent on an order already ``'refunded'``.

    Raises ``OrderError`` if the order doesn't exist or is in any state other than
    ``'paid'`` / ``'refunded'`` (e.g. a released order can never be refunded here).
    """
    if not order_id:
        raise OrderError("order_id required")
    with db.connect() as c:
        row = _get_order_row(c, order_id)
    if row is None:
        raise OrderError("no such order")
    if row["state"] == "refunded":
        return dict(row)
    if row["state"] != "paid":
        raise OrderError(f"cannot fail order in state '{row['state']}'")

    # Re-assert stripe's 'held' vocabulary so its guard accepts the refund, then
    # refund the buyer's held funds (seller gets nothing). refund() sets
    # state -> 'refunded'.
    now = int(time.time())
    with db.connect() as c:
        c.execute("UPDATE orders SET state='held' WHERE id=?", (order_id,))
    stripe_connect.refund(order_id)

    with db.connect() as c:
        c.execute(
            "UPDATE orders SET updated_at=? WHERE id=?",
            (now, order_id),
        )
        if row["listing_id"]:
            c.execute(
                "UPDATE listings SET status='active' WHERE id=?",
                (row["listing_id"],),
            )
        out = _get_order_row(c, order_id)
    return dict(out)


def get_order(order_id: str) -> dict | None:
    """Return the order row as a dict, or None if it doesn't exist."""
    with db.connect() as c:
        return _row_to_dict(_get_order_row(c, order_id))
