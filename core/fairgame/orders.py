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

``orders.state`` is one column shared with stripe_connect. ``stripe_connect``
writes the terminal ``'released'`` / ``'refunded'`` state itself; both its payout
and refund guards accept our ``'paid'`` vocabulary directly (as well as their own
legacy ``'held'``), so confirm/fail call straight through with no fragile
"re-assert 'held'" dance and never leave the order in an observable intermediate
state. A crash mid-transition leaves the order in ``'paid'`` (retryable) — never
stranded — and confirm/fail are idempotent on their terminal state.

Concurrency: the seat is the single point of contention. ``create_order`` claims
the listing with a conditional ``UPDATE ... WHERE status='active'`` BEFORE any
money is held; only the buyer who wins that claim (rowcount == 1) creates the
held payment + order, so two simultaneous buyers can never both pay for one seat.

Any other transition (e.g. confirming an already-refunded order, or operating on
a missing order) raises ``OrderError``.

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

    Atomically CLAIMS the listing (``UPDATE listings SET status='sold' WHERE
    id=? AND status='active'``) and only proceeds if it won the claim — this
    conditional update is the lock, so two concurrent buyers can never both pass
    an "is active" check and both get a held payment. Only the winner then holds
    the buyer's funds via Stripe (``create_held_payment``), prices the order at
    the listing's locked ``buyer_total_cents`` (face + $15 — never a caller-
    supplied amount), and advances the order to ``'paid'``. Returns the stored
    order row.

    Raises ``OrderError`` if the buyer is missing, the listing doesn't exist, or
    the listing is not available (already sold/cancelled, or lost the race).
    """
    if not buyer_fan_id:
        raise OrderError("buyer_fan_id required")
    if not listing_id:
        raise OrderError("listing_id required")

    listing = listings.get_listing(listing_id)
    if listing is None:
        raise OrderError("no such listing")

    amount = int(listing["buyer_total_cents"])
    order_id = "ord_" + uuid.uuid4().hex[:12]
    now = int(time.time())

    # Claim the seat atomically BEFORE touching any money. The conditional UPDATE
    # is the lock: exactly one concurrent buyer can flip 'active' -> 'sold', and
    # only that winner (rowcount == 1) is allowed to hold funds / create an order.
    with db.connect() as c:
        cur = c.execute(
            "UPDATE listings SET status='sold' WHERE id=? AND status='active'",
            (listing_id,),
        )
        if cur.rowcount != 1:
            raise OrderError(
                f"listing not available (status '{listing['status']}')"
            )

    # We won the claim — now hold the buyer's money (sim by default; real Stripe
    # behind FAIRGAME_STRIPE_KEY). create_held_payment creates the order row and
    # records the payment ref. Then advance our vocabulary to 'paid'.
    stripe_connect.create_held_payment(
        order_id,
        amount,
        listing_id=listing_id,
        buyer_fan_id=buyer_fan_id,
    )

    with db.connect() as c:
        c.execute(
            "UPDATE orders SET state='paid', updated_at=? WHERE id=?",
            (now, order_id),
        )
        out = _get_order_row(c, order_id)
    return dict(out)


def confirm_transfer(order_id: str) -> dict:
    """Transfer confirmed on TM rails — release the seller's proceeds.

    Only a ``'paid'`` order can be confirmed. Initiates + confirms the (simulated)
    Ticketmaster transfer, records its ref on the order, releases the held funds
    to the seller (``release_to_seller``), and advances the order to ``'released'``.
    Idempotent on an order already ``'released'``.

    Accepts ``'paid'`` (and legacy ``'held'``, so a transition that crashed before
    stripe advanced state can be retried to completion). Idempotent on
    ``'released'``. Raises ``OrderError`` if the order doesn't exist or is in any
    other state (e.g. a refunded order can never be released).
    """
    if not order_id:
        raise OrderError("order_id required")
    with db.connect() as c:
        row = _get_order_row(c, order_id)
    if row is None:
        raise OrderError("no such order")
    if row["state"] == "released":
        return dict(row)
    if row["state"] not in ("paid", "held"):
        raise OrderError(f"cannot confirm order in state '{row['state']}'")

    # Move the ticket on TM's rails (simulated): initiate then confirm.
    tm_transfer.initiate(order_id)
    transfer = tm_transfer.confirm(order_id)

    # Release the escrowed funds to the seller (keeps Rod's $5). release_to_seller
    # accepts our 'paid'/'held' vocabulary and sets state -> 'released' itself, so
    # there is no observable intermediate state to strand the order in.
    now = int(time.time())
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

    Accepts ``'paid'`` (and legacy ``'held'``, so a transition that crashed before
    stripe advanced state can be retried to completion). Idempotent on
    ``'refunded'``. Raises ``OrderError`` if the order doesn't exist or is in any
    other state (e.g. a released order can never be refunded here).
    """
    if not order_id:
        raise OrderError("order_id required")
    with db.connect() as c:
        row = _get_order_row(c, order_id)
    if row is None:
        raise OrderError("no such order")
    if row["state"] == "refunded":
        return dict(row)
    if row["state"] not in ("paid", "held"):
        raise OrderError(f"cannot fail order in state '{row['state']}'")

    # Refund the buyer's held funds (seller gets nothing). refund() accepts our
    # 'paid'/'held' vocabulary and sets state -> 'refunded' itself, so there is no
    # observable intermediate state to strand the order in.
    now = int(time.time())
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
