"""Fair Game — Ticketmaster transfer (SIMULATED).

Fair Game rides Ticketmaster's rails: TM issues the barcode and handles entry,
while Fair Game owns the fan, the cap, and the resale fee. The actual ticket
moves between fans on TM's transfer system; the money moves separately on Stripe.
The two never touch — Fair Game holds the buyer's funds until the transfer is
confirmed, then pays the seller (escrow, handled by the orders flow).

>>> THIS IS A SIMULATION. <<<
There is no real Ticketmaster integration in v1. `initiate`, `confirm`, and
`status` model the transfer state machine against the local `transfers` table so
the resale escrow flow can be built and demoed end-to-end. When the real TM
Partner API is wired up, swap ONLY the bodies of `initiate` / `confirm` (and have
`status` read TM's reported state) — keep these signatures and the
'initiated' -> 'confirmed' state contract stable so callers don't change.

State machine:
    (no row)  --initiate-->  'initiated'  --confirm-->  'confirmed'

Schema is owned by the Foundation phase (core/fairgame/db.py); this module only
reads/writes the `transfers` table, it never alters it. One transfer row per order.
"""
from __future__ import annotations

import time
import uuid

from . import db


class TransferError(Exception):
    """Raised on an illegal transfer state transition."""


def _row_to_dict(row):
    return dict(row) if row else None


def _get_by_order(c, order_id: str):
    return c.execute(
        "SELECT * FROM transfers WHERE order_id=? ORDER BY created_at DESC LIMIT 1",
        (order_id,),
    ).fetchone()


def initiate(order_id: str) -> dict:
    """Begin the (simulated) TM transfer for an order.

    Creates a `transfers` row in state 'initiated', or returns the existing one
    if a transfer for this order is already initiated (idempotent). Returns the
    transfer row; the `id` is the transfer ref callers store on the order.

    REAL TM: replace the body with a call to the TM Partner transfer-initiate
    endpoint and persist the TM reference here.
    """
    if not order_id:
        raise TransferError("order_id required")
    now = int(time.time())
    with db.connect() as c:
        existing = _get_by_order(c, order_id)
        if existing is not None:
            if existing["state"] == "confirmed":
                raise TransferError("transfer already confirmed")
            # Already initiated — idempotent.
            return dict(existing)
        ref = "tx_" + uuid.uuid4().hex[:12]
        c.execute(
            "INSERT INTO transfers(id,order_id,state,created_at,updated_at) "
            "VALUES(?,?,?,?,?)",
            (ref, order_id, "initiated", now, now),
        )
        row = c.execute("SELECT * FROM transfers WHERE id=?", (ref,)).fetchone()
    return dict(row)


def confirm(order_id: str) -> dict:
    """Confirm the (simulated) TM transfer — the buyer now holds the ticket.

    Moves the transfer to state 'confirmed'. This is the escrow trigger: only a
    confirmed transfer releases the seller's funds. Idempotent on an already
    confirmed transfer. Raises TransferError if no transfer was initiated.

    REAL TM: replace the body with a poll/webhook against TM's transfer status
    and only flip to 'confirmed' once TM reports the recipient accepted.
    """
    if not order_id:
        raise TransferError("order_id required")
    now = int(time.time())
    with db.connect() as c:
        row = _get_by_order(c, order_id)
        if row is None:
            raise TransferError("no transfer initiated for order")
        if row["state"] == "confirmed":
            return dict(row)
        c.execute(
            "UPDATE transfers SET state='confirmed', updated_at=? WHERE id=?",
            (now, row["id"]),
        )
        out = c.execute("SELECT * FROM transfers WHERE id=?", (row["id"],)).fetchone()
    return dict(out)


def status(order_id: str):
    """Return the current transfer row for an order, or None if none exists.

    REAL TM: have this reflect TM's reported transfer state.
    """
    with db.connect() as c:
        return _row_to_dict(_get_by_order(c, order_id))
