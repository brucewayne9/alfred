"""Fair Game — Delivery operator queue.

Assembles the two-sided delivery queue for the human-in-the-loop transfer model:
  - resale ORDERS (paid/released via the transfers state machine)
  - primary ACCESS GRANTS (Rod-held seats granted directly to fans)

Each item exposes the buyer's TM email, what they bought, the show, and the
current delivery state. The operator does the actual TM transfer, then marks it
delivered here.

Real TM automation stays out — this is the supervised queue.
"""
from __future__ import annotations

import time

from . import db, tm_transfer


def queue() -> list[dict]:
    """Return all deliverable purchase items, orders and grants combined.

    Item shape:
        kind            'order' | 'grant'
        id              order.id or access_grants.id
        buyer_tm_email  captured at checkout
        show_id
        city
        show_date
        detail          section/seat for orders, "{qty} ticket(s)" for grants
        state           'delivered' | 'pending'
    """
    items = []
    with db.connect() as c:
        # ---- orders: paid, held, released (i.e. anything that has a buyer) ----
        order_rows = c.execute(
            """
            SELECT o.id, o.buyer_fan_id, o.tm_email, o.state AS order_state,
                   l.show_id, l.section,
                   s.city, s.show_date,
                   t.state AS transfer_state
            FROM orders o
            JOIN listings l ON l.id = o.listing_id
            LEFT JOIN shows s ON s.id = l.show_id
            LEFT JOIN transfers t ON t.order_id = o.id
            WHERE o.state IN ('paid', 'held', 'released')
            ORDER BY o.created_at
            """
        ).fetchall()

        for row in order_rows:
            delivered = (row["transfer_state"] == "confirmed")
            items.append({
                "kind": "order",
                "id": row["id"],
                "buyer_tm_email": row["tm_email"] or "",
                "show_id": row["show_id"] or "",
                "city": row["city"] or "",
                "show_date": row["show_date"] or "",
                "detail": row["section"] or "Resale ticket",
                "state": "delivered" if delivered else "pending",
            })

        # ---- grants: all primary access grants ----
        grant_rows = c.execute(
            """
            SELECT g.id, g.fan_id, g.show_id, g.qty, g.delivered_at,
                   g.tm_email,
                   s.city, s.show_date
            FROM access_grants g
            LEFT JOIN shows s ON s.id = g.show_id
            ORDER BY g.created_at
            """
        ).fetchall()

        for row in grant_rows:
            items.append({
                "kind": "grant",
                "id": row["id"],
                "buyer_tm_email": row["tm_email"] or "",
                "show_id": row["show_id"] or "",
                "city": row["city"] or "",
                "show_date": row["show_date"] or "",
                "detail": f"{row['qty']} ticket(s)",
                "state": "delivered" if row["delivered_at"] else "pending",
            })

    return items


def mark_delivered(kind: str, item_id: str) -> dict:
    """Mark a queue item as delivered.

    For 'order': drives tm_transfer.initiate + confirm (simulated TM transfer).
    For 'grant': stamps delivered_at on access_grants.

    Returns: {kind, id, state: 'delivered'}
    Raises ValueError on unknown kind or missing/invalid item_id.
    """
    if kind == "order":
        if not item_id:
            raise ValueError("item_id required for order delivery")
        with db.connect() as c:
            row = c.execute("SELECT id FROM orders WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"order not found: {item_id}")
        try:
            tm_transfer.initiate(item_id)
        except tm_transfer.TransferError:
            # Already initiated or confirmed — idempotent, continue to confirm.
            pass
        try:
            tm_transfer.confirm(item_id)
        except tm_transfer.TransferError:
            # Already confirmed — idempotent.
            pass
        return {"kind": "order", "id": item_id, "state": "delivered"}

    elif kind == "grant":
        if not item_id:
            raise ValueError("item_id required for grant delivery")
        now = int(time.time())
        with db.connect() as c:
            row = c.execute("SELECT id FROM access_grants WHERE id=?", (item_id,)).fetchone()
            if not row:
                raise ValueError(f"grant not found: {item_id}")
            c.execute(
                "UPDATE access_grants SET delivered_at=? WHERE id=?",
                (now, item_id),
            )
        return {"kind": "grant", "id": item_id, "state": "delivered"}

    else:
        raise ValueError(f"unknown kind: {kind!r}. Must be 'order' or 'grant'.")
