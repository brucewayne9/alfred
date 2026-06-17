"""Fair Game resale escrow state machine tests — all in SIM mode (no Stripe keys).

Covers the full happy path (paid -> released) and the refund path
(paid -> refunded, listing reopened), plus invalid transitions and validation.
"""
import importlib
import os
import tempfile

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    # Force the Stripe simulator: no key, sim on. Tests run with zero secrets.
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    from core.fairgame import (
        db,
        listings,
        stripe_connect,
        tm_transfer,
        orders,
    )
    importlib.reload(db)
    importlib.reload(listings)
    importlib.reload(stripe_connect)
    importlib.reload(tm_transfer)
    importlib.reload(orders)
    db.init_db()
    return listings, stripe_connect, tm_transfer, orders


def _listing(listings, face=6000):
    return listings.create_listing("seller_1", "show_1", "Floor", face)


# --------------------------------------------------------------------------- #
# create_order
# --------------------------------------------------------------------------- #

def test_create_order_holds_funds_and_marks_sold(monkeypatch):
    listings, stripe_connect, tm_transfer, orders = _setup(monkeypatch)
    lst = _listing(listings, face=6000)

    order = orders.create_order("buyer_1", lst["id"])

    assert order["id"].startswith("ord_")
    assert order["state"] == "paid"
    assert order["buyer_fan_id"] == "buyer_1"
    assert order["listing_id"] == lst["id"]
    # Buyer always pays the LOCKED cap (face + $15), never a free amount.
    assert order["amount_cents"] == 7500
    # Funds are held in escrow (a payment ref exists).
    assert order["payment_ref"]
    # The seat is now off the market.
    assert listings.get_listing(lst["id"])["status"] == "sold"


def test_create_order_unknown_listing(monkeypatch):
    _, _, _, orders = _setup(monkeypatch)
    with pytest.raises(orders.OrderError):
        orders.create_order("buyer_1", "lst_nope")


def test_create_order_requires_buyer(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    with pytest.raises(orders.OrderError):
        orders.create_order("", lst["id"])


def test_cannot_buy_same_listing_twice(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    orders.create_order("buyer_1", lst["id"])
    # Second buyer can't claim a sold seat.
    with pytest.raises(orders.OrderError):
        orders.create_order("buyer_2", lst["id"])


def test_atomic_claim_no_held_payment_for_loser(monkeypatch):
    """The loser of the seat race must never get funds held (escrow integrity).

    create_order claims the listing with a conditional UPDATE BEFORE holding any
    money, so a second buyer of an already-sold seat is rejected and no order /
    held payment is ever created for them.
    """
    listings, stripe_connect, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    o1 = orders.create_order("buyer_1", lst["id"])
    assert o1["state"] == "paid"
    with pytest.raises(orders.OrderError):
        orders.create_order("buyer_2", lst["id"])
    # Exactly one order/held payment exists for this seat — no double-hold.
    from core.fairgame import db
    with db.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE listing_id=?", (lst["id"],)
        ).fetchone()["n"]
    assert n == 1


def test_held_order_can_be_driven_to_terminal(monkeypatch):
    """A crash that left an order stuck in legacy 'held' must still be resumable.

    Simulates a process death mid-transition (state forced to 'held' with no
    stripe advance) and asserts confirm_transfer can still drive it to a terminal
    state — the buyer's funds are never frozen forever.
    """
    listings, stripe_connect, _, orders = _setup(monkeypatch)
    from core.fairgame import db
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    # Force the unrecoverable-looking intermediate state a crash could leave.
    with db.connect() as conn:
        conn.execute("UPDATE orders SET state='held' WHERE id=?", (order["id"],))
    # Recovery: confirm still works (held is accepted as resumable).
    released = orders.confirm_transfer(order["id"])
    assert released["state"] == "released"


def test_held_order_can_be_refunded(monkeypatch):
    """Same recovery guarantee on the refund side."""
    listings, stripe_connect, _, orders = _setup(monkeypatch)
    from core.fairgame import db
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    with db.connect() as conn:
        conn.execute("UPDATE orders SET state='held' WHERE id=?", (order["id"],))
    refunded = orders.fail_transfer(order["id"])
    assert refunded["state"] == "refunded"


# --------------------------------------------------------------------------- #
# Happy path: paid -> released
# --------------------------------------------------------------------------- #

def test_full_happy_path_paid_to_released(monkeypatch):
    listings, stripe_connect, tm_transfer, orders = _setup(monkeypatch)
    lst = _listing(listings, face=6000)

    order = orders.create_order("buyer_1", lst["id"])
    assert order["state"] == "paid"

    released = orders.confirm_transfer(order["id"])

    assert released["state"] == "released"
    # The TM transfer ref is recorded on the order.
    assert released["transfer_ref"]
    assert released["transfer_ref"].startswith("tx_")
    # TM transfer is confirmed.
    assert tm_transfer.status(order["id"])["state"] == "confirmed"
    # Stripe escrow released the seller's funds.
    assert stripe_connect.get_order(order["id"])["state"] == "released"
    # Sold seat stays sold.
    assert listings.get_listing(lst["id"])["status"] == "sold"


def test_confirm_transfer_idempotent(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    first = orders.confirm_transfer(order["id"])
    second = orders.confirm_transfer(order["id"])
    assert first["state"] == second["state"] == "released"
    assert first["transfer_ref"] == second["transfer_ref"]


# --------------------------------------------------------------------------- #
# Refund path: paid -> refunded (listing reopened)
# --------------------------------------------------------------------------- #

def test_refund_path_paid_to_refunded_and_reopens_listing(monkeypatch):
    listings, stripe_connect, tm_transfer, orders = _setup(monkeypatch)
    lst = _listing(listings, face=6000)

    order = orders.create_order("buyer_1", lst["id"])
    assert listings.get_listing(lst["id"])["status"] == "sold"

    refunded = orders.fail_transfer(order["id"])

    assert refunded["state"] == "refunded"
    # Buyer's held funds are refunded in Stripe escrow.
    assert stripe_connect.get_order(order["id"])["state"] == "refunded"
    # The seat is back on the market for someone else.
    assert listings.get_listing(lst["id"])["status"] == "active"


def test_fail_transfer_idempotent(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    first = orders.fail_transfer(order["id"])
    second = orders.fail_transfer(order["id"])
    assert first["state"] == second["state"] == "refunded"


def test_reopened_listing_can_be_rebought(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    o1 = orders.create_order("buyer_1", lst["id"])
    orders.fail_transfer(o1["id"])
    # A new buyer can now take the reopened seat.
    o2 = orders.create_order("buyer_2", lst["id"])
    assert o2["state"] == "paid"
    assert o2["id"] != o1["id"]


# --------------------------------------------------------------------------- #
# Invalid transitions
# --------------------------------------------------------------------------- #

def test_cannot_refund_a_released_order(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    orders.confirm_transfer(order["id"])
    with pytest.raises(orders.OrderError):
        orders.fail_transfer(order["id"])


def test_cannot_confirm_a_refunded_order(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    orders.fail_transfer(order["id"])
    with pytest.raises(orders.OrderError):
        orders.confirm_transfer(order["id"])


def test_confirm_unknown_order(monkeypatch):
    _, _, _, orders = _setup(monkeypatch)
    with pytest.raises(orders.OrderError):
        orders.confirm_transfer("ord_nope")


def test_fail_unknown_order(monkeypatch):
    _, _, _, orders = _setup(monkeypatch)
    with pytest.raises(orders.OrderError):
        orders.fail_transfer("ord_nope")


# --------------------------------------------------------------------------- #
# get_order
# --------------------------------------------------------------------------- #

def test_get_order_roundtrip(monkeypatch):
    listings, _, _, orders = _setup(monkeypatch)
    lst = _listing(listings)
    order = orders.create_order("buyer_1", lst["id"])
    got = orders.get_order(order["id"])
    assert got["id"] == order["id"]
    assert got["state"] == "paid"


def test_get_order_missing(monkeypatch):
    _, _, _, orders = _setup(monkeypatch)
    assert orders.get_order("ord_nope") is None
