"""Delivery operator queue tests.

Seeds orders (via test_orders.py pattern) and access grants (via test_access.py
pattern) to verify:
  - a paid order appears in the queue as pending
  - mark_delivered('order', id) drives the TM transfer and flips state
  - an access grant appears in the queue as pending
  - mark_delivered('grant', id) stamps delivered_at and flips state
  - delivered_at migration is idempotent (calling init_db twice is safe)
  - unknown kind raises ValueError
"""
import importlib
import os
import tempfile
import time

import pytest


# ---- shared setup --------------------------------------------------------- #

def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)

    from core.fairgame import (
        db, identity, events, access, listings,
        stripe_connect, tm_transfer, orders, delivery,
    )
    for mod in (db, identity, events, access, listings,
                stripe_connect, tm_transfer, orders, delivery):
        importlib.reload(mod)

    db.init_db()
    # Seed a minimal show directly (tour JSON file is not in repo)
    with db.connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO shows(id, idx, city, venue, show_date, status, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            ("show_1", 1, "Atlanta", "State Farm Arena", "2026-09-12", "on_sale", int(time.time())),
        )
    return db, identity, events, access, listings, orders, tm_transfer, delivery


def _verified_fan(db, identity, email, phone, priority=0):
    fan = identity.upsert_fan(email, phone, None, None)
    with db.connect() as c:
        c.execute(
            "UPDATE fans SET status='verified', priority=? WHERE id=?",
            (priority, fan["id"]),
        )
    return identity.get_fan(fan["id"])


def _make_order(db, listings, orders, identity, buyer_email="buyer@test.com"):
    """Seed: show_1 listing -> paid order with a tm_email."""
    lst = listings.create_listing("seller_1", "show_1", "Floor", 6000)
    order = orders.create_order("buyer_1", lst["id"])
    # stamp tm_email directly (checkout path does this; here we seed it)
    with db.connect() as c:
        c.execute(
            "UPDATE orders SET tm_email=? WHERE id=?",
            (buyer_email, order["id"]),
        )
    return orders.get_order(order["id"])


def _make_grant(db, identity, events, access, tm_email="fan@test.com", qty=2):
    """Seed: verified fan -> access grant with a tm_email."""
    events.add_inventory("show_1", "Floor", 20, 15000)
    access.create_wave("show_1", "Presale", 0, 9_999_999_999, max_qty_per_fan=4)
    fan = _verified_fan(db, identity, "fan@test.com", "+15550001")
    grant = access.grant_access(fan["id"], "show_1", qty, now=1)
    # stamp tm_email directly
    with db.connect() as c:
        c.execute(
            "UPDATE access_grants SET tm_email=? WHERE id=?",
            (tm_email, grant["id"]),
        )
    return grant


# ---- migration idempotency ------------------------------------------------ #

def test_delivered_at_migration_idempotent(monkeypatch):
    """Calling init_db twice (which runs ensure_delivery_columns twice) must not raise."""
    db, *_ = _setup(monkeypatch)
    db.init_db()  # second call — should be a no-op
    # Verify column exists
    with db.connect() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(access_grants)").fetchall()}
    assert "delivered_at" in cols


# ---- order queue flow ----------------------------------------------------- #

def test_paid_order_appears_in_queue_as_pending(monkeypatch):
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    order = _make_order(db, listings, orders, identity)

    q = delivery.queue()
    order_items = [i for i in q if i["kind"] == "order"]
    assert len(order_items) == 1
    item = order_items[0]
    assert item["id"] == order["id"]
    assert item["state"] == "pending"
    assert item["buyer_tm_email"] == "buyer@test.com"
    assert item["show_id"] == "show_1"
    assert item["detail"] == "Floor"


def test_mark_delivered_order_flips_to_delivered(monkeypatch):
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    order = _make_order(db, listings, orders, identity)

    result = delivery.mark_delivered("order", order["id"])

    assert result["kind"] == "order"
    assert result["id"] == order["id"]
    assert result["state"] == "delivered"

    # Queue should now show delivered
    q = delivery.queue()
    item = next(i for i in q if i["kind"] == "order" and i["id"] == order["id"])
    assert item["state"] == "delivered"

    # TM transfer was confirmed
    transfer = tm_transfer.status(order["id"])
    assert transfer is not None
    assert transfer["state"] == "confirmed"


def test_mark_delivered_order_idempotent(monkeypatch):
    """Calling mark_delivered twice on the same order must not raise."""
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    order = _make_order(db, listings, orders, identity)

    delivery.mark_delivered("order", order["id"])
    result = delivery.mark_delivered("order", order["id"])
    assert result["state"] == "delivered"


def test_mark_delivered_order_not_found_raises(monkeypatch):
    db, *_, delivery = _setup(monkeypatch)
    with pytest.raises(ValueError, match="order not found"):
        delivery.mark_delivered("order", "ord_nope")


# ---- grant queue flow ----------------------------------------------------- #

def test_grant_appears_in_queue_as_pending(monkeypatch):
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    grant = _make_grant(db, identity, events, access, tm_email="fanx@test.com", qty=3)

    q = delivery.queue()
    grant_items = [i for i in q if i["kind"] == "grant"]
    assert len(grant_items) == 1
    item = grant_items[0]
    assert item["id"] == grant["id"]
    assert item["state"] == "pending"
    assert item["buyer_tm_email"] == "fanx@test.com"
    assert item["detail"] == "3 ticket(s)"


def test_mark_delivered_grant_flips_to_delivered(monkeypatch):
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    grant = _make_grant(db, identity, events, access)

    result = delivery.mark_delivered("grant", grant["id"])

    assert result["kind"] == "grant"
    assert result["id"] == grant["id"]
    assert result["state"] == "delivered"

    # Queue shows delivered
    q = delivery.queue()
    item = next(i for i in q if i["kind"] == "grant" and i["id"] == grant["id"])
    assert item["state"] == "delivered"

    # delivered_at is stamped in DB
    with db.connect() as c:
        row = c.execute(
            "SELECT delivered_at FROM access_grants WHERE id=?", (grant["id"],)
        ).fetchone()
    assert row["delivered_at"] is not None
    assert row["delivered_at"] > 0


def test_mark_delivered_grant_idempotent(monkeypatch):
    """Calling mark_delivered twice on the same grant is fine (overwrites timestamp)."""
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    grant = _make_grant(db, identity, events, access)

    delivery.mark_delivered("grant", grant["id"])
    result = delivery.mark_delivered("grant", grant["id"])
    assert result["state"] == "delivered"


def test_mark_delivered_grant_not_found_raises(monkeypatch):
    db, *_, delivery = _setup(monkeypatch)
    with pytest.raises(ValueError, match="grant not found"):
        delivery.mark_delivered("grant", "grnt_nope")


# ---- bad kind ------------------------------------------------------------- #

def test_mark_delivered_unknown_kind_raises(monkeypatch):
    db, *_, delivery = _setup(monkeypatch)
    with pytest.raises(ValueError, match="unknown kind"):
        delivery.mark_delivered("ticket", "some_id")


# ---- mixed queue ---------------------------------------------------------- #

def test_queue_contains_both_orders_and_grants(monkeypatch):
    db, identity, events, access, listings, orders, tm_transfer, delivery = _setup(monkeypatch)
    order = _make_order(db, listings, orders, identity)
    grant = _make_grant(db, identity, events, access)

    q = delivery.queue()
    kinds = {i["kind"] for i in q}
    assert "order" in kinds
    assert "grant" in kinds
    assert len(q) == 2
