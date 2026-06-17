"""Fair Game tour-admin query tests.

Seeds a little of everything — fans (verified / priority / plain), a show with
inventory, capped resale listings, and orders walked through the escrow state
machine — then asserts the dashboard rollups. Runs entirely in Stripe SIM mode
with no secrets.
"""
import importlib
import os
import tempfile
import time

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    from core.fairgame import (
        db,
        listings,
        stripe_connect,
        tm_transfer,
        orders,
        admin,
    )
    importlib.reload(db)
    importlib.reload(listings)
    importlib.reload(stripe_connect)
    importlib.reload(tm_transfer)
    importlib.reload(orders)
    importlib.reload(admin)
    db.init_db()
    return db, listings, orders, admin


def _add_fan(db, fid, status="pending", priority=0):
    now = int(time.time())
    with db.connect() as c:
        c.execute(
            "INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,"
            "priority,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (fid, f"{fid}@x.com", f"+1555{fid}", f"eh_{fid}", f"ph_{fid}",
             status, priority, now, now),
        )


def _add_show(db, sid="show_1"):
    now = int(time.time())
    with db.connect() as c:
        c.execute(
            "INSERT INTO shows(id,idx,city,venue,show_date,status,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (sid, 1, "Atlanta", "State Farm Arena", "Nov 18", "on_sale", now),
        )
        c.execute(
            "INSERT INTO inventory(id,show_id,section,qty_total,qty_available,"
            "face_price_cents,created_at) VALUES(?,?,?,?,?,?,?)",
            ("inv_1", sid, "Floor", 100, 80, 15000, now),
        )
        c.execute(
            "INSERT INTO inventory(id,show_id,section,qty_total,qty_available,"
            "face_price_cents,created_at) VALUES(?,?,?,?,?,?,?)",
            ("inv_2", sid, "Upper", 200, 150, 5500, now),
        )


def test_stats_rollups(monkeypatch):
    db, listings, orders, admin = _setup(monkeypatch)

    # 4 fans: 2 verified, 1 of those priority; 1 pending; 1 priority pending.
    _add_fan(db, "fA", status="verified", priority=1)
    _add_fan(db, "fB", status="verified", priority=0)
    _add_fan(db, "fC", status="pending", priority=0)
    _add_fan(db, "fD", status="pending", priority=1)

    _add_show(db)  # inventory_remaining = 80 + 150 = 230

    # 3 listings @ $60 face -> each fee = $5 (500c). Two get bought.
    l1 = listings.create_listing("fB", "show_1", "Floor", 6000)
    l2 = listings.create_listing("fB", "show_1", "Floor", 6000)
    l3 = listings.create_listing("fA", "show_1", "Upper", 5500)  # stays active

    # Order on l1 -> released (settled). Order on l2 -> paid (held, not settled).
    o1 = orders.create_order("fC", l1["id"])
    orders.confirm_transfer(o1["id"])  # -> released
    o2 = orders.create_order("fD", l2["id"])  # stays 'paid'

    s = admin.stats()

    assert s["fans_total"] == 4
    assert s["fans_verified"] == 2
    assert s["fans_priority"] == 2
    assert s["shows"] == 1
    assert s["inventory_remaining"] == 230
    # l1 + l2 are sold; only l3 remains active.
    assert s["listings_active"] == 1
    assert s["orders_by_state"] == {"released": 1, "paid": 1}
    # Only the released order earns Rod's $5 fee.
    assert s["gross_platform_fees_cents"] == 500


def test_stats_empty(monkeypatch):
    db, listings, orders, admin = _setup(monkeypatch)
    s = admin.stats()
    assert s["fans_total"] == 0
    assert s["inventory_remaining"] == 0
    assert s["listings_active"] == 0
    assert s["orders_by_state"] == {}
    assert s["gross_platform_fees_cents"] == 0


def test_list_fans_limit_and_order(monkeypatch):
    db, listings, orders, admin = _setup(monkeypatch)
    for i in range(5):
        _add_fan(db, f"f{i}")
        time.sleep(0.001)
    fans = admin.list_fans(limit=3)
    assert len(fans) == 3
    # Most-recent first.
    assert fans[0]["id"] == "f4"


def test_list_orders_limit(monkeypatch):
    db, listings, orders, admin = _setup(monkeypatch)
    _add_show(db)
    _add_fan(db, "buyer")
    for _ in range(4):
        l = listings.create_listing("seller", "show_1", "Floor", 6000)
        orders.create_order("buyer", l["id"])
    got = admin.list_orders(limit=2)
    assert len(got) == 2
    assert all("state" in o for o in got)


def test_flag_broker(monkeypatch):
    db, listings, orders, admin = _setup(monkeypatch)
    _add_fan(db, "scalper", status="verified")
    updated = admin.flag_broker("scalper")
    assert updated["status"] == "flagged"
    # Idempotent + persisted.
    assert admin.flag_broker("scalper")["status"] == "flagged"


def test_flag_broker_missing(monkeypatch):
    db, listings, orders, admin = _setup(monkeypatch)
    assert admin.flag_broker("nope") is None
