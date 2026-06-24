import importlib
import os
import tempfile

import pytest


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def seeded_show_and_fan(monkeypatch):
    """Verified fan + show_1 with open wave + inventory. Returns (fan_id, show_id)."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, identity, events, access
    importlib.reload(db)
    importlib.reload(identity)
    importlib.reload(events)
    importlib.reload(access)
    db.init_db()
    events.seed_shows()
    events.add_inventory("show_1", "Floor", 10, 15000)
    access.create_wave("show_1", "Presale", 0, 3_000_000_000, max_qty_per_fan=4)
    fan = identity.upsert_fan("fan@tm.com", "+15550001", None, None)
    with db.connect() as c:
        c.execute("UPDATE fans SET status='verified' WHERE id=?", (fan["id"],))
    return fan["id"], "show_1"


@pytest.fixture()
def seeded_listing_and_buyer(monkeypatch):
    """Active listing + buyer fan id. Returns (buyer_id, listing_id)."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    from core.fairgame import db, listings, stripe_connect, tm_transfer, orders
    importlib.reload(db)
    importlib.reload(listings)
    importlib.reload(stripe_connect)
    importlib.reload(tm_transfer)
    importlib.reload(orders)
    db.init_db()
    lst = listings.create_listing("seller_1", "show_1", "Floor", 6000)
    return "buyer_1", lst["id"]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_valid_tm_email_helper():
    from core.api.fairgame import _valid_tm_email
    assert _valid_tm_email("fan@example.com")
    assert not _valid_tm_email("nope")
    assert not _valid_tm_email("a@b")
    assert not _valid_tm_email("")
    assert not _valid_tm_email(None)


def test_access_grant_stores_tm_email_and_ack(seeded_show_and_fan):
    from core.fairgame import access
    fan_id, show_id = seeded_show_and_fan
    g = access.grant_access(fan_id, show_id, 1, tm_email="fan@tm.com", final_sale_ack=True)
    assert g["tm_email"] == "fan@tm.com"
    assert g["final_sale_ack"] == 1   # stored as int


def test_order_stores_tm_email_and_ack(seeded_listing_and_buyer):
    from core.fairgame import orders
    buyer_id, listing_id = seeded_listing_and_buyer
    o = orders.create_order(buyer_id, listing_id, tm_email="buy@tm.com", final_sale_ack=True)
    assert o["tm_email"] == "buy@tm.com"
    assert o["final_sale_ack"] == 1


def test_columns_are_idempotent():
    from core.fairgame import db
    with db.connect() as c:
        db.ensure_checkout_columns(c)
        db.ensure_checkout_columns(c)
