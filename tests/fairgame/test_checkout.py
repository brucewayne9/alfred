import importlib
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# HTTP-layer client helper (mirrors test_api._client)
# --------------------------------------------------------------------------- #

def _http_client(monkeypatch) -> TestClient:
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    monkeypatch.setenv("FAIRGAME_DEV_ECHO", "1")
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.setenv("FAIRGAME_ADMIN_TOKEN", "test-admin")
    import core.fairgame.db as db
    import core.fairgame.waitlist as waitlist
    import core.fairgame.identity as identity
    import core.fairgame.verify as verify
    import core.fairgame.sessions as sessions
    import core.fairgame.events as events
    import core.fairgame.access as access
    import core.fairgame.listings as listings
    import core.fairgame.stripe_connect as stripe_connect
    import core.fairgame.tm_transfer as tm_transfer
    import core.fairgame.orders as orders
    import core.fairgame.admin as admin
    for m in (db, waitlist, identity, verify, sessions, events, access,
              listings, stripe_connect, tm_transfer, orders, admin):
        importlib.reload(m)
    import core.api.fairgame as fg
    importlib.reload(fg)
    return TestClient(fg.app)


def _verify_fan_http(c: TestClient, email="fan@x.com", phone="+15550001"):
    """Register → verify SMS → verify email; return (fan_id, bearer_token)."""
    r = c.post("/fairgame/api/register",
               json={"email": email, "phone": phone, "device_fp": "fp"})
    assert r.status_code == 200, r.text
    fid = r.json()["fan_id"]
    sms = r.json()["dev_code"]
    r = c.post("/fairgame/api/verify", json={"fan_id": fid, "code": sms})
    assert r.status_code == 200, r.text
    email_code = r.json()["dev_code"]
    r = c.post("/fairgame/api/verify-email",
               json={"fan_id": fid, "code": email_code})
    assert r.status_code == 200, r.text
    return fid, r.json()["token"]


def _create_listing_http(c: TestClient, seller_tok: str, show_id="show_1") -> str:
    """Create an active listing for seller; return listing_id."""
    r = c.post("/fairgame/api/listings",
               headers={"Authorization": f"Bearer {seller_tok}"},
               json={"show_id": show_id, "section": "Lower",
                     "face_price_cents": 6000})
    assert r.status_code == 200, r.text
    return r.json()["listing"]["id"]


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


# --------------------------------------------------------------------------- #
# HTTP-level checkout gate rejection tests
# These prove the _require_checkout() call is wired into the real API handlers.
# If someone removed that call, the suite would catch it here (not just via the
# _valid_tm_email unit test above which only tests the helper in isolation).
# --------------------------------------------------------------------------- #

def test_access_rejects_missing_tm_email(monkeypatch):
    """POST /access with a valid fan + no tm_email → 400."""
    c = _http_client(monkeypatch)
    _, tok = _verify_fan_http(c)
    r = c.post("/fairgame/api/access",
               headers={"Authorization": f"Bearer {tok}"},
               json={"show_id": "show_1", "qty": 1, "final_sale_ack": True})
    assert r.status_code == 400, r.text


def test_access_rejects_invalid_tm_email(monkeypatch):
    """POST /access with a valid fan + malformed email → 400."""
    c = _http_client(monkeypatch)
    _, tok = _verify_fan_http(c)
    r = c.post("/fairgame/api/access",
               headers={"Authorization": f"Bearer {tok}"},
               json={"show_id": "show_1", "qty": 1,
                     "tm_email": "not-an-email", "final_sale_ack": True})
    assert r.status_code == 400, r.text


def test_access_rejects_missing_final_sale_ack(monkeypatch):
    """POST /access with valid email but no final_sale_ack → 400."""
    c = _http_client(monkeypatch)
    _, tok = _verify_fan_http(c)
    r = c.post("/fairgame/api/access",
               headers={"Authorization": f"Bearer {tok}"},
               json={"show_id": "show_1", "qty": 1,
                     "tm_email": "fan@x.com"})
    assert r.status_code == 400, r.text


def test_access_rejects_false_final_sale_ack(monkeypatch):
    """POST /access with valid email but final_sale_ack=false → 400."""
    c = _http_client(monkeypatch)
    _, tok = _verify_fan_http(c)
    r = c.post("/fairgame/api/access",
               headers={"Authorization": f"Bearer {tok}"},
               json={"show_id": "show_1", "qty": 1,
                     "tm_email": "fan@x.com", "final_sale_ack": False})
    assert r.status_code == 400, r.text


def test_buy_rejects_missing_tm_email(monkeypatch):
    """POST /buy with valid fan + listing but no tm_email → 400."""
    c = _http_client(monkeypatch)
    _, seller_tok = _verify_fan_http(c, "seller@x.com", "+15550100")
    _, buyer_tok = _verify_fan_http(c, "buyer@x.com", "+15550200")
    listing_id = _create_listing_http(c, seller_tok)
    r = c.post("/fairgame/api/buy",
               headers={"Authorization": f"Bearer {buyer_tok}"},
               json={"listing_id": listing_id, "final_sale_ack": True})
    assert r.status_code == 400, r.text


def test_buy_rejects_false_final_sale_ack(monkeypatch):
    """POST /buy with valid email but final_sale_ack=false → 400."""
    c = _http_client(monkeypatch)
    _, seller_tok = _verify_fan_http(c, "seller2@x.com", "+15550300")
    _, buyer_tok = _verify_fan_http(c, "buyer2@x.com", "+15550400")
    listing_id = _create_listing_http(c, seller_tok)
    r = c.post("/fairgame/api/buy",
               headers={"Authorization": f"Bearer {buyer_tok}"},
               json={"listing_id": listing_id,
                     "tm_email": "buyer2@x.com", "final_sale_ack": False})
    assert r.status_code == 400, r.text


def test_access_happy_path_control(monkeypatch):
    """POST /access with valid email + ack=true → 200 (proves 400s are gate, not auth)."""
    c = _http_client(monkeypatch)
    _, tok = _verify_fan_http(c)
    r = c.post("/fairgame/api/access",
               headers={"Authorization": f"Bearer {tok}"},
               json={"show_id": "show_1", "qty": 1,
                     "tm_email": "fan@x.com", "final_sale_ack": True})
    assert r.status_code == 200, r.text
    assert r.json()["grant"]["qty"] == 1
