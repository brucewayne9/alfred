"""Fair Game public API — end-to-end tests against the FastAPI sub-app.

Covers the full fan + resale journey through the HTTP surface:
  register -> verify (sms) -> verify-email -> session
  -> list shows -> create listing -> buy (held) -> confirm (released)
plus auth gating on /me and admin token gating.

FAIRGAME_DEV_ECHO=1 echoes the verification codes in the response so the flow
completes with no SMS/email provider; FAIRGAME_STRIPE_SIM=1 keeps escrow off the
network.
"""
import importlib
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    monkeypatch.setenv("FAIRGAME_DEV_ECHO", "1")
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.setenv("FAIRGAME_ADMIN_TOKEN", "test-admin")
    # Reload the module graph so every module binds to the fresh DB path.
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


def _verify_fan(c, email="fan@x.com", phone="+15550001"):
    """Run register -> verify -> verify-email; return (fan_id, bearer_token)."""
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


# --------------------------------------------------------------------------- #
# Identity / verification / session
# --------------------------------------------------------------------------- #

def test_full_identity_flow_and_me(monkeypatch):
    c = _client(monkeypatch)
    fid, tok = _verify_fan(c)
    r = c.get("/fairgame/api/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["fan"]["email"] == "fan@x.com"
    assert r.json()["fan"]["status"] == "verified"


def test_me_requires_auth(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/fairgame/api/me").status_code == 401
    assert c.get("/fairgame/api/me",
                 headers={"Authorization": "Bearer nope"}).status_code == 401


def test_register_requires_email_and_phone(monkeypatch):
    c = _client(monkeypatch)
    assert c.post("/fairgame/api/register", json={"email": "a@x.com"}).status_code == 400


def test_wrong_sms_code_rejected(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/fairgame/api/register",
               json={"email": "a@x.com", "phone": "+15550009"})
    fid = r.json()["fan_id"]
    assert c.post("/fairgame/api/verify",
                  json={"fan_id": fid, "code": "000000"}).status_code == 400


# --------------------------------------------------------------------------- #
# Shows
# --------------------------------------------------------------------------- #

def test_shows_seeded_with_inventory(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/fairgame/api/shows")
    assert r.status_code == 200
    shows = r.json()["shows"]
    assert len(shows) == 25
    first = shows[0]
    assert first["remaining"] > 0
    assert first["min_price_cents"] is not None
    detail = c.get(f"/fairgame/api/shows/{first['id']}")
    assert detail.status_code == 200
    assert len(detail.json()["inventory"]) == 3


# --------------------------------------------------------------------------- #
# Primary access (capped buy)
# --------------------------------------------------------------------------- #

def test_access_grant_primary(monkeypatch):
    c = _client(monkeypatch)
    fid, tok = _verify_fan(c)
    r = c.post("/fairgame/api/access",
               json={"fan_id": fid, "show_id": "show_1", "qty": 2})
    assert r.status_code == 200, r.text
    assert r.json()["grant"]["qty"] == 2


def test_access_over_cap_rejected(monkeypatch):
    c = _client(monkeypatch)
    fid, tok = _verify_fan(c)
    r = c.post("/fairgame/api/access",
               json={"fan_id": fid, "show_id": "show_1", "qty": 99})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Resale escrow — the headline end-to-end
# --------------------------------------------------------------------------- #

def test_resale_full_escrow_release(monkeypatch):
    c = _client(monkeypatch)
    _, seller_tok = _verify_fan(c, "seller@x.com", "+15550100")
    _, buyer_tok = _verify_fan(c, "buyer@x.com", "+15550200")

    # Seller lists a $60 (6000c) seat -> buyer pays face + $15 = 7500c.
    r = c.post("/fairgame/api/listings",
               headers={"Authorization": f"Bearer {seller_tok}"},
               json={"show_id": "show_1", "section": "Lower",
                     "face_price_cents": 6000})
    assert r.status_code == 200, r.text
    lst = r.json()["listing"]
    assert lst["buyer_total_cents"] == 7500
    assert lst["seller_proceeds_cents"] == 7000
    assert lst["platform_fee_cents"] == 500
    listing_id = lst["id"]

    # Exchange shows the active listing.
    ex = c.get("/fairgame/api/exchange/show_1")
    assert any(l["id"] == listing_id for l in ex.json()["listings"])

    # Buyer buys -> funds held.
    r = c.post("/fairgame/api/buy",
               headers={"Authorization": f"Bearer {buyer_tok}"},
               json={"listing_id": listing_id})
    assert r.status_code == 200, r.text
    order = r.json()["order"]
    assert order["state"] == "paid"
    assert order["amount_cents"] == 7500
    order_id = order["id"]

    # Order state reflects held/paid.
    r = c.get(f"/fairgame/api/orders/{order_id}")
    assert r.json()["order"]["state"] == "paid"

    # Confirm transfer (sim TM receipt) -> released.
    r = c.post(f"/fairgame/api/orders/{order_id}/confirm")
    assert r.status_code == 200, r.text
    assert r.json()["order"]["state"] == "released"

    # Final state is released.
    r = c.get(f"/fairgame/api/orders/{order_id}")
    assert r.json()["order"]["state"] == "released"

    # Listing is now sold (not active on the exchange).
    ex = c.get("/fairgame/api/exchange/show_1")
    assert not any(l["id"] == listing_id for l in ex.json()["listings"])


def test_resale_failed_transfer_refunds(monkeypatch):
    c = _client(monkeypatch)
    _, seller_tok = _verify_fan(c, "seller2@x.com", "+15550300")
    _, buyer_tok = _verify_fan(c, "buyer2@x.com", "+15550400")
    r = c.post("/fairgame/api/listings",
               headers={"Authorization": f"Bearer {seller_tok}"},
               json={"show_id": "show_2", "section": "Floor",
                     "face_price_cents": 15000})
    listing_id = r.json()["listing"]["id"]
    r = c.post("/fairgame/api/buy",
               headers={"Authorization": f"Bearer {buyer_tok}"},
               json={"listing_id": listing_id})
    order_id = r.json()["order"]["id"]
    r = c.post(f"/fairgame/api/orders/{order_id}/fail")
    assert r.status_code == 200, r.text
    assert r.json()["order"]["state"] == "refunded"
    # Listing reopened.
    ex = c.get("/fairgame/api/exchange/show_2")
    assert any(l["id"] == listing_id for l in ex.json()["listings"])


def test_listing_requires_auth(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/fairgame/api/listings",
               json={"show_id": "show_1", "section": "Lower",
                     "face_price_cents": 6000})
    assert r.status_code == 401


def test_buy_requires_auth(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/fairgame/api/buy", json={"listing_id": "lst_x"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #

def test_admin_requires_token(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/fairgame/api/admin/stats").status_code == 401
    assert c.get("/fairgame/api/admin/stats",
                 headers={"X-Fairgame-Admin": "wrong"}).status_code == 401


def test_admin_stats_reflects_activity(monkeypatch):
    c = _client(monkeypatch)
    _, seller_tok = _verify_fan(c, "s3@x.com", "+15550500")
    _, buyer_tok = _verify_fan(c, "b3@x.com", "+15550600")
    r = c.post("/fairgame/api/listings",
               headers={"Authorization": f"Bearer {seller_tok}"},
               json={"show_id": "show_3", "section": "Upper",
                     "face_price_cents": 5500})
    listing_id = r.json()["listing"]["id"]
    r = c.post("/fairgame/api/buy",
               headers={"Authorization": f"Bearer {buyer_tok}"},
               json={"listing_id": listing_id})
    order_id = r.json()["order"]["id"]
    c.post(f"/fairgame/api/orders/{order_id}/confirm")

    hdr = {"X-Fairgame-Admin": "test-admin"}
    stats = c.get("/fairgame/api/admin/stats", headers=hdr).json()
    assert stats["fans_verified"] == 2
    assert stats["shows"] == 25
    assert stats["gross_platform_fees_cents"] == 500  # Rod's $5 on the settled resale
    assert stats["orders_by_state"].get("released") == 1

    fans = c.get("/fairgame/api/admin/fans", headers=hdr).json()["fans"]
    assert len(fans) == 2
    ords = c.get("/fairgame/api/admin/orders", headers=hdr).json()["orders"]
    assert len(ords) == 1
