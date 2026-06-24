"""Tests for inventory CMS endpoints (admin price/qty edit).

Covers:
  - events.update_inventory() unit tests
  - PATCH /fairgame/api/admin/inventory/{inv_id}
  - GET  /fairgame/api/admin/shows/{show_id}/inventory
  - POST /fairgame/api/admin/shows/{show_id}/inventory
"""
import importlib
import os
import tempfile
import time

import pytest


# ---------------------------------------------------------------------------
# Test setup helpers (mirrors test_admin.py pattern)
# ---------------------------------------------------------------------------

def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.setenv("FAIRGAME_ADMIN_TOKEN", "TestAdminToken2026")
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    from core.fairgame import db, events
    importlib.reload(db)
    importlib.reload(events)
    db.init_db()
    return db, events


def _add_show_with_inventory(db_mod, events_mod, sid="show_1"):
    now = int(time.time())
    with db_mod.connect() as c:
        c.execute(
            "INSERT INTO shows(id,idx,city,venue,show_date,status,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (sid, 1, "Atlanta", "State Farm Arena", "Nov 18", "on_sale", now),
        )
        c.execute(
            "INSERT INTO inventory(id,show_id,section,qty_total,qty_available,"
            "face_price_cents,created_at) VALUES(?,?,?,?,?,?,?)",
            ("inv_test_1", sid, "Floor", 100, 80, 15000, now),
        )
    return "inv_test_1"


# ---------------------------------------------------------------------------
# events.update_inventory() unit tests
# ---------------------------------------------------------------------------

def test_update_inventory_changes_price_and_qty(monkeypatch):
    db_mod, events_mod = _setup(monkeypatch)
    inv_id = _add_show_with_inventory(db_mod, events_mod)
    row = events_mod.update_inventory(inv_id, face_price_cents=12345, qty_available=7)
    assert row is not None
    assert row["face_price_cents"] == 12345
    assert row["qty_available"] == 7


def test_update_inventory_partial_only_touches_given(monkeypatch):
    db_mod, events_mod = _setup(monkeypatch)
    inv_id = _add_show_with_inventory(db_mod, events_mod)
    # Only update qty_total; price should remain 15000
    row = events_mod.update_inventory(inv_id, qty_total=999)
    assert row is not None
    assert row["qty_total"] == 999
    assert row["face_price_cents"] == 15000  # unchanged from seed


def test_update_inventory_missing_returns_none(monkeypatch):
    db_mod, events_mod = _setup(monkeypatch)
    result = events_mod.update_inventory("inv_does_not_exist", face_price_cents=100)
    assert result is None


def test_update_inventory_no_fields_returns_existing(monkeypatch):
    db_mod, events_mod = _setup(monkeypatch)
    inv_id = _add_show_with_inventory(db_mod, events_mod)
    row = events_mod.update_inventory(inv_id)  # no fields — should return the row unchanged
    assert row is not None
    assert row["id"] == inv_id


# ---------------------------------------------------------------------------
# Admin endpoint tests via TestClient
# ---------------------------------------------------------------------------

def _setup_api(monkeypatch):
    """Reload the API module (which bootstraps shows+inventory), then return client + a known inv_id."""
    db_mod, events_mod = _setup(monkeypatch)
    import core.api.fairgame as fg_mod
    importlib.reload(fg_mod)
    from fastapi.testclient import TestClient
    client = TestClient(fg_mod.app, raise_server_exceptions=True)
    # Bootstrap already seeded show_1 and its standard inventory; grab the first inventory row id.
    inv = events_mod.get_inventory("show_1")
    assert inv, "bootstrap must seed show_1 inventory"
    inv_id = inv[0]["id"]
    return client, inv_id


ADMIN_HDR = {"X-Fairgame-Admin": "TestAdminToken2026"}


def test_patch_inventory_updates_price(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        f"/fairgame/api/admin/inventory/{inv_id}",
        json={"face_price_cents": 8888},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["inventory"]["face_price_cents"] == 8888
    assert data["inventory"]["qty_available"] == 200  # unchanged from seeded Floor section


def test_patch_inventory_updates_qty(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        f"/fairgame/api/admin/inventory/{inv_id}",
        json={"qty_available": 55, "qty_total": 200},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 200, r.text
    inv = r.json()["inventory"]
    assert inv["qty_available"] == 55
    assert inv["qty_total"] == 200


def test_patch_inventory_without_admin_header_returns_401(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        f"/fairgame/api/admin/inventory/{inv_id}",
        json={"face_price_cents": 5000},
    )
    assert r.status_code == 401


def test_patch_inventory_wrong_token_returns_401(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        f"/fairgame/api/admin/inventory/{inv_id}",
        json={"face_price_cents": 5000},
        headers={"X-Fairgame-Admin": "wrongtoken"},
    )
    assert r.status_code == 401


def test_patch_inventory_negative_price_returns_400(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        f"/fairgame/api/admin/inventory/{inv_id}",
        json={"face_price_cents": -1},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 400


def test_patch_inventory_non_int_returns_400(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        f"/fairgame/api/admin/inventory/{inv_id}",
        json={"face_price_cents": 99.99},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 400


def test_patch_inventory_missing_row_returns_404(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.patch(
        "/fairgame/api/admin/inventory/inv_nope",
        json={"face_price_cents": 5000},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 404


def test_get_admin_show_inventory(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.get("/fairgame/api/admin/shows/show_1/inventory", headers=ADMIN_HDR)
    assert r.status_code == 200, r.text
    inv = r.json()["inventory"]
    assert isinstance(inv, list)
    assert len(inv) >= 1  # bootstrap seeds 3 sections (Floor/Lower/Upper)
    ids = [row["id"] for row in inv]
    assert inv_id in ids
    sections = [row["section"] for row in inv]
    assert "Floor" in sections


def test_get_admin_show_inventory_no_auth_returns_401(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.get("/fairgame/api/admin/shows/show_1/inventory")
    assert r.status_code == 401


def test_get_admin_show_inventory_missing_show_returns_404(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.get("/fairgame/api/admin/shows/show_nope/inventory", headers=ADMIN_HDR)
    assert r.status_code == 404


def test_post_admin_add_inventory(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.post(
        "/fairgame/api/admin/shows/show_1/inventory",
        json={"section": "VIP", "qty": 50, "face_price_cents": 25000},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 200, r.text
    inv = r.json()["inventory"]
    assert inv["section"] == "VIP"
    assert inv["qty_total"] == 50
    assert inv["qty_available"] == 50
    assert inv["face_price_cents"] == 25000


def test_post_admin_add_inventory_no_auth_returns_401(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.post(
        "/fairgame/api/admin/shows/show_1/inventory",
        json={"section": "VIP", "qty": 10, "face_price_cents": 5000},
    )
    assert r.status_code == 401


def test_post_admin_add_inventory_negative_price_returns_400(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.post(
        "/fairgame/api/admin/shows/show_1/inventory",
        json={"section": "VIP", "qty": 10, "face_price_cents": -100},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 400


def test_post_admin_add_inventory_missing_show_returns_404(monkeypatch):
    client, inv_id = _setup_api(monkeypatch)
    r = client.post(
        "/fairgame/api/admin/shows/show_nope/inventory",
        json={"section": "VIP", "qty": 10, "face_price_cents": 5000},
        headers=ADMIN_HDR,
    )
    assert r.status_code == 404
