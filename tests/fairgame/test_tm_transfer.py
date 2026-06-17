import importlib
import os
import tempfile

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, tm_transfer
    importlib.reload(db)
    importlib.reload(tm_transfer)
    db.init_db()
    return tm_transfer


def test_status_none_before_initiate(monkeypatch):
    tm = _setup(monkeypatch)
    assert tm.status("ord_1") is None


def test_initiate_creates_initiated_row(monkeypatch):
    tm = _setup(monkeypatch)
    t = tm.initiate("ord_1")
    assert t["order_id"] == "ord_1"
    assert t["state"] == "initiated"
    assert t["id"].startswith("tx_")
    # ref is queryable via status
    assert tm.status("ord_1")["id"] == t["id"]


def test_confirm_transitions_to_confirmed(monkeypatch):
    tm = _setup(monkeypatch)
    tm.initiate("ord_1")
    c = tm.confirm("ord_1")
    assert c["state"] == "confirmed"
    assert tm.status("ord_1")["state"] == "confirmed"


def test_confirm_updates_timestamp(monkeypatch):
    tm = _setup(monkeypatch)
    t = tm.initiate("ord_1")
    c = tm.confirm("ord_1")
    assert c["updated_at"] >= t["created_at"]


def test_initiate_is_idempotent(monkeypatch):
    tm = _setup(monkeypatch)
    a = tm.initiate("ord_1")
    b = tm.initiate("ord_1")
    assert a["id"] == b["id"]
    assert b["state"] == "initiated"


def test_confirm_is_idempotent(monkeypatch):
    tm = _setup(monkeypatch)
    tm.initiate("ord_1")
    a = tm.confirm("ord_1")
    b = tm.confirm("ord_1")
    assert a["id"] == b["id"]
    assert b["state"] == "confirmed"


def test_confirm_without_initiate_raises(monkeypatch):
    tm = _setup(monkeypatch)
    with pytest.raises(tm.TransferError):
        tm.confirm("ord_missing")


def test_initiate_after_confirm_raises(monkeypatch):
    tm = _setup(monkeypatch)
    tm.initiate("ord_1")
    tm.confirm("ord_1")
    with pytest.raises(tm.TransferError):
        tm.initiate("ord_1")


def test_empty_order_id_raises(monkeypatch):
    tm = _setup(monkeypatch)
    with pytest.raises(tm.TransferError):
        tm.initiate("")
    with pytest.raises(tm.TransferError):
        tm.confirm("")


def test_transfers_are_per_order(monkeypatch):
    tm = _setup(monkeypatch)
    a = tm.initiate("ord_a")
    b = tm.initiate("ord_b")
    assert a["id"] != b["id"]
    tm.confirm("ord_a")
    assert tm.status("ord_a")["state"] == "confirmed"
    assert tm.status("ord_b")["state"] == "initiated"
