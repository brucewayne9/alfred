import importlib
import os
import tempfile

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, identity, events, access
    importlib.reload(db)
    importlib.reload(identity)
    importlib.reload(events)
    importlib.reload(access)
    db.init_db()
    events.seed_shows()
    return db, identity, events, access


def _verified_fan(db, identity, email, phone, priority=0):
    fan = identity.upsert_fan(email, phone, None, None)
    with db.connect() as c:
        c.execute(
            "UPDATE fans SET status='verified', priority=? WHERE id=?",
            (priority, fan["id"]),
        )
    return identity.get_fan(fan["id"])


# ---- waves ----

def test_create_and_open_waves(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    access.create_wave("show_1", "Presale", 100, 200, priority_only=False, max_qty_per_fan=4)
    # closed (in the future)
    access.create_wave("show_1", "Later", 500, 600)
    open_now = access.open_waves("show_1", now=150)
    assert len(open_now) == 1
    assert open_now[0]["name"] == "Presale"
    assert open_now[0]["max_qty_per_fan"] == 4
    # boundaries inclusive
    assert len(access.open_waves("show_1", now=100)) == 1
    assert len(access.open_waves("show_1", now=200)) == 1
    assert access.open_waves("show_1", now=201) == []


# ---- grant happy path + inventory decrement ----

def test_grant_decrements_inventory(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 10, 15000)
    events.add_inventory("show_1", "Upper", 10, 5500)
    access.create_wave("show_1", "Presale", 0, 1000, max_qty_per_fan=4)
    fan = _verified_fan(db, identity, "a@x.com", "+15550001")
    before = events.remaining("show_1")
    grant = access.grant_access(fan["id"], "show_1", 3, now=10)
    assert grant["qty"] == 3
    assert grant["fan_id"] == fan["id"]
    assert grant["show_id"] == "show_1"
    assert events.remaining("show_1") == before - 3
    # cheapest section (Upper) drained first
    inv = {r["section"]: r["qty_available"] for r in events.get_inventory("show_1")}
    assert inv["Upper"] == 7
    assert inv["Floor"] == 10


# ---- verification gate ----

def test_unverified_fan_rejected(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 10, 15000)
    access.create_wave("show_1", "Presale", 0, 1000)
    fan = identity.upsert_fan("p@x.com", "+15550009", None, None)  # status 'pending'
    with pytest.raises(access.AccessError):
        access.grant_access(fan["id"], "show_1", 1, now=10)
    assert access.can_purchase(fan["id"], "show_1", 1, now=10) is False


def test_missing_fan_rejected(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 10, 15000)
    access.create_wave("show_1", "Presale", 0, 1000)
    with pytest.raises(access.AccessError):
        access.grant_access("fan_nope", "show_1", 1, now=10)
    assert access.can_purchase("fan_nope", "show_1", 1, now=10) is False


# ---- no open wave ----

def test_no_open_wave_rejected(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 10, 15000)
    access.create_wave("show_1", "Future", 500, 600)
    fan = _verified_fan(db, identity, "a@x.com", "+15550001")
    with pytest.raises(access.AccessError):
        access.grant_access(fan["id"], "show_1", 1, now=10)
    assert access.can_purchase(fan["id"], "show_1", 1, now=10) is False


# ---- priority gating ----

def test_priority_only_blocks_non_priority(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 10, 15000)
    access.create_wave("show_1", "FanClub", 0, 1000, priority_only=True, max_qty_per_fan=4)
    nonprio = _verified_fan(db, identity, "a@x.com", "+15550001", priority=0)
    with pytest.raises(access.AccessError):
        access.grant_access(nonprio["id"], "show_1", 1, now=10)
    assert access.can_purchase(nonprio["id"], "show_1", 1, now=10) is False


def test_priority_only_admits_priority(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 10, 15000)
    access.create_wave("show_1", "FanClub", 0, 1000, priority_only=True, max_qty_per_fan=4)
    prio = _verified_fan(db, identity, "v@x.com", "+15559999", priority=1)
    grant = access.grant_access(prio["id"], "show_1", 2, now=10)
    assert grant["qty"] == 2
    assert access.can_purchase(prio["id"], "show_1", 2, now=10) is True


# ---- qty caps ----

def test_qty_exceeds_wave_cap(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 100, 15000)
    access.create_wave("show_1", "Presale", 0, 1000, max_qty_per_fan=4)
    fan = _verified_fan(db, identity, "a@x.com", "+15550001")
    with pytest.raises(access.AccessError):
        access.grant_access(fan["id"], "show_1", 5, now=10)
    assert access.can_purchase(fan["id"], "show_1", 5, now=10) is False
    assert access.can_purchase(fan["id"], "show_1", 4, now=10) is True


def test_qty_below_one_rejected(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 100, 15000)
    access.create_wave("show_1", "Presale", 0, 1000, max_qty_per_fan=4)
    fan = _verified_fan(db, identity, "a@x.com", "+15550001")
    with pytest.raises(access.AccessError):
        access.grant_access(fan["id"], "show_1", 0, now=10)
    assert access.can_purchase(fan["id"], "show_1", 0, now=10) is False


# ---- sold out ----

def test_sold_out_rejected(monkeypatch):
    db, identity, events, access = _setup(monkeypatch)
    events.add_inventory("show_1", "Floor", 2, 15000)
    access.create_wave("show_1", "Presale", 0, 1000, max_qty_per_fan=4)
    fan = _verified_fan(db, identity, "a@x.com", "+15550001")
    # qty exceeds remaining inventory
    with pytest.raises(access.AccessError):
        access.grant_access(fan["id"], "show_1", 3, now=10)
    assert access.can_purchase(fan["id"], "show_1", 3, now=10) is False
    # drain it exactly
    access.grant_access(fan["id"], "show_1", 2, now=10)
    assert events.remaining("show_1") == 0
    # now fully sold out for a second buyer
    fan2 = _verified_fan(db, identity, "b@x.com", "+15550002")
    with pytest.raises(access.AccessError):
        access.grant_access(fan2["id"], "show_1", 1, now=10)
    assert access.can_purchase(fan2["id"], "show_1", 1, now=10) is False
