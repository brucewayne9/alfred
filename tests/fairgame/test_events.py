import importlib
import json
import os
import tempfile


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, events
    importlib.reload(db)
    importlib.reload(events)
    db.init_db()
    return events


# ---- seeding from the REAL tour JSON ----

def test_seed_shows_from_real_json(monkeypatch):
    events = _setup(monkeypatch)
    n = events.seed_shows()
    shows = events.list_shows()
    assert n == len(shows) == 25
    # Ordered by idx, carries city/venue/date from the file.
    first = shows[0]
    assert first["idx"] == 1
    assert first["city"] == "Philadelphia"
    assert first["venue"] == "Xfinity Mobile Arena"
    assert first["show_date"] == "Sep 12 & 14"
    assert first["status"] == "on_sale"
    # Atlanta (home show) is last.
    atl = events.get_show("show_25")
    assert atl["city"] == "Atlanta"
    assert atl["show_date"] == "Nov 18 & 19"


def test_seed_shows_idempotent(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    before = events.get_show("show_1")["created_at"]
    n2 = events.seed_shows()  # re-run
    assert n2 == 25
    assert len(events.list_shows()) == 25  # no duplicates
    # created_at preserved across re-seed.
    assert events.get_show("show_1")["created_at"] == before


def test_get_show_missing(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    assert events.get_show("nope") is None


# ---- inventory math ----

def test_add_inventory_sets_available(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    row = events.add_inventory("show_1", "Floor", 100, 15000)
    assert row["qty_total"] == 100
    assert row["qty_available"] == 100
    assert row["face_price_cents"] == 15000
    assert row["section"] == "Floor"


def test_remaining_sums_available(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    assert events.remaining("show_1") == 0  # nothing yet
    events.add_inventory("show_1", "Floor", 100, 15000)
    events.add_inventory("show_1", "Lower", 250, 9500)
    assert events.remaining("show_1") == 350
    # scoped per show
    assert events.remaining("show_2") == 0


def test_get_inventory_scoped_and_ordered(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    events.add_inventory("show_1", "Upper", 600, 5500)
    events.add_inventory("show_1", "Floor", 200, 15000)
    inv = events.get_inventory("show_1")
    assert [r["section"] for r in inv] == ["Floor", "Upper"]  # priciest first
    assert all(r["show_id"] == "show_1" for r in inv)


# ---- demo inventory ----

def test_seed_demo_inventory(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    created = events.seed_demo_inventory()
    assert created == 25 * 3  # 3 sections per show
    inv = events.get_inventory("show_1")
    assert {r["section"] for r in inv} == {"Floor", "Lower", "Upper"}
    prices = {r["section"]: r["face_price_cents"] for r in inv}
    assert prices == {"Floor": 15000, "Lower": 9500, "Upper": 5500}
    # remaining = 200 + 400 + 600
    assert events.remaining("show_1") == 1200


def test_seed_demo_inventory_idempotent(monkeypatch):
    events = _setup(monkeypatch)
    events.seed_shows()
    events.seed_demo_inventory()
    second = events.seed_demo_inventory()  # re-run adds nothing
    assert second == 0
    assert len(events.get_inventory("show_1")) == 3
