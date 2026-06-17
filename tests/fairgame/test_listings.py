import importlib
import os
import tempfile

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, listings
    importlib.reload(db)
    importlib.reload(listings)
    db.init_db()
    return listings


# ---- the locked cap split: $60 face -> buyer 7500 / seller 7000 / platform 500 ----

def test_constants_are_locked():
    from core.fairgame import listings
    assert listings.SELLER_MARKUP_CENTS == 1000
    assert listings.PLATFORM_FEE_CENTS == 500


def test_quote_split_for_60_dollar_face(monkeypatch):
    listings = _setup(monkeypatch)
    q = listings.quote(6000)
    assert q["face_price_cents"] == 6000
    assert q["seller_proceeds_cents"] == 7000   # face + $10
    assert q["platform_fee_cents"] == 500       # Rod's $5
    assert q["buyer_total_cents"] == 7500        # face + $15


def test_create_listing_computes_exact_split(monkeypatch):
    listings = _setup(monkeypatch)
    row = listings.create_listing("fan_1", "show_1", "Floor", 6000)
    assert row["seller_fan_id"] == "fan_1"
    assert row["show_id"] == "show_1"
    assert row["section"] == "Floor"
    assert row["face_price_cents"] == 6000
    assert row["seller_proceeds_cents"] == 7000
    assert row["platform_fee_cents"] == 500
    assert row["buyer_total_cents"] == 7500
    assert row["status"] == "active"
    assert row["id"].startswith("lst_")


def test_seller_always_made_whole_plus_ten(monkeypatch):
    listings = _setup(monkeypatch)
    for face in (0, 5500, 9500, 15000):
        q = listings.quote(face)
        # seller never loses on face and always pockets exactly $10
        assert q["seller_proceeds_cents"] == face + 1000
        # Rod's cut is always a flat $5, regardless of face
        assert q["platform_fee_cents"] == 500
        # buyer total = seller proceeds + platform fee
        assert q["buyer_total_cents"] == q["seller_proceeds_cents"] + 500


# ---- get / list / cancel ----

def test_get_listing_roundtrip(monkeypatch):
    listings = _setup(monkeypatch)
    row = listings.create_listing("fan_1", "show_1", "Lower", 9500)
    got = listings.get_listing(row["id"])
    assert got["id"] == row["id"]
    assert got["face_price_cents"] == 9500


def test_get_listing_missing(monkeypatch):
    listings = _setup(monkeypatch)
    assert listings.get_listing("nope") is None


def test_list_active_scoped_and_cheapest_first(monkeypatch):
    listings = _setup(monkeypatch)
    listings.create_listing("fan_1", "show_1", "Floor", 15000)
    listings.create_listing("fan_2", "show_1", "Upper", 5500)
    listings.create_listing("fan_3", "show_2", "Floor", 9000)  # other show
    active = listings.list_active("show_1")
    assert [r["face_price_cents"] for r in active] == [5500, 15000]  # cheapest up top
    assert all(r["show_id"] == "show_1" for r in active)


def test_cancel_listing_drops_it_from_active(monkeypatch):
    listings = _setup(monkeypatch)
    a = listings.create_listing("fan_1", "show_1", "Floor", 6000)
    b = listings.create_listing("fan_2", "show_1", "Lower", 9500)
    updated = listings.cancel_listing(a["id"])
    assert updated["status"] == "cancelled"
    active = listings.list_active("show_1")
    assert [r["id"] for r in active] == [b["id"]]


def test_cancel_missing_returns_none(monkeypatch):
    listings = _setup(monkeypatch)
    assert listings.cancel_listing("nope") is None


def test_negative_face_rejected(monkeypatch):
    listings = _setup(monkeypatch)
    with pytest.raises(ValueError):
        listings.create_listing("fan_1", "show_1", "Floor", -100)


# ---- anti-gouge: declared face cannot exceed the section's true primary face ----

def _setup_with_inventory(monkeypatch):
    listings = _setup(monkeypatch)
    from core.fairgame import events
    importlib.reload(events)
    events.add_inventory("show_9", "Lower", 100, 9500)  # true face = $95
    return listings, events


def test_face_above_true_face_rejected(monkeypatch):
    listings, _ = _setup_with_inventory(monkeypatch)
    # Scalper declares a $300 "face" on a $95 seat -> rejected, cap can't be gamed.
    with pytest.raises(ValueError):
        listings.create_listing("scalper", "show_9", "Lower", 30000)


def test_face_at_or_below_true_face_allowed(monkeypatch):
    listings, _ = _setup_with_inventory(monkeypatch)
    # Exactly face is fine; selling under face (fan-friendly) is fine too.
    at = listings.create_listing("fan_a", "show_9", "Lower", 9500)
    assert at["face_price_cents"] == 9500
    under = listings.create_listing("fan_b", "show_9", "Lower", 6000)
    assert under["face_price_cents"] == 6000
