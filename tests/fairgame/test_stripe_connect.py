"""Tests for core/fairgame/stripe_connect.py — runs FULLY in SIM mode (no keys)."""
import importlib
import os
import tempfile

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    # Force the simulator; never touch the network.
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    from core.fairgame import db, stripe_connect
    importlib.reload(db)
    importlib.reload(stripe_connect)
    db.init_db()
    return stripe_connect


# ---- sim-mode selection ----

def test_sim_is_default_without_key(monkeypatch):
    monkeypatch.delenv("FAIRGAME_STRIPE_SIM", raising=False)
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    from core.fairgame import stripe_connect
    importlib.reload(stripe_connect)
    assert stripe_connect._sim_mode() is True


def test_key_present_disables_sim(monkeypatch):
    monkeypatch.delenv("FAIRGAME_STRIPE_SIM", raising=False)
    monkeypatch.setenv("FAIRGAME_STRIPE_KEY", "sk_test_xxx")
    from core.fairgame import stripe_connect
    importlib.reload(stripe_connect)
    assert stripe_connect._sim_mode() is False


def test_explicit_sim_flag_wins_over_key(monkeypatch):
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    monkeypatch.setenv("FAIRGAME_STRIPE_KEY", "sk_test_xxx")
    from core.fairgame import stripe_connect
    importlib.reload(stripe_connect)
    assert stripe_connect._sim_mode() is True


# ---- onboard_seller ----

def test_onboard_creates_account_and_url(monkeypatch):
    sc = _setup(monkeypatch)
    acct = sc.onboard_seller("fan_1")
    assert acct["fan_id"] == "fan_1"
    assert acct["stripe_account_id"].startswith("acct_sim_")
    assert acct["onboarded"] == 0
    assert acct["onboarding_url"].startswith("https://")
    assert acct["stripe_account_id"] in acct["onboarding_url"]


def test_onboard_is_idempotent(monkeypatch):
    sc = _setup(monkeypatch)
    a = sc.onboard_seller("fan_1")
    b = sc.onboard_seller("fan_1")
    assert a["stripe_account_id"] == b["stripe_account_id"]
    # only one row exists
    assert sc.get_account("fan_1")["stripe_account_id"] == a["stripe_account_id"]


def test_onboard_requires_fan_id(monkeypatch):
    sc = _setup(monkeypatch)
    with pytest.raises(sc.StripeError):
        sc.onboard_seller("")


def test_mark_onboarded(monkeypatch):
    sc = _setup(monkeypatch)
    sc.onboard_seller("fan_1")
    row = sc.mark_onboarded("fan_1")
    assert row["onboarded"] == 1
    # idempotent
    assert sc.mark_onboarded("fan_1")["onboarded"] == 1


def test_get_account_missing(monkeypatch):
    sc = _setup(monkeypatch)
    assert sc.get_account("nobody") is None


# ---- create_held_payment ----

def test_create_held_payment_holds_funds(monkeypatch):
    sc = _setup(monkeypatch)
    o = sc.create_held_payment("ord_1", 7500, buyer_ref="fan_buyer")
    assert o["id"] == "ord_1"
    assert o["amount_cents"] == 7500
    assert o["state"] == "held"
    assert o["payment_ref"].startswith("pi_sim_")


def test_create_held_payment_persists_listing_and_buyer(monkeypatch):
    sc = _setup(monkeypatch)
    o = sc.create_held_payment(
        "ord_1", 7500, buyer_ref="ref", listing_id="lst_9", buyer_fan_id="fan_b"
    )
    assert o["listing_id"] == "lst_9"
    assert o["buyer_fan_id"] == "fan_b"


def test_create_held_payment_is_idempotent(monkeypatch):
    sc = _setup(monkeypatch)
    a = sc.create_held_payment("ord_1", 7500)
    b = sc.create_held_payment("ord_1", 7500)
    assert a["payment_ref"] == b["payment_ref"]  # same held ref, no double charge
    assert b["state"] == "held"


def test_create_held_payment_rejects_negative(monkeypatch):
    sc = _setup(monkeypatch)
    with pytest.raises(sc.StripeError):
        sc.create_held_payment("ord_1", -1)


def test_create_held_payment_requires_order_id(monkeypatch):
    sc = _setup(monkeypatch)
    with pytest.raises(sc.StripeError):
        sc.create_held_payment("", 7500)


# ---- release_to_seller ----

def test_release_marks_released(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500)
    o = sc.release_to_seller("ord_1")
    assert o["state"] == "released"


def test_release_is_idempotent(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500)
    a = sc.release_to_seller("ord_1")
    b = sc.release_to_seller("ord_1")
    assert a["payment_ref"] == b["payment_ref"]
    assert b["state"] == "released"


def test_release_without_order_raises(monkeypatch):
    sc = _setup(monkeypatch)
    with pytest.raises(sc.StripeError):
        sc.release_to_seller("ord_missing")


def test_release_after_refund_raises(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500)
    sc.refund("ord_1")
    with pytest.raises(sc.StripeError):
        sc.release_to_seller("ord_1")


# ---- refund ----

def test_refund_marks_refunded(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500)
    o = sc.refund("ord_1")
    assert o["state"] == "refunded"


def test_refund_is_idempotent(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500)
    a = sc.refund("ord_1")
    b = sc.refund("ord_1")
    assert a["state"] == b["state"] == "refunded"


def test_refund_without_order_raises(monkeypatch):
    sc = _setup(monkeypatch)
    with pytest.raises(sc.StripeError):
        sc.refund("ord_missing")


def test_refund_after_release_raises(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500)
    sc.release_to_seller("ord_1")
    with pytest.raises(sc.StripeError):
        sc.refund("ord_1")


# ---- full escrow happy path: hold -> transfer confirmed -> release ----

def test_full_escrow_flow_release(monkeypatch):
    sc = _setup(monkeypatch)
    from core.fairgame import tm_transfer
    importlib.reload(tm_transfer)
    held = sc.create_held_payment("ord_1", 7500, buyer_ref="fan_b")
    assert held["state"] == "held"
    # ticket moves on TM rails
    tm_transfer.initiate("ord_1")
    tm_transfer.confirm("ord_1")
    # only now do we pay out
    released = sc.release_to_seller("ord_1")
    assert released["state"] == "released"


def test_full_escrow_flow_failed_transfer_refunds(monkeypatch):
    sc = _setup(monkeypatch)
    sc.create_held_payment("ord_1", 7500, buyer_ref="fan_b")
    # transfer never confirmed -> auto-refund buyer, seller gets nothing
    refunded = sc.refund("ord_1")
    assert refunded["state"] == "refunded"
