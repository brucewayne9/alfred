"""Tests for the $1 Discover unlock endpoint (POST /fairgame/api/discover/unlock).

Sim mode (default when no FAIRGAME_STRIPE_KEY) returns unlocked immediately.
Live mode (key present + SIM=0) would return a checkout_url — that branch is
tested via a monkeypatch so no real Stripe calls are made.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi.testclient import TestClient
from core.api.fairgame import app


def test_discover_unlock_sim_returns_unlocked(monkeypatch):
    """Sim mode: no Stripe key → instant unlock."""
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    c = TestClient(app)
    r = c.post("/fairgame/api/discover/unlock", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["unlocked"] is True
    assert d["sim"] is True
    assert d["amount_cents"] == 100


def test_discover_unlock_sim_default_no_key(monkeypatch):
    """When FAIRGAME_STRIPE_KEY is absent and no SIM flag, sim activates by default."""
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    monkeypatch.delenv("FAIRGAME_STRIPE_SIM", raising=False)
    c = TestClient(app)
    r = c.post("/fairgame/api/discover/unlock", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["unlocked"] is True
    assert d["sim"] is True
    assert d["amount_cents"] == 100


def test_discover_unlock_live_returns_checkout_url(monkeypatch):
    """Live mode: Stripe key present → returns checkout_url (Stripe call mocked)."""
    monkeypatch.setenv("FAIRGAME_STRIPE_KEY", "sk_test_fake_key_for_testing")
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "0")

    # Stub out the checkout creation so no real network call is made.
    import core.fairgame.stripe_connect as sc
    monkeypatch.setattr(sc, "create_unlock_checkout", lambda amount: "https://checkout.stripe.com/fake")

    c = TestClient(app)
    r = c.post("/fairgame/api/discover/unlock", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["unlocked"] is False
    assert "checkout_url" in d
    assert d["checkout_url"].startswith("https://")
    assert d["amount_cents"] == 100


def test_discover_unlock_empty_body(monkeypatch):
    """Endpoint tolerates an empty body (no required fields)."""
    monkeypatch.delenv("FAIRGAME_STRIPE_KEY", raising=False)
    monkeypatch.setenv("FAIRGAME_STRIPE_SIM", "1")
    c = TestClient(app)
    r = c.post("/fairgame/api/discover/unlock", content=b"{}", headers={"Content-Type": "application/json"})
    assert r.status_code == 200
    assert r.json()["unlocked"] is True
