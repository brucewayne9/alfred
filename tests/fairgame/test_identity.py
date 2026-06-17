import importlib
import os
import tempfile


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, identity, waitlist
    importlib.reload(db)
    importlib.reload(waitlist)
    importlib.reload(identity)
    # is_priority is bound into identity's namespace at import; patch it there.
    monkeypatch.setattr(
        identity, "is_priority", lambda email: email.endswith("@vip.com"), raising=False
    )
    db.init_db()
    return identity


def test_normalize_phone(monkeypatch):
    identity = _setup(monkeypatch)
    assert identity.normalize_phone("(555) 000-1234") == "+5550001234"


def test_hash_value_stable_and_case_insensitive(monkeypatch):
    identity = _setup(monkeypatch)
    assert identity.hash_value("A@X.com ") == identity.hash_value("a@x.com")
    assert identity.hash_value("a@x.com") != identity.hash_value("b@x.com")


def test_upsert_dedupes_on_phone(monkeypatch):
    identity = _setup(monkeypatch)
    a = identity.upsert_fan("a@x.com", "+15550001", "fp1", "1.1.1.1")
    b = identity.upsert_fan("different@x.com", "+15550001", "fp2", "2.2.2.2")
    assert a["id"] == b["id"]  # same phone -> same fan


def test_upsert_dedupes_on_email(monkeypatch):
    identity = _setup(monkeypatch)
    a = identity.upsert_fan("same@x.com", "+15550001", None, None)
    b = identity.upsert_fan("same@x.com", "+15559999", None, None)
    assert a["id"] == b["id"]  # same email -> same fan


def test_priority_flag(monkeypatch):
    identity = _setup(monkeypatch)
    f = identity.upsert_fan("fan@vip.com", "+15559999", None, None)
    assert f["priority"] == 1


def test_get_fan_roundtrip(monkeypatch):
    identity = _setup(monkeypatch)
    f = identity.upsert_fan("a@x.com", "+15550001", "fp1", "1.1.1.1")
    got = identity.get_fan(f["id"])
    assert got["email"] == "a@x.com"
    assert identity.get_fan("nope") is None
