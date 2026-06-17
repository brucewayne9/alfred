import importlib
import os
import tempfile
import time


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, sessions
    importlib.reload(db)
    importlib.reload(sessions)
    db.init_db()
    return sessions


def test_issue_resolve(monkeypatch):
    s = _setup(monkeypatch)
    tok = s.issue("fan_1", "fp", "1.1.1.1")
    got = s.resolve(tok)
    assert got["fan_id"] == "fan_1"
    assert got["device_fp"] == "fp"
    assert got["ip"] == "1.1.1.1"


def test_revoke(monkeypatch):
    s = _setup(monkeypatch)
    tok = s.issue("fan_1")
    s.revoke(tok)
    assert s.resolve(tok) is None


def test_bad_token(monkeypatch):
    s = _setup(monkeypatch)
    assert s.resolve("nope") is None


def test_expired_session(monkeypatch):
    s = _setup(monkeypatch)
    monkeypatch.setattr(s, "SESSION_TTL", -1)
    tok = s.issue("fan_1")
    assert s.resolve(tok) is None
