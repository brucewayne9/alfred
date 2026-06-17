import importlib
import os
import sqlite3
import tempfile

import pytest


def _fresh_db(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db
    importlib.reload(db)
    db.init_db()
    return db


def test_init_db_creates_tables(monkeypatch):
    db = _fresh_db(monkeypatch)
    with db.connect() as c:
        names = {
            r["name"]
            for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"fans", "verification_codes", "sessions", "device_events"} <= names


def test_full_schema_present(monkeypatch):
    db = _fresh_db(monkeypatch)
    with db.connect() as c:
        names = {
            r["name"]
            for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    expected = {
        "fans", "verification_codes", "sessions", "device_events",
        "shows", "inventory", "access_waves", "access_grants",
        "listings", "orders", "connect_accounts", "transfers",
    }
    assert expected <= names


def test_email_hash_unique(monkeypatch):
    db = _fresh_db(monkeypatch)
    with db.connect() as c:
        c.execute(
            "INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,priority,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            ("f1", "a@b.com", "+15550001", "H1", "P1", "verified", 0, 1, 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            c.execute(
                "INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,priority,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                ("f2", "c@d.com", "+15550002", "H1", "P2", "verified", 0, 1, 1),
            )


def test_db_path_honors_env(monkeypatch):
    monkeypatch.setenv("FAIRGAME_DB_PATH", "/tmp/fairgame_custom.db")
    from core.fairgame import db
    importlib.reload(db)
    assert str(db.db_path()) == "/tmp/fairgame_custom.db"
