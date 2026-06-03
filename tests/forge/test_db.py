import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def forge_db(tmp_path, monkeypatch):
    db_file = tmp_path / "forge.db"
    monkeypatch.setenv("FORGE_DB_PATH", str(db_file))
    from core.forge import db
    db.init_db()
    return db_file


def test_db_path_honors_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "forge.db"
    monkeypatch.setenv("FORGE_DB_PATH", str(target))
    from core.forge import db
    assert db._db_path() == target


def test_init_db_creates_jobs_table(forge_db):
    conn = sqlite3.connect(str(forge_db))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
    ).fetchall()
    conn.close()
    assert rows == [("jobs",)]


def test_init_db_is_idempotent(forge_db):
    from core.forge import db
    db.init_db()  # second call must not raise
    db.init_db()


def test_jobs_table_has_expected_columns(forge_db):
    conn = sqlite3.connect(str(forge_db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    conn.close()
    assert cols == {
        "id", "job_type", "status", "params",
        "result", "error", "created_by", "created_at", "updated_at",
    }
