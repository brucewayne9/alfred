"""Mainstay Forge — job store (SQLite, WAL). Mirrors core/api/arcade_scores.py."""
import os
import sqlite3
from pathlib import Path


def _db_path() -> Path:
    override = os.environ.get("FORGE_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "data" / "forge.db"


def _conn() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                job_type    TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                params      TEXT NOT NULL DEFAULT '{}',
                result      TEXT,
                error       TEXT,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status
                ON jobs(status, created_at);

            -- Library soft-delete trash. One row per delete *action* (a single
            -- file, or a whole batch's dirs). 'items' is JSON [{orig, trash}].
            -- 'job_id' is set for batch deletes so the card can be re-shown on undo.
            CREATE TABLE IF NOT EXISTS trash (
                token       TEXT PRIMARY KEY,
                kind        TEXT NOT NULL,
                items       TEXT NOT NULL,
                job_id      TEXT,
                label       TEXT,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trash_created
                ON trash(created_at);
            """
        )
