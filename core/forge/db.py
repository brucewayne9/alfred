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
                created_by  TEXT,
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

            -- Long-form ingest: one row per source file / URL (stable across
            -- multiple job enqueue attempts — survives reconcile_orphans).
            CREATE TABLE IF NOT EXISTS sources (
                id           TEXT PRIMARY KEY,
                kind         TEXT NOT NULL,
                spec         TEXT NOT NULL,
                file_path    TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                duration_s   REAL,
                language     TEXT,
                error        TEXT,
                created_at   INTEGER NOT NULL,
                updated_at   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sources_status
                ON sources(status, created_at);

            -- Whisper segments keyed to a source; persists after the ingest job
            -- is gone. 'words' is JSON: [{"word","start","end"}, ...].
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                seq          INTEGER NOT NULL,
                start_s      REAL NOT NULL,
                end_s        REAL NOT NULL,
                text         TEXT NOT NULL,
                speaker      TEXT,
                words        TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_segments_source
                ON transcript_segments(source_id, seq);

            -- Phase 08 "Auto-Clips": viral-scored clip candidates for a source.
            -- Re-scoring a source replaces its rows (delete-before-insert).
            -- 'rendered'/'posted' close the loop back to intel.py engagement.
            CREATE TABLE IF NOT EXISTS clip_candidates (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id    TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                start_s      REAL NOT NULL,
                end_s        REAL NOT NULL,
                score        INTEGER NOT NULL,
                hook         TEXT,
                emotion      TEXT,
                reason       TEXT,
                caption      TEXT,
                judge_model  TEXT,
                rendered     INTEGER NOT NULL DEFAULT 0,
                posted       INTEGER NOT NULL DEFAULT 0,
                created_at   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_candidates_source
                ON clip_candidates(source_id, score DESC);

            -- Multi-tenancy: one row per company (tenant).
            CREATE TABLE IF NOT EXISTS orgs (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at INTEGER NOT NULL DEFAULT 0
            );

            -- Distribution post ledger (was implicit; make it explicit + org-scoped).
            CREATE TABLE IF NOT EXISTS dist_posts (
                post_id   TEXT PRIMARY KEY,
                posted    INTEGER NOT NULL DEFAULT 0,
                posted_at INTEGER,
                org_id    TEXT NOT NULL DEFAULT 'mainstay'
            );
            """
        )
        # Idempotent migration: add jobs.created_by to pre-existing DBs.
        cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)").fetchall()}
        if "created_by" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN created_by TEXT")
        # Multi-tenancy: add org_id to scoped tables (DEFAULT backfills old rows
        # to 'mainstay' automatically). Idempotent — skip if already present.
        for table in ("sources", "jobs", "clip_candidates", "dist_posts"):
            tcols = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
            if "org_id" not in tcols:
                c.execute(
                    f"ALTER TABLE {table} ADD COLUMN org_id TEXT NOT NULL DEFAULT 'mainstay'"
                )
        # Index for org-filtered listing.
        c.execute("CREATE INDEX IF NOT EXISTS idx_sources_org ON sources(org_id, status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_jobs_org ON jobs(org_id, created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_candidates_org ON clip_candidates(org_id)")
        # Seed the three known orgs (idempotent via INSERT OR IGNORE).
        for oid, name in (("mainstay", "Mainstay Music Group"),
                          ("rucktalk", "RuckTalk"),
                          ("groundrush", "Ground Rush")):
            c.execute("INSERT OR IGNORE INTO orgs (id, name, created_at) VALUES (?, ?, 0)",
                      (oid, name))
