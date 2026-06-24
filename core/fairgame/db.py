"""Fair Game persistence — sqlite (WAL). Mirrors core/forge/db.py.

The FULL forward-looking schema is created here in M1 so later milestones
(events/presale, resale exchange, Stripe escrow, admin console) only *use*
these tables and never ALTER them. Money is integer cents everywhere.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def db_path() -> Path:
    override = os.environ.get("FAIRGAME_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "data" / "fairgame.db"


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c


def ensure_checkout_columns(c) -> None:
    """Idempotently add tm_email + final_sale_ack to access_grants and orders."""
    for table in ("access_grants", "orders"):
        cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        if "tm_email" not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN tm_email TEXT")
        if "final_sale_ack" not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN final_sale_ack INTEGER DEFAULT 0")


def init_db() -> None:
    with connect() as c:
        c.executescript(
            """
            -- ===== M1: Identity & verification =====
            CREATE TABLE IF NOT EXISTS fans(
                id          TEXT PRIMARY KEY,
                email       TEXT,
                phone       TEXT,
                email_hash  TEXT,
                phone_hash  TEXT,
                status      TEXT NOT NULL DEFAULT 'pending',
                priority    INTEGER NOT NULL DEFAULT 0,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_fans_email_hash ON fans(email_hash);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_fans_phone_hash ON fans(phone_hash);

            CREATE TABLE IF NOT EXISTS verification_codes(
                id          TEXT PRIMARY KEY,
                fan_id      TEXT NOT NULL,
                channel     TEXT NOT NULL,
                code_hash   TEXT NOT NULL,
                expires_at  INTEGER NOT NULL,
                attempts    INTEGER NOT NULL DEFAULT 0,
                consumed    INTEGER NOT NULL DEFAULT 0,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vc_fan
                ON verification_codes(fan_id, channel, consumed);

            CREATE TABLE IF NOT EXISTS sessions(
                token       TEXT PRIMARY KEY,
                fan_id      TEXT NOT NULL,
                device_fp   TEXT,
                ip          TEXT,
                expires_at  INTEGER NOT NULL,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_fan ON sessions(fan_id);

            CREATE TABLE IF NOT EXISTS device_events(
                id          TEXT PRIMARY KEY,
                fan_id      TEXT,
                device_fp   TEXT,
                ip          TEXT,
                event       TEXT NOT NULL,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_dev_fp ON device_events(device_fp, created_at);

            -- ===== M2: Shows, inventory, presale access =====
            CREATE TABLE IF NOT EXISTS shows(
                id          TEXT PRIMARY KEY,
                idx         INTEGER,
                city        TEXT,
                venue       TEXT,
                show_date   TEXT,
                status      TEXT DEFAULT 'on_sale',
                created_at  INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_shows_idx ON shows(idx);

            CREATE TABLE IF NOT EXISTS inventory(
                id               TEXT PRIMARY KEY,
                show_id          TEXT,
                section          TEXT,
                qty_total        INTEGER,
                qty_available    INTEGER,
                face_price_cents INTEGER,
                created_at       INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_inventory_show ON inventory(show_id);

            CREATE TABLE IF NOT EXISTS access_waves(
                id              TEXT PRIMARY KEY,
                show_id         TEXT,
                name            TEXT,
                starts_at       INTEGER,
                ends_at         INTEGER,
                priority_only   INTEGER DEFAULT 0,
                max_qty_per_fan INTEGER DEFAULT 4,
                created_at      INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_waves_show ON access_waves(show_id);

            CREATE TABLE IF NOT EXISTS access_grants(
                id          TEXT PRIMARY KEY,
                fan_id      TEXT,
                show_id     TEXT,
                wave_id     TEXT,
                qty         INTEGER,
                created_at  INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_grants_fan ON access_grants(fan_id, show_id);

            -- ===== M3: Capped resale exchange + Stripe escrow =====
            CREATE TABLE IF NOT EXISTS listings(
                id                   TEXT PRIMARY KEY,
                show_id              TEXT,
                seller_fan_id        TEXT,
                section              TEXT,
                face_price_cents     INTEGER,
                seller_proceeds_cents INTEGER,
                platform_fee_cents   INTEGER,
                buyer_total_cents    INTEGER,
                status               TEXT DEFAULT 'active',
                created_at           INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_listings_show ON listings(show_id, status);
            CREATE INDEX IF NOT EXISTS idx_listings_seller ON listings(seller_fan_id);

            CREATE TABLE IF NOT EXISTS orders(
                id            TEXT PRIMARY KEY,
                listing_id    TEXT,
                buyer_fan_id  TEXT,
                amount_cents  INTEGER,
                state         TEXT DEFAULT 'pending',
                payment_ref   TEXT,
                transfer_ref  TEXT,
                created_at    INTEGER,
                updated_at    INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_orders_buyer ON orders(buyer_fan_id);
            CREATE INDEX IF NOT EXISTS idx_orders_listing ON orders(listing_id);

            CREATE TABLE IF NOT EXISTS connect_accounts(
                fan_id             TEXT PRIMARY KEY,
                stripe_account_id  TEXT,
                onboarded          INTEGER DEFAULT 0,
                created_at         INTEGER
            );

            CREATE TABLE IF NOT EXISTS transfers(
                id          TEXT PRIMARY KEY,
                order_id    TEXT,
                state       TEXT DEFAULT 'pending',
                created_at  INTEGER,
                updated_at  INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_transfers_order ON transfers(order_id);
            """
        )
        ensure_checkout_columns(c)
