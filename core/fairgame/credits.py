"""Fans First — Discover credits ledger.

Fans buy a credit pack (one Stripe charge, so the fee is paid once) and spend
1 credit to reveal an event's lowest verified price + buy link. Reveals are
permanent — a fan never pays twice for the same event. Balance is the SUM of the
ledger; every purchase and every reveal is a row, so the account page shows the
full history. Packs are defaults here and editable later (like ticket prices).
"""
from __future__ import annotations

import time
import uuid

from . import db

# Default credit packs — one Stripe charge each. Editable later via admin.
PACKS = {
    "starter":  {"credits": 6,  "cents": 500,  "label": "Starter"},
    "fan":      {"credits": 14, "cents": 1000, "label": "Fan"},
    "superfan": {"credits": 30, "cents": 2000, "label": "Superfan"},
}


def _now() -> int:
    return int(time.time())


def balance(fan_id: str) -> int:
    """Current credit balance = sum of the fan's ledger deltas."""
    with db.connect() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(delta),0) AS bal FROM credit_ledger WHERE fan_id=?",
            (fan_id,),
        ).fetchone()
    return int(row["bal"]) if row else 0


def grant(fan_id: str, credits: int, *, kind: str, ref: str,
          amount_cents: int | None = None) -> int:
    """Append a ledger row (+credits on purchase, -1 on reveal). Idempotent on
    (fan, kind, ref) so a re-confirmed Stripe session never double-grants.
    Returns the new balance."""
    with db.connect() as c:
        if ref:
            dup = c.execute(
                "SELECT 1 FROM credit_ledger WHERE fan_id=? AND kind=? AND ref=?",
                (fan_id, kind, ref),
            ).fetchone()
            if dup:
                row = c.execute(
                    "SELECT COALESCE(SUM(delta),0) AS bal FROM credit_ledger WHERE fan_id=?",
                    (fan_id,),
                ).fetchone()
                return int(row["bal"])
        c.execute(
            "INSERT INTO credit_ledger(id,fan_id,delta,kind,ref,amount_cents,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            ("cl_" + uuid.uuid4().hex[:16], fan_id, int(credits), kind, ref,
             amount_cents, _now()),
        )
        row = c.execute(
            "SELECT COALESCE(SUM(delta),0) AS bal FROM credit_ledger WHERE fan_id=?",
            (fan_id,),
        ).fetchone()
    return int(row["bal"])


def has_revealed(fan_id: str, event_key: str) -> bool:
    """True if the fan already paid to reveal this event (so re-views are free)."""
    with db.connect() as c:
        row = c.execute(
            "SELECT 1 FROM credit_ledger WHERE fan_id=? AND kind='reveal' AND ref=?",
            (fan_id, event_key),
        ).fetchone()
    return bool(row)


def revealed_keys(fan_id: str) -> list[str]:
    """Every event key this fan has already unlocked."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT ref FROM credit_ledger WHERE fan_id=? AND kind='reveal'",
            (fan_id,),
        ).fetchall()
    return [r["ref"] for r in rows if r["ref"]]


def reveal(fan_id: str, event_key: str) -> dict:
    """Spend 1 credit to reveal an event (free if already revealed). Returns
    {revealed, already, balance[, need_credits]}."""
    if has_revealed(fan_id, event_key):
        return {"revealed": True, "already": True, "balance": balance(fan_id)}
    if balance(fan_id) < 1:
        return {"revealed": False, "already": False, "balance": 0, "need_credits": True}
    new_bal = grant(fan_id, -1, kind="reveal", ref=event_key)
    return {"revealed": True, "already": False, "balance": new_bal}


def history(fan_id: str, limit: int = 100) -> list[dict]:
    """Full ledger newest-first for the account page."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT * FROM credit_ledger WHERE fan_id=? ORDER BY created_at DESC LIMIT ?",
            (fan_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
