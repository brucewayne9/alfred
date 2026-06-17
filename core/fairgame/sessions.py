"""Fair Game fan sessions — opaque DB-backed tokens.

A verified fan gets a long-lived opaque token (no JWT secret to manage in v1);
the row is the source of truth, so revoke is a hard delete and expiry is checked
on every resolve.
"""
from __future__ import annotations

import secrets
import time

from . import db

SESSION_TTL = 30 * 24 * 3600


def issue(fan_id, device_fp=None, ip=None) -> str:
    tok = secrets.token_urlsafe(32)
    now = int(time.time())
    with db.connect() as c:
        c.execute(
            "INSERT INTO sessions(token,fan_id,device_fp,ip,expires_at,created_at) VALUES(?,?,?,?,?,?)",
            (tok, fan_id, device_fp, ip, now + SESSION_TTL, now),
        )
    return tok


def resolve(token):
    now = int(time.time())
    with db.connect() as c:
        r = c.execute(
            "SELECT * FROM sessions WHERE token=? AND expires_at>?", (token, now)
        ).fetchone()
    return {"fan_id": r["fan_id"], "device_fp": r["device_fp"], "ip": r["ip"]} if r else None


def revoke(token) -> None:
    with db.connect() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))
