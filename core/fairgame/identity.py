"""Fair Game fan identity — dedupe, device fingerprint, waitlist priority.

One identity = one account: dedupe on salted email_hash OR phone_hash. Raw
email/phone are stored in M1 (flagged for at-rest encryption in M5); the hashes
are what we look up on, so a fan who re-registers with a new email but the same
phone (or vice versa) maps back to the existing record.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import uuid

from . import db
from .waitlist import is_priority

_SALT = os.environ.get("FAIRGAME_HASH_SALT", "fairgame-v1-salt")


def hash_value(raw: str) -> str:
    return hashlib.sha256((_SALT + (raw or "").strip().lower()).encode()).hexdigest()


def normalize_phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw or "")
    if digits and not digits.startswith("+"):
        digits = "+" + digits
    return digits


def get_fan(fan_id: str):
    with db.connect() as c:
        row = c.execute("SELECT * FROM fans WHERE id=?", (fan_id,)).fetchone()
    return dict(row) if row else None


def upsert_fan(email: str, phone: str, device_fp=None, ip=None) -> dict:
    email = (email or "").strip().lower()
    phone = normalize_phone(phone)
    eh, ph = hash_value(email), hash_value(phone)
    now = int(time.time())
    with db.connect() as c:
        row = c.execute(
            "SELECT * FROM fans WHERE email_hash=? OR phone_hash=?", (eh, ph)
        ).fetchone()
        if row:
            fid = row["id"]
            c.execute(
                "UPDATE fans SET email=?,phone=?,email_hash=?,phone_hash=?,updated_at=? WHERE id=?",
                (email, phone, eh, ph, now, fid),
            )
        else:
            fid = "fan_" + uuid.uuid4().hex[:12]
            prio = 1 if is_priority(email) else 0
            c.execute(
                "INSERT INTO fans(id,email,phone,email_hash,phone_hash,status,priority,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (fid, email, phone, eh, ph, "pending", prio, now, now),
            )
        c.execute(
            "INSERT INTO device_events(id,fan_id,device_fp,ip,event,created_at) VALUES(?,?,?,?,?,?)",
            ("ev_" + uuid.uuid4().hex[:12], fid, device_fp, ip, "register", now),
        )
        out = c.execute("SELECT * FROM fans WHERE id=?", (fid,)).fetchone()
    return dict(out)
