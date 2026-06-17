"""Fair Game verification codes — channel-agnostic; send_fn injected by caller.

6-digit codes stored as salted hashes with a TTL. Throttles resends and caps
attempts so the verification gate is anti-bot without government ID (the
"medium strength" lever in the spec). The API layer wires the real Twilio/email
send via send_fn; tests inject a spy.
"""
from __future__ import annotations

import secrets
import time
import uuid

from . import db
from .identity import hash_value

CODE_TTL = 600
MAX_ATTEMPTS = 5
RESEND_COOLDOWN = 30


class VerifyError(Exception):
    ...


def _newest(c, fan_id, channel):
    return c.execute(
        "SELECT * FROM verification_codes WHERE fan_id=? AND channel=? AND consumed=0 "
        "ORDER BY created_at DESC LIMIT 1",
        (fan_id, channel),
    ).fetchone()


def start_verification(fan_id, channel, send_fn) -> dict:
    if channel not in ("sms", "email"):
        raise VerifyError("bad channel")
    now = int(time.time())
    with db.connect() as c:
        last = _newest(c, fan_id, channel)
        if last and now - last["created_at"] < RESEND_COOLDOWN:
            raise VerifyError("resend too soon")
        code = f"{secrets.randbelow(1000000):06d}"
        c.execute(
            "INSERT INTO verification_codes(id,fan_id,channel,code_hash,expires_at,attempts,consumed,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            ("vc_" + uuid.uuid4().hex[:12], fan_id, channel, hash_value(code), now + CODE_TTL, 0, 0, now),
        )
    send_fn(code)
    return {"sent": True, "channel": channel}


def check_code(fan_id, channel, code) -> bool:
    now = int(time.time())
    with db.connect() as c:
        row = _newest(c, fan_id, channel)
        if not row:
            raise VerifyError("no code")
        if row["attempts"] >= MAX_ATTEMPTS:
            raise VerifyError("too many attempts")
        if now > row["expires_at"]:
            raise VerifyError("expired")
        c.execute(
            "UPDATE verification_codes SET attempts=attempts+1 WHERE id=?", (row["id"],)
        )
        if row["code_hash"] == hash_value(code):
            c.execute(
                "UPDATE verification_codes SET consumed=1 WHERE id=?", (row["id"],)
            )
            return True
    return False
