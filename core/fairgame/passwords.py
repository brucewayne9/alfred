"""Fair Game password hashing — stdlib only (pbkdf2-hmac-sha256).

Stored format: ``pbkdf2_sha256$<iters>$<salt_hex>$<hash_hex>``.
No external deps; constant-time verify; transparently re-hashable later.
"""
from __future__ import annotations

import hashlib
import hmac
import os

_ITERS = 200_000
_ALGO = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERS)
    return f"{_ALGO}${_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not password or not stored:
        return False
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        if algo != _ALGO:
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, TypeError):
        return False
