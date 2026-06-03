"""Forge user store — per-person logins managed in-app (admin panel).

Separate from the main Alfred app users (config/users.json).  Backs the
Caddy ``forward_auth`` gate: Caddy asks /forge/authcheck, which verifies the
Basic-auth credentials against this store.  Adding/removing a user takes
effect immediately — no Caddy reload.

File: data/forge_users.json  {username: {password_hash, role}}.
Roles: "admin" (can manage users) | "team".
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from passlib.context import CryptContext

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _users_file() -> Path:
    return Path(os.environ.get("FORGE_USERS_FILE", "data/forge_users.json"))


def load_users() -> dict:
    f = _users_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_users(users: dict) -> None:
    f = _users_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(users, indent=2))
    try:
        f.chmod(0o600)
    except Exception:  # noqa: BLE001
        pass


def create_user(username: str, password: str, role: str = "team") -> bool:
    """Add or update a user.  Returns True.  Role is 'admin' or 'team'."""
    username = (username or "").strip().lower()
    if not username or not password:
        raise ValueError("username and password are required")
    role = "admin" if role == "admin" else "team"
    users = load_users()
    users[username] = {"password_hash": _pwd.hash(password), "role": role}
    save_users(users)
    return True


def delete_user(username: str) -> bool:
    username = (username or "").strip().lower()
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        return True
    return False


def verify_user(username: str, password: str) -> dict | None:
    """Return {'username','role'} if the credentials are valid, else None."""
    username = (username or "").strip().lower()
    user = load_users().get(username)
    if not user:
        return None
    try:
        if not _pwd.verify(password, user["password_hash"]):
            return None
    except Exception:  # noqa: BLE001
        return None
    return {"username": username, "role": user.get("role", "team")}


def list_users() -> list[dict]:
    """Public roster (no hashes) — newest sort by username."""
    return [
        {"username": u, "role": d.get("role", "team")}
        for u, d in sorted(load_users().items())
    ]


# Seed the store once with the accounts already issued (so the credentials we
# emailed keep working after the forward_auth cutover).  Only runs if empty.
_SEED = [
    ("mike", "Mike-Boss0619!", "admin"),
    ("mainstay", "RodWave0619!", "admin"),
    ("jordan", "Jordan-Anthem26!", "team"),
    ("mello", "Mello-Studio72!", "team"),
]


def ensure_seeded() -> None:
    if not load_users():
        for u, p, r in _SEED:
            create_user(u, p, r)
