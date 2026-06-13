"""Forge user store — per-person logins managed in-app (admin panel).

Separate from the main Alfred app users (config/users.json).  Backs the
Caddy ``forward_auth`` gate: Caddy asks /forge/authcheck, which verifies the
Basic-auth credentials against this store.  Adding/removing a user takes
effect immediately — no Caddy reload.

File: data/forge_users.json  {username: {password_hash, role, org}}.
Roles: "member" | "org_admin" | "super_admin".
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


_ROLES = ("member", "org_admin", "super_admin")


def create_user(username: str, password: str, role: str = "member",
                org: str = "mainstay") -> bool:
    """Add or update a user. role in {member, org_admin, super_admin}."""
    username = (username or "").strip().lower()
    if not username or not password:
        raise ValueError("username and password are required")
    role = role if role in _ROLES else "member"
    org = (org or "mainstay").strip().lower()
    users = load_users()
    users[username] = {"password_hash": _pwd.hash(password), "role": role, "org": org}
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
    username = (username or "").strip().lower()
    user = load_users().get(username)
    if not user:
        return None
    try:
        if not _pwd.verify(password, user["password_hash"]):
            return None
    except Exception:  # noqa: BLE001
        return None
    return {
        "username": username,
        "role": user.get("role", "member"),
        "org": user.get("org", "mainstay"),
    }


def list_users() -> list[dict]:
    """Public roster (no hashes), sorted by username."""
    return [
        {"username": u, "role": d.get("role", "member"), "org": d.get("org", "mainstay")}
        for u, d in sorted(load_users().items())
    ]


# Seed once with the accounts already issued, now org/role aware.
_SEED = [
    ("mike", "Mike-Boss0619!", "super_admin", "*"),
    ("mainstay", "RodWave0619!", "org_admin", "mainstay"),
    ("jordan", "Jordan-Anthem26!", "member", "mainstay"),
    ("mello", "Mello-Studio72!", "member", "mainstay"),
]


def ensure_seeded() -> None:
    if not load_users():
        for u, p, r, o in _SEED:
            create_user(u, p, r, o)
