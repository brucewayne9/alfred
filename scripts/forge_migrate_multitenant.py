"""One-time migration: rewrite the Forge user store to org/role-aware records.

DB columns self-migrate via core.forge.db.init_db() (ALTER ... DEFAULT 'mainstay').
This handles the JSON user store. Idempotent — safe to run repeatedly.

Run:  python scripts/forge_migrate_multitenant.py
"""
import json
import os
from pathlib import Path

# Legacy username -> (role, org). Unlisted users default to (member, mainstay).
_KNOWN = {
    "mike":     ("super_admin", "*"),
    "mainstay": ("org_admin", "mainstay"),
}
_LEGACY_ROLE = {"admin": "org_admin", "team": "member"}


def _users_file() -> Path:
    return Path(os.environ.get("FORGE_USERS_FILE", "data/forge_users.json"))


def migrate_users() -> None:
    f = _users_file()
    if not f.exists():
        print(f"[migrate] no user store at {f} — nothing to do")
        return
    data = json.loads(f.read_text())
    changed = False
    for username, rec in data.items():
        if username in _KNOWN:
            role, org = _KNOWN[username]
        else:
            role = _LEGACY_ROLE.get(rec.get("role"), rec.get("role", "member"))
            if role not in ("member", "org_admin", "super_admin"):
                role = "member"
            org = rec.get("org", "mainstay")
        if rec.get("role") != role or rec.get("org") != org:
            rec["role"], rec["org"] = role, org
            changed = True
    if changed:
        f.write_text(json.dumps(data, indent=2))
        print(f"[migrate] rewrote {f}")
    else:
        print("[migrate] already current — no changes")


if __name__ == "__main__":
    from core.forge import db
    db.init_db()          # self-migrates the SQLite schema + seeds orgs
    migrate_users()
    print("[migrate] done")
