import json
import pytest


def _migrate_module():
    # import the migration module in whatever way works in this repo
    from scripts.forge_migrate_multitenant import migrate_users
    return migrate_users


def test_migration_maps_legacy_roles(tmp_path, monkeypatch):
    f = tmp_path / "forge_users.json"
    f.write_text(json.dumps({
        "mike":     {"password_hash": "h", "role": "admin"},
        "mainstay": {"password_hash": "h", "role": "admin"},
        "jordan":   {"password_hash": "h", "role": "team"},
    }))
    monkeypatch.setenv("FORGE_USERS_FILE", str(f))
    migrate_users = _migrate_module()
    migrate_users()
    data = json.loads(f.read_text())
    assert data["mike"]["role"] == "super_admin" and data["mike"]["org"] == "*"
    assert data["mainstay"]["role"] == "org_admin" and data["mainstay"]["org"] == "mainstay"
    assert data["jordan"]["role"] == "member" and data["jordan"]["org"] == "mainstay"


def test_migration_is_idempotent(tmp_path, monkeypatch):
    f = tmp_path / "forge_users.json"
    f.write_text(json.dumps({"mike": {"password_hash": "h", "role": "admin"}}))
    monkeypatch.setenv("FORGE_USERS_FILE", str(f))
    migrate_users = _migrate_module()
    migrate_users(); first = f.read_text()
    migrate_users(); assert f.read_text() == first
