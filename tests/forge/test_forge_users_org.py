import pytest
from core.forge import users


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_USERS_FILE", str(tmp_path / "forge_users.json"))


def test_create_user_stores_org_and_role():
    users.create_user("alice", "pw123", role="member", org="rucktalk")
    u = users.verify_user("alice", "pw123")
    assert u == {"username": "alice", "role": "member", "org": "rucktalk"}


def test_unknown_role_falls_back_to_member():
    users.create_user("bob", "pw123", role="banana", org="mainstay")
    assert users.verify_user("bob", "pw123")["role"] == "member"


def test_super_admin_role_is_allowed():
    users.create_user("mike", "pw123", role="super_admin", org="*")
    assert users.verify_user("mike", "pw123")["role"] == "super_admin"


def test_list_users_includes_org():
    users.create_user("alice", "pw123", role="member", org="rucktalk")
    roster = users.list_users()
    assert {"username": "alice", "role": "member", "org": "rucktalk"} in roster
