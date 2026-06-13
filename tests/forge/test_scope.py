from core.forge.scope import Scope, scope_from_user


def test_member_is_pinned_to_own_org():
    s = scope_from_user({"username": "alice", "role": "member", "org": "rucktalk"})
    assert s.org == "rucktalk"
    assert s.view_all is False
    assert s.can_write_org("rucktalk") is True
    assert s.can_write_org("mainstay") is False


def test_super_admin_view_all_by_default():
    s = scope_from_user({"username": "mike", "role": "super_admin", "org": "*"})
    assert s.view_all is True
    assert s.can_write_org("mainstay") is True
    assert s.can_write_org("rucktalk") is True


def test_super_admin_can_focus_one_org():
    s = scope_from_user(
        {"username": "mike", "role": "super_admin", "org": "*"},
        requested_org="mainstay",
    )
    assert s.view_all is False
    assert s.org == "mainstay"


def test_member_cannot_escape_org_via_requested_org():
    s = scope_from_user(
        {"username": "alice", "role": "member", "org": "rucktalk"},
        requested_org="mainstay",
    )
    assert s.org == "rucktalk"
    assert s.view_all is False
