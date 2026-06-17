import importlib
import os
import tempfile

import pytest


def _setup(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("FAIRGAME_DB_PATH", os.path.join(d, "fg.db"))
    from core.fairgame import db, identity, verify
    importlib.reload(db)
    importlib.reload(identity)
    importlib.reload(verify)
    db.init_db()
    fan = identity.upsert_fan("a@x.com", "+15550001", None, None)
    return verify, fan["id"]


def test_send_and_verify(monkeypatch):
    verify, fid = _setup(monkeypatch)
    sent = {}
    verify.start_verification(fid, "sms", lambda code: sent.update(code=code))
    assert len(sent["code"]) == 6 and sent["code"].isdigit()
    assert verify.check_code(fid, "sms", sent["code"]) is True


def test_consumed_code_cannot_reverify(monkeypatch):
    verify, fid = _setup(monkeypatch)
    sent = {}
    verify.start_verification(fid, "sms", lambda code: sent.update(code=code))
    assert verify.check_code(fid, "sms", sent["code"]) is True
    with pytest.raises(verify.VerifyError):
        verify.check_code(fid, "sms", sent["code"])  # no unconsumed code left


def test_wrong_code_fails(monkeypatch):
    verify, fid = _setup(monkeypatch)
    sent = {}
    verify.start_verification(fid, "sms", lambda code: sent.update(code=code))
    assert verify.check_code(fid, "sms", "000000") is False


def test_max_attempts(monkeypatch):
    verify, fid = _setup(monkeypatch)
    verify.start_verification(fid, "sms", lambda code: None)
    for _ in range(verify.MAX_ATTEMPTS):
        verify.check_code(fid, "sms", "000000")
    with pytest.raises(verify.VerifyError):
        verify.check_code(fid, "sms", "000000")


def test_resend_cooldown(monkeypatch):
    verify, fid = _setup(monkeypatch)
    verify.start_verification(fid, "sms", lambda code: None)
    with pytest.raises(verify.VerifyError):
        verify.start_verification(fid, "sms", lambda code: None)


def test_bad_channel(monkeypatch):
    verify, fid = _setup(monkeypatch)
    with pytest.raises(verify.VerifyError):
        verify.start_verification(fid, "carrier-pigeon", lambda code: None)
