import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def _cand():
    return {"start_s": 1.0, "end_s": 9.0, "score": 80, "hook": "h",
            "emotion": "e", "reason": "r", "caption": "c"}


def test_save_candidates_stamps_org(db):
    from core.forge import scorer
    scorer.save_candidates("src1", [_cand()], org="rucktalk")
    rows = scorer.get_candidates("src1")
    assert rows and rows[0]["org_id"] == "rucktalk"


def test_get_candidate_exposes_org_for_ownership_check(db):
    from core.forge import scorer
    scorer.save_candidates("src1", [_cand()], org="rucktalk")
    cid = scorer.get_candidates("src1")[0]["id"]
    assert scorer.get_candidate(cid)["org_id"] == "rucktalk"
