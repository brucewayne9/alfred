import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import db as _db
    _db.init_db()
    return _db


def test_create_source_stamps_org(db):
    from core.forge import ingest
    sid = ingest.create_source("url", "https://x", None, org="rucktalk")
    assert ingest.get_source(sid)["org_id"] == "rucktalk"


def test_create_source_defaults_to_mainstay(db):
    from core.forge import ingest
    sid = ingest.create_source("url", "https://x", None)
    assert ingest.get_source(sid)["org_id"] == "mainstay"


def test_list_sources_filters_by_org(db):
    from core.forge import ingest
    ingest.create_source("url", "a", None, org="mainstay")
    ingest.create_source("url", "b", None, org="rucktalk")
    mainstay = [s for s in ingest.list_sources(org="mainstay")]
    rucktalk = [s for s in ingest.list_sources(org="rucktalk")]
    assert len(mainstay) == 1 and mainstay[0]["spec"] == "a"
    assert len(rucktalk) == 1 and rucktalk[0]["spec"] == "b"


def test_list_sources_org_none_returns_all(db):
    from core.forge import ingest
    ingest.create_source("url", "a", None, org="mainstay")
    ingest.create_source("url", "b", None, org="rucktalk")
    assert len(ingest.list_sources(org=None)) == 2
