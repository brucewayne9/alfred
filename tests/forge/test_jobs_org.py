import pytest


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs as _jobs
    _jobs.init_db()
    return _jobs


def test_enqueue_stamps_org(forge):
    jid = forge.enqueue("echo", {"x": 1}, org="rucktalk")
    assert forge.get_job(jid)["org_id"] == "rucktalk"


def test_list_jobs_filters_by_org(forge):
    forge.enqueue("echo", {}, org="mainstay")
    forge.enqueue("echo", {}, org="rucktalk")
    assert len(forge.list_jobs(org="rucktalk")) == 1
    assert len(forge.list_jobs(org=None)) == 2
