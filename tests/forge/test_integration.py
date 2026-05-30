import pytest


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs
    jobs._HANDLERS.clear()
    return jobs


def test_register_default_handlers_adds_echo(forge):
    from core.forge.handlers import register_default_handlers
    register_default_handlers()
    assert "echo" in forge._HANDLERS


def test_echo_job_runs_through_claim_and_execute(forge):
    from core.forge.handlers import register_default_handlers
    register_default_handlers()
    job_id = forge.enqueue("echo", {"hello": "world"}, now=1)
    claimed = forge.claim_next_pending(now=2)
    assert claimed["id"] == job_id
    assert claimed["status"] == "running"
    done = forge._execute(job_id, now=3)
    assert done["status"] == "done"
    assert done["result"] == {"echo": {"hello": "world"}}
