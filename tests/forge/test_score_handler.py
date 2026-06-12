import pytest


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs, db
    db.init_db()
    jobs._HANDLERS.clear()
    return jobs


def test_score_source_handler_is_registered(forge):
    from core.forge.handlers import register_default_handlers
    register_default_handlers()
    assert "score_source" in forge._HANDLERS


def test_score_source_handler_scores_and_summarises(forge, monkeypatch):
    from core.forge import handlers, scorer
    monkeypatch.setattr(scorer, "score_source", lambda sid, **kw: [
        {"score": 92}, {"score": 71},
    ])
    handlers.register_default_handlers()
    jid = forge.enqueue("score_source", {"source_id": "src1"}, now=1)
    out = forge.run_job(jid, now=2)
    assert out["status"] == "done"
    assert out["result"]["source_id"] == "src1"
    assert out["result"]["candidates"] == 2
    assert out["result"]["top_score"] == 92


def test_score_source_handler_requires_source_id(forge, monkeypatch):
    from core.forge import handlers, scorer
    monkeypatch.setattr(scorer, "score_source", lambda sid, **kw: [])
    handlers.register_default_handlers()
    jid = forge.enqueue("score_source", {}, now=1)
    out = forge.run_job(jid, now=2)
    assert out["status"] == "error"
    assert "source_id" in out["error"]


def test_transcribe_handler_auto_enqueues_a_score_job_on_success(forge, monkeypatch):
    from core.forge import handlers, ingest
    monkeypatch.setattr(ingest, "transcribe_handler",
                        lambda params: {"ok": True, "segments": 12})
    out = handlers._ingest_transcribe_handler({"source_id": "src9"})
    assert out["ok"] is True
    pending = forge.list_jobs(status="pending")
    score_jobs = [j for j in pending if j["job_type"] == "score_source"]
    assert len(score_jobs) == 1
    assert score_jobs[0]["params"]["source_id"] == "src9"
