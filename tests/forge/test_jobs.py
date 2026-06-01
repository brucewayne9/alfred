import pytest


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs
    # clear the module-level handler registry between tests
    jobs._HANDLERS.clear()
    return jobs


def test_enqueue_creates_pending_job(forge):
    job_id = forge.enqueue("echo", {"msg": "hi"}, now=1000)
    job = forge.get_job(job_id)
    assert job["status"] == "pending"
    assert job["job_type"] == "echo"
    assert job["params"] == {"msg": "hi"}
    assert job["created_at"] == 1000


def test_get_job_returns_none_for_unknown(forge):
    assert forge.get_job("nope") is None


def test_list_jobs_filters_by_status(forge):
    a = forge.enqueue("echo", {}, now=1)
    b = forge.enqueue("echo", {}, now=2)
    forge.register_handler("echo", lambda p: {"ok": True})
    forge.run_job(a, now=3)  # -> done
    pending = forge.list_jobs(status="pending")
    done = forge.list_jobs(status="done")
    assert [j["id"] for j in pending] == [b]
    assert [j["id"] for j in done] == [a]


def test_run_job_runs_handler_and_stores_result(forge):
    forge.register_handler("echo", lambda params: {"echo": params})
    job_id = forge.enqueue("echo", {"x": 1}, now=10)
    result = forge.run_job(job_id, now=20)
    assert result["status"] == "done"
    assert result["result"] == {"echo": {"x": 1}}
    assert result["error"] is None
    assert result["updated_at"] == 20


def test_run_job_records_handler_exception(forge):
    def boom(params):
        raise ValueError("kaboom")

    forge.register_handler("bad", boom)
    job_id = forge.enqueue("bad", {}, now=10)
    result = forge.run_job(job_id, now=20)
    assert result["status"] == "error"
    assert "kaboom" in result["error"]


def test_run_job_errors_when_no_handler(forge):
    job_id = forge.enqueue("unregistered", {}, now=10)
    result = forge.run_job(job_id, now=20)
    assert result["status"] == "error"
    assert "no handler" in result["error"]


def test_claim_next_pending_marks_running_and_is_fifo(forge):
    first = forge.enqueue("echo", {}, now=1)
    forge.enqueue("echo", {}, now=2)
    claimed = forge.claim_next_pending(now=5)
    assert claimed["id"] == first
    assert claimed["status"] == "running"


def test_claim_next_pending_returns_none_when_empty(forge):
    assert forge.claim_next_pending(now=5) is None


def test_list_jobs_on_fresh_db_returns_empty(forge):
    assert forge.list_jobs() == []


# ── cancel / delete / reconcile ───────────────────────────────────────────

def test_cancel_pending_marks_cancelled(forge):
    jid = forge.enqueue("echo", {}, now=1)
    out = forge.cancel_job(jid, now=2)
    assert out["status"] == "cancelled"
    assert forge.get_job(jid)["status"] == "cancelled"


def test_cancel_running_requests_cancellation(forge):
    jid = forge.enqueue("echo", {}, now=1)
    forge._update(jid, status="running", now=2)
    out = forge.cancel_job(jid, now=3)
    assert out["status"] == "cancelling"
    assert forge.is_cancel_requested(jid)


def test_cancel_unknown_returns_none(forge):
    assert forge.cancel_job("nope") is None


def test_cancel_done_job_is_noop(forge):
    forge.register_handler("echo", lambda p: {"ok": True})
    jid = forge.enqueue("echo", {}, now=1)
    forge.run_job(jid, now=2)
    out = forge.cancel_job(jid, now=3)
    assert out["status"] == "done"


def test_delete_job_removes_row(forge):
    jid = forge.enqueue("echo", {}, now=1)
    assert forge.delete_job(jid) is True
    assert forge.get_job(jid) is None
    assert forge.delete_job(jid) is False  # already gone


def test_reconcile_orphans_fails_running_jobs(forge):
    a = forge.enqueue("echo", {}, now=1)
    b = forge.enqueue("echo", {}, now=2)
    forge._update(a, status="running", now=3)
    forge._update(b, status="pending", now=3)
    n = forge.reconcile_orphans(now=10)
    assert n == 1
    ja = forge.get_job(a)
    assert ja["status"] == "error"
    assert "interrupted" in (ja["error"] or "")
    assert forge.get_job(b)["status"] == "pending"  # untouched


def test_execute_marks_cancelled_when_requested_mid_run(forge):
    # Handler that requests its own cancellation while running.
    def slow(params):
        forge.request_cancel(forge.current_job_id())
        return {"ok": True}

    forge.register_handler("slow", slow)
    jid = forge.enqueue("slow", {}, now=1)
    forge._update(jid, status="running", now=2)
    forge._execute(jid, now=3)
    assert forge.get_job(jid)["status"] == "cancelled"
