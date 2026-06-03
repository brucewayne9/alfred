"""Mainstay Forge — job queue, handler registry, and async worker.

GPU work is serialized: a single worker_loop claims one pending job at a time
and runs its handler in a thread executor. No Celery/RQ — matches the repo's
asyncio-only job pattern (see core/audit/lead_router.py).
"""
import asyncio
import contextvars
import json
import time
import uuid
from typing import Callable, Optional

from core.forge.db import _conn, init_db

# job_type -> handler(params: dict) -> dict
_HANDLERS: dict[str, Callable[[dict], dict]] = {}

# job_ids for which cancellation has been requested (in-memory; survives a single
# process only — orphaned 'running' rows are reconciled on startup instead).
_CANCEL_REQUESTED: set[str] = set()


# Set for the duration of each handler run so handlers can checkpoint without
# threading the job_id through their signature (keeps params clean).
_CURRENT_JOB: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "forge_current_job", default=None
)


def current_job_id() -> Optional[str]:
    return _CURRENT_JOB.get()


class JobCancelled(Exception):
    """Raised by check_cancel() so a long handler can abort cooperatively."""


def register_handler(job_type: str, fn: Callable[[dict], dict]) -> None:
    _HANDLERS[job_type] = fn


def request_cancel(job_id: str) -> None:
    _CANCEL_REQUESTED.add(job_id)


def is_cancel_requested(job_id: str) -> bool:
    return job_id in _CANCEL_REQUESTED


def clear_cancel(job_id: str) -> None:
    _CANCEL_REQUESTED.discard(job_id)


def check_cancel(job_id: str | None = None) -> None:
    """Handlers call this at step boundaries; raises if a stop was requested.
    With no argument it uses the current job's id."""
    jid = job_id or _CURRENT_JOB.get()
    if jid and jid in _CANCEL_REQUESTED:
        raise JobCancelled(jid)


def _now() -> int:
    return int(time.time())


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.get("params") or "{}")
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d


def enqueue(job_type: str, params: Optional[dict] = None, now: Optional[int] = None,
            created_by: Optional[str] = None) -> str:
    init_db()
    job_id = uuid.uuid4().hex
    ts = now if now is not None else _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, job_type, status, params, created_by, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?, ?)",
            (job_id, job_type, json.dumps(params or {}), created_by, ts, ts),
        )
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    init_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    init_db()
    with _conn() as c:
        if status:
            rows = c.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _update(job_id: str, now: Optional[int] = None, **fields) -> None:
    fields["updated_at"] = now if now is not None else _now()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id = ?", vals)


def claim_next_pending(now: Optional[int] = None) -> Optional[dict]:
    init_db()
    ts = now if now is not None else _now()
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        cur = c.execute(
            "UPDATE jobs SET status = 'running', updated_at = ? "
            "WHERE id = ? AND status = 'pending'",
            (ts, row["id"]),
        )
        if cur.rowcount == 0:
            return None
    return get_job(row["id"])


def _execute(job_id: str, now: Optional[int] = None) -> dict:
    job = get_job(job_id)
    if job is None:
        raise ValueError(f"job {job_id} not found")
    handler = _HANDLERS.get(job["job_type"])
    if handler is None:
        _update(job_id, status="error", error=f"no handler for '{job['job_type']}'", now=now)
        return get_job(job_id)
    # Stopped before it even started running.
    if is_cancel_requested(job_id):
        _update(job_id, status="cancelled", error="cancelled before start", now=now)
        clear_cancel(job_id)
        return get_job(job_id)
    # Expose job_id to the handler (via contextvar) so long renders can checkpoint
    # with check_cancel() without polluting their params.
    token = _CURRENT_JOB.set(job_id)
    try:
        result = handler(job["params"])
        if is_cancel_requested(job_id):
            _update(job_id, status="cancelled", error="cancelled", now=now)
        else:
            _update(job_id, status="done", result=json.dumps(result or {}), error=None, now=now)
    except JobCancelled:
        _update(job_id, status="cancelled", error="cancelled", now=now)
    except Exception as e:  # noqa: BLE001 — record any handler failure
        status = "cancelled" if is_cancel_requested(job_id) else "error"
        _update(job_id, status=status, error=str(e), now=now)
    finally:
        clear_cancel(job_id)
        _CURRENT_JOB.reset(token)
    return get_job(job_id)


def run_job(job_id: str, now: Optional[int] = None) -> dict:
    """Synchronous run: mark running, then execute the handler. Used by tests.

    Caller is responsible for only invoking this on a 'pending' job — it does not
    guard against re-running an already running/done job (would re-execute the handler).
    """
    _update(job_id, status="running", now=now)
    return _execute(job_id, now=now)


def cancel_job(job_id: str, now: Optional[int] = None) -> Optional[dict]:
    """Stop a job. Pending -> cancelled instantly; running -> request cooperative
    cancel (handler halts at its next checkpoint). Finished jobs are left as-is."""
    init_db()
    job = get_job(job_id)
    if job is None:
        return None
    if job["status"] == "pending":
        _update(job_id, status="cancelled", error="cancelled", now=now)
    elif job["status"] in ("running", "cancelling"):
        request_cancel(job_id)
        _update(job_id, status="cancelling", now=now)
    return get_job(job_id)


def delete_job(job_id: str) -> bool:
    """Remove a job row entirely. If it's running, also request cancellation so the
    worker stops touching it. Returns False if the job didn't exist."""
    init_db()
    request_cancel(job_id)
    with _conn() as c:
        cur = c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cur.rowcount > 0


def reconcile_orphans(now: Optional[int] = None) -> int:
    """On startup, fail any job left 'running'/'cancelling' — its worker died with
    the previous process. Returns how many were reconciled."""
    init_db()
    ts = now if now is not None else _now()
    with _conn() as c:
        cur = c.execute(
            "UPDATE jobs SET status = 'error', error = 'interrupted (service restart)', "
            "updated_at = ? WHERE status IN ('running', 'cancelling')",
            (ts,),
        )
    return cur.rowcount


async def worker_loop(poll_interval: float = 2.0) -> None:
    """Background loop: claim one pending job at a time, run it off the event loop."""
    init_db()
    loop = asyncio.get_running_loop()
    while True:
        job = claim_next_pending()
        if job is None:
            await asyncio.sleep(poll_interval)
            continue
        await loop.run_in_executor(None, _execute, job["id"])
