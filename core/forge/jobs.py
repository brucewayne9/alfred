"""Mainstay Forge — job queue, handler registry, and async worker.

GPU work is serialized: a single worker_loop claims one pending job at a time
and runs its handler in a thread executor. No Celery/RQ — matches the repo's
asyncio-only job pattern (see core/audit/lead_router.py).
"""
import asyncio
import json
import time
import uuid
from typing import Callable, Optional

from core.forge.db import _conn, init_db

# job_type -> handler(params: dict) -> dict
_HANDLERS: dict[str, Callable[[dict], dict]] = {}


def register_handler(job_type: str, fn: Callable[[dict], dict]) -> None:
    _HANDLERS[job_type] = fn


def _now() -> int:
    return int(time.time())


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.get("params") or "{}")
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d


def enqueue(job_type: str, params: Optional[dict] = None, now: Optional[int] = None) -> str:
    init_db()
    job_id = uuid.uuid4().hex
    ts = now if now is not None else _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO jobs (id, job_type, status, params, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?)",
            (job_id, job_type, json.dumps(params or {}), ts, ts),
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
    try:
        result = handler(job["params"])
        _update(job_id, status="done", result=json.dumps(result or {}), error=None, now=now)
    except Exception as e:  # noqa: BLE001 — record any handler failure
        _update(job_id, status="error", error=str(e), now=now)
    return get_job(job_id)


def run_job(job_id: str, now: Optional[int] = None) -> dict:
    """Synchronous run: mark running, then execute the handler. Used by tests.

    Caller is responsible for only invoking this on a 'pending' job — it does not
    guard against re-running an already running/done job (would re-execute the handler).
    """
    _update(job_id, status="running", now=now)
    return _execute(job_id, now=now)


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
