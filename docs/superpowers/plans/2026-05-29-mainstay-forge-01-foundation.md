# Mainstay Forge — Plan 1: Foundation (Backend Spine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend spine of Mainstay Forge — a SQLite-backed GPU job queue with a background worker, a Nextcloud delivery helper, and an authenticated job API — so any later subsystem (Creation, Multiplication, etc.) can enqueue a render and have it run, track status, and deliver the artifact.

**Architecture:** A new `core/forge/` Python package owns the job store (`db.py`), the queue + worker (`jobs.py`), and delivery (`delivery.py`). A `core/api/forge.py` router exposes job endpoints and is wired into the existing FastAPI app via the established `register(app)` pattern. Render work is GPU-serial (one job at a time) — a single async `worker_loop` polls SQLite, claims the next pending job, and runs its registered handler in a thread executor. This matches the repo's existing patterns: `sqlite3` like `core/api/arcade_scores.py`, `asyncio` jobs like `core/audit/lead_router.py` (no Celery/RQ in this codebase), and `require_auth` from `core/security/auth.py`.

**Tech Stack:** Python 3, FastAPI, sqlite3 (WAL), asyncio, pytest + FastAPI `TestClient`, existing `integrations/nextcloud/client.py`.

**Reference spec:** `docs/superpowers/specs/2026-05-29-mainstay-forge-design.md` (§4.7 render workers, §4.5 delivery, §6 data model).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `core/forge/__init__.py` | Package marker |
| `core/forge/db.py` | SQLite connection + schema (the `jobs` table). Owns path resolution + `init_db()`. |
| `core/forge/jobs.py` | Queue logic: enqueue / get / list / claim / execute, handler registry, async `worker_loop`. |
| `core/forge/delivery.py` | Deliver a local artifact to the team's Nextcloud folders. |
| `core/api/forge.py` | FastAPI router (`register(app)`): `/forge/health`, `/forge/jobs` CRUD. |
| `core/api/main.py` | MODIFY: import + call `register` for forge; start `worker_loop` in lifespan; register demo `echo` handler. |
| `tests/forge/test_db.py` | Schema + connection tests. |
| `tests/forge/test_jobs.py` | Queue + handler + execute tests. |
| `tests/forge/test_delivery.py` | Delivery helper tests (Nextcloud mocked). |
| `tests/forge/test_api.py` | Endpoint tests (auth overridden, isolated app). |

**Test isolation:** every DB-touching test sets `FORGE_DB_PATH` to a `tmp_path` file via `monkeypatch.setenv` so tests never touch `data/forge.db`.

---

### Task 1: Forge package + job store (`db.py`)

**Files:**
- Create: `core/forge/__init__.py`
- Create: `core/forge/db.py`
- Test: `tests/forge/__init__.py`, `tests/forge/test_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/forge/__init__.py` (empty file) and `tests/forge/test_db.py`:

```python
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def forge_db(tmp_path, monkeypatch):
    db_file = tmp_path / "forge.db"
    monkeypatch.setenv("FORGE_DB_PATH", str(db_file))
    from core.forge import db
    db.init_db()
    return db_file


def test_db_path_honors_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "forge.db"
    monkeypatch.setenv("FORGE_DB_PATH", str(target))
    from core.forge import db
    assert db._db_path() == target


def test_init_db_creates_jobs_table(forge_db):
    conn = sqlite3.connect(str(forge_db))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
    ).fetchall()
    conn.close()
    assert rows == [("jobs",)]


def test_init_db_is_idempotent(forge_db):
    from core.forge import db
    db.init_db()  # second call must not raise
    db.init_db()


def test_jobs_table_has_expected_columns(forge_db):
    conn = sqlite3.connect(str(forge_db))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    conn.close()
    assert cols == {
        "id", "job_type", "status", "params",
        "result", "error", "created_at", "updated_at",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.forge'`

- [ ] **Step 3: Write minimal implementation**

Create `core/forge/__init__.py` (empty file).

Create `core/forge/db.py`:

```python
"""Mainstay Forge — job store (SQLite, WAL). Mirrors core/api/arcade_scores.py."""
import os
import sqlite3
from pathlib import Path


def _db_path() -> Path:
    override = os.environ.get("FORGE_DB_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "data" / "forge.db"


def _conn() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                job_type    TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                params      TEXT NOT NULL DEFAULT '{}',
                result      TEXT,
                error       TEXT,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status
                ON jobs(status, created_at);
            """
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_db.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add core/forge/__init__.py core/forge/db.py tests/forge/__init__.py tests/forge/test_db.py
git commit -m "feat(forge): job store schema + sqlite connection (Plan 1 Task 1)"
```

---

### Task 2: Queue logic + handler registry (`jobs.py`)

**Files:**
- Create: `core/forge/jobs.py`
- Test: `tests/forge/test_jobs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/forge/test_jobs.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_jobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.forge.jobs'`

- [ ] **Step 3: Write minimal implementation**

Create `core/forge/jobs.py`:

```python
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
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_jobs(status: Optional[str] = None, limit: int = 100) -> list[dict]:
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
    """Synchronous run (mark running, execute handler). Used by tests + worker."""
    _update(job_id, status="running", now=now)
    return _execute(job_id, now=now)


async def worker_loop(poll_interval: float = 2.0) -> None:
    """Background loop: claim one pending job at a time, run it off the event loop."""
    init_db()
    loop = asyncio.get_event_loop()
    while True:
        job = claim_next_pending()
        if job is None:
            await asyncio.sleep(poll_interval)
            continue
        await loop.run_in_executor(None, _execute, job["id"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_jobs.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add core/forge/jobs.py tests/forge/test_jobs.py
git commit -m "feat(forge): job queue, handler registry, async worker loop (Plan 1 Task 2)"
```

---

### Task 3: Nextcloud delivery helper (`delivery.py`)

**Files:**
- Create: `core/forge/delivery.py`
- Test: `tests/forge/test_delivery.py`

**Context:** `integrations/nextcloud/client.py` exposes `create_folder(path) -> bool` (WebDAV MKCOL) and `upload_file(local_path: Path, remote_path: str) -> bool` (WebDAV PUT). The team's delivery root `/Content/Mainstay-RodWave/{Viral Album Videos,Podcast & Episode Clips,Ideas}` already exists. MKCOL creates only one level — parents must exist, which they do for our known subfolders.

- [ ] **Step 1: Write the failing test**

Create `tests/forge/test_delivery.py`:

```python
from pathlib import Path

import pytest


def test_deliver_creates_folder_and_uploads(monkeypatch, tmp_path):
    calls = {"folders": [], "uploads": []}

    def fake_create_folder(path):
        calls["folders"].append(path)
        return True

    def fake_upload_file(local_path, remote_path):
        calls["uploads"].append((str(local_path), remote_path))
        return True

    monkeypatch.setattr("core.forge.delivery.create_folder", fake_create_folder)
    monkeypatch.setattr("core.forge.delivery.upload_file", fake_upload_file)

    from core.forge import delivery
    local = tmp_path / "clip.mp4"
    local.write_bytes(b"x")
    remote = delivery.deliver(local, "Viral Album Videos/Processed")

    assert remote == "/Content/Mainstay-RodWave/Viral Album Videos/Processed/clip.mp4"
    assert calls["folders"] == ["/Content/Mainstay-RodWave/Viral Album Videos/Processed"]
    assert calls["uploads"] == [(str(local), remote)]


def test_deliver_honors_explicit_filename(monkeypatch, tmp_path):
    monkeypatch.setattr("core.forge.delivery.create_folder", lambda p: True)
    captured = {}
    monkeypatch.setattr(
        "core.forge.delivery.upload_file",
        lambda lp, rp: captured.setdefault("remote", rp) or True,
    )
    from core.forge import delivery
    local = tmp_path / "raw.mp4"
    local.write_bytes(b"x")
    remote = delivery.deliver(local, "Ideas", filename="renamed.mp4")
    assert remote.endswith("/Ideas/renamed.mp4")
    assert captured["remote"].endswith("/Ideas/renamed.mp4")


def test_deliver_raises_on_upload_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("core.forge.delivery.create_folder", lambda p: True)
    monkeypatch.setattr("core.forge.delivery.upload_file", lambda lp, rp: False)
    from core.forge import delivery
    local = tmp_path / "f.mp4"
    local.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        delivery.deliver(local, "Ideas")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_delivery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.forge.delivery'`

- [ ] **Step 3: Write minimal implementation**

Create `core/forge/delivery.py`:

```python
"""Mainstay Forge — deliver rendered artifacts to the team's Nextcloud folders."""
from pathlib import Path

from integrations.nextcloud.client import create_folder, upload_file

DELIVERY_ROOT = "/Content/Mainstay-RodWave"


def deliver(local_path: Path, subfolder: str, filename: str | None = None) -> str:
    """Upload `local_path` into DELIVERY_ROOT/subfolder. Returns the remote path.

    Raises RuntimeError if the upload fails.
    """
    local_path = Path(local_path)
    remote_dir = f"{DELIVERY_ROOT}/{subfolder}".rstrip("/")
    create_folder(remote_dir)  # idempotent; parents already exist for known subfolders
    name = filename or local_path.name
    remote_path = f"{remote_dir}/{name}"
    if not upload_file(local_path, remote_path):
        raise RuntimeError(f"Nextcloud upload failed: {remote_path}")
    return remote_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_delivery.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add core/forge/delivery.py tests/forge/test_delivery.py
git commit -m "feat(forge): Nextcloud delivery helper (Plan 1 Task 3)"
```

---

### Task 4: Job API router (`core/api/forge.py`)

**Files:**
- Create: `core/api/forge.py`
- Test: `tests/forge/test_api.py`

**Context:** Follow the repo router pattern — a module-level `register(app: FastAPI)` that defines endpoints. Protect mutating/listing endpoints with `require_auth` from `core/security/auth.py`. Tests build an isolated `FastAPI()` app (not the 332KB `main.py`) and override the auth dependency.

- [ ] **Step 1: Write the failing test**

Create `tests/forge/test_api.py`:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security.auth import require_auth


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.api import forge
    app = FastAPI()
    forge.register(app)
    app.dependency_overrides[require_auth] = lambda: {"username": "tester", "role": "admin"}
    return TestClient(app)


def test_health_is_public(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.api import forge
    app = FastAPI()
    forge.register(app)
    c = TestClient(app)
    r = c.get("/forge/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_job_returns_pending(client):
    r = client.post("/forge/jobs", json={"job_type": "echo", "params": {"a": 1}})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["job_type"] == "echo"
    assert body["params"] == {"a": 1}


def test_create_job_requires_job_type(client):
    r = client.post("/forge/jobs", json={"params": {}})
    assert r.status_code == 400


def test_get_job_roundtrip(client):
    created = client.post("/forge/jobs", json={"job_type": "echo"}).json()
    r = client.get(f"/forge/jobs/{created['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]


def test_get_unknown_job_404(client):
    assert client.get("/forge/jobs/does-not-exist").status_code == 404


def test_list_jobs_and_status_filter(client):
    client.post("/forge/jobs", json={"job_type": "echo"})
    client.post("/forge/jobs", json={"job_type": "echo"})
    r = client.get("/forge/jobs?status=pending")
    assert r.status_code == 200
    assert len(r.json()["jobs"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'forge' from 'core.api'`

- [ ] **Step 3: Write minimal implementation**

Create `core/api/forge.py`:

```python
"""Mainstay Forge — job API router. Wired via register(app) in core/api/main.py."""
from fastapi import Body, Depends, FastAPI, HTTPException

from core.forge import jobs as forge_jobs
from core.security.auth import require_auth


def register(app: FastAPI) -> None:
    @app.get("/forge/health")
    async def forge_health():
        return {"status": "ok", "service": "mainstay-forge"}

    @app.post("/forge/jobs")
    async def create_job(payload: dict = Body(...), user: dict = Depends(require_auth)):
        job_type = payload.get("job_type")
        if not job_type:
            raise HTTPException(status_code=400, detail="job_type is required")
        job_id = forge_jobs.enqueue(job_type, payload.get("params") or {})
        return forge_jobs.get_job(job_id)

    @app.get("/forge/jobs")
    async def list_jobs(status: str | None = None, user: dict = Depends(require_auth)):
        return {"jobs": forge_jobs.list_jobs(status=status)}

    @app.get("/forge/jobs/{job_id}")
    async def get_job(job_id: str, user: dict = Depends(require_auth)):
        job = forge_jobs.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_api.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add core/api/forge.py tests/forge/test_api.py
git commit -m "feat(forge): authenticated job API router (Plan 1 Task 4)"
```

---

### Task 5: Wire into the app + prove end-to-end (`main.py`)

**Files:**
- Modify: `core/api/main.py` (router registration block ~lines 154-189; lifespan ~lines 64-100)
- Test: `tests/forge/test_integration.py`

**Goal:** Register the forge router in the real app, start `worker_loop` in the lifespan, and register a demo `echo` handler so a job runs end-to-end through the queue (not just `run_job`). The integration test exercises the queue + worker contract without standing up the whole app.

- [ ] **Step 1: Write the failing test**

Create `tests/forge/test_integration.py`:

```python
import pytest


@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_DB_PATH", str(tmp_path / "forge.db"))
    from core.forge import jobs
    jobs._HANDLERS.clear()
    return jobs


def test_register_forge_handlers_adds_echo(forge):
    from core.api.main import register_forge_handlers
    register_forge_handlers()
    assert "echo" in forge._HANDLERS


def test_echo_job_runs_through_claim_and_execute(forge):
    from core.api.main import register_forge_handlers
    register_forge_handlers()
    job_id = forge.enqueue("echo", {"hello": "world"}, now=1)
    claimed = forge.claim_next_pending(now=2)
    assert claimed["id"] == job_id
    assert claimed["status"] == "running"
    done = forge._execute(job_id, now=3)
    assert done["status"] == "done"
    assert done["result"] == {"echo": {"hello": "world"}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_integration.py -v`
Expected: FAIL — `ImportError: cannot import name 'register_forge_handlers' from 'core.api.main'`

- [ ] **Step 3: Write minimal implementation**

In `core/api/main.py`, near the other endpoint-module imports (around line 154), add:

```python
from core.api.forge import register as _register_forge
from core.forge import jobs as _forge_jobs


def register_forge_handlers() -> None:
    """Register Mainstay Forge job handlers. Demo 'echo' proves the queue end-to-end;
    real format renderers (kinetic-lyric, montage, leak-graphic) register here in later plans."""
    _forge_jobs.register_handler("echo", lambda params: {"echo": params})
```

In the router-registration block (around line 189, alongside `_register_arcade(app)` etc.), add:

```python
_register_forge(app)
register_forge_handlers()
```

In the `lifespan` async context manager (around line 64-100), after the existing startup work and **before** `yield`, add:

```python
    # Mainstay Forge — start the GPU job worker (one job at a time)
    import asyncio as _asyncio
    forge_worker = _asyncio.create_task(_forge_jobs.worker_loop())
```

And after `yield` (shutdown), add:

```python
    forge_worker.cancel()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/test_integration.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full forge suite + import-check the app**

Run: `cd /home/aialfred/alfred && python -m pytest tests/forge/ -v`
Expected: PASS (all forge tests, 23 total)

Run: `cd /home/aialfred/alfred && python -c "import core.api.main"`
Expected: no error (the app module imports cleanly with the new wiring)

- [ ] **Step 6: Commit**

```bash
git add core/api/main.py tests/forge/test_integration.py
git commit -m "feat(forge): wire job router + worker into app, echo handler proves queue (Plan 1 Task 5)"
```

---

## Self-Review

**Spec coverage (§ from the design spec):**
- §4.7 Render workers / job queue → Tasks 2 + 5 (queue, worker_loop, GPU-serial). ✓
- §4.5/§3 Delivery to Nextcloud → Task 3. ✓
- §6 Data model (`jobs` first; `projects/renders/variations/...` come with their subsystems) → Task 1 establishes the store + pattern. ✓
- Auth (mobile dashboard will call these) → Task 4 uses `require_auth`. ✓
- Foundation goal "any later subsystem can enqueue a render" → handler registry (Task 2) + `register_forge_handlers()` extension point (Task 5). ✓
- **Deferred (correctly out of this plan):** the format renderers, multiplication, distribution, intelligence, and the React dashboard shell — each its own plan. Plan 1b (frontend shell + Queue/Create tabs) is the immediate next plan.

**Placeholder scan:** No TBD/TODO; every step has full code and exact commands. ✓

**Type/name consistency:** `enqueue`, `get_job`, `list_jobs`, `claim_next_pending`, `run_job`, `_execute`, `register_handler`, `worker_loop`, `_HANDLERS`, `deliver`, `register` — names used identically across `jobs.py`, `delivery.py`, `forge.py`, `main.py`, and all tests. `init_db`/`_conn`/`_db_path` consistent in `db.py` and tests. ✓

**Notes for the executor:**
- Run everything from `/home/aialfred/alfred` so `core.*` imports resolve.
- If `python -m pytest` isn't found, use the repo venv: `/home/aialfred/alfred/venv/bin/python -m pytest`.
- These commits land on branch `feat/mainstay-forge` (created during brainstorming).
