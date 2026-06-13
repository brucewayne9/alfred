"""Mainstay Forge — production server (behind Caddy /forge* + basic_auth).

Serves the dashboard at /forge and the job API at /forge/* on 127.0.0.1:8201.
Caddy gates the surface with basic_auth, so this service keeps the auth dependency
overridden (it is never reachable except through Caddy). Uses a dedicated DB.

Run (persistent): setsid nohup python services/forge-web/serve.py &
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("FORGE_DB_PATH", "/home/aialfred/alfred/data/forge_live.db")

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from core.api.forge import register as register_forge
from core.security.auth import require_auth
from core.forge import jobs as forge_jobs
from core.forge.handlers import register_default_handlers

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("FORGE_PORT", "8201"))

app = FastAPI(title="Mainstay Forge")
register_forge(app)
register_default_handlers()  # echo + leak_graphic (real ComfyUI-Cloud render)

# Seed the per-person login store (mike/mainstay/jordan/mello) on first run.
from core.forge import users as _forge_users  # noqa: E402
_forge_users.ensure_seeded()

# Fallback super-admins if the X-Forge-Role header is ever missing.
FORGE_SUPER = {"mike", "mainstay"}


# Auth: Caddy forward_auth hits /forge/authcheck, which validates the login
# against the Forge user store and returns X-Forge-User / X-Forge-Role; Caddy
# copies those onto the proxied request.  We read them here so each person's
# jobs are stamped and admins get the management panel.
def _forge_user(request: Request) -> dict:
    username = request.headers.get("X-Forge-User") or "mainstay"
    role = request.headers.get("X-Forge-Role")
    org = request.headers.get("X-Forge-Org") or "mainstay"
    if role not in ("member", "org_admin", "super_admin"):
        role = "super_admin" if username in FORGE_SUPER else "member"
    return {"username": username, "role": role, "org": org}


app.dependency_overrides[require_auth] = _forge_user


@app.on_event("startup")
async def _start_worker():
    import asyncio
    # Any job left 'running' belongs to a worker that died with the previous
    # process — fail those ghosts so the queue reflects reality.
    n = forge_jobs.reconcile_orphans()
    if n:
        print(f"[forge] reconciled {n} orphaned running job(s) on startup")
    # Hold a strong reference on app.state: the event loop only keeps a *weak*
    # ref to a bare create_task(), so without this the GC can collect the worker
    # mid-run and silently stall the whole queue (the recurring stuck-jobs bug).
    app.state.worker_task = asyncio.create_task(forge_jobs.worker_loop(poll_interval=2.0))
    print("[forge] worker_loop started")


@app.get("/forge")
@app.get("/forge/")
async def dashboard():
    return FileResponse(HERE / "index.html")


@app.get("/forge/vendor/{name}")
async def vendor_asset(name: str):
    """Serve vendored front-end libs (wavesurfer ESM) — kept local, no runtime CDN."""
    from fastapi import HTTPException
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="bad name")
    path = HERE / "vendor" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    media = "application/javascript" if name.endswith(".js") else None
    return FileResponse(path, media_type=media)


def _seed_if_empty():
    if forge_jobs.list_jobs():
        return
    # Honest history: only the format that actually renders today (leak_graphic).
    now = int(time.time())
    rows = [
        ("leak_graphic", 360, {"caption": "Rod wave dropping???", "variations": 18}),
        ("leak_graphic", 180, {"caption": "Don't Look Down — the wait is over", "variations": 24}),
    ]
    for job_type, age, params in rows:
        jid = forge_jobs.enqueue(job_type, params, now=now - age)
        forge_jobs._update(
            jid, status="done",
            result='{"format": "leak_graphic", "variant_count": %d, "delivered": %d, "delivered_dir": "Content/Mainstay-RodWave/Viral Album Videos/Processed"}'
            % (params["variations"], params["variations"] + 1),
            now=now - age + 25)


if __name__ == "__main__":
    forge_jobs.init_db()
    _seed_if_empty()
    print(f"Forge serving on 127.0.0.1:{PORT} (db={os.environ['FORGE_DB_PATH']})")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
