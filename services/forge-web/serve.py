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
from fastapi import FastAPI
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
# Surface is gated by Caddy basic_auth; service is bound to localhost only.
app.dependency_overrides[require_auth] = lambda: {"username": "mainstay", "role": "team"}


@app.on_event("startup")
async def _start_worker():
    import asyncio
    # Any job left 'running' belongs to a worker that died with the previous
    # process — fail those ghosts so the queue reflects reality.
    n = forge_jobs.reconcile_orphans()
    if n:
        print(f"[forge] reconciled {n} orphaned running job(s) on startup")
    asyncio.create_task(forge_jobs.worker_loop(poll_interval=2.0))


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
