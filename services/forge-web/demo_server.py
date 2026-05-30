"""Mainstay Forge — local demo/screenshot harness.

Stands up an ISOLATED Forge API + serves the dashboard on 127.0.0.1:8099.
Never touches alfred.service (:8000) or the real forge.db — uses a throwaway DB.
Auth is overridden for the demo so the dashboard loads without a login.
Seeds a realistic queue so screenshots show real content.

Run: python services/forge-web/demo_server.py
"""
import os
import sys
import time
from pathlib import Path

# Make repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Throwaway DB BEFORE importing forge modules (db path is read at call time).
_DB = "/tmp/forge_demo.db"
for _ext in ("", "-wal", "-shm"):
    try:
        os.remove(_DB + _ext)
    except FileNotFoundError:
        pass
os.environ["FORGE_DB_PATH"] = _DB

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from core.api.forge import register as register_forge
from core.security.auth import require_auth
from core.forge import jobs as forge_jobs

HERE = Path(__file__).resolve().parent

app = FastAPI(title="Mainstay Forge — demo")
register_forge(app)
# Demo-only: bypass login so the dashboard renders with data.
app.dependency_overrides[require_auth] = lambda: {"username": "demo", "role": "admin"}


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


def _seed():
    now = int(time.time())
    rows = [
        ("leak_graphic", "done",    360, {"caption": "Rod wave dropping???", "variations": 18}),
        ("kinetic_lyric", "done",   240, {"caption": "I ain't mad at you", "variations": 24}),
        ("film_montage", "running",  95, {"caption": "while I still can't get over you", "variations": 18}),
        ("kinetic_lyric", "pending", 35, {"caption": "still waiting for that new album", "variations": 12}),
        ("leak_graphic", "pending",  12, {"caption": "Don't Look Down — full tracklist", "variations": 20}),
    ]
    for job_type, status, age, params in rows:
        jid = forge_jobs.enqueue(job_type, params, now=now - age)
        if status == "done":
            forge_jobs._update(jid, status="done",
                               result='{"delivered": "Content/Mainstay-RodWave/Viral Album Videos/Processed"}',
                               now=now - age + 30)
        elif status != "pending":
            forge_jobs._update(jid, status=status, now=now - age + 10)


if __name__ == "__main__":
    forge_jobs.init_db()
    _seed()
    print(f"Forge demo on http://127.0.0.1:8099  (db={_DB})")
    uvicorn.run(app, host="127.0.0.1", port=8099, log_level="warning")
