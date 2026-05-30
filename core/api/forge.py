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
