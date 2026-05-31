"""Mainstay Forge — job API router. Wired via register(app) in core/api/main.py."""
from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, Response, UploadFile

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

    @app.get("/forge/library")
    async def library_index(user: dict = Depends(require_auth)):
        from core.forge import library
        return {"jobs": library.list_done_jobs()}

    @app.get("/forge/library/files")
    async def library_files(dir: str = Query(...), user: dict = Depends(require_auth)):
        from core.forge import library
        try:
            return {"files": library.list_dir_files(dir)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/forge/library/file")
    async def library_file(path: str = Query(...), user: dict = Depends(require_auth)):
        from core.forge import library
        try:
            data, ctype = library.read_file(path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return Response(content=data, media_type=ctype)

    @app.post("/forge/uploads")
    async def create_upload(file: UploadFile = File(...), user: dict = Depends(require_auth)):
        from core.forge import uploads
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty upload")
        uid = uploads.save_upload(content, file.filename or "upload.bin")
        return {"upload_id": uid, "filename": file.filename}

    @app.get("/forge/distribution/accounts")
    async def dist_accounts(user: dict = Depends(require_auth)):
        from core.forge import distribution
        return {"accounts": distribution.get_accounts()}

    @app.post("/forge/distribution/accounts")
    async def dist_set_accounts(payload: dict = Body(...), user: dict = Depends(require_auth)):
        from core.forge import distribution
        return {"accounts": distribution.set_accounts(payload.get("accounts") or [])}

    @app.get("/forge/distribution/pack")
    async def dist_pack(job_id: str = Query(...), user: dict = Depends(require_auth)):
        from core.forge import distribution
        return distribution.build_pack(job_id)

    @app.post("/forge/distribution/posted")
    async def dist_posted(payload: dict = Body(...), user: dict = Depends(require_auth)):
        from core.forge import distribution
        pid = payload.get("post_id")
        if not pid:
            raise HTTPException(status_code=400, detail="post_id required")
        distribution.mark_posted(pid, bool(payload.get("posted", True)))
        return {"ok": True}
