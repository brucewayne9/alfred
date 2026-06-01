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

    @app.post("/forge/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, user: dict = Depends(require_auth)):
        job = forge_jobs.cancel_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.delete("/forge/jobs/{job_id}")
    async def delete_job(job_id: str, user: dict = Depends(require_auth)):
        if not forge_jobs.delete_job(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return {"ok": True, "deleted": job_id}

    @app.get("/forge/library")
    async def library_index(user: dict = Depends(require_auth)):
        from core.forge import library
        return {"jobs": library.list_done_jobs(), "undo": library.trash_state()}

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

    @app.delete("/forge/library/file")
    async def library_delete(path: str = Query(...), user: dict = Depends(require_auth)):
        """Soft-delete a single file: move to trash (recoverable via undo)."""
        from core.forge import library
        label = (path or "").rstrip("/").split("/")[-1]
        try:
            res = library.soft_delete([path], kind="file", label=label)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, "undo": library.trash_state(), **res}

    @app.post("/forge/library/batch-delete")
    async def library_batch_delete(payload: dict = Body(...), user: dict = Depends(require_auth)):
        """Soft-delete a whole batch (its dirs) as one undoable action; hide the card."""
        from core.forge import library
        dirs = payload.get("dirs") or []
        if not dirs:
            raise HTTPException(status_code=400, detail="dirs required")
        try:
            res = library.soft_delete(
                dirs, kind="batch",
                job_id=payload.get("job_id") or None,
                label=payload.get("label") or "batch",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, "undo": library.trash_state(), **res}

    @app.post("/forge/library/undo")
    async def library_undo(user: dict = Depends(require_auth)):
        """Restore the most recent delete action."""
        from core.forge import library
        restored = library.undo_last()
        if restored is None:
            raise HTTPException(status_code=404, detail="nothing to undo")
        return {"ok": True, "restored": restored, "undo": library.trash_state()}

    @app.post("/forge/uploads")
    async def create_upload(file: UploadFile = File(...), user: dict = Depends(require_auth)):
        from core.forge import uploads, ingest
        from core.forge import jobs as _forge_jobs
        uid = await uploads.save_upload_stream(file, file.filename or "upload.bin")
        path = uploads.get_upload_path(uid)
        if path is None or path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="empty upload")
        source_id = ingest.create_source("upload", file.filename or "upload.bin", str(path))
        job_id = _forge_jobs.enqueue("ingest_transcribe", {"source_id": source_id})
        return {
            "upload_id": uid,
            "source_id": source_id,
            "job_id": job_id,
            "filename": file.filename,
        }

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

    @app.post("/forge/distribution/postiz")
    async def dist_postiz(payload: dict = Body(...), user: dict = Depends(require_auth)):
        from core.forge import distribution
        job_id = payload.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id required")
        return distribution.push_to_postiz(job_id)
