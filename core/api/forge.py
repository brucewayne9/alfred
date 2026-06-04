"""Mainstay Forge — job API router. Wired via register(app) in core/api/main.py."""
import base64
import binascii

from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile

from core.forge import jobs as forge_jobs
from core.forge import users as forge_users
from core.security.auth import require_auth


def _require_admin(user: dict) -> None:
    if (user or {}).get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin only")


def register(app: FastAPI) -> None:
    @app.get("/forge/health")
    async def forge_health():
        return {"status": "ok", "service": "mainstay-forge"}

    @app.get("/forge/caption-styles")
    async def forge_caption_styles(user: dict = Depends(require_auth)):
        """Caption-style catalog for the visual gallery picker (all 3 formats)."""
        from core.forge import caption_styles
        return {"styles": caption_styles.list_styles(),
                "families": caption_styles.families(),
                "default": caption_styles.DEFAULT_STYLE}

    @app.get("/forge/authcheck")
    async def forge_authcheck(request: Request):
        """Caddy forward_auth target: validate Basic credentials against the
        Forge user store.  200 + X-Forge-User/Role on success; 401 otherwise.
        """
        unauth = Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Mainstay Forge"'},
        )
        hdr = request.headers.get("Authorization", "")
        if not hdr.startswith("Basic "):
            return unauth
        try:
            raw = base64.b64decode(hdr[6:]).decode("utf-8", "replace")
            username, _, password = raw.partition(":")
        except (binascii.Error, ValueError):
            return unauth
        u = forge_users.verify_user(username, password)
        if not u:
            return unauth
        return Response(
            status_code=200,
            headers={"X-Forge-User": u["username"], "X-Forge-Role": u["role"]},
        )

    # ---- User management (admin only) -------------------------------------
    @app.get("/forge/users")
    async def list_forge_users(user: dict = Depends(require_auth)):
        _require_admin(user)
        return {"users": forge_users.list_users(), "me": user.get("username")}

    @app.post("/forge/users")
    async def add_forge_user(payload: dict = Body(...), user: dict = Depends(require_auth)):
        _require_admin(user)
        username = (payload.get("username") or "").strip().lower()
        password = payload.get("password") or ""
        role = "admin" if payload.get("role") == "admin" else "team"
        if not username or not password:
            raise HTTPException(status_code=400, detail="username and password required")
        forge_users.create_user(username, password, role)
        return {"ok": True, "users": forge_users.list_users()}

    @app.delete("/forge/users/{username}")
    async def remove_forge_user(username: str, user: dict = Depends(require_auth)):
        _require_admin(user)
        username = (username or "").strip().lower()
        if username == user.get("username"):
            raise HTTPException(status_code=400, detail="can't remove your own login")
        roster = forge_users.list_users()
        admins = [u for u in roster if u["role"] == "admin"]
        target = next((u for u in roster if u["username"] == username), None)
        if target and target["role"] == "admin" and len(admins) <= 1:
            raise HTTPException(status_code=400, detail="can't remove the last admin")
        forge_users.delete_user(username)
        return {"ok": True, "users": forge_users.list_users()}

    @app.get("/forge/me")
    async def forge_me(user: dict = Depends(require_auth)):
        """Who am I — for the 'Signed in as…' indicator."""
        return {"username": user.get("username"), "role": user.get("role", "team")}

    @app.post("/forge/jobs")
    async def create_job(payload: dict = Body(...), user: dict = Depends(require_auth)):
        job_type = payload.get("job_type")
        if not job_type:
            raise HTTPException(status_code=400, detail="job_type is required")
        job_id = forge_jobs.enqueue(
            job_type, payload.get("params") or {}, created_by=user.get("username")
        )
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
    async def library_file(path: str = Query(...), request: Request = None,
                           user: dict = Depends(require_auth)):
        from core.forge import library
        try:
            data, ctype = library.read_file(path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        total = len(data)
        rng = (request.headers.get("range") if request else None) or ""
        # HTTP Range so <video> plays + seeks everywhere (iOS Safari REQUIRES it).
        if rng.startswith("bytes="):
            first, _, last = rng[6:].split(",")[0].strip().partition("-")
            start = int(first) if first else 0
            end = int(last) if last else total - 1
            start = max(0, start)
            end = min(end, total - 1)
            if start > end:
                start, end = 0, total - 1
            chunk = data[start:end + 1]
            return Response(content=chunk, status_code=206, media_type=ctype, headers={
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(chunk)),
            })
        return Response(content=data, media_type=ctype,
                        headers={"Accept-Ranges": "bytes", "Content-Length": str(total)})

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

    @app.get("/forge/ingest/cloud-sources")
    async def ingest_cloud_sources(user: dict = Depends(require_auth)):
        from core.forge import library
        return {"files": library.list_source_files()}

    @app.post("/forge/ingest/cloud")
    async def ingest_cloud(payload: dict = Body(...), user: dict = Depends(require_auth)):
        from core.forge import library, ingest
        from core.forge import jobs as _forge_jobs
        path = (payload.get("path") or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="path is required")
        try:
            safe = library._safe_library_path(path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not safe.startswith(library.SOURCES_ROOT):
            raise HTTPException(status_code=400, detail="path must be inside Sources folder")
        name = path.rstrip("/").split("/")[-1]
        source_id = ingest.create_source("cloud", name, path)
        job_id = _forge_jobs.enqueue("ingest_transcribe", {"source_id": source_id, "cloud_path": path})
        return {"source_id": source_id, "job_id": job_id}

    @app.post("/forge/ingest/url")
    async def ingest_url(payload: dict = Body(...), user: dict = Depends(require_auth)):
        from core.forge import clips, ingest
        from core.forge import jobs as _forge_jobs
        url = (payload.get("url") or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="url is required")
        target, kind = clips.resolve_source(url)
        source_id = ingest.create_source("url", url, None)
        job_id = _forge_jobs.enqueue("ingest_transcribe", {"source_id": source_id, "url": url})
        return {"source_id": source_id, "job_id": job_id, "resolved": target}

    @app.get("/forge/sources")
    async def list_sources_endpoint(status: str | None = None, user: dict = Depends(require_auth)):
        """Return sources newest-first. Use ?status=done to populate the source picker."""
        from core.forge import ingest
        return {"sources": ingest.list_sources(status=status)}

    @app.get("/forge/sources/{source_id}")
    async def get_source_status(source_id: str, user: dict = Depends(require_auth)):
        """Return the source row (status, duration_s, language, error, …).

        Lets the UI poll status: pending → extracting → transcribing → done/error.
        """
        from core.forge import ingest
        source = ingest.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        return source

    @app.get("/forge/sources/{source_id}/search")
    async def search_source_segments(
        source_id: str,
        q: str = Query(..., description="Topic or theme to search for"),
        top_k: int = Query(default=10, ge=1, le=50),
        speaker: str | None = Query(default=None),
        threshold: float = Query(default=0.45, ge=0.0, le=1.0),
        user: dict = Depends(require_auth),
    ):
        """Search a transcribed source for a topic/theme.

        Returns ranked windows with transcript text inline (TOPIC-02 preview).
        Lazy-backfills embedding for Phase-10 sources that predate Phase-11.

        Errors:
            404 — source_id not found
            409 — source not yet done (status != 'done')
        """
        from core.forge import ingest, search as forge_search
        source = ingest.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        if source.get("status") != "done":
            raise HTTPException(
                status_code=409,
                detail=f"source not ready: {source.get('status')}",
            )
        # Lazy backfill — Phase-10 sources ingested before embedding existed
        if not forge_search.has_windows(source_id):
            forge_search.embed_source_windows(source_id)
        results = forge_search.search_segments(
            source_id=source_id,
            query=q,
            top_k=top_k,
            speaker=speaker or None,
            score_threshold=threshold,
        )
        return {"source_id": source_id, "query": q, "results": results}

    @app.get("/forge/sources/{source_id}/transcript")
    async def get_source_transcript(source_id: str, user: dict = Depends(require_auth)):
        """Return ordered, speaker-labelled transcript segments for a source.

        Returns an empty segments list if the source exists but transcription is
        not yet complete.  404 if the source_id is unknown.
        """
        from core.forge import ingest
        source = ingest.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        segments = ingest.get_segments(source_id)
        return {"source_id": source_id, "segments": segments}

    @app.get("/forge/sources/{source_id}/preview")
    async def preview_source_window(
        source_id: str,
        start: float = Query(..., ge=0.0),
        end: float = Query(..., gt=0.0),
        user: dict = Depends(require_auth),
    ):
        """Stream a small mono MP3 of a source's [start, end] window so the topic
        UI can let the user drag the in/out handles and *listen* before assembling.

        The audio is what drives the cut (video follows the audio), so previewing
        the window audio is enough to set the clip. Fast-seek extract, capped at
        75s, clamped to the source duration. Returns audio/mpeg bytes.
        """
        import subprocess
        import tempfile
        from pathlib import Path as _P
        from core.forge import ingest

        source = ingest.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="source not found")
        fp = source.get("file_path")
        if not fp or not _P(fp).exists():
            raise HTTPException(status_code=409, detail="source media missing on disk")
        dur = float(source.get("duration_s") or 0.0)
        s = max(0.0, float(start))
        e = float(end)
        if dur > 0:
            e = min(e, dur)
        if e <= s:
            raise HTTPException(status_code=400, detail="end must be after start")
        length = min(75.0, e - s)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            out = tf.name
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-i", str(fp),
                 "-t", f"{length:.3f}", "-vn", "-ac", "1", "-ar", "44100",
                 "-b:a", "96k", "-f", "mp3", out],
                capture_output=True, text=True)
            if proc.returncode != 0 or not _P(out).exists():
                raise HTTPException(status_code=500,
                                    detail=f"preview extract failed: {proc.stderr[-300:]}")
            data = _P(out).read_bytes()
        finally:
            try:
                _P(out).unlink()
            except OSError:
                pass
        return Response(content=data, media_type="audio/mpeg",
                        headers={"Cache-Control": "no-store"})

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
