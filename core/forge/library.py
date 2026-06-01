"""Forge Library — index delivered assets (job DB) + list/stream from Nextcloud (guarded)."""
from __future__ import annotations
import os
from core.forge import jobs as forge_jobs

DELIVERY_ROOT = "Content/Mainstay-RodWave"
VIDEO_EXT = {".mp4", ".mov", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MEDIA_EXT = VIDEO_EXT | IMAGE_EXT


def _safe_library_path(path: str) -> str:
    """Normalize and confine a path to the delivery root; raise on escape."""
    p = (path or "").strip().lstrip("/")
    parts = p.split("/")
    if ".." in parts or not p.startswith(DELIVERY_ROOT):
        raise ValueError(f"path outside library root: {path!r}")
    return p


def list_done_jobs(limit: int = 100) -> list[dict]:
    """Done jobs as library cards (newest first)."""
    out: list[dict] = []
    for j in forge_jobs.list_jobs(status="done", limit=limit):
        res = j.get("result") or {}
        dirs = res.get("delivered_dirs")
        if not dirs:
            dirs = [res["delivered_dir"]] if res.get("delivered_dir") else []
        out.append({
            "id": j["id"],
            "format": res.get("format") or j.get("job_type"),
            "caption": (j.get("params") or {}).get("caption", ""),
            "created_at": j.get("created_at"),
            "remix_looks": res.get("remix_looks"),
            "variations_each": res.get("variations_each") or res.get("variant_count"),
            "delivered": res.get("delivered"),
            "dirs": dirs,
        })
    return out


def list_dir_files(dir_path: str) -> list[dict]:
    """Media files directly inside a delivery dir (path-guarded)."""
    import requests
    from integrations.nextcloud import client as nc
    safe = _safe_library_path(dir_path)
    files: list[dict] = []
    try:
        listing = nc.list_files(safe, depth=1)
    except requests.exceptions.HTTPError as e:
        # Folder gone (e.g. a job recorded a dir whose delivery failed) -> empty,
        # not a 500. Re-raise anything that isn't a clean "not found".
        if getattr(e.response, "status_code", None) == 404:
            return []
        raise
    for f in listing:
        path = f.get("path") or ""
        name = f.get("name") or path.rstrip("/").split("/")[-1]
        if not name or path.rstrip("/").endswith(safe.rstrip("/")):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in MEDIA_EXT:
            continue
        files.append({
            "name": name,
            "path": path,
            "size": f.get("size"),
            "kind": "video" if ext in VIDEO_EXT else "image",
        })
    return files


def read_file(path: str) -> tuple[bytes, str]:
    """Stream one library file's bytes + content-type (path-guarded)."""
    import mimetypes
    from integrations.nextcloud import client as nc
    safe = _safe_library_path(path)
    data = nc.download_file(safe)
    ctype = mimetypes.guess_type(safe)[0] or "application/octet-stream"
    return data, ctype


def delete_path(path: str) -> dict:
    """Delete a library file or whole batch folder from Nextcloud (path-guarded).

    Refuses to delete the delivery root itself — only things beneath it.
    """
    from integrations.nextcloud import client as nc
    safe = _safe_library_path(path)
    if safe.rstrip("/") == DELIVERY_ROOT:
        raise ValueError("refusing to delete the delivery root")
    return nc.delete_file(safe)
