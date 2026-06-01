"""Forge Library — index delivered assets (job DB) + list/stream from Nextcloud (guarded)."""
from __future__ import annotations
import json
import os
import time
import uuid

from core.forge import jobs as forge_jobs
from core.forge.db import _conn, init_db

DELIVERY_ROOT = "Content/Mainstay-RodWave"
TRASH_ROOT = f"{DELIVERY_ROOT}/.forge-trash"
VIDEO_EXT = {".mp4", ".mov", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MEDIA_EXT = VIDEO_EXT | IMAGE_EXT

# How many delete actions we keep recoverable. Beyond this the oldest are purged
# from Nextcloud for good (so the trash folder can't grow without bound). The UI
# Undo button walks back through whatever is still here, newest first.
TRASH_RETAIN = 10


def _safe_library_path(path: str) -> str:
    """Normalize and confine a path to the delivery root; raise on escape."""
    p = (path or "").strip().lstrip("/")
    parts = p.split("/")
    if ".." in parts or not p.startswith(DELIVERY_ROOT):
        raise ValueError(f"path outside library root: {path!r}")
    return p


def list_done_jobs(limit: int = 100) -> list[dict]:
    """Done jobs as library cards (newest first).

    Batches the user has deleted live in trash (recoverable via undo); their job
    rows still exist, so we filter them out here. Without this a deleted batch
    card reappears on reload — the exact bug undo replaces.
    """
    hidden = _trashed_job_ids()
    out: list[dict] = []
    for j in forge_jobs.list_jobs(status="done", limit=limit):
        if j["id"] in hidden:
            continue
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


def _hard_delete(path: str) -> dict:
    """Permanently remove a library path from Nextcloud (path-guarded).

    Refuses to delete the delivery root itself — only things beneath it.
    """
    from integrations.nextcloud import client as nc
    safe = _safe_library_path(path)
    if safe.rstrip("/") == DELIVERY_ROOT:
        raise ValueError("refusing to delete the delivery root")
    return nc.delete_file(safe)


# ---- soft-delete / undo ---------------------------------------------------
# Delete moves the item into a hidden .forge-trash folder and records the move,
# so the team can undo (move it back) instead of losing a render permanently.

def _trashed_job_ids() -> set[str]:
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT job_id FROM trash WHERE job_id IS NOT NULL").fetchall()
    return {r["job_id"] for r in rows}


def trash_state() -> dict:
    """Undo-button state: how many delete actions are recoverable + newest label."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT label FROM trash ORDER BY rowid DESC"
        ).fetchall()
    return {"count": len(rows), "label": (rows[0]["label"] if rows else None)}


def _ensure_folder(path: str) -> None:
    """MKCOL a folder, tolerating 'already exists' (405)."""
    import requests
    from integrations.nextcloud import client as nc
    try:
        nc.create_folder(path)
    except requests.exceptions.HTTPError as e:
        if getattr(e.response, "status_code", None) not in (405, 409):
            raise


def soft_delete(paths: list[str], kind: str, job_id: str | None = None,
                label: str | None = None) -> dict:
    """Move one or more library paths into trash as a single undoable action.

    `kind` is 'file' or 'batch'. For a batch, `job_id` hides the card until undo.
    Returns {token, count}.
    """
    from integrations.nextcloud import client as nc
    safe_paths = [_safe_library_path(p) for p in paths if (p or "").strip()]
    safe_paths = [p for p in safe_paths if p.rstrip("/") != DELIVERY_ROOT]
    if not safe_paths:
        raise ValueError("nothing to delete")

    token = uuid.uuid4().hex[:12]
    dest_dir = f"{TRASH_ROOT}/{token}"
    _ensure_folder(TRASH_ROOT)
    _ensure_folder(dest_dir)

    items: list[dict] = []
    for i, src in enumerate(safe_paths):
        base = src.rstrip("/").split("/")[-1]
        dest = f"{dest_dir}/{i:02d}_{base}"
        nc.move_file(src, dest)
        items.append({"orig": src, "trash": dest})

    with _conn() as c:
        c.execute(
            "INSERT INTO trash (token, kind, items, job_id, label, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (token, kind, json.dumps(items), job_id, label, int(time.time())),
        )
    _prune_trash()
    return {"token": token, "count": len(items)}


def undo_last() -> dict | None:
    """Restore the most recent delete action: move its items back, drop the row.

    Returns {token, kind, restored, job_id, label} or None if trash is empty.
    """
    from integrations.nextcloud import client as nc
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM trash ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None

    items = json.loads(row["items"])
    restored = 0
    for it in items:
        parent = it["orig"].rstrip("/").rsplit("/", 1)[0]
        _ensure_folder(parent)
        nc.move_file(it["trash"], it["orig"])
        restored += 1

    with _conn() as c:
        c.execute("DELETE FROM trash WHERE token = ?", (row["token"],))
    # Best-effort: tidy the now-empty per-action trash folder.
    try:
        _hard_delete(f"{TRASH_ROOT}/{row['token']}")
    except Exception:  # noqa: BLE001
        pass
    return {"token": row["token"], "kind": row["kind"], "restored": restored,
            "job_id": row["job_id"], "label": row["label"]}


def _prune_trash() -> None:
    """Permanently purge delete actions older than the newest TRASH_RETAIN."""
    init_db()
    with _conn() as c:
        old = c.execute(
            "SELECT token, items FROM trash ORDER BY rowid DESC "
            "LIMIT -1 OFFSET ?",
            (TRASH_RETAIN,),
        ).fetchall()
        for row in old:
            for it in json.loads(row["items"]):
                try:
                    _hard_delete(it["trash"])
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
            try:
                _hard_delete(f"{TRASH_ROOT}/{row['token']}")
            except Exception:  # noqa: BLE001
                pass
            c.execute("DELETE FROM trash WHERE token = ?", (row["token"],))
