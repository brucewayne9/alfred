"""Mirror Forge deliveries into Google Drive alongside Nextcloud.

Dual-save during the Nextcloud -> Google Drive migration trial (Mike, 2026-06-05).
Nextcloud stays PRIMARY; this mirror is strictly best-effort — a Drive failure must
never break or delay the primary delivery. Mirrors the same folder tree the team sees
on Nextcloud (``Mainstay-RodWave/<subfolder>``) using find-or-create semantics, on the
mjohnson@groundrushinc.com Workspace Drive (~16 TB).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Drive root for Forge deliveries — mirrors Nextcloud's Content/Mainstay-RodWave tree.
DRIVE_ROOT = "Mainstay-RodWave"

# path-in-drive -> folder id (process cache; safe to lose on restart, just re-resolves)
_folder_cache: dict[str, str] = {}


def _find_or_create(name: str, parent_id: str | None) -> str:
    """Return an existing child folder id by exact name, or create it (proven pattern
    mirrored from scripts/backup/drive_manager.py)."""
    from integrations.google_drive.client import create_folder, list_files

    for f in (list_files(folder_id=parent_id, query=name, file_type="folder") or []):
        if f.get("name") == name:
            return f["id"]
    return create_folder(name, parent_id=parent_id)["id"]


def _ensure_path(path: str) -> str:
    """Find-or-create the nested folder chain (DRIVE_ROOT/...); return the leaf id."""
    if path in _folder_cache:
        return _folder_cache[path]
    parent: str | None = None
    accum = ""
    for seg in [p for p in path.split("/") if p]:
        accum = f"{accum}/{seg}".lstrip("/")
        cached = _folder_cache.get(accum)
        if cached:
            parent = cached
            continue
        parent = _find_or_create(seg, parent)
        _folder_cache[accum] = parent
    return parent  # type: ignore[return-value]


def deliver(local_path: Path | str, subfolder: str, filename: str | None = None) -> str | None:
    """Mirror one artifact into Drive ``DRIVE_ROOT/subfolder``.

    Returns the Drive file id on success, or None on any failure (logged, non-fatal).
    """
    try:
        from integrations.google_drive.client import upload_file

        local_path = Path(local_path)
        full = f"{DRIVE_ROOT}/{subfolder.strip('/')}".rstrip("/")
        folder_id = _ensure_path(full)
        name = filename or local_path.name
        res = upload_file(str(local_path), name=name, folder_id=folder_id)
        fid = res.get("id") if isinstance(res, dict) else None
        logger.info("Forge Drive mirror OK: %s/%s (id=%s)", full, name, fid)
        return fid
    except Exception as exc:  # noqa: BLE001 — mirror is best-effort, never fatal
        logger.warning("Forge Drive mirror failed (non-fatal, Nextcloud is primary): %s",
                       str(exc)[:200])
        return None
