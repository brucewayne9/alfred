"""Forge upload store — opaque-id addressed files under data/forge_uploads/<id>/."""
from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import Optional


def _root() -> Path:
    root = Path(os.environ.get("FORGE_UPLOAD_DIR", "data/forge_uploads"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_upload(content: bytes, filename: str) -> str:
    """Store bytes under a fresh id; preserve only the basename's extension."""
    ext = Path(filename or "").suffix.lower()[:12]
    uid = uuid.uuid4().hex
    dest_dir = _root() / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"file{ext}").write_bytes(content)
    return uid


def get_upload_path(uid: str) -> Optional[Path]:
    """Resolve the stored file for an id, or None. Ignores any path separators in uid."""
    safe = Path(uid).name
    d = _root() / safe
    if not d.is_dir():
        return None
    files = [p for p in d.iterdir() if p.is_file()]
    return files[0] if files else None
