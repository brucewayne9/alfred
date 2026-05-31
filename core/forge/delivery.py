"""Mainstay Forge — deliver rendered artifacts to the team's Nextcloud folders."""
from pathlib import Path

from integrations.nextcloud.client import create_folder, upload_file

# WebDAV path relative to the Nextcloud user's files root (no leading slash).
DELIVERY_ROOT = "Content/Mainstay-RodWave"


def deliver(local_path: Path, subfolder: str, filename: str | None = None) -> str:
    """Upload `local_path` into DELIVERY_ROOT/subfolder on Nextcloud.

    Returns the remote WebDAV path. Raises RuntimeError if the upload fails.
    Target subfolders normally already exist, so folder creation is best-effort
    (the client raises on MKCOL of an existing folder — that is not fatal).
    """
    local_path = Path(local_path)
    remote_dir = f"{DELIVERY_ROOT}/{subfolder.strip('/')}".rstrip("/")
    parts = [p for p in subfolder.strip("/").split("/") if p]
    accum = DELIVERY_ROOT
    for seg in parts:
        accum = f"{accum}/{seg}"
        try:
            create_folder(accum)
        except Exception:  # noqa: BLE001 — folder may already exist (405)
            pass
    name = filename or local_path.name
    remote_path = f"{remote_dir}/{name}"
    try:
        upload_file(remote_path, local_path.read_bytes())
    except Exception as e:
        raise RuntimeError(f"Nextcloud upload failed: {remote_path}: {e}") from e
    return remote_path
