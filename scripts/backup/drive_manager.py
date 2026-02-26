"""
Alfred Backup — Google Drive folder manager.

Wraps integrations.google_drive.client for backup-specific folder management.
Provides find-or-create semantics for the backup folder hierarchy:

    Alfred Backups/
      {server_name}/
        daily/
          YYYY-MM-DD/
            *.tar.gz
        weekly/
          YYYY-MM-DD/
            *.tar.gz

Usage (standalone):
    python3 drive_manager.py --prune 30
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone

# Resolve project root so imports work when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from integrations.google_drive.client import (
    create_folder,
    delete_file,
    list_files,
    upload_file,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: avoids repeated Drive API calls within a single run
# ---------------------------------------------------------------------------
_root_folder_id: str | None = None
_server_folder_ids: dict[str, str] = {}      # {server_name: folder_id}
_type_folder_ids: dict[str, str] = {}         # {server_name/backup_type: folder_id}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_folder(name: str, parent_id: str | None = None) -> str | None:
    """Search for a folder by exact name under a parent (or root).

    Returns the folder ID if found, else None.
    """
    folders = list_files(folder_id=parent_id, query=name, file_type="folder")
    for f in folders:
        if f["name"] == name:
            return f["id"]
    return None


def _find_or_create_folder(name: str, parent_id: str | None = None) -> str:
    """Return existing folder ID or create the folder and return new ID."""
    folder_id = _find_folder(name, parent_id)
    if folder_id:
        logger.debug("Found existing folder '%s' (id=%s)", name, folder_id)
        return folder_id
    result = create_folder(name, parent_id=parent_id)
    logger.info("Created Drive folder '%s' (id=%s)", name, result["id"])
    return result["id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_root_folder() -> str:
    """Find or create the 'Alfred Backups' root folder in Drive.

    The result is cached for the lifetime of the process.

    Returns:
        Folder ID string.
    """
    global _root_folder_id
    if _root_folder_id:
        return _root_folder_id
    _root_folder_id = _find_or_create_folder("Alfred Backups")
    logger.info("Root backup folder id=%s", _root_folder_id)
    return _root_folder_id


def ensure_server_folder(server_name: str) -> str:
    """Find or create 'Alfred Backups/{server_name}/' folder.

    Args:
        server_name: Short server identifier (e.g. "alfred-labs", "alfred-claw").

    Returns:
        Folder ID string.
    """
    if server_name in _server_folder_ids:
        return _server_folder_ids[server_name]
    root_id = ensure_root_folder()
    folder_id = _find_or_create_folder(server_name, parent_id=root_id)
    _server_folder_ids[server_name] = folder_id
    return folder_id


def _ensure_type_folder(server_name: str, backup_type: str) -> str:
    """Find or create 'Alfred Backups/{server_name}/{backup_type}/' folder.

    Args:
        server_name: Short server identifier.
        backup_type: "daily" or "weekly".

    Returns:
        Folder ID string.
    """
    cache_key = f"{server_name}/{backup_type}"
    if cache_key in _type_folder_ids:
        return _type_folder_ids[cache_key]
    server_id = ensure_server_folder(server_name)
    folder_id = _find_or_create_folder(backup_type, parent_id=server_id)
    _type_folder_ids[cache_key] = folder_id
    return folder_id


def upload_backup(local_path: str, server_name: str, backup_type: str) -> dict:
    """Upload a tar.gz file to 'Alfred Backups/{server_name}/{backup_type}/YYYY-MM-DD/'.

    Creates the date-stamped folder if it doesn't exist, then uploads the file.

    Args:
        local_path: Absolute path to the local .tar.gz file.
        server_name: Short server identifier (e.g. "alfred-labs").
        backup_type: "daily" or "weekly".

    Returns:
        Drive file metadata dict with keys: id, name, link.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    type_folder_id = _ensure_type_folder(server_name, backup_type)
    date_folder_id = _find_or_create_folder(today, parent_id=type_folder_id)

    file_name = os.path.basename(local_path)
    logger.info(
        "Uploading %s -> Alfred Backups/%s/%s/%s/%s",
        local_path, server_name, backup_type, today, file_name,
    )
    result = upload_file(
        local_path=local_path,
        name=file_name,
        folder_id=date_folder_id,
        mime_type="application/gzip",
    )
    logger.info("Upload complete: %s (id=%s)", result.get("name"), result.get("id"))
    return result


def list_date_folders(server_name: str, backup_type: str) -> list[dict]:
    """List all date-stamped folders under 'Alfred Backups/{server_name}/{backup_type}/'.

    Args:
        server_name: Short server identifier.
        backup_type: "daily" or "weekly".

    Returns:
        List of dicts with keys: id, name, modified.
        Sorted by name (date string) ascending.
    """
    type_folder_id = _ensure_type_folder(server_name, backup_type)
    raw = list_files(folder_id=type_folder_id, file_type="folder", max_results=100)
    folders = [
        {"id": f["id"], "name": f["name"], "modified": f.get("modified")}
        for f in raw
    ]
    folders.sort(key=lambda x: x["name"])
    return folders


def delete_folder_recursive(folder_id: str) -> dict:
    """Trash a Drive folder (Drive cascades trash to all children).

    Args:
        folder_id: The Drive folder ID to trash.

    Returns:
        Result dict from delete_file.
    """
    logger.info("Trashing folder id=%s (including all contents)", folder_id)
    return delete_file(folder_id)


def prune_old_backups(
    retention_days: int = 30,
    server_names: list[str] | None = None,
    backup_types: list[str] | None = None,
) -> dict:
    """Delete backup date-folders older than retention_days.

    Iterates over all servers and backup types (daily/weekly), parses
    folder names as YYYY-MM-DD dates, and trashes folders older than
    the retention window.

    Args:
        retention_days: Delete folders older than this many days. Default 30.
        server_names: Restrict pruning to these server names. If None, uses
                      all servers defined in backup_utils.SERVERS.
        backup_types: Restrict to these backup types. Defaults to ["daily", "weekly"].

    Returns:
        Summary dict: {server_name: {"daily": {deleted: [...], kept: [...]}, "weekly": {...}}}
    """
    from scripts.backup.backup_utils import SERVERS  # local import avoids circular

    if server_names is None:
        server_names = [s["name"] for s in SERVERS]
    if backup_types is None:
        backup_types = ["daily", "weekly"]

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    summary: dict = {}

    for server_name in server_names:
        summary[server_name] = {}
        for btype in backup_types:
            deleted = []
            kept = []
            try:
                folders = list_date_folders(server_name, btype)
            except Exception as exc:
                logger.warning("Could not list %s/%s folders: %s", server_name, btype, exc)
                summary[server_name][btype] = {"deleted": [], "kept": [], "error": str(exc)}
                continue

            for folder in folders:
                folder_name = folder["name"]
                try:
                    folder_date = datetime.strptime(folder_name, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    logger.warning(
                        "Skipping non-date folder '%s' in %s/%s",
                        folder_name, server_name, btype,
                    )
                    kept.append(folder_name)
                    continue

                if folder_date < cutoff:
                    logger.info(
                        "Pruning %s/%s/%s (older than %d days)",
                        server_name, btype, folder_name, retention_days,
                    )
                    try:
                        delete_folder_recursive(folder["id"])
                        deleted.append(folder_name)
                    except Exception as exc:
                        logger.error(
                            "Failed to delete %s/%s/%s: %s",
                            server_name, btype, folder_name, exc,
                        )
                        kept.append(folder_name)
                else:
                    kept.append(folder_name)

            summary[server_name][btype] = {"deleted": deleted, "kept": kept}
            if deleted:
                logger.info(
                    "Pruned %d folder(s) from %s/%s (kept %d)",
                    len(deleted), server_name, btype, len(kept),
                )

    return summary


# ---------------------------------------------------------------------------
# CLI entry point for standalone pruning runs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Alfred Drive backup manager")
    parser.add_argument(
        "--prune",
        type=int,
        metavar="DAYS",
        help="Delete backup folders older than DAYS days",
    )
    parser.add_argument(
        "--list",
        metavar="SERVER",
        help="List date folders for a server (shows both daily and weekly)",
    )
    args = parser.parse_args()

    if args.prune:
        import json
        result = prune_old_backups(retention_days=args.prune)
        print(json.dumps(result, indent=2))
    elif args.list:
        import json
        for btype in ["daily", "weekly"]:
            folders = list_date_folders(args.list, btype)
            print(f"\n{args.list}/{btype}:")
            print(json.dumps(folders, indent=2))
    else:
        parser.print_help()
