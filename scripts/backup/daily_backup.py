#!/usr/bin/env python3
"""
Alfred Daily Config Backup

Runs at 2 AM every day (via cron) to:
1. SSH into each of the 7 infrastructure servers
2. Collect config files, env files, crontabs, systemd units, and database dumps
3. Pack per-server artifacts into a tar.gz
4. Upload to Google Drive: Alfred Backups/{server}/daily/YYYY-MM-DD/

Usage:
    python3 scripts/backup/daily_backup.py
    /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/backup/daily_backup.py
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

# Resolve project root so imports work from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from scripts.backup.backup_utils import (
    DAILY_TARGETS,
    SERVERS,
    STAGING_DIR,
    cleanup_staging,
    collect_remote_files,
    pack_tarball,
    run_cmd,
)
from scripts.backup.drive_manager import upload_backup

# ---------------------------------------------------------------------------
# Logging setup — called from main() after we know the log path
# ---------------------------------------------------------------------------

LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "backup_daily.log"
)


def _configure_logging() -> None:
    """Configure logging to stdout and the rotating log file."""
    log_path = os.path.abspath(LOG_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove any pre-existing handlers (e.g. from pytest)
    root.handlers.clear()

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(ch)

    # File (append)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(fh)


logger = logging.getLogger("daily_backup")


# ---------------------------------------------------------------------------
# Per-server backup logic
# ---------------------------------------------------------------------------

def backup_server(server: dict, date_str: str) -> dict:
    """Backup a single server — SSH collect, pack, upload, clean.

    Args:
        server: Entry from SERVERS list (keys: name, alias, ip, description).
        date_str: Today's date string in YYYY-MM-DD format (for folder naming).

    Returns:
        Result dict with keys: server, success, files_collected, tarball_size,
        drive_id, drive_link, error.
    """
    server_name = server["name"]
    alias = server["alias"]

    result: dict = {
        "server": server_name,
        "success": False,
        "files_collected": 0,
        "tarball_size": "0 KB",
        "drive_id": None,
        "drive_link": None,
        "error": None,
    }

    staging_server_dir = os.path.join(STAGING_DIR, "daily", date_str, server_name)
    os.makedirs(staging_server_dir, exist_ok=True)

    targets = DAILY_TARGETS.get(server_name)
    if not targets:
        logger.warning("[%s] No DAILY_TARGETS entry — skipping", server_name)
        result["error"] = "No DAILY_TARGETS entry defined"
        return result

    logger.info("[%s] Starting daily backup (alias=%s)", server_name, alias or "local")

    # -----------------------------------------------------------------------
    # 1. Collect static files via scp / cp
    # -----------------------------------------------------------------------
    static_files = targets.get("files", [])
    if static_files:
        collected = collect_remote_files(alias, static_files, staging_server_dir)
        result["files_collected"] += len(collected)
        logger.info(
            "[%s] Collected %d/%d static files",
            server_name, len(collected), len(static_files),
        )
    else:
        logger.debug("[%s] No static files configured", server_name)

    # -----------------------------------------------------------------------
    # 2. Run command-based collections
    # -----------------------------------------------------------------------
    commands = targets.get("commands", [])
    for output_filename, cmd in commands:
        dest_path = os.path.join(staging_server_dir, output_filename)
        try:
            output = run_cmd(alias, cmd, timeout=120)
            with open(dest_path, "w", encoding="utf-8") as fh:
                fh.write(output)
                if output and not output.endswith("\n"):
                    fh.write("\n")
            result["files_collected"] += 1
            logger.debug("[%s] Captured: %s (%d bytes)", server_name, output_filename, len(output))
        except Exception as exc:
            # Command failures are non-fatal — log and continue
            logger.warning(
                "[%s] Command failed for %s: %s", server_name, output_filename, exc
            )
            # Write an error marker so the tarball shows what was attempted
            try:
                with open(dest_path, "w", encoding="utf-8") as fh:
                    fh.write(f"(command failed: {exc})\n")
            except OSError:
                pass

    logger.info(
        "[%s] Collection complete — %d artifact(s) staged",
        server_name, result["files_collected"],
    )

    # -----------------------------------------------------------------------
    # 3. Pack tarball
    # -----------------------------------------------------------------------
    tarball_name = f"{server_name}-daily-{date_str}.tar.gz"
    tarball_dir = os.path.join(STAGING_DIR, "daily", date_str)
    tarball_path = os.path.join(tarball_dir, tarball_name)

    try:
        pack_tarball(staging_server_dir, tarball_path)
        size_bytes = os.path.getsize(tarball_path)
        if size_bytes >= 1024 * 1024:
            result["tarball_size"] = f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            result["tarball_size"] = f"{size_bytes // 1024} KB"
        logger.info("[%s] Tarball: %s (%s)", server_name, tarball_name, result["tarball_size"])
    except Exception as exc:
        logger.error("[%s] Failed to pack tarball: %s", server_name, exc)
        result["error"] = f"pack_tarball failed: {exc}"
        return result

    # -----------------------------------------------------------------------
    # 4. Upload to Drive — failure IS fatal for this server (no local keeping)
    # -----------------------------------------------------------------------
    try:
        drive_result = upload_backup(tarball_path, server_name, "daily")
        result["drive_id"] = drive_result.get("id")
        result["drive_link"] = drive_result.get("link")
        result["success"] = True
        logger.info(
            "[%s] Uploaded to Drive (id=%s)", server_name, result["drive_id"]
        )
    except Exception as exc:
        logger.error("[%s] Drive upload failed: %s", server_name, exc)
        result["error"] = f"Drive upload failed: {exc}"
        return result

    # -----------------------------------------------------------------------
    # 5. Clean up staging dir for this server (tarball + source dir)
    # -----------------------------------------------------------------------
    try:
        cleanup_staging(staging_server_dir)
        if os.path.exists(tarball_path):
            os.remove(tarball_path)
            logger.debug("[%s] Removed local tarball: %s", server_name, tarball_path)
    except Exception as exc:
        logger.warning("[%s] Cleanup warning: %s", server_name, exc)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run daily backup across all servers.

    Returns:
        Exit code: 0 if all servers succeeded, 1 if any failed.
    """
    _configure_logging()

    run_start = time.time()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("Alfred Daily Config Backup — %s", date_str)
    logger.info("Servers: %d", len(SERVERS))
    logger.info("=" * 60)

    results = []
    for server in SERVERS:
        try:
            r = backup_server(server, date_str)
        except Exception as exc:
            # Unexpected error — don't let one server crash the whole run
            logger.exception("[%s] Unexpected error: %s", server["name"], exc)
            r = {
                "server": server["name"],
                "success": False,
                "files_collected": 0,
                "tarball_size": "0 KB",
                "drive_id": None,
                "drive_link": None,
                "error": f"Unexpected: {exc}",
            }
        results.append(r)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    duration = time.time() - run_start
    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info("=" * 60)
    logger.info("Daily backup complete in %.1fs", duration)
    logger.info("Succeeded: %d/%d", len(succeeded), len(SERVERS))
    if succeeded:
        logger.info("  %s", ", ".join(r["server"] for r in succeeded))
    if failed:
        logger.warning("Failed: %d/%d", len(failed), len(SERVERS))
        for r in failed:
            logger.warning("  %s — %s", r["server"], r["error"])
    logger.info("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
