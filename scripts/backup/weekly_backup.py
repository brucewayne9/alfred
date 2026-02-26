#!/usr/bin/env python3
"""
Alfred Backup — weekly full backup script.

Runs at 2 AM every Sunday (via cron). Collects everything the daily script
does PLUS Docker volume exports, app data, and package lists, then uploads
per-server tarballs to Google Drive and prunes backups older than 30 days.

Drive path: Alfred Backups/{server_name}/weekly/YYYY-MM-DD/
Retention:  30 days (both daily and weekly folders pruned after each run)

Usage:
    python3 scripts/backup/weekly_backup.py
    /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/backup/weekly_backup.py
"""

import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Resolve project root so imports work when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from scripts.backup.backup_alerting import send_backup_alert
from scripts.backup.backup_utils import (
    DAILY_TARGETS,
    SERVERS,
    STAGING_DIR,
    WEEKLY_EXTRAS,
    cleanup_staging,
    collect_remote_files,
    pack_tarball,
    run_cmd,
)
from scripts.backup.drive_manager import prune_old_backups, upload_backup

# ---------------------------------------------------------------------------
# Logging setup — called from main()
# ---------------------------------------------------------------------------

LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "backup_weekly.log"
)


def _configure_logging() -> None:
    """Configure logging to stdout and the log file (append mode)."""
    log_path = os.path.abspath(LOG_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(fh)


logger = logging.getLogger("weekly_backup")

# ---------------------------------------------------------------------------
# Docker volume export helpers
# ---------------------------------------------------------------------------

# Pattern for anonymous Docker volumes (64-char hex strings)
_ANON_VOLUME_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

# Volume name patterns to SKIP (case-insensitive)
_SKIP_PATTERNS = ("cache", "tmp", "log")

# Patterns to include for labsliveserver (55 containers — selective only)
_LABSLIVE_INCLUDE_PATTERNS = (
    "_db_", "_data_", "postgres", "mysql", "redis", "mongo",
)

# Per-volume export timeout (seconds) — prevents hanging on huge volumes
VOLUME_EXPORT_TIMEOUT = 300


def _is_anonymous_volume(volume_name: str) -> bool:
    """Return True if the volume name looks like a Docker anonymous volume hash."""
    return bool(_ANON_VOLUME_RE.match(volume_name))


def _should_skip_volume(volume_name: str, server_name: str) -> bool:
    """Return True if this volume should be skipped.

    Skips:
    - Anonymous volumes (64-char hex hashes)
    - Names containing 'cache', 'tmp', or 'log'
    - For labsliveserver: volumes not matching the include patterns
    """
    if _is_anonymous_volume(volume_name):
        return True

    name_lower = volume_name.lower()
    for pat in _SKIP_PATTERNS:
        if pat in name_lower:
            return True

    # labsliveserver has 55+ containers — only export key volumes
    if server_name == "labsliveserver":
        if not any(pat in name_lower for pat in _LABSLIVE_INCLUDE_PATTERNS):
            return True

    return False


def _export_docker_volume(
    alias: str | None,
    volume_name: str,
    staging_dir: str,
    server_name: str,
) -> bool:
    """Export a single Docker volume to a .tar.gz in staging_dir.

    Uses an Alpine container to tar the volume contents without stopping
    the running container that uses the volume.

    Args:
        alias: SSH alias (None = local).
        volume_name: Docker volume name.
        staging_dir: Local directory to write the export tarball to.
        server_name: Server identifier for logging.

    Returns:
        True on success, False on failure (failure is non-fatal).
    """
    # Sanitize volume name for use as a filename
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", volume_name)
    remote_tarball = f"/tmp/{safe_name}.tar.gz"
    local_tarball = os.path.join(staging_dir, f"docker-vol-{safe_name}.tar.gz")

    export_cmd = (
        f"docker run --rm "
        f"-v {volume_name}:/data "
        f"-v /tmp:/backup "
        f"alpine tar czf /backup/{safe_name}.tar.gz /data 2>/dev/null && "
        f"echo OK"
    )

    try:
        result = run_cmd(alias, export_cmd, timeout=VOLUME_EXPORT_TIMEOUT)
        if "OK" not in result:
            logger.warning(
                "[%s] Volume export did not confirm OK for '%s'",
                server_name, volume_name,
            )
            return False

        # Now scp/copy the tarball to staging
        if alias is None:
            # Local: just move the file
            if os.path.exists(remote_tarball):
                import shutil
                shutil.move(remote_tarball, local_tarball)
                logger.debug(
                    "[%s] Moved local volume export: %s (%d KB)",
                    server_name, safe_name, os.path.getsize(local_tarball) // 1024,
                )
                return True
            else:
                logger.warning(
                    "[%s] Volume export file not found locally: %s",
                    server_name, remote_tarball,
                )
                return False
        else:
            # Remote: scp the file down
            scp_result = subprocess.run(
                [
                    "scp",
                    "-o", "ConnectTimeout=10",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                    f"{alias}:{remote_tarball}",
                    local_tarball,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            # Cleanup remote temp file (non-fatal if this fails)
            try:
                run_cmd(alias, f"rm -f {remote_tarball}", timeout=10)
            except Exception:
                pass

            if scp_result.returncode == 0 and os.path.exists(local_tarball):
                logger.debug(
                    "[%s] SCP'd volume export: %s (%d KB)",
                    server_name, safe_name,
                    os.path.getsize(local_tarball) // 1024,
                )
                return True
            else:
                logger.warning(
                    "[%s] SCP failed for volume export '%s': %s",
                    server_name, volume_name, scp_result.stderr.strip(),
                )
                return False

    except Exception as exc:
        logger.warning(
            "[%s] Volume export failed for '%s': %s",
            server_name, volume_name, exc,
        )
        return False


def _collect_docker_volumes(
    alias: str | None,
    server_name: str,
    staging_dir: str,
) -> int:
    """Discover and export all eligible Docker volumes on a server.

    Discovers volumes live via `docker volume ls -q`, filters out anonymous
    and unwanted volumes, then exports each eligible named volume using
    an Alpine container.

    Args:
        alias: SSH alias (None = local).
        server_name: Server identifier (used for filtering + logging).
        staging_dir: Local directory to write volume export tarballs.

    Returns:
        Number of volumes successfully exported.
    """
    # Discover volumes live
    try:
        raw = run_cmd(alias, "docker volume ls -q 2>/dev/null", timeout=30)
    except Exception as exc:
        logger.warning("[%s] Could not list Docker volumes: %s", server_name, exc)
        return 0

    all_volumes = [v.strip() for v in raw.splitlines() if v.strip()]
    if not all_volumes:
        logger.info("[%s] No Docker volumes found", server_name)
        return 0

    eligible = [v for v in all_volumes if not _should_skip_volume(v, server_name)]
    skipped = len(all_volumes) - len(eligible)
    logger.info(
        "[%s] Docker volumes: %d total, %d eligible, %d skipped",
        server_name, len(all_volumes), len(eligible), skipped,
    )

    exported = 0
    for volume_name in eligible:
        logger.info("[%s] Exporting Docker volume: %s", server_name, volume_name)
        success = _export_docker_volume(alias, volume_name, staging_dir, server_name)
        if success:
            exported += 1
        else:
            logger.warning(
                "[%s] Volume export failed (non-fatal): %s", server_name, volume_name
            )

    logger.info("[%s] Docker volume exports: %d/%d succeeded", server_name, exported, len(eligible))
    return exported


def _collect_alfred_labs_data(staging_dir: str) -> int:
    """Copy key SQLite DBs and ChromaDB from alfred-labs data/ directory.

    Returns number of artifacts collected.
    """
    alfred_data = "/home/aialfred/alfred/data"
    collected = 0

    # Copy specific SQLite databases
    sqlite_files = ["conversations.db", "learning.db"]
    for db_file in sqlite_files:
        src = os.path.join(alfred_data, db_file)
        dst = os.path.join(staging_dir, db_file)
        if os.path.exists(src):
            try:
                import shutil
                shutil.copy2(src, dst)
                size_kb = os.path.getsize(dst) // 1024
                logger.info("[alfred-labs] Copied %s (%d KB)", db_file, size_kb)
                collected += 1
            except Exception as exc:
                logger.warning("[alfred-labs] Failed to copy %s: %s", db_file, exc)
        else:
            logger.debug("[alfred-labs] %s not found — skipping", src)

    # Archive ChromaDB directory (entire dir)
    chromadb_dir = os.path.join(alfred_data, "chromadb")
    if os.path.isdir(chromadb_dir):
        chroma_archive = os.path.join(staging_dir, "chromadb.tar.gz")
        try:
            import tarfile
            with tarfile.open(chroma_archive, "w:gz") as tar:
                tar.add(chromadb_dir, arcname="chromadb")
            size_kb = os.path.getsize(chroma_archive) // 1024
            logger.info("[alfred-labs] Archived chromadb/ (%d KB)", size_kb)
            collected += 1
        except Exception as exc:
            logger.warning("[alfred-labs] Failed to archive chromadb/: %s", exc)
    else:
        logger.debug("[alfred-labs] chromadb/ not found at %s — skipping", chromadb_dir)

    return collected


# ---------------------------------------------------------------------------
# Per-server weekly backup logic
# ---------------------------------------------------------------------------

def backup_server(server: dict, date_str: str) -> dict:
    """Full weekly backup for one server.

    Collects everything daily does (DAILY_TARGETS) plus weekly extras
    (WEEKLY_EXTRAS): package lists, Docker volume inventories, and
    Docker volume exports.

    Args:
        server: Entry from SERVERS list (keys: name, alias, ip, description).
        date_str: Today's date string in YYYY-MM-DD format.

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

    staging_server_dir = os.path.join(STAGING_DIR, "weekly", date_str, server_name)
    os.makedirs(staging_server_dir, exist_ok=True)

    logger.info("[%s] Starting weekly backup (alias=%s)", server_name, alias or "local")

    # -----------------------------------------------------------------------
    # Phase 1: Collect everything from DAILY_TARGETS
    # -----------------------------------------------------------------------
    daily_targets = DAILY_TARGETS.get(server_name, {"files": [], "commands": []})

    # 1a. Static files
    static_files = daily_targets.get("files", [])
    if static_files:
        copied = collect_remote_files(alias, static_files, staging_server_dir)
        result["files_collected"] += len(copied)
        logger.info(
            "[%s] Daily phase: collected %d/%d static files",
            server_name, len(copied), len(static_files),
        )

    # 1b. Command outputs
    daily_commands = daily_targets.get("commands", [])
    for output_filename, cmd in daily_commands:
        dest_path = os.path.join(staging_server_dir, output_filename)
        try:
            output = run_cmd(alias, cmd, timeout=120)
            with open(dest_path, "w", encoding="utf-8") as fh:
                fh.write(output)
                if output and not output.endswith("\n"):
                    fh.write("\n")
            result["files_collected"] += 1
            logger.debug("[%s] Captured: %s", server_name, output_filename)
        except Exception as exc:
            logger.warning("[%s] Daily command failed for %s: %s", server_name, output_filename, exc)
            # Write error marker so tarball shows what was attempted
            try:
                with open(dest_path, "w", encoding="utf-8") as fh:
                    fh.write(f"(command failed: {exc})\n")
            except OSError:
                pass

    logger.info(
        "[%s] Daily phase complete — %d artifact(s)",
        server_name, result["files_collected"],
    )

    # -----------------------------------------------------------------------
    # Phase 2: Collect WEEKLY_EXTRAS (package lists, Docker inventory)
    # -----------------------------------------------------------------------
    weekly_extras = WEEKLY_EXTRAS.get(server_name, {"files": [], "commands": []})

    extra_files = weekly_extras.get("files", [])
    if extra_files:
        copied_extra = collect_remote_files(alias, extra_files, staging_server_dir)
        result["files_collected"] += len(copied_extra)

    weekly_commands = weekly_extras.get("commands", [])
    for output_filename, cmd in weekly_commands:
        dest_path = os.path.join(staging_server_dir, output_filename)
        try:
            output = run_cmd(alias, cmd, timeout=120)
            with open(dest_path, "w", encoding="utf-8") as fh:
                fh.write(output)
                if output and not output.endswith("\n"):
                    fh.write("\n")
            result["files_collected"] += 1
            logger.debug("[%s] Weekly extra captured: %s", server_name, output_filename)
        except Exception as exc:
            logger.warning(
                "[%s] Weekly command failed for %s: %s", server_name, output_filename, exc
            )

    # -----------------------------------------------------------------------
    # Phase 3: Docker volume exports (live discovery, non-fatal per volume)
    # -----------------------------------------------------------------------
    volumes_subdir = os.path.join(staging_server_dir, "docker-volumes")
    os.makedirs(volumes_subdir, exist_ok=True)
    exported_count = _collect_docker_volumes(alias, server_name, volumes_subdir)
    result["files_collected"] += exported_count

    # -----------------------------------------------------------------------
    # Phase 4: alfred-labs specific — copy SQLite DBs and ChromaDB
    # -----------------------------------------------------------------------
    if server_name == "alfred-labs":
        data_count = _collect_alfred_labs_data(staging_server_dir)
        result["files_collected"] += data_count

    logger.info(
        "[%s] Weekly collection complete — %d total artifact(s)",
        server_name, result["files_collected"],
    )

    # -----------------------------------------------------------------------
    # Pack tarball
    # -----------------------------------------------------------------------
    tarball_name = f"{server_name}-weekly-{date_str}.tar.gz"
    tarball_dir = os.path.join(STAGING_DIR, "weekly", date_str)
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
    # Upload to Drive
    # -----------------------------------------------------------------------
    try:
        drive_result = upload_backup(tarball_path, server_name, "weekly")
        result["drive_id"] = drive_result.get("id")
        result["drive_link"] = drive_result.get("link")
        result["success"] = True
        logger.info("[%s] Uploaded to Drive (id=%s)", server_name, result["drive_id"])
    except Exception as exc:
        logger.error("[%s] Drive upload failed: %s", server_name, exc)
        result["error"] = f"Drive upload failed: {exc}"
        return result

    # -----------------------------------------------------------------------
    # Clean up staging dir and local tarball
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
    """Run weekly backup across all servers, then prune old backups.

    Returns:
        Exit code: 0 if all servers succeeded, 1 if any failed.
    """
    _configure_logging()

    run_start = time.time()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("Alfred Weekly Full Backup — %s", date_str)
    logger.info("Servers: %d", len(SERVERS))
    logger.info("Retention: 30 days (daily + weekly)")
    logger.info("=" * 60)

    results = []
    for server in SERVERS:
        try:
            r = backup_server(server, date_str)
        except Exception as exc:
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
    # Retention cleanup — prune daily AND weekly folders older than 30 days
    # -----------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("Running 30-day retention cleanup (daily + weekly)...")
    try:
        prune_summary = prune_old_backups(retention_days=30)
        total_deleted = sum(
            len(btypes.get(btype, {}).get("deleted", []))
            for server_data in prune_summary.values()
            for btype in ["daily", "weekly"]
            for btypes in [server_data]
        )
        logger.info("Retention cleanup complete — %d folder(s) pruned", total_deleted)
        for server_name, btypes in prune_summary.items():
            for btype, info in btypes.items():
                deleted = info.get("deleted", [])
                if deleted:
                    logger.info(
                        "  Pruned %s/%s: %s",
                        server_name, btype, ", ".join(deleted),
                    )
    except Exception as exc:
        logger.error("Retention cleanup failed: %s", exc)
        # Non-fatal — backups succeeded even if cleanup fails

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    duration = time.time() - run_start
    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info("=" * 60)
    logger.info("Weekly backup complete in %.1fs", duration)
    logger.info("Succeeded: %d/%d", len(succeeded), len(SERVERS))
    if succeeded:
        logger.info("  %s", ", ".join(r["server"] for r in succeeded))
    if failed:
        logger.warning("Failed: %d/%d", len(failed), len(SERVERS))
        for r in failed:
            logger.warning("  %s — %s", r["server"], r.get("error", "unknown"))
    logger.info("=" * 60)

    if failed:
        send_backup_alert("weekly", results)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
