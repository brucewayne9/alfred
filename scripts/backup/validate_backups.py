#!/usr/bin/env python3
"""
Alfred Backup — validation script.

Checks Google Drive to verify that each server's latest daily and weekly
backups are present, recent, and non-empty.

Runs at 5 AM daily (cron) — 3 hours after the 2 AM backup window so all
Drive uploads have time to complete.

Output:
    data/backup_status.json — machine-readable per-server status dict
    data/backup_validate.log — human-readable log

Usage:
    python3 scripts/backup/validate_backups.py
    /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/backup/validate_backups.py
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Resolve project root so imports work from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from scripts.backup.backup_alerting import send_backup_alert
from scripts.backup.backup_utils import SERVERS
from scripts.backup.drive_manager import list_date_folders

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

LOG_FILE = os.path.join(_PROJECT_ROOT, "data", "backup_validate.log")
STATUS_FILE = os.path.join(_PROJECT_ROOT, "data", "backup_status.json")

# ---------------------------------------------------------------------------
# Logging setup — matches daily_backup.py pattern
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure logging to stdout and the log file."""
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


logger = logging.getLogger("validate_backups")

# ---------------------------------------------------------------------------
# Staleness thresholds
# ---------------------------------------------------------------------------

# Daily: allow up to 26 hours to account for timezone drift + upload time
DAILY_MAX_AGE_HOURS = 26

# Weekly: allow up to 170 hours (~7 days + 2h buffer)
WEEKLY_MAX_AGE_HOURS = 170


# ---------------------------------------------------------------------------
# Per-server validation
# ---------------------------------------------------------------------------


def validate_server(server_name: str, backup_type: str, expected_within_hours: int) -> dict:
    """Check Drive for the latest backup of a single server+type.

    Args:
        server_name: Short server identifier (e.g. "alfred-labs").
        backup_type: "daily" or "weekly".
        expected_within_hours: Max age in hours before flagging as "stale".

    Returns:
        Dict with keys:
            server, backup_type, latest_date (str|None), age_hours (float|None),
            file_count (int), total_size_bytes (int), status (str)

        Status values:
            "ok"      — latest folder found, recent, and has files
            "stale"   — latest folder found but older than expected_within_hours
            "empty"   — latest folder found but has no files
            "missing" — no date folders found at all
    """
    result: dict = {
        "server": server_name,
        "backup_type": backup_type,
        "latest_date": None,
        "age_hours": None,
        "file_count": 0,
        "total_size_bytes": 0,
        "status": "missing",
    }

    try:
        folders = list_date_folders(server_name, backup_type)
    except Exception as exc:
        logger.warning(
            "[%s/%s] Failed to list date folders: %s",
            server_name, backup_type, exc,
        )
        result["status"] = "missing"
        return result

    if not folders:
        logger.warning("[%s/%s] No backup folders found", server_name, backup_type)
        result["status"] = "missing"
        return result

    # latest folder is last when sorted by name (YYYY-MM-DD ascending)
    latest = folders[-1]
    latest_date_str = latest["name"]
    result["latest_date"] = latest_date_str

    # Parse the date and compute age
    now_utc = datetime.now(timezone.utc)
    try:
        folder_date = datetime.strptime(latest_date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        age_hours = (now_utc - folder_date).total_seconds() / 3600
        result["age_hours"] = round(age_hours, 1)
    except ValueError as exc:
        logger.warning(
            "[%s/%s] Could not parse folder date '%s': %s",
            server_name, backup_type, latest_date_str, exc,
        )
        result["status"] = "stale"
        return result

    # List files inside the latest date folder
    try:
        from integrations.google_drive.client import list_files
        files_in_folder = list_files(folder_id=latest["id"], max_results=50)
    except Exception as exc:
        logger.warning(
            "[%s/%s] Failed to list files in folder %s: %s",
            server_name, backup_type, latest_date_str, exc,
        )
        # Can't confirm files — treat as empty (conservative)
        result["status"] = "empty"
        return result

    file_count = len(files_in_folder)
    total_size = 0
    for f in files_in_folder:
        size_str = f.get("size")
        if size_str:
            try:
                total_size += int(size_str)
            except (ValueError, TypeError):
                pass

    result["file_count"] = file_count
    result["total_size_bytes"] = total_size

    logger.info(
        "[%s/%s] Latest: %s (%.1fh ago) — %d file(s), %d bytes",
        server_name, backup_type, latest_date_str,
        result["age_hours"], file_count, total_size,
    )

    # Determine status
    if file_count == 0:
        result["status"] = "empty"
    elif age_hours > expected_within_hours:
        result["status"] = "stale"
    else:
        result["status"] = "ok"

    return result


# ---------------------------------------------------------------------------
# Full validation sweep
# ---------------------------------------------------------------------------


def validate_all_servers() -> dict:
    """Validate daily and weekly backups for all 7 servers.

    Returns:
        Dict with keys:
            timestamp (str): ISO8601 UTC timestamp
            servers (dict): {server_name: {daily: {...}, weekly: {...}}}
            summary (dict): {total_checks, ok, stale, empty, missing}
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    servers_result: dict = {}
    summary = {"total_checks": 0, "ok": 0, "stale": 0, "empty": 0, "missing": 0}

    # Determine the day of week (0=Monday, 6=Sunday) to know if a weekly
    # backup has had a chance to run.  Weekly cron fires Sunday 2 AM UTC.
    now_utc = datetime.now(timezone.utc)
    # Consider weekly expected only if today is Monday or later in the week
    # following a Sunday, i.e. at least one Sunday has passed since system
    # setup.  We detect this by checking if any weekly folder exists for
    # *any* server; if none do, we skip weekly validation entirely.
    any_weekly_exists = False

    for server in SERVERS:
        server_name = server["name"]
        servers_result[server_name] = {}

        for backup_type, max_hours in [
            ("daily", DAILY_MAX_AGE_HOURS),
            ("weekly", WEEKLY_MAX_AGE_HOURS),
        ]:
            check = validate_server(server_name, backup_type, max_hours)
            servers_result[server_name][backup_type] = check
            summary["total_checks"] += 1

            if backup_type == "weekly" and check["status"] != "missing":
                any_weekly_exists = True

            status = check["status"]
            if status in summary:
                summary[status] += 1

    # If NO weekly backups exist for ANY server, the weekly cron hasn't
    # fired yet.  Downgrade all weekly "missing" to "ok" so we don't send
    # a false-alarm alert before the first Sunday run.
    if not any_weekly_exists:
        logger.info("No weekly backups found for any server — first Sunday "
                     "run has not occurred yet, suppressing weekly alerts")
        for server_name in servers_result:
            wk = servers_result[server_name].get("weekly", {})
            if wk.get("status") == "missing":
                wk["status"] = "pending_first_run"
                summary["missing"] -= 1
                summary["ok"] += 1

    return {
        "timestamp": timestamp,
        "servers": servers_result,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run full backup validation, write status JSON, alert if issues found.

    Returns:
        Exit code: 0 if all backups ok, 1 if any issues.
    """
    _configure_logging()

    logger.info("=" * 60)
    logger.info("Alfred Backup Validation — %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    logger.info("Checking %d servers (daily + weekly = %d checks)", len(SERVERS), len(SERVERS) * 2)
    logger.info("=" * 60)

    validation = validate_all_servers()
    summary = validation["summary"]

    # Write status JSON atomically (write to .tmp then rename)
    status_path = os.path.abspath(STATUS_FILE)
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    tmp_path = status_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(validation, fh, indent=2)
        os.rename(tmp_path, status_path)
        logger.info("Status written to %s", status_path)
    except Exception as exc:
        logger.error("Failed to write backup_status.json: %s", exc)
        # Non-fatal — continue to alert even if file write fails

    # Log summary
    logger.info("=" * 60)
    logger.info(
        "Validation complete — %d total checks: %d ok, %d stale, %d empty, %d missing",
        summary["total_checks"],
        summary["ok"],
        summary["stale"],
        summary["empty"],
        summary["missing"],
    )

    # Collect all issues for alerting
    issues_count = summary["stale"] + summary["empty"] + summary["missing"]

    if issues_count > 0:
        logger.warning("Found %d issue(s) — sending Telegram alert", issues_count)

        # Build a flat results_list compatible with send_backup_alert()
        # Format each issue as a pseudo-result dict
        alert_results = []
        for server_name, btypes in validation["servers"].items():
            for btype, check in btypes.items():
                if check["status"] not in ("ok", "pending_first_run"):
                    age_info = ""
                    if check.get("age_hours") is not None:
                        age_info = f" ({check['age_hours']:.0f}h old)"
                    error_msg = (
                        f"{btype} backup {check['status']}{age_info}"
                        f" — latest: {check.get('latest_date', 'none')}"
                        f", {check.get('file_count', 0)} file(s)"
                    )
                    alert_results.append({
                        "server": server_name,
                        "success": False,
                        "error": error_msg,
                    })

        send_backup_alert("validation", alert_results)
        logger.info("=" * 60)
        return 1

    logger.info("All backups validated successfully")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
