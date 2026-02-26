"""
Alfred Backup — Telegram failure alerting module.

Sends Telegram notifications to Mike when backup jobs fail.
Uses Alfred Claw's Telegram channel via SSH to avoid requiring
a separate Telegram bot on Alfred Labs.

Usage:
    from scripts.backup.backup_alerting import send_backup_alert
    if failed_servers:
        send_backup_alert("daily", results)
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

# Mike's Telegram chat ID (used by openclaw message send --target)
MIKE_TELEGRAM_ID = "7582976864"

# SSH alias for Alfred Claw (defined in ~/.ssh/config from Phase 6)
CLAW_SSH_ALIAS = "claw"

# Timeout for the SSH + openclaw call (seconds)
ALERT_SSH_TIMEOUT = 30


def send_backup_alert(backup_type: str, results: list[dict]) -> bool:
    """Send a Telegram alert to Mike for backup failures.

    Filters the results list for failed servers and sends a concise
    Telegram message via Alfred Claw's openclaw CLI.

    Alert failure is non-fatal — it is wrapped in try/except so the
    backup script continues even if Telegram delivery fails.

    Args:
        backup_type: One of "daily", "weekly", or "validation".
        results: List of result dicts, each with keys:
                 - server (str): server name
                 - success (bool): whether the backup succeeded
                 - error (str|None): error message if failed

    Returns:
        True if the alert was sent successfully, False if no failures
        or if the send attempt failed.
    """
    try:
        # Filter for failed entries
        failures = [r for r in results if not r.get("success", True)]
        if not failures:
            logger.debug("send_backup_alert: no failures in results — skipping alert")
            return False

        # Build the alert message
        failure_count = len(failures)
        total_count = len(results)
        lines = [
            f"Backup Alert: {backup_type} backup had {failure_count}/{total_count} failure(s)"
        ]
        for r in failures:
            server = r.get("server", "unknown")
            error = r.get("error") or "unknown error"
            lines.append(f"- {server}: {error}")

        message = "\n".join(lines)

        logger.info(
            "Sending backup alert via Telegram (%s failures in %s backup)",
            failure_count, backup_type,
        )

        # Send via SSH to Alfred Claw's openclaw CLI
        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            CLAW_SSH_ALIAS,
            f"openclaw message send --channel telegram --target {MIKE_TELEGRAM_ID} -m {repr(message)}",
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=ALERT_SSH_TIMEOUT,
        )

        if result.returncode == 0:
            logger.info("Backup alert sent successfully via Telegram")
            return True
        else:
            logger.warning(
                "Backup alert SSH command returned non-zero (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return False

    except subprocess.TimeoutExpired:
        logger.warning(
            "Backup alert timed out after %ds — alert not delivered",
            ALERT_SSH_TIMEOUT,
        )
        return False
    except Exception as exc:
        logger.warning("Backup alert failed (non-fatal): %s", exc)
        return False
