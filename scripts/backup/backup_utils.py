"""
Alfred Backup — shared utilities for daily and weekly backup scripts.

Provides:
- SERVERS: list of all infrastructure servers with backup metadata
- DAILY_TARGETS: per-server file/command targets for daily config backups
- WEEKLY_EXTRAS: per-server additional targets for weekly full backups
- run_cmd(): execute commands locally or via SSH
- collect_remote_files(): copy files from remote/local servers to staging dir
- pack_tarball(): create a tar.gz from a staging directory
- cleanup_staging(): remove a staging directory after upload
- STAGING_DIR: base temp directory for in-progress backup staging
"""

import logging
import os
import shutil
import subprocess
import sys
import tarfile

# Resolve project root so imports work when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base staging directory — each backup run creates a dated subdirectory
# ---------------------------------------------------------------------------
STAGING_DIR = "/tmp/alfred-backups"

# ---------------------------------------------------------------------------
# Server registry
# Enriches the audit.py SERVERS list with backup-specific metadata.
# alias=None means the command runs locally (alfred-labs, 105)
# ---------------------------------------------------------------------------
SERVERS = [
    {
        "alias": None,
        "ip": "75.43.156.105",
        "name": "alfred-labs",
        "description": "Alfred Labs (local)",
    },
    {
        "alias": "server-98",
        "ip": "75.43.156.98",
        "name": "groundrush-radio",
        "description": "GroundRushRadio",
    },
    {
        "alias": "server-100",
        "ip": "75.43.156.100",
        "name": "labs-edge",
        "description": "labs-edge-server",
    },
    {
        "alias": "claw",
        "ip": "75.43.156.101",
        "name": "alfred-claw",
        "description": "Alfred Claw",
    },
    {
        "alias": "server-104",
        "ip": "75.43.156.104",
        "name": "labsliveserver",
        "description": "labsliveserver",
    },
    {
        "alias": "lonewolf",
        "ip": "75.43.156.117",
        "name": "lonewolf",
        "description": "Lonewolf/Dokploy",
    },
    {
        "alias": "server-121",
        "ip": "75.43.156.121",
        "name": "cloud-mail",
        "description": "gloundrush-cloud-mail",
    },
]

# ---------------------------------------------------------------------------
# Per-server daily backup targets
# Each value is a dict with:
#   "files":    list of file paths to collect via scp/cp
#   "commands": list of (output_filename, shell_command) tuples — command
#               stdout is captured and saved as output_filename
# ---------------------------------------------------------------------------

# Common targets applied to all servers
_COMMON_FILES = [
    "/etc/crontab",
]
_COMMON_COMMANDS = [
    ("crontab-user.txt", "crontab -l 2>/dev/null || echo '(none)'"),
    ("systemd-custom-services.txt",
     "find /etc/systemd/system -maxdepth 1 -name '*.service' "
     "-not -name '*.wants' 2>/dev/null | xargs ls -la 2>/dev/null || echo '(none)'"),
]

DAILY_TARGETS: dict[str, dict] = {
    "alfred-labs": {
        "files": _COMMON_FILES + [
            "/home/aialfred/alfred/config/.env",
            "/home/aialfred/alfred/config/settings.py",
            "/home/aialfred/alfred/config/users.json",
            "/home/aialfred/alfred/config/google_token.json",
            "/var/lib/redis/dump.rdb",
        ],
        "commands": _COMMON_COMMANDS + [
            ("postgresql-dump.sql", "pg_dumpall -U postgres 2>/dev/null"),
        ],
    },
    "groundrush-radio": {
        "files": _COMMON_FILES,
        "commands": _COMMON_COMMANDS + [
            ("dotenv-files.txt",
             "find /opt /var/www /home -maxdepth 4 -name '.env' 2>/dev/null "
             "| head -20"),
            ("azuracast-config.txt",
             "find /var/azuracast /opt/azuracast /home/azuracast -maxdepth 3 "
             "-name '*.conf' -o -name '*.env' 2>/dev/null | head -20"),
        ],
    },
    "labs-edge": {
        "files": _COMMON_FILES,
        "commands": _COMMON_COMMANDS + [
            ("mysql-dump.sql",
             "mysqldump --all-databases 2>/dev/null "
             "|| mysqldump --all-databases --no-tablespaces 2>/dev/null "
             "|| echo '(mysql dump failed)'"),
            ("dotenv-files.txt",
             "find /opt -maxdepth 4 -name '.env' 2>/dev/null | head -20"),
        ],
    },
    "alfred-claw": {
        "files": _COMMON_FILES + [
            "/root/.openclaw/openclaw.json",
            "/root/.openclaw/workspace/USER.md",
            "/root/.openclaw/workspace/SOUL.md",
            "/root/.openclaw/workspace/AGENTS.md",
            "/root/.openclaw/workspace/TOOLS.md",
            "/root/.openclaw/workspace/HEARTBEAT.md",
            "/root/.openclaw/workspace/QUEUE.md",
        ],
        "commands": _COMMON_COMMANDS,
    },
    "labsliveserver": {
        # 55 containers — do NOT export all volumes daily (too heavy)
        "files": _COMMON_FILES,
        "commands": _COMMON_COMMANDS + [
            ("mysql-dump.sql",
             "mysqldump --all-databases 2>/dev/null "
             "|| mysqldump --all-databases --no-tablespaces 2>/dev/null "
             "|| echo '(mysql dump failed)'"),
            ("dotenv-files.txt",
             "find /opt -maxdepth 4 -name '.env' 2>/dev/null | head -40"),
        ],
    },
    "lonewolf": {
        "files": _COMMON_FILES,
        "commands": _COMMON_COMMANDS + [
            ("dokploy-config.txt",
             "find /opt/dokploy /var/lib/dokploy 2>/dev/null "
             "-name '*.json' -o -name '*.yml' | head -20"),
            ("traefik-config.txt",
             "find /etc/traefik /opt/traefik 2>/dev/null "
             "-name '*.yml' -o -name '*.yaml' -o -name '*.toml' | head -20"),
        ],
    },
    "cloud-mail": {
        "files": _COMMON_FILES,
        "commands": _COMMON_COMMANDS + [
            ("mail-config.txt",
             "find /etc/postfix /etc/dovecot /etc/exim4 /opt/mail 2>/dev/null "
             "-name '*.conf' -o -name '*.cf' | head -30"),
            ("dotenv-files.txt",
             "find /opt /var/www /home -maxdepth 4 -name '.env' 2>/dev/null "
             "| head -20"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Per-server weekly extras (in addition to DAILY_TARGETS)
# Same schema: files + commands
# ---------------------------------------------------------------------------

_COMMON_WEEKLY_COMMANDS = [
    ("packages-dpkg.txt", "dpkg --list 2>/dev/null || rpm -qa 2>/dev/null || echo '(none)'"),
    ("docker-volumes.txt", "docker volume ls 2>/dev/null || echo '(none)'"),
    ("docker-images.txt", "docker image ls 2>/dev/null || echo '(none)'"),
]

WEEKLY_EXTRAS: dict[str, dict] = {
    "alfred-labs": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS + [
            # Full data directory — SQLite, ChromaDB
            ("data-files.txt",
             "find /home/aialfred/alfred/data -type f 2>/dev/null | head -100"),
            ("docker-volume-export.sh",
             "docker volume ls -q 2>/dev/null | while read v; do "
             "echo \"=== $v ===\"; docker inspect $v 2>/dev/null | grep Mountpoint; "
             "done"),
        ],
    },
    "groundrush-radio": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS,
    },
    "labs-edge": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS,
    },
    "alfred-claw": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS,
    },
    "labsliveserver": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS + [
            # Export key service volumes (databases, app data)
            ("docker-volume-export.sh",
             "docker volume ls -q 2>/dev/null | while read v; do "
             "echo \"=== $v ===\"; docker inspect $v 2>/dev/null | grep Mountpoint; "
             "done"),
        ],
    },
    "lonewolf": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS + [
            ("dokploy-app-data.txt",
             "find /opt/dokploy /var/lib/dokploy 2>/dev/null "
             "-type f -name '*.json' | head -30"),
            ("docker-volume-export.sh",
             "docker volume ls -q 2>/dev/null | while read v; do "
             "echo \"=== $v ===\"; docker inspect $v 2>/dev/null | grep Mountpoint; "
             "done"),
        ],
    },
    "cloud-mail": {
        "files": [],
        "commands": _COMMON_WEEKLY_COMMANDS + [
            ("docker-volume-export.sh",
             "docker volume ls -q 2>/dev/null | while read v; do "
             "echo \"=== $v ===\"; docker inspect $v 2>/dev/null | grep Mountpoint; "
             "done"),
        ],
    },
}

# ---------------------------------------------------------------------------
# SSH / local execution helpers
# ---------------------------------------------------------------------------

CONNECT_TIMEOUT = 10
DEFAULT_TIMEOUT = 60


def run_cmd(alias: str | None, cmd: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Execute a shell command locally or via SSH.

    Args:
        alias: SSH config alias (e.g. "claw", "lonewolf"). None means localhost.
        cmd: Shell command string to run.
        timeout: Seconds before raising TimeoutError. Default 60.

    Returns:
        Command stdout as a string (stripped).

    Raises:
        RuntimeError: On timeout or non-zero exit with stderr output.
    """
    if alias is None:
        # Run locally
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0 and result.stderr:
                logger.debug("local cmd stderr: %s", result.stderr.strip())
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Local command timed out after {timeout}s: {cmd!r}")
        except Exception as exc:
            raise RuntimeError(f"Local command error: {exc} — cmd={cmd!r}") from exc
    else:
        # Run via SSH
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={CONNECT_TIMEOUT}",
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=no",
                    alias,
                    cmd,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0 and result.stderr:
                logger.debug("ssh %s stderr: %s", alias, result.stderr.strip())
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"SSH command timed out after {timeout}s — alias={alias!r}, cmd={cmd!r}"
            )
        except Exception as exc:
            raise RuntimeError(
                f"SSH command error: {exc} — alias={alias!r}, cmd={cmd!r}"
            ) from exc


def collect_remote_files(
    alias: str | None,
    remote_paths: list[str],
    local_dest: str,
) -> list[str]:
    """Copy files from a remote (or local) server into local_dest/.

    For remote servers, uses scp. For local (alias=None), uses cp.
    Skips files that don't exist on the source — logs a warning for each.

    Args:
        alias: SSH config alias, or None for local.
        remote_paths: List of absolute paths on the source server.
        local_dest: Local directory to copy files into (must exist).

    Returns:
        List of successfully copied local file paths.
    """
    os.makedirs(local_dest, exist_ok=True)
    collected = []

    for remote_path in remote_paths:
        filename = os.path.basename(remote_path)
        if not filename:
            logger.warning("Skipping path with no filename: %s", remote_path)
            continue

        dest_file = os.path.join(local_dest, filename)

        if alias is None:
            # Local copy
            if not os.path.exists(remote_path):
                logger.warning("File not found locally, skipping: %s", remote_path)
                continue
            try:
                shutil.copy2(remote_path, dest_file)
                logger.debug("Copied local: %s -> %s", remote_path, dest_file)
                collected.append(dest_file)
            except Exception as exc:
                logger.warning("Failed to copy %s: %s", remote_path, exc)
        else:
            # SCP from remote
            try:
                result = subprocess.run(
                    [
                        "scp",
                        "-o", f"ConnectTimeout={CONNECT_TIMEOUT}",
                        "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no",
                        f"{alias}:{remote_path}",
                        dest_file,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_TIMEOUT,
                )
                if result.returncode == 0:
                    logger.debug("SCP: %s:%s -> %s", alias, remote_path, dest_file)
                    collected.append(dest_file)
                else:
                    logger.warning(
                        "SCP failed for %s:%s — %s",
                        alias, remote_path, result.stderr.strip(),
                    )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "SCP timed out for %s:%s", alias, remote_path
                )
            except Exception as exc:
                logger.warning("SCP error for %s:%s — %s", alias, remote_path, exc)

    return collected


def pack_tarball(source_dir: str, output_path: str) -> str:
    """Create a gzip-compressed tar archive of source_dir.

    Args:
        source_dir: Directory to archive (all contents included).
        output_path: Destination file path for the .tar.gz archive.

    Returns:
        The output_path (same as input, for chaining convenience).

    Raises:
        FileNotFoundError: If source_dir does not exist.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

    size_kb = os.path.getsize(output_path) // 1024
    logger.info("Packed tarball: %s (%d KB)", output_path, size_kb)
    return output_path


def cleanup_staging(staging_path: str) -> None:
    """Remove a staging directory after successful upload.

    Args:
        staging_path: Path to the staging directory to delete.
    """
    if os.path.isdir(staging_path):
        shutil.rmtree(staging_path)
        logger.info("Cleaned up staging dir: %s", staging_path)
    else:
        logger.debug("cleanup_staging: path not found (already removed?): %s", staging_path)
