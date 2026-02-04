"""Server management - SSH connections and Docker control for remote servers."""

import json
import logging
from pathlib import Path

import paramiko

from config.settings import settings

logger = logging.getLogger(__name__)

SERVERS_FILE = Path(settings.base_dir) / "config" / "servers.json"


def _load_servers() -> dict:
    if SERVERS_FILE.exists():
        return json.loads(SERVERS_FILE.read_text())
    return {}


def _save_servers(servers: dict):
    SERVERS_FILE.write_text(json.dumps(servers, indent=2))
    SERVERS_FILE.chmod(0o600)


def add_server(
    name: str,
    host: str,
    username: str = "root",
    port: int = 22,
    key_path: str = "",
    description: str = "",
) -> bool:
    """Register a server for management."""
    servers = _load_servers()
    servers[name] = {
        "host": host,
        "username": username,
        "port": port,
        "key_path": key_path,
        "description": description,
    }
    _save_servers(servers)
    logger.info(f"Server registered: {name} ({host})")
    return True


def list_servers() -> list[dict]:
    """List all registered servers."""
    servers = _load_servers()
    return [{"name": k, **v} for k, v in servers.items()]


def _connect(server_name: str) -> paramiko.SSHClient:
    """Create an SSH connection to a named server."""
    servers = _load_servers()
    if server_name not in servers:
        raise ValueError(f"Unknown server: {server_name}")

    server = servers[server_name]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": server["host"],
        "username": server["username"],
        "port": server.get("port", 22),
        "timeout": 10,
    }

    if server.get("key_path"):
        connect_kwargs["key_filename"] = server["key_path"]

    client.connect(**connect_kwargs)
    return client


def run_command(server_name: str, command: str, timeout: int = 30) -> dict:
    """Run a command on a remote server."""
    logger.info(f"Running on {server_name}: {command}")
    try:
        client = _connect(server_name)
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        client.close()

        return {
            "server": server_name,
            "command": command,
            "output": output,
            "error": error,
            "exit_code": exit_code,
        }
    except Exception as e:
        logger.error(f"Command failed on {server_name}: {e}")
        return {
            "server": server_name,
            "command": command,
            "output": "",
            "error": str(e),
            "exit_code": -1,
        }


def get_server_status(server_name: str) -> dict:
    """Get system status of a remote server."""
    results = {}

    # Uptime + load
    r = run_command(server_name, "uptime")
    results["uptime"] = r["output"].strip()

    # Disk usage
    r = run_command(server_name, "df -h / | tail -1")
    results["disk"] = r["output"].strip()

    # Memory
    r = run_command(server_name, "free -h | grep Mem")
    results["memory"] = r["output"].strip()

    # Docker containers
    r = run_command(server_name, "docker ps --format '{{.Names}}: {{.Status}}' 2>/dev/null || echo 'No Docker'")
    results["docker"] = r["output"].strip()

    return {"server": server_name, "status": results}


def docker_ps(server_name: str) -> str:
    """List Docker containers on a server."""
    r = run_command(server_name, "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'")
    return r["output"]


def docker_restart(server_name: str, container: str) -> str:
    """Restart a Docker container on a server."""
    r = run_command(server_name, f"docker restart {container}")
    return r["output"] if r["exit_code"] == 0 else f"Error: {r['error']}"


# ==================== SAFE UPDATE WORKFLOW ====================

def check_updates(server_name: str) -> dict:
    """Check what updates are available on a server WITHOUT installing them.

    Returns list of upgradable packages and summary.
    """
    servers = _load_servers()
    if server_name not in servers:
        return {"error": f"Unknown server: {server_name}"}

    server = servers[server_name]
    is_production = "prod" in server_name.lower()

    # Update package lists
    run_command(server_name, "sudo apt update -qq", timeout=120)

    # Get upgradable packages
    r = run_command(server_name, "apt list --upgradable 2>/dev/null | grep -v 'Listing'")
    packages = [line.strip() for line in r["output"].strip().split("\n") if line.strip()]

    # Check if reboot is required
    reboot_check = run_command(server_name, "[ -f /var/run/reboot-required ] && echo 'yes' || echo 'no'")
    reboot_required = reboot_check["output"].strip() == "yes"

    # Get security updates count
    sec_check = run_command(server_name, "apt list --upgradable 2>/dev/null | grep -i security | wc -l")
    security_count = int(sec_check["output"].strip() or 0)

    return {
        "server": server_name,
        "description": server.get("description", ""),
        "is_production": is_production,
        "packages_available": len(packages),
        "security_updates": security_count,
        "reboot_required": reboot_required,
        "packages": packages[:20],  # Limit to first 20 for readability
        "total_packages": len(packages),
        "warning": "⚠️ PRODUCTION SERVER - Confirm before updating" if is_production else None
    }


def check_all_updates() -> list[dict]:
    """Check updates available on ALL servers."""
    servers = _load_servers()
    results = []
    for name in servers:
        results.append(check_updates(name))
    return results


def run_updates(server_name: str, confirm_production: bool = False) -> dict:
    """Run apt update && apt upgrade on a server.

    Args:
        server_name: Server to update
        confirm_production: Must be True to update production servers

    Returns:
        Update results
    """
    servers = _load_servers()
    if server_name not in servers:
        return {"error": f"Unknown server: {server_name}"}

    server = servers[server_name]
    is_production = "prod" in server_name.lower()

    # Safety check for production
    if is_production and not confirm_production:
        return {
            "error": "BLOCKED: Production server update requires explicit confirmation",
            "server": server_name,
            "is_production": True,
            "action_required": "Set confirm_production=True to proceed, or ask user to confirm"
        }

    logger.info(f"Running updates on {server_name} (production={is_production})")

    # Run the update
    result = run_command(
        server_name,
        "sudo DEBIAN_FRONTEND=noninteractive apt update && sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y",
        timeout=600  # 10 minutes for updates
    )

    # Check if reboot needed after update
    reboot_check = run_command(server_name, "[ -f /var/run/reboot-required ] && echo 'yes' || echo 'no'")
    reboot_required = reboot_check["output"].strip() == "yes"

    # Get summary of what was done
    success = result["exit_code"] == 0

    return {
        "server": server_name,
        "success": success,
        "is_production": is_production,
        "output": result["output"][-2000:] if success else result["error"],  # Last 2000 chars
        "reboot_required": reboot_required,
        "message": f"✅ Updates completed on {server_name}" if success else f"❌ Update failed on {server_name}"
    }


def run_autoremove(server_name: str) -> dict:
    """Remove unused packages on a server."""
    result = run_command(server_name, "sudo apt autoremove -y", timeout=120)
    return {
        "server": server_name,
        "success": result["exit_code"] == 0,
        "output": result["output"] if result["exit_code"] == 0 else result["error"]
    }


def reboot_server(server_name: str, confirm: bool = False) -> dict:
    """Reboot a server. Requires explicit confirmation.

    Args:
        server_name: Server to reboot
        confirm: Must be True to proceed
    """
    if not confirm:
        return {
            "error": "BLOCKED: Reboot requires explicit confirmation",
            "server": server_name,
            "action_required": "Set confirm=True to proceed"
        }

    servers = _load_servers()
    if server_name not in servers:
        return {"error": f"Unknown server: {server_name}"}

    logger.info(f"Rebooting server: {server_name}")
    run_command(server_name, "sudo reboot", timeout=5)

    return {
        "server": server_name,
        "success": True,
        "message": f"🔄 Reboot initiated on {server_name}. Server will be back in ~1-2 minutes."
    }


# ==================== MAILCOW SPECIFIC ====================

MAILCOW_PATH = "/opt/mailcow-dockerized"


def mailcow_check_updates() -> dict:
    """Check if Mailcow has updates available.

    Returns current version, latest version, and list of pending updates.
    """
    server = "mailcow-prod"

    # Ensure git safe directory
    run_command(server, f"git config --global --add safe.directory {MAILCOW_PATH}")

    # Get current version
    r = run_command(server, f"cd {MAILCOW_PATH} && git describe --tags 2>/dev/null || git log -1 --format='%h'")
    current_version = r["output"].strip()

    # Fetch updates (using sudo to handle permissions)
    run_command(server, f"cd {MAILCOW_PATH} && sudo git fetch --all --tags 2>/dev/null", timeout=30)

    # Get latest tag
    r = run_command(server, f"cd {MAILCOW_PATH} && git describe --tags $(git rev-list --tags --max-count=1) 2>/dev/null")
    latest_tag = r["output"].strip() or "unknown"

    # Get pending commits
    r = run_command(server, f"cd {MAILCOW_PATH} && git log HEAD..origin/master --oneline 2>/dev/null")
    pending_commits = [line.strip() for line in r["output"].strip().split("\n") if line.strip()]

    # Check Docker image updates
    r = run_command(server, f"cd {MAILCOW_PATH} && docker compose pull --dry-run 2>&1 | grep -i 'pull' | head -5 || echo ''", timeout=60)
    docker_updates = r["output"].strip()

    has_updates = len(pending_commits) > 0 or bool(docker_updates)

    return {
        "server": server,
        "current_version": current_version,
        "latest_release": latest_tag,
        "has_updates": has_updates,
        "pending_commits": len(pending_commits),
        "commits": pending_commits[:10],  # Show first 10
        "docker_updates": docker_updates if docker_updates else "No Docker image updates",
        "message": f"⚠️ Mailcow update available: {current_version} → {latest_tag}" if has_updates else f"✅ Mailcow is up to date ({current_version})"
    }


def mailcow_update(confirm: bool = False) -> dict:
    """Update Mailcow to the latest version.

    This runs the official update.sh script which:
    1. Pulls latest code from git
    2. Pulls new Docker images
    3. Recreates containers

    Args:
        confirm: Must be True to proceed (Mailcow is production mail server)
    """
    server = "mailcow-prod"

    if not confirm:
        # First check what updates are available
        check = mailcow_check_updates()
        return {
            "error": "BLOCKED: Mailcow update requires explicit confirmation",
            "server": server,
            "current_version": check.get("current_version"),
            "latest_version": check.get("latest_release"),
            "pending_updates": check.get("pending_commits", 0),
            "action_required": "This is your PRODUCTION mail server. Confirm to proceed.",
            "warning": "⚠️ Mail service may be briefly interrupted during update"
        }

    logger.info("Starting Mailcow update on mailcow-prod")

    # Run the official update script with auto-confirm
    # The -f flag forces the update without interactive prompts
    result = run_command(
        server,
        f"cd {MAILCOW_PATH} && sudo ./update.sh -f 2>&1 | tail -50",
        timeout=600  # 10 minutes for update
    )

    success = result["exit_code"] == 0

    # Get new version after update
    r = run_command(server, f"cd {MAILCOW_PATH} && git describe --tags 2>/dev/null")
    new_version = r["output"].strip()

    return {
        "server": server,
        "success": success,
        "new_version": new_version,
        "output": result["output"][-3000:],  # Last 3000 chars
        "message": f"✅ Mailcow updated to {new_version}" if success else "❌ Mailcow update failed - check output"
    }


def mailcow_restart() -> dict:
    """Restart all Mailcow containers."""
    server = "mailcow-prod"

    logger.info("Restarting Mailcow containers")

    result = run_command(
        server,
        f"cd {MAILCOW_PATH} && docker compose restart 2>&1",
        timeout=180
    )

    return {
        "server": server,
        "success": result["exit_code"] == 0,
        "output": result["output"] if result["exit_code"] == 0 else result["error"],
        "message": "✅ Mailcow containers restarted" if result["exit_code"] == 0 else "❌ Restart failed"
    }


def mailcow_status() -> dict:
    """Get Mailcow container status and health."""
    server = "mailcow-prod"

    # Get container status
    r = run_command(server, "docker ps --filter 'name=mailcow' --format 'table {{.Names}}\t{{.Status}}' | head -20")
    containers = r["output"]

    # Get disk usage
    r = run_command(server, f"du -sh {MAILCOW_PATH}/data/*/  2>/dev/null | sort -hr | head -5")
    disk_usage = r["output"]

    # Get mail queue
    r = run_command(server, "docker exec $(docker ps -qf 'name=postfix-mailcow') postqueue -p 2>/dev/null | tail -5 || echo 'Queue check failed'")
    mail_queue = r["output"]

    return {
        "server": server,
        "containers": containers,
        "disk_usage": disk_usage,
        "mail_queue": mail_queue
    }


# ==================== HOME ASSISTANT ====================

HOMEASSISTANT_PATH = "/opt/homeassistant"
HOMEASSISTANT_SERVER = "lonewolf-dev"


def homeassistant_status() -> dict:
    """Get Home Assistant status."""
    server = HOMEASSISTANT_SERVER

    # Container status
    r = run_command(server, "docker ps --filter 'name=homeassistant' --format '{{.Status}}'")
    container_status = r["output"].strip() or "Not running"

    # Check if responding
    r = run_command(server, "curl -s -o /dev/null -w '%{http_code}' http://localhost:8123 2>/dev/null || echo '000'")
    http_status = r["output"].strip()

    # Get recent logs
    r = run_command(server, "docker logs homeassistant 2>&1 | tail -10")
    logs = r["output"]

    return {
        "server": server,
        "url": "http://75.43.156.117:8123",
        "container_status": container_status,
        "http_responding": http_status == "200",
        "http_status": http_status,
        "recent_logs": logs
    }


def homeassistant_restart() -> dict:
    """Restart Home Assistant container."""
    server = HOMEASSISTANT_SERVER

    r = run_command(server, f"cd {HOMEASSISTANT_PATH} && docker compose restart 2>&1", timeout=120)

    return {
        "success": r["exit_code"] == 0,
        "output": r["output"] if r["exit_code"] == 0 else r["error"],
        "message": "✅ Home Assistant restarted" if r["exit_code"] == 0 else "❌ Restart failed"
    }


def homeassistant_update() -> dict:
    """Update Home Assistant to latest version."""
    server = HOMEASSISTANT_SERVER

    # Pull latest image
    r = run_command(server, f"cd {HOMEASSISTANT_PATH} && docker compose pull 2>&1", timeout=300)
    pull_output = r["output"]

    # Recreate container
    r = run_command(server, f"cd {HOMEASSISTANT_PATH} && docker compose up -d 2>&1", timeout=120)

    return {
        "success": r["exit_code"] == 0,
        "pull_output": pull_output[-1000:],
        "message": "✅ Home Assistant updated" if r["exit_code"] == 0 else "❌ Update failed"
    }


def homeassistant_logs(lines: int = 50) -> dict:
    """Get Home Assistant logs."""
    server = HOMEASSISTANT_SERVER

    r = run_command(server, f"docker logs homeassistant 2>&1 | tail -{lines}")

    return {
        "server": server,
        "logs": r["output"]
    }
