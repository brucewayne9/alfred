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
