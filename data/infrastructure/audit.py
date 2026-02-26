#!/usr/bin/env python3
"""
Alfred Infrastructure Audit Script
Collects system information from all infrastructure servers via SSH.

Usage:
  python3 audit.py               # Full audit of all servers
  python3 audit.py --markdown-only  # Regenerate markdown from existing JSON
"""

import json
import subprocess
import sys
import os
from datetime import datetime, timezone

# Output paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INVENTORY_JSON = os.path.join(SCRIPT_DIR, "inventory.json")
INVENTORY_MD = os.path.join(SCRIPT_DIR, "inventory.md")

# Server list: (alias, ip, description)
# alias=None means local (105)
SERVERS = [
    (None,         "75.43.156.105", "Alfred Labs (local)"),
    ("server-98",  "75.43.156.98",  "GroundRushRadio"),
    ("server-100", "75.43.156.100", "labs-edge-server"),
    ("claw",       "75.43.156.101", "Alfred Claw"),
    ("server-104", "75.43.156.104", "labsliveserver"),
    ("lonewolf",   "75.43.156.117", "Lonewolf/Dokploy"),
    ("server-121", "75.43.156.121", "gloundrush-cloud-mail"),
]

TIMEOUT = 30


def run_local(cmd: str) -> str:
    """Run a shell command locally and return stdout."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"


def run_remote(alias: str, cmd: str) -> str:
    """Run a shell command on a remote server via SSH and return stdout."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", alias, cmd],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"


def run_cmd(alias: str | None, cmd: str) -> str:
    """Run command locally or remotely based on alias."""
    if alias is None:
        return run_local(cmd)
    return run_remote(alias, cmd)


def audit_server(alias: str | None, ip: str, description: str) -> dict:
    """Collect all data for a single server."""
    print(f"  Auditing {description} ({ip})...", flush=True)

    def r(cmd):
        return run_cmd(alias, cmd)

    data = {
        "alias": alias or "localhost",
        "ip": ip,
        "description": description,
    }

    # --- System info ---
    hostname = r("hostname")
    os_info = r("cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'")
    kernel = r("uname -r")
    uptime = r("uptime -p 2>/dev/null || uptime")
    arch = r("uname -m")

    data["system"] = {
        "hostname": hostname,
        "os": os_info,
        "kernel": kernel,
        "uptime": uptime,
        "arch": arch,
    }

    # --- Disk usage ---
    disk_raw = r("df -h --output=source,size,used,avail,pcent,target 2>/dev/null | grep -v tmpfs | grep -v udev | grep -v loop")
    disk_lines = []
    for line in disk_raw.splitlines():
        parts = line.split()
        if len(parts) >= 6:
            disk_lines.append({
                "filesystem": parts[0],
                "size": parts[1],
                "used": parts[2],
                "avail": parts[3],
                "use_pct": parts[4],
                "mount": parts[5],
            })
    data["disk_usage"] = disk_lines

    # --- Memory ---
    mem_raw = r("free -h 2>/dev/null")
    data["memory_raw"] = mem_raw

    mem_total = ""
    mem_used = ""
    for line in mem_raw.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 3:
                mem_total = parts[1]
                mem_used = parts[2]
    data["memory"] = {"total": mem_total, "used": mem_used}

    # --- Python/Node versions ---
    data["python_version"] = r("python3 --version 2>/dev/null")
    data["node_version"] = r("node --version 2>/dev/null || echo 'not installed'")
    data["package_count"] = r("dpkg --list 2>/dev/null | wc -l || rpm -qa 2>/dev/null | wc -l || echo '?'")

    # --- Docker containers ---
    docker_raw = r("docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null")
    containers = []
    if docker_raw and "(error" not in docker_raw and "(timeout" not in docker_raw:
        for line in docker_raw.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                containers.append({
                    "name": parts[0] if len(parts) > 0 else "",
                    "image": parts[1] if len(parts) > 1 else "",
                    "status": parts[2] if len(parts) > 2 else "",
                    "ports": parts[3] if len(parts) > 3 else "",
                })
    data["docker_containers"] = containers

    # Docker not installed check
    docker_check = r("which docker 2>/dev/null")
    data["docker_installed"] = bool(docker_check and "(error" not in docker_check)

    # Docker compose projects
    compose_raw = r("docker compose ls --format json 2>/dev/null")
    compose_projects = []
    if compose_raw and compose_raw.startswith("["):
        try:
            compose_projects = json.loads(compose_raw)
        except Exception:
            compose_projects = []
    data["docker_compose_projects"] = compose_projects

    # --- Systemd services (running, non-system) ---
    services_raw = r(
        "systemctl list-units --type=service --state=running --no-pager --plain 2>/dev/null"
        " | awk 'NR>1 && /running/{print $1}'"
        " | grep -v '^$'"
    )
    services = [s for s in services_raw.splitlines() if s.strip() and not s.startswith("UNIT")]
    data["services"] = services

    # --- Listening ports ---
    ports_raw = r("ss -tlnp 2>/dev/null")
    port_lines = []
    for line in ports_raw.splitlines():
        if "LISTEN" in line or "Local Address" in line:
            port_lines.append(line)
    data["listening_ports_raw"] = "\n".join(port_lines)

    # --- Databases ---
    databases = {}

    def is_active(status: str) -> bool:
        """Check if systemctl output indicates a service is active (not inactive/failed)."""
        if not status:
            return False
        first_line = status.strip().splitlines()[0].strip()
        return first_line == "active"

    pg_status = r("systemctl is-active postgresql 2>/dev/null || pg_lsclusters 2>/dev/null | tail -1 | awk '{print $4}'")
    databases["postgresql"] = {"detected": is_active(pg_status), "status": pg_status}

    mysql_status = r("systemctl is-active mysql 2>/dev/null || systemctl is-active mariadb 2>/dev/null")
    databases["mysql"] = {"detected": is_active(mysql_status), "status": mysql_status}

    redis_status = r("systemctl is-active redis 2>/dev/null || systemctl is-active redis-server 2>/dev/null")
    databases["redis"] = {"detected": is_active(redis_status), "status": redis_status}

    mongo_status = r("systemctl is-active mongod 2>/dev/null")
    databases["mongodb"] = {"detected": is_active(mongo_status), "status": mongo_status}

    # SQLite files in common locations
    sqlite_files = r(
        "find /opt /home /var/lib /srv 2>/dev/null -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' 2>/dev/null"
        " | grep -v '/proc' | head -20"
    )
    databases["sqlite_files"] = [f for f in sqlite_files.splitlines() if f.strip()]

    data["databases"] = databases

    # --- Cron jobs ---
    user_cron = r("crontab -l 2>/dev/null || echo '(none)'")
    cron_d = r("ls /etc/cron.d/ 2>/dev/null || echo '(empty)'")
    cron_daily = r("ls /etc/cron.daily/ 2>/dev/null || echo '(empty)'")

    data["cron_jobs"] = {
        "user_crontab": user_cron,
        "cron_d_files": cron_d,
        "cron_daily_files": cron_daily,
    }

    # --- Cross-server connection detection ---
    connections = []

    # /etc/hosts references to 75.43.156.*
    hosts_refs = r("grep '75\\.43\\.156\\.' /etc/hosts 2>/dev/null")
    if hosts_refs:
        connections.append({"source": "/etc/hosts", "content": hosts_refs})

    # env files referencing other servers
    env_refs = r(
        "grep -r '75\\.43\\.156\\.' ~/.env /etc/environment /opt/*/.env 2>/dev/null"
        " | grep -v Binary | head -20"
    )
    if env_refs:
        connections.append({"source": "env files", "content": env_refs})

    # Docker container env vars
    if containers:
        docker_env_refs = r(
            "docker inspect $(docker ps -q) 2>/dev/null"
            " | grep -o '\"[^\"]*75\\.43\\.156\\.[^\"]*\"' | sort -u | head -20"
        )
        if docker_env_refs:
            connections.append({"source": "docker env", "content": docker_env_refs})

    # Traefik config (server 117)
    if alias == "lonewolf":
        traefik_refs = r(
            "grep -r '75\\.43\\.156\\.' /etc/traefik/ /opt/traefik/ ~/.config/traefik/ 2>/dev/null"
            " | grep -v Binary | head -30"
        )
        if traefik_refs:
            connections.append({"source": "traefik config", "content": traefik_refs})

        # Dokploy config
        dokploy_refs = r(
            "find /opt/dokploy /var/lib/dokploy 2>/dev/null -name '*.json' -o -name '*.yml' 2>/dev/null"
            " | xargs grep -l '75\\.43\\.156\\.' 2>/dev/null | head -10"
        )
        if dokploy_refs:
            connections.append({"source": "dokploy config", "content": dokploy_refs})

    # Health monitor on 105 (local)
    if alias is None:
        monitor_refs = run_local(
            "grep -o '75\\.43\\.156\\.[0-9]*' /home/aialfred/alfred/scripts/alfred_claw_monitor.py 2>/dev/null | sort -u"
        )
        if monitor_refs:
            connections.append({"source": "alfred_claw_monitor.py", "content": monitor_refs})

    data["cross_server_connections"] = connections

    # Notable findings (auto-detected patterns)
    findings = []

    if len(containers) > 10:
        findings.append(f"High container count: {len(containers)} running Docker containers")
    if any(c.get("use_pct", "0%").rstrip("%").isdigit() and int(c.get("use_pct", "0%").rstrip("%")) > 80 for c in disk_lines):
        findings.append("Disk usage > 80% on one or more filesystems")
    if databases["postgresql"]["detected"]:
        findings.append("PostgreSQL is running")
    if databases["mysql"]["detected"]:
        findings.append("MySQL/MariaDB is running")
    if databases["redis"]["detected"]:
        findings.append("Redis is running")
    if databases["mongodb"]["detected"]:
        findings.append("MongoDB is running")
    if databases["sqlite_files"]:
        findings.append(f"SQLite databases found: {len(databases['sqlite_files'])} files")
    if not data["docker_installed"]:
        findings.append("Docker not installed")
    if compose_projects:
        findings.append(f"Docker Compose projects: {len(compose_projects)}")

    data["notable_findings"] = findings

    return data


def run_full_audit() -> dict:
    """Run the full audit against all servers."""
    print("Alfred Infrastructure Audit", flush=True)
    print("=" * 40, flush=True)
    print(f"Starting at {datetime.now(timezone.utc).isoformat()}", flush=True)
    print(flush=True)

    servers_data = []
    for alias, ip, description in SERVERS:
        try:
            server_data = audit_server(alias, ip, description)
            servers_data.append(server_data)
            print(f"  -> Done: {server_data['system'].get('hostname', ip)}", flush=True)
        except Exception as e:
            print(f"  -> FAILED: {ip} ({e})", flush=True)
            servers_data.append({
                "alias": alias or "localhost",
                "ip": ip,
                "description": description,
                "error": str(e),
                "system": {"hostname": ip},
                "disk_usage": [],
                "memory": {},
                "docker_containers": [],
                "services": [],
                "databases": {},
                "cron_jobs": {},
                "cross_server_connections": [],
                "notable_findings": [f"Audit failed: {e}"],
            })
        print(flush=True)

    # Build cross-server connections map (global)
    all_connections = {}
    for s in servers_data:
        server_key = s.get("system", {}).get("hostname") or s["ip"]
        if s.get("cross_server_connections"):
            all_connections[server_key] = s["cross_server_connections"]

    inventory = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "server_count": len(servers_data),
        "servers": servers_data,
        "connections": all_connections,
    }

    with open(INVENTORY_JSON, "w") as f:
        json.dump(inventory, f, indent=2)

    print(f"Written: {INVENTORY_JSON}", flush=True)
    return inventory


def generate_markdown(inventory: dict) -> None:
    """Generate human-readable markdown from inventory JSON."""
    ts = inventory.get("audit_timestamp", "unknown")
    servers = inventory.get("servers", [])

    lines = []
    lines.append("# Server Inventory")
    lines.append(f"*Generated: {ts}*")
    lines.append(f"*Source: data/infrastructure/inventory.json*")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Server | IP | OS | Docker | Services | Disk Used |")
    lines.append("|--------|----|----|--------|----------|-----------|")

    for s in servers:
        hostname = s.get("system", {}).get("hostname", s.get("ip", "?"))
        ip = s.get("ip", "?")
        os_ver = s.get("system", {}).get("os", "?")
        # Shorten OS string
        if "Ubuntu" in os_ver:
            os_short = os_ver.split("Ubuntu")[1][:20].strip() if "Ubuntu" in os_ver else os_ver[:20]
            os_short = f"Ubuntu {os_short}"
        elif os_ver:
            os_short = os_ver[:25]
        else:
            os_short = "?"

        container_count = len(s.get("docker_containers", []))
        if not s.get("docker_installed", True):
            container_str = "not installed"
        else:
            container_str = f"{container_count} containers"

        service_count = len(s.get("services", []))

        # Primary disk usage
        disk_pct = "?"
        for d in s.get("disk_usage", []):
            if d.get("mount") == "/":
                disk_pct = f"{d.get('used', '?')}/{d.get('size', '?')} ({d.get('use_pct', '?')})"
                break

        lines.append(f"| {hostname} | {ip} | {os_short} | {container_str} | {service_count} | {disk_pct} |")

    lines.append("")

    # Cross-server connections
    lines.append("## Cross-Server Connections")
    lines.append("")
    connections = inventory.get("connections", {})
    if connections:
        for server, conn_list in connections.items():
            lines.append(f"**{server}** references:")
            for conn in conn_list:
                lines.append(f"  - Source: `{conn['source']}`")
                for ref_line in conn.get("content", "").splitlines()[:5]:
                    lines.append(f"    ```")
                    lines.append(f"    {ref_line}")
                    lines.append(f"    ```")
    else:
        lines.append("No cross-server IP references detected in checked locations.")
    lines.append("")

    # Per-server sections
    for s in servers:
        hostname = s.get("system", {}).get("hostname", s.get("ip", "?"))
        ip = s.get("ip", "?")
        alias = s.get("alias", "?")
        description = s.get("description", "")

        lines.append(f"## Server: {hostname} ({ip})")
        if description:
            lines.append(f"*{description} — alias: `{alias}`*")
        lines.append("")

        # Error case
        if "error" in s:
            lines.append(f"> **Audit failed:** {s['error']}")
            lines.append("")
            continue

        sys_info = s.get("system", {})

        # System
        lines.append("### System")
        lines.append(f"- **OS:** {sys_info.get('os', '?')}")
        lines.append(f"- **Kernel:** {sys_info.get('kernel', '?')}")
        lines.append(f"- **Arch:** {sys_info.get('arch', '?')}")
        lines.append(f"- **Uptime:** {sys_info.get('uptime', '?')}")
        mem = s.get("memory", {})
        lines.append(f"- **Memory:** {mem.get('used', '?')} used / {mem.get('total', '?')} total")
        lines.append(f"- **Python:** {s.get('python_version', '?')}")
        lines.append(f"- **Node:** {s.get('node_version', '?')}")
        lines.append(f"- **Packages:** {s.get('package_count', '?')}")
        lines.append("")

        # Disk usage
        lines.append("### Disk Usage")
        disk_lines = s.get("disk_usage", [])
        if disk_lines:
            lines.append("")
            lines.append("| Filesystem | Size | Used | Avail | Use% | Mount |")
            lines.append("|------------|------|------|-------|------|-------|")
            for d in disk_lines:
                lines.append(
                    f"| {d.get('filesystem','')} | {d.get('size','')} | {d.get('used','')} | {d.get('avail','')} | {d.get('use_pct','')} | {d.get('mount','')} |"
                )
        else:
            lines.append("No disk data collected.")
        lines.append("")

        # Docker
        lines.append("### Docker Containers")
        containers = s.get("docker_containers", [])
        if not s.get("docker_installed", True):
            lines.append("Docker not installed on this server.")
        elif containers:
            lines.append("")
            lines.append("| Container | Image | Status | Ports |")
            lines.append("|-----------|-------|--------|-------|")
            for c in containers:
                lines.append(
                    f"| {c.get('name','')} | {c.get('image','')} | {c.get('status','')} | {c.get('ports','')} |"
                )
        else:
            lines.append("No running Docker containers.")

        compose_projects = s.get("docker_compose_projects", [])
        if compose_projects:
            lines.append("")
            lines.append("**Docker Compose Projects:**")
            for p in compose_projects:
                if isinstance(p, dict):
                    lines.append(f"- {p.get('Name', p)}: {p.get('Status', '')} ({p.get('ConfigFiles', '')})")
                else:
                    lines.append(f"- {p}")
        lines.append("")

        # Services
        lines.append("### Running Services")
        services = s.get("services", [])
        if services:
            for svc in services:
                lines.append(f"- `{svc}`")
        else:
            lines.append("No running services detected (or systemctl not available).")
        lines.append("")

        # Ports
        lines.append("### Listening Ports")
        ports_raw = s.get("listening_ports_raw", "")
        if ports_raw:
            lines.append("")
            lines.append("```")
            lines.append(ports_raw)
            lines.append("```")
        else:
            lines.append("No port data collected.")
        lines.append("")

        # Databases
        lines.append("### Databases")
        dbs = s.get("databases", {})
        db_found = False
        for db_name, db_info in dbs.items():
            if db_name == "sqlite_files":
                if db_info:
                    lines.append(f"**SQLite files found:**")
                    for f_path in db_info:
                        lines.append(f"  - `{f_path}`")
                    db_found = True
            elif isinstance(db_info, dict) and db_info.get("detected"):
                lines.append(f"- **{db_name.capitalize()}:** running (`{db_info.get('status', '?')}`)")
                db_found = True
        if not db_found:
            lines.append("No databases detected.")
        lines.append("")

        # Cron jobs
        lines.append("### Cron Jobs")
        cron = s.get("cron_jobs", {})
        user_cron = cron.get("user_crontab", "(none)")
        lines.append(f"**User crontab:**")
        lines.append("```")
        lines.append(user_cron if user_cron else "(empty)")
        lines.append("```")
        lines.append(f"**cron.d files:** {cron.get('cron_d_files', '(empty)')}")
        lines.append(f"**cron.daily files:** {cron.get('cron_daily_files', '(empty)')}")
        lines.append("")

        # Notable findings
        lines.append("### Notable Findings")
        findings = s.get("notable_findings", [])
        if findings:
            for finding in findings:
                lines.append(f"- {finding}")
        else:
            lines.append("No unusual configurations detected.")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(INVENTORY_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"Written: {INVENTORY_MD}", flush=True)


def main():
    markdown_only = "--markdown-only" in sys.argv

    if markdown_only:
        print("Regenerating markdown from existing JSON...", flush=True)
        if not os.path.exists(INVENTORY_JSON):
            print(f"ERROR: {INVENTORY_JSON} not found. Run without --markdown-only first.", file=sys.stderr)
            sys.exit(1)
        with open(INVENTORY_JSON) as f:
            inventory = json.load(f)
    else:
        inventory = run_full_audit()

    generate_markdown(inventory)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
