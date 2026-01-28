"""Alfred MCP Server - Exposes Alfred's tools to Claude Code / Claude Desktop."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from mcp.server.fastmcp import FastMCP

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alfred-mcp")

mcp = FastMCP(
    "Alfred",
    instructions="You are Alfred, a personal AI assistant for Bruce Johnson, owner of Ground Rush Inc. "
    "Use these tools to manage email, calendar, servers, and memory. "
    "Be concise and professional. Address Bruce as 'sir' or by name.",
)


# ==================== EMAIL TOOLS ====================

@mcp.tool()
def check_email(max_results: int = 5, query: str = "") -> str:
    """Check inbox for recent emails. Returns subject, sender, and snippet.

    Args:
        max_results: Number of emails to return (default 5)
        query: Gmail search query to filter emails (optional)
    """
    from integrations.gmail.client import get_inbox
    results = get_inbox(max_results=max_results, query=query)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def read_email(message_id: str) -> str:
    """Read the full content of a specific email by its ID.

    Args:
        message_id: The email message ID from check_email results
    """
    from integrations.gmail.client import read_email as _read
    result = _read(message_id)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body text
    """
    from integrations.gmail.client import send_email as _send
    result = _send(to, subject, body)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def unread_email_count() -> str:
    """Get the number of unread emails in inbox."""
    from integrations.gmail.client import get_unread_count
    return json.dumps({"unread": get_unread_count()})


# ==================== CALENDAR TOOLS ====================

@mcp.tool()
def check_calendar(days_ahead: int = 7, max_results: int = 10) -> str:
    """Get upcoming calendar events for the next N days.

    Args:
        days_ahead: Number of days to look ahead (default 7)
        max_results: Maximum number of events to return (default 10)
    """
    from integrations.calendar.client import get_upcoming_events
    results = get_upcoming_events(max_results=max_results, days_ahead=days_ahead)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def today_schedule() -> str:
    """Get today's schedule - all events for today."""
    from integrations.calendar.client import get_today_events
    results = get_today_events()
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
) -> str:
    """Create a new calendar event.

    Args:
        summary: Event title
        start_time: ISO datetime (e.g. 2026-01-28T10:00:00-05:00)
        end_time: ISO datetime
        description: Event description (optional)
        location: Event location (optional)
    """
    from integrations.calendar.client import create_event as _create
    result = _create(summary, start_time, end_time, description, location)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def find_free_time(date: str, duration_minutes: int = 60) -> str:
    """Find available time slots on a given date.

    Args:
        date: Date string (YYYY-MM-DD)
        duration_minutes: Desired meeting duration in minutes (default 60)
    """
    from integrations.calendar.client import find_free_time as _find
    results = _find(date, duration_minutes)
    return json.dumps(results, indent=2, default=str)


# ==================== SERVER TOOLS ====================

@mcp.tool()
def list_servers() -> str:
    """List all registered servers that Alfred can manage."""
    from integrations.servers.manager import list_servers as _list
    results = _list()
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
def server_status(server_name: str) -> str:
    """Get the status of a specific server (uptime, disk, memory, docker containers).

    Args:
        server_name: Name of the registered server (e.g. loovacast-dev, groundrush-prod)
    """
    from integrations.servers.manager import get_server_status
    result = get_server_status(server_name)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def server_command(server_name: str, command: str) -> str:
    """Run a shell command on a remote server.

    Args:
        server_name: Name of the registered server
        command: The shell command to run
    """
    from integrations.servers.manager import run_command
    result = run_command(server_name, command)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def docker_containers(server_name: str) -> str:
    """List Docker containers on a remote server.

    Args:
        server_name: Name of the registered server
    """
    from integrations.servers.manager import docker_ps
    return docker_ps(server_name)


@mcp.tool()
def docker_restart(server_name: str, container: str) -> str:
    """Restart a Docker container on a remote server.

    Args:
        server_name: Name of the registered server
        container: Docker container name to restart
    """
    from integrations.servers.manager import docker_restart as _restart
    return _restart(server_name, container)


# ==================== MEMORY TOOLS ====================

@mcp.tool()
def remember(text: str, category: str = "general") -> str:
    """Store a piece of information in Alfred's long-term memory for later recall.

    Args:
        text: The information to remember
        category: Category - general, business, personal, or financial
    """
    from core.memory.store import store_memory
    doc_id = store_memory(text, category)
    return json.dumps({"stored": True, "id": doc_id})


@mcp.tool()
def recall(query: str, category: str = "general") -> str:
    """Search Alfred's memory for relevant information.

    Args:
        query: What to search for
        category: Category to search in - general, business, personal, or financial
    """
    from core.memory.store import recall as _recall
    results = _recall(query, category)
    return json.dumps(results, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
