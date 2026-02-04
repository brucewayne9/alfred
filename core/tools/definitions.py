"""Tool definitions - registers all available tools for the LLM."""

from core.tools.registry import tool


# ==================== EMAIL TOOLS ====================

@tool(
    name="check_email",
    description="Check inbox for recent emails. Returns subject, sender, and snippet.",
    parameters={"max_results": "int (default 5)", "query": "string - Gmail search query (optional)"},
)
def check_email(max_results: int = 5, query: str = "") -> list[dict]:
    from integrations.gmail.client import get_inbox
    return get_inbox(max_results=max_results, query=query)


@tool(
    name="read_email",
    description="Read the full content of a specific email by its ID.",
    parameters={"message_id": "string - the email message ID"},
)
def read_email(message_id: str) -> dict:
    from integrations.gmail.client import read_email as _read
    return _read(message_id)


@tool(
    name="send_email",
    description="Send an email to someone.",
    parameters={"to": "string - recipient email", "subject": "string", "body": "string"},
)
def send_email(to: str, subject: str, body: str) -> dict:
    from integrations.gmail.client import send_email as _send
    return _send(to, subject, body)


@tool(
    name="unread_email_count",
    description="Get the number of unread emails in inbox.",
    parameters={},
)
def unread_email_count() -> dict:
    from integrations.gmail.client import get_unread_count
    return {"unread": get_unread_count()}


# ==================== MULTI-ACCOUNT EMAIL (IMAP/SMTP) ====================

@tool(
    name="email_list_accounts",
    description="List all configured email accounts (Groundrush Labs, Ruck Talk, LoovaCast).",
    parameters={},
)
def email_list_accounts() -> dict:
    from integrations.email.client import email_client
    return email_client.list_accounts()


@tool(
    name="email_inbox",
    description="Get recent emails from a specific account's inbox. Accounts: groundrush, rucktalk, loovacast, lumabot, support, groundrush info.",
    parameters={
        "account": {"type": "string", "description": "Account name: groundrush, rucktalk, loovacast, lumabot, support, or 'groundrush info'", "required": True},
        "limit": {"type": "integer", "description": "Max emails to return (default 10)"},
    },
)
def email_inbox(account: str, limit: int = 10) -> dict:
    from integrations.email.client import email_client
    return email_client.get_inbox(account, limit)


@tool(
    name="email_read",
    description="Read the full content of an email by ID from a specific account.",
    parameters={
        "account": {"type": "string", "description": "Account name: groundrush, rucktalk, loovacast, lumabot, support, or 'groundrush info'", "required": True},
        "message_id": {"type": "string", "description": "The email message ID", "required": True},
    },
)
def email_read(account: str, message_id: str) -> dict:
    from integrations.email.client import email_client
    return email_client.read_email(account, message_id)


@tool(
    name="email_send",
    description="Send an email from a specific account. Accounts: groundrush (mjohnson@), rucktalk (info@rucktalk), loovacast (info@loovacast), lumabot, support (support@loovacast), groundrush info (info@groundrushlabs).",
    parameters={
        "account": {"type": "string", "description": "Account to send from: groundrush, rucktalk, loovacast, lumabot, support, or 'groundrush info'", "required": True},
        "to": {"type": "string", "description": "Recipient email address", "required": True},
        "subject": {"type": "string", "description": "Email subject", "required": True},
        "body": {"type": "string", "description": "Email body text", "required": True},
    },
)
def email_send(account: str, to: str, subject: str, body: str) -> dict:
    from integrations.email.client import email_client
    return email_client.send_email(account, to, subject, body)


@tool(
    name="email_search",
    description="Search emails in a specific account by keyword.",
    parameters={
        "account": {"type": "string", "description": "Account name: groundrush, rucktalk, loovacast, lumabot, support, or 'groundrush info'", "required": True},
        "query": {"type": "string", "description": "Search keyword", "required": True},
        "limit": {"type": "integer", "description": "Max results (default 10)"},
    },
)
def email_search(account: str, query: str, limit: int = 10) -> dict:
    from integrations.email.client import email_client
    return email_client.search_emails(account, query, limit)


@tool(
    name="email_unread",
    description="Get unread email count for a specific account or all accounts.",
    parameters={
        "account": {"type": "string", "description": "Account name (optional - omit to get all accounts)"},
    },
)
def email_unread(account: str = None) -> dict:
    from integrations.email.client import email_client
    if account:
        return email_client.get_unread_count(account)
    return email_client.get_all_unread_counts()


@tool(
    name="email_trash",
    description="Move an email to trash. Requires the message ID from email_inbox or email_read.",
    parameters={
        "account": {"type": "string", "description": "Account name: groundrush, rucktalk, loovacast, lumabot, support, or 'groundrush info'", "required": True},
        "message_id": {"type": "string", "description": "The email message ID to trash", "required": True},
    },
)
def email_trash(account: str, message_id: str) -> dict:
    from integrations.email.client import email_client
    return email_client.trash_email(account, message_id)


@tool(
    name="email_mark_read",
    description="Mark an email as read.",
    parameters={
        "account": {"type": "string", "description": "Account name", "required": True},
        "message_id": {"type": "string", "description": "The email message ID", "required": True},
    },
)
def email_mark_read(account: str, message_id: str) -> dict:
    from integrations.email.client import email_client
    return email_client.mark_read(account, message_id)


@tool(
    name="email_mark_unread",
    description="Mark an email as unread.",
    parameters={
        "account": {"type": "string", "description": "Account name", "required": True},
        "message_id": {"type": "string", "description": "The email message ID", "required": True},
    },
)
def email_mark_unread(account: str, message_id: str) -> dict:
    from integrations.email.client import email_client
    return email_client.mark_unread(account, message_id)


# ==================== CALENDAR TOOLS ====================

@tool(
    name="check_calendar",
    description="Get upcoming calendar events for the next N days.",
    parameters={"days_ahead": "int (default 7)", "max_results": "int (default 10)"},
)
def check_calendar(days_ahead: int = 7, max_results: int = 10) -> list[dict]:
    from integrations.calendar.client import get_upcoming_events
    return get_upcoming_events(max_results=max_results, days_ahead=days_ahead)


@tool(
    name="today_schedule",
    description="Get today's schedule - all events for today.",
    parameters={},
)
def today_schedule() -> list[dict]:
    from integrations.calendar.client import get_today_events
    return get_today_events()


@tool(
    name="create_event",
    description="Create a new calendar event.",
    parameters={
        "summary": "string - event title",
        "start_time": "string - ISO datetime (e.g. 2026-01-28T10:00:00-05:00)",
        "end_time": "string - ISO datetime",
        "description": "string (optional)",
        "location": "string (optional)",
    },
)
def create_event(summary: str, start_time: str, end_time: str, description: str = "", location: str = "") -> dict:
    from integrations.calendar.client import create_event as _create
    return _create(summary, start_time, end_time, description, location)


@tool(
    name="find_free_time",
    description="Find available time slots on a given date.",
    parameters={"date": "string - YYYY-MM-DD", "duration_minutes": "int (default 60)"},
)
def find_free_time(date: str, duration_minutes: int = 60) -> list[dict]:
    from integrations.calendar.client import find_free_time as _find
    return _find(date, duration_minutes)


# ==================== SERVER TOOLS ====================

@tool(
    name="list_servers",
    description="List all registered servers that Alfred can manage.",
    parameters={},
)
def list_servers() -> list[dict]:
    from integrations.servers.manager import list_servers as _list
    return _list()


@tool(
    name="server_status",
    description="Get the status of a specific server (uptime, disk, memory, docker containers).",
    parameters={"server_name": "string - name of the registered server"},
)
def server_status(server_name: str) -> dict:
    from integrations.servers.manager import get_server_status
    return get_server_status(server_name)


@tool(
    name="server_command",
    description="Run a shell command on a remote server. Use with caution.",
    parameters={"server_name": "string", "command": "string - the command to run"},
)
def server_command(server_name: str, command: str) -> dict:
    from integrations.servers.manager import run_command
    return run_command(server_name, command)


@tool(
    name="docker_containers",
    description="List Docker containers on a remote server.",
    parameters={"server_name": "string"},
)
def docker_containers(server_name: str) -> str:
    from integrations.servers.manager import docker_ps
    return docker_ps(server_name)


@tool(
    name="docker_restart",
    description="Restart a Docker container on a remote server.",
    parameters={"server_name": "string", "container": "string - container name"},
)
def docker_restart(server_name: str, container: str) -> str:
    from integrations.servers.manager import docker_restart as _restart
    return _restart(server_name, container)


# ==================== SERVER UPDATE TOOLS ====================

@tool(
    name="server_check_updates",
    description="Check what updates are available on a server WITHOUT installing. Shows package count, security updates, and if reboot is needed. Always use this BEFORE running updates.",
    parameters={"server_name": "string - name of the server (e.g., 'lonewolf-dev', 'groundrush-prod')"},
)
def server_check_updates(server_name: str) -> dict:
    from integrations.servers.manager import check_updates
    return check_updates(server_name)


@tool(
    name="server_check_all_updates",
    description="Check updates available on ALL servers at once. Returns summary for each server.",
    parameters={},
)
def server_check_all_updates() -> list[dict]:
    from integrations.servers.manager import check_all_updates
    return check_all_updates()


@tool(
    name="server_run_updates",
    description="Run apt update && apt upgrade on a server. For PRODUCTION servers, confirm_production must be True (requires explicit user confirmation first).",
    parameters={
        "server_name": "string - name of the server",
        "confirm_production": "boolean - must be True to update production servers (default False)"
    },
)
def server_run_updates(server_name: str, confirm_production: bool = False) -> dict:
    from integrations.servers.manager import run_updates
    return run_updates(server_name, confirm_production)


@tool(
    name="server_autoremove",
    description="Remove unused packages (apt autoremove) on a server.",
    parameters={"server_name": "string - name of the server"},
)
def server_autoremove(server_name: str) -> dict:
    from integrations.servers.manager import run_autoremove
    return run_autoremove(server_name)


@tool(
    name="server_reboot",
    description="Reboot a server. Requires explicit confirmation. Only use when reboot is needed (e.g., after kernel updates).",
    parameters={
        "server_name": "string - name of the server",
        "confirm": "boolean - must be True to proceed with reboot"
    },
)
def server_reboot(server_name: str, confirm: bool = False) -> dict:
    from integrations.servers.manager import reboot_server
    return reboot_server(server_name, confirm)


# ==================== MAILCOW TOOLS ====================

@tool(
    name="mailcow_check_updates",
    description="Check if Mailcow mail server has updates available. Shows current version, latest version, and pending updates.",
    parameters={},
)
def mailcow_check_updates() -> dict:
    from integrations.servers.manager import mailcow_check_updates as _check
    return _check()


@tool(
    name="mailcow_update",
    description="Update Mailcow to the latest version. REQUIRES CONFIRMATION as this is the production mail server. Brief mail interruption may occur.",
    parameters={
        "confirm": "boolean - must be True to proceed with update"
    },
)
def mailcow_update(confirm: bool = False) -> dict:
    from integrations.servers.manager import mailcow_update as _update
    return _update(confirm)


@tool(
    name="mailcow_restart",
    description="Restart all Mailcow Docker containers. Use if mail services are having issues.",
    parameters={},
)
def mailcow_restart() -> dict:
    from integrations.servers.manager import mailcow_restart as _restart
    return _restart()


@tool(
    name="mailcow_status",
    description="Get Mailcow status including container health, disk usage, and mail queue.",
    parameters={},
)
def mailcow_status() -> dict:
    from integrations.servers.manager import mailcow_status as _status
    return _status()


# ==================== HOME ASSISTANT TOOLS ====================

@tool(
    name="homeassistant_status",
    description="Get Home Assistant status - container health, URL, and recent logs.",
    parameters={},
)
def homeassistant_status() -> dict:
    from integrations.servers.manager import homeassistant_status as _status
    return _status()


@tool(
    name="homeassistant_restart",
    description="Restart the Home Assistant container.",
    parameters={},
)
def homeassistant_restart() -> dict:
    from integrations.servers.manager import homeassistant_restart as _restart
    return _restart()


@tool(
    name="homeassistant_update",
    description="Update Home Assistant to the latest version (pulls new image and restarts).",
    parameters={},
)
def homeassistant_update() -> dict:
    from integrations.servers.manager import homeassistant_update as _update
    return _update()


@tool(
    name="homeassistant_logs",
    description="Get recent Home Assistant logs for debugging.",
    parameters={"lines": "int - number of log lines (default 50)"},
)
def homeassistant_logs(lines: int = 50) -> dict:
    from integrations.servers.manager import homeassistant_logs as _logs
    return _logs(lines)


# ==================== HOME ASSISTANT SMART HOME CONTROL ====================

@tool(
    name="ha_list_devices",
    description="List all smart home devices (lights, switches, media players, climate, sensors) with their current states.",
    parameters={},
)
def ha_list_devices() -> dict:
    from integrations.homeassistant.client import list_devices
    return list_devices()


@tool(
    name="ha_turn_on",
    description="Turn on a smart home device (light, switch, media player, etc.) by entity ID or name.",
    parameters={
        "entity_id": "string - entity ID (e.g., 'light.kitchen') or device name (e.g., 'kitchen lights')",
    },
)
def ha_turn_on(entity_id: str) -> dict:
    from integrations.homeassistant.client import turn_on, find_entity
    # Check if it's a name or entity_id
    if "." not in entity_id:
        found = find_entity(entity_id)
        if found:
            entity_id = found
        else:
            return {"error": f"Device '{entity_id}' not found"}
    return turn_on(entity_id)


@tool(
    name="ha_turn_off",
    description="Turn off a smart home device (light, switch, media player, etc.).",
    parameters={
        "entity_id": "string - entity ID or device name",
    },
)
def ha_turn_off(entity_id: str) -> dict:
    from integrations.homeassistant.client import turn_off, find_entity
    if "." not in entity_id:
        found = find_entity(entity_id)
        if found:
            entity_id = found
        else:
            return {"error": f"Device '{entity_id}' not found"}
    return turn_off(entity_id)


# Room groups stored in memory for quick access
_ROOM_GROUPS = {
    "living room": [
        "light.freddy_s_light",
        "light.the_skull",
        "light.overhead_4",
        "switch.johnson_sign",
    ],
    "kitchen": [
        "light.kitchen_overhead_1",
        "light.kitchen_overhead_2",
        "light.kitchen_overhead_3",
        "light.kitchen_overhead_4",
        "light.stove_1",
        "light.stove_3",  # Stove Light
        "light.tp_link_smart_bulb_7f8d",  # Stove 2
        "light.kitchen_window",
    ],
    "front yard": [
        "light.outdoor_front_left_1",
        "light.outdoor_front_left_2",
        "light.outdoor_front_right_1",
        "light.outdoor_front_right_2",
        "light.left_side_light",
    ],
    "backyard": [
        "light.outdoor_back_1",
        "light.outdoor_back_2",
        "light.studio_outside",
    ],
    "overheads": [
        "light.overhead_1",
        "light.overhead_2",
        "light.overhead_3",
        "light.overhead_4",
    ],
}


@tool(
    name="ha_room_control",
    description="Turn on or off all lights/devices in a room at once. Supports: living room, kitchen, front yard, backyard, overheads. Use this for room-wide control.",
    parameters={
        "room": "string - room name (living room, kitchen, front yard, backyard, overheads)",
        "action": "string - 'on' or 'off'",
    },
)
def ha_room_control(room: str, action: str) -> dict:
    """Control all devices in a room at once."""
    from integrations.homeassistant.client import turn_on, turn_off

    room_lower = room.lower().strip()
    if room_lower not in _ROOM_GROUPS:
        return {
            "error": f"Unknown room: {room}. Available rooms: {list(_ROOM_GROUPS.keys())}",
            "hint": "Use ha_add_room_group to create a new room group"
        }

    entities = _ROOM_GROUPS[room_lower]
    results = []
    errors = []

    for entity_id in entities:
        try:
            if action.lower() == "on":
                result = turn_on(entity_id)
            else:
                result = turn_off(entity_id)
            results.append({"entity": entity_id, "success": True})
        except Exception as e:
            errors.append({"entity": entity_id, "error": str(e)})

    return {
        "room": room,
        "action": action,
        "devices_controlled": len(results),
        "successes": results,
        "errors": errors if errors else None,
    }


@tool(
    name="ha_add_room_group",
    description="Add or update a room group for controlling multiple devices at once. Store in memory so 'turn on [room] lights' works.",
    parameters={
        "room": "string - room name (e.g., 'living room', 'bedroom')",
        "entities": "list - list of entity IDs to include in this room",
    },
)
def ha_add_room_group(room: str, entities: list) -> dict:
    """Add or update a room group."""
    room_lower = room.lower().strip()
    _ROOM_GROUPS[room_lower] = entities

    # Also store to memory for persistence
    try:
        from core.memory.store import add_memory
        memory_text = f"Room group '{room}' contains: {', '.join(entities)}"
        add_memory(memory_text, category="personal")
    except Exception:
        pass  # Memory storage is optional

    return {
        "success": True,
        "room": room,
        "entities": entities,
        "message": f"Room group '{room}' saved with {len(entities)} devices. Use ha_room_control to control them all at once.",
    }


@tool(
    name="ha_list_room_groups",
    description="List all configured room groups and their devices.",
    parameters={},
)
def ha_list_room_groups() -> dict:
    """List all room groups."""
    return {
        "rooms": {room: entities for room, entities in _ROOM_GROUPS.items()},
        "count": len(_ROOM_GROUPS),
    }


@tool(
    name="ha_set_light",
    description="Control a light - set brightness, color, or color temperature.",
    parameters={
        "entity_id": "string - light entity ID or name",
        "brightness": "int - brightness 0-255 (optional)",
        "color": "string - color name like 'red', 'blue' or hex '#FF0000' (optional)",
        "temperature": "int - color temperature in mireds, lower=cooler (optional)",
    },
)
def ha_set_light(entity_id: str, brightness: int = None, color: str = None, temperature: int = None) -> dict:
    from integrations.homeassistant.client import set_light, find_entity
    if "." not in entity_id:
        found = find_entity(entity_id)
        if found:
            entity_id = found
        else:
            return {"error": f"Light '{entity_id}' not found"}
    return set_light(entity_id, brightness, color, temperature)


@tool(
    name="ha_media_players",
    description="Get all media players (TVs, speakers, displays) with their current state, volume, and now playing info.",
    parameters={},
)
def ha_media_players() -> list[dict]:
    from integrations.homeassistant.client import get_media_players
    return get_media_players()


@tool(
    name="ha_media_control",
    description="Control media playback on a device (play, pause, stop, next, previous).",
    parameters={
        "entity_id": "string - media player entity ID or name",
        "action": "string - 'play', 'pause', 'stop', 'next', 'previous'",
    },
)
def ha_media_control(entity_id: str, action: str) -> dict:
    from integrations.homeassistant.client import media_play, media_pause, media_stop, media_next, media_previous, find_entity
    if "." not in entity_id:
        found = find_entity(entity_id)
        if found:
            entity_id = found
        else:
            return {"error": f"Media player '{entity_id}' not found"}

    actions = {
        "play": media_play,
        "pause": media_pause,
        "stop": media_stop,
        "next": media_next,
        "previous": media_previous,
    }
    if action not in actions:
        return {"error": f"Unknown action '{action}'. Use: play, pause, stop, next, previous"}
    return actions[action](entity_id)


@tool(
    name="ha_set_volume",
    description="Set volume on a media player (0.0 to 1.0).",
    parameters={
        "entity_id": "string - media player entity ID or name",
        "volume": "float - volume level 0.0 to 1.0",
    },
)
def ha_set_volume(entity_id: str, volume: float) -> dict:
    from integrations.homeassistant.client import set_volume, find_entity
    if "." not in entity_id:
        found = find_entity(entity_id)
        if found:
            entity_id = found
        else:
            return {"error": f"Media player '{entity_id}' not found"}
    return set_volume(entity_id, volume)


@tool(
    name="ha_set_thermostat",
    description="Control a thermostat - set temperature or mode.",
    parameters={
        "entity_id": "string - climate entity ID or name",
        "temperature": "float - target temperature (optional)",
        "hvac_mode": "string - 'heat', 'cool', 'heat_cool', 'off', 'auto' (optional)",
    },
)
def ha_set_thermostat(entity_id: str, temperature: float = None, hvac_mode: str = None) -> dict:
    from integrations.homeassistant.client import set_thermostat, find_entity
    if "." not in entity_id:
        found = find_entity(entity_id)
        if found:
            entity_id = found
        else:
            return {"error": f"Thermostat '{entity_id}' not found"}
    return set_thermostat(entity_id, temperature, hvac_mode)


@tool(
    name="ha_get_weather",
    description="Get current weather and forecast from Home Assistant.",
    parameters={},
)
def ha_get_weather() -> dict:
    from integrations.homeassistant.client import get_weather
    return get_weather()


@tool(
    name="ha_activate_scene",
    description="Activate a Home Assistant scene (e.g., 'movie_mode', 'good_night').",
    parameters={
        "scene": "string - scene name or ID",
    },
)
def ha_activate_scene(scene: str) -> dict:
    from integrations.homeassistant.client import activate_scene
    return activate_scene(scene)


@tool(
    name="ha_run_script",
    description="Run a Home Assistant script/automation.",
    parameters={
        "script": "string - script name or ID",
    },
)
def ha_run_script(script: str) -> dict:
    from integrations.homeassistant.client import run_script
    return run_script(script)


@tool(
    name="ha_get_state",
    description="Get the current state of any Home Assistant entity.",
    parameters={
        "entity_id": "string - entity ID (e.g., 'sensor.temperature', 'binary_sensor.front_door')",
    },
)
def ha_get_state(entity_id: str) -> dict:
    from integrations.homeassistant.client import get_state
    return get_state(entity_id)


# ==================== CRM TOOLS (Twenty CRM) ====================

@tool(
    name="crm_list_people",
    description="List contacts/people in the CRM.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_people(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_people
    return list_people(limit=limit)


@tool(
    name="crm_search_people",
    description="Search for a person in the CRM by name or email.",
    parameters={"query": "string - name or email to search for"},
)
def crm_search_people(query: str) -> list[dict]:
    from integrations.base_crm.client import search_people
    return search_people(query)


@tool(
    name="crm_get_person",
    description="Get full details of a specific person/contact in the CRM by their ID.",
    parameters={"person_id": "string - UUID of the person"},
)
def crm_get_person(person_id: str) -> dict:
    from integrations.base_crm.client import get_person
    return get_person(person_id)


@tool(
    name="crm_create_person",
    description="Add a new contact/person to the CRM.",
    parameters={
        "first_name": "string",
        "last_name": "string",
        "email": "string (optional)",
        "phone": "string (optional)",
        "job_title": "string (optional)",
        "city": "string (optional)",
    },
)
def crm_create_person(first_name: str, last_name: str, email: str = "",
                       phone: str = "", job_title: str = "", city: str = "") -> dict:
    from integrations.base_crm.client import create_person
    return create_person(first_name, last_name, email, phone, job_title, city)


@tool(
    name="crm_update_person",
    description="Update an existing person/contact in the CRM. Only provide fields you want to change.",
    parameters={
        "person_id": "string - UUID of the person",
        "first_name": "string (optional)",
        "last_name": "string (optional)",
        "email": "string (optional)",
        "phone": "string (optional)",
        "job_title": "string (optional)",
        "city": "string (optional)",
    },
)
def crm_update_person(person_id: str, first_name: str = "", last_name: str = "",
                       email: str = "", phone: str = "", job_title: str = "", city: str = "") -> dict:
    from integrations.base_crm.client import update_person
    return update_person(person_id, first_name, last_name, email, phone, job_title, city)


@tool(
    name="crm_delete_person",
    description="Delete a person/contact from the CRM.",
    parameters={"person_id": "string - UUID of the person to delete"},
)
def crm_delete_person(person_id: str) -> dict:
    from integrations.base_crm.client import delete_person
    return delete_person(person_id)


@tool(
    name="crm_list_companies",
    description="List companies in the CRM.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_companies(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_companies
    return list_companies(limit=limit)


@tool(
    name="crm_search_companies",
    description="Search for a company in the CRM by name.",
    parameters={"query": "string - company name to search for"},
)
def crm_search_companies(query: str) -> list[dict]:
    from integrations.base_crm.client import search_companies
    return search_companies(query)


@tool(
    name="crm_get_company",
    description="Get full details of a specific company in the CRM by its ID.",
    parameters={"company_id": "string - UUID of the company"},
)
def crm_get_company(company_id: str) -> dict:
    from integrations.base_crm.client import get_company
    return get_company(company_id)


@tool(
    name="crm_create_company",
    description="Add a new company to the CRM.",
    parameters={
        "name": "string - company name",
        "domain": "string - website URL (optional)",
        "employees": "int (optional)",
        "city": "string (optional)",
    },
)
def crm_create_company(name: str, domain: str = "", employees: int = 0, city: str = "") -> dict:
    from integrations.base_crm.client import create_company
    return create_company(name, domain, employees, city)


@tool(
    name="crm_update_company",
    description="Update an existing company in the CRM. Only provide fields you want to change.",
    parameters={
        "company_id": "string - UUID of the company",
        "name": "string (optional)",
        "domain": "string - website URL (optional)",
        "employees": "int (optional)",
        "city": "string (optional)",
    },
)
def crm_update_company(company_id: str, name: str = "", domain: str = "",
                        employees: int = 0, city: str = "") -> dict:
    from integrations.base_crm.client import update_company
    return update_company(company_id, name, domain, employees, city)


@tool(
    name="crm_delete_company",
    description="Delete a company from the CRM.",
    parameters={"company_id": "string - UUID of the company to delete"},
)
def crm_delete_company(company_id: str) -> dict:
    from integrations.base_crm.client import delete_company
    return delete_company(company_id)


@tool(
    name="crm_list_opportunities",
    description="List deals/opportunities in the CRM pipeline.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_opportunities(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_opportunities
    return list_opportunities(limit=limit)


@tool(
    name="crm_search_opportunities",
    description="Search deals/opportunities in the CRM by name or stage.",
    parameters={"query": "string - deal name or stage to search for"},
)
def crm_search_opportunities(query: str) -> list[dict]:
    from integrations.base_crm.client import search_opportunities
    return search_opportunities(query)


@tool(
    name="crm_get_opportunity",
    description="Get full details of a specific deal/opportunity by its ID.",
    parameters={"opportunity_id": "string - UUID of the opportunity"},
)
def crm_get_opportunity(opportunity_id: str) -> dict:
    from integrations.base_crm.client import get_opportunity
    return get_opportunity(opportunity_id)


@tool(
    name="crm_create_opportunity",
    description="Create a new deal/opportunity in the CRM.",
    parameters={
        "name": "string - deal name",
        "stage": "string - pipeline stage (MEETING, PROPOSAL, CUSTOMER, etc.)",
        "amount": "float - deal value in dollars (optional)",
        "company_id": "string - UUID of the company (optional)",
        "contact_id": "string - UUID of the point of contact (optional)",
    },
)
def crm_create_opportunity(name: str, stage: str = "MEETING", amount: float = 0,
                            company_id: str = "", contact_id: str = "") -> dict:
    from integrations.base_crm.client import create_opportunity
    return create_opportunity(name, stage, amount, company_id, contact_id)


@tool(
    name="crm_update_deal_stage",
    description="Move a deal/opportunity to a new pipeline stage.",
    parameters={
        "opportunity_id": "string - UUID of the opportunity",
        "stage": "string - new stage (MEETING, PROPOSAL, CUSTOMER, etc.)",
    },
)
def crm_update_deal_stage(opportunity_id: str, stage: str) -> dict:
    from integrations.base_crm.client import update_opportunity_stage
    return update_opportunity_stage(opportunity_id, stage)


@tool(
    name="crm_delete_opportunity",
    description="Delete a deal/opportunity from the CRM.",
    parameters={"opportunity_id": "string - UUID of the opportunity to delete"},
)
def crm_delete_opportunity(opportunity_id: str) -> dict:
    from integrations.base_crm.client import delete_opportunity
    return delete_opportunity(opportunity_id)


@tool(
    name="crm_pipeline_summary",
    description="Get a summary of the deal pipeline: deal count and total dollar value per stage.",
    parameters={},
)
def crm_pipeline_summary() -> dict:
    from integrations.base_crm.client import pipeline_summary
    return pipeline_summary()


@tool(
    name="crm_list_tasks",
    description="List tasks in the CRM.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_tasks(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_tasks
    return list_tasks(limit=limit)


@tool(
    name="crm_create_task",
    description="Create a new task in the CRM.",
    parameters={
        "title": "string - task title",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_create_task(title: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task
    return create_task(title, status, due_date)


@tool(
    name="crm_update_task",
    description="Update a CRM task. Change its title, status (TODO/DONE), or due date.",
    parameters={
        "task_id": "string - UUID of the task",
        "title": "string (optional)",
        "status": "string - TODO or DONE (optional)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_update_task(task_id: str, title: str = "", status: str = "", due_date: str = "") -> dict:
    from integrations.base_crm.client import update_task
    return update_task(task_id, title, status, due_date)


@tool(
    name="crm_delete_task",
    description="Delete a task from the CRM.",
    parameters={"task_id": "string - UUID of the task to delete"},
)
def crm_delete_task(task_id: str) -> dict:
    from integrations.base_crm.client import delete_task
    return delete_task(task_id)


@tool(
    name="crm_add_note_to_person",
    description="Add a note linked to a specific person/contact in the CRM.",
    parameters={
        "title": "string - note title",
        "person_id": "string - UUID of the person",
        "body": "string - note content (optional)",
    },
)
def crm_add_note_to_person(title: str, person_id: str, body: str = "") -> dict:
    from integrations.base_crm.client import create_note_for_person
    return create_note_for_person(title, person_id, body)


@tool(
    name="crm_add_note_to_company",
    description="Add a note linked to a specific company in the CRM.",
    parameters={
        "title": "string - note title",
        "company_id": "string - UUID of the company",
        "body": "string - note content (optional)",
    },
)
def crm_add_note_to_company(title: str, company_id: str, body: str = "") -> dict:
    from integrations.base_crm.client import create_note_for_company
    return create_note_for_company(title, company_id, body)


@tool(
    name="crm_add_note_to_deal",
    description="Add a note linked to a specific deal/opportunity in the CRM.",
    parameters={
        "title": "string - note title",
        "opportunity_id": "string - UUID of the opportunity",
        "body": "string - note content (optional)",
    },
)
def crm_add_note_to_deal(title: str, opportunity_id: str, body: str = "") -> dict:
    from integrations.base_crm.client import create_note_for_opportunity
    return create_note_for_opportunity(title, opportunity_id, body)


@tool(
    name="crm_add_task_to_person",
    description="Create a task linked to a specific person/contact in the CRM.",
    parameters={
        "title": "string - task title",
        "person_id": "string - UUID of the person",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_add_task_to_person(title: str, person_id: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task_for_person
    return create_task_for_person(title, person_id, status, due_date)


@tool(
    name="crm_add_task_to_company",
    description="Create a task linked to a specific company in the CRM.",
    parameters={
        "title": "string - task title",
        "company_id": "string - UUID of the company",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_add_task_to_company(title: str, company_id: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task_for_company
    return create_task_for_company(title, company_id, status, due_date)


@tool(
    name="crm_add_task_to_deal",
    description="Create a task linked to a specific deal/opportunity in the CRM.",
    parameters={
        "title": "string - task title",
        "opportunity_id": "string - UUID of the opportunity",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_add_task_to_deal(title: str, opportunity_id: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task_for_opportunity
    return create_task_for_opportunity(title, opportunity_id, status, due_date)


# ==================== MEMORY TOOLS ====================

@tool(
    name="remember",
    description="Store a piece of information in long-term memory for later recall.",
    parameters={"text": "string - the information to remember", "category": "string - general/business/personal/financial"},
)
def remember(text: str, category: str = "general") -> dict:
    from core.memory.store import store_memory
    doc_id = store_memory(text, category)
    return {"stored": True, "id": doc_id}


@tool(
    name="recall",
    description="Search memory for relevant information based on a query.",
    parameters={"query": "string - what to search for", "category": "string (optional)"},
)
def recall_memory(query: str, category: str = "general") -> list[dict]:
    from core.memory.store import recall
    return recall(query, category)


# ==================== DOCUMENT TOOLS ====================

@tool(
    name="analyze_document",
    description="Analyze an uploaded document (PDF, Word, Excel, CSV, TXT, etc). Returns the extracted text content.",
    parameters={"file_path": "string - path to the uploaded document"},
)
def analyze_document(file_path: str) -> dict:
    from core.tools.files import parse_document
    return parse_document(file_path)


@tool(
    name="create_document",
    description="Create a document file that the user can download. Supports txt, md, csv, pdf, docx, xlsx, json formats.",
    parameters={
        "content": "string - the content to put in the document",
        "filename": "string - base name for the file (no extension)",
        "format": "string - output format: txt, md, csv, pdf, docx, xlsx, or json",
    },
)
def create_document_tool(content: str, filename: str, format: str = "txt") -> dict:
    from core.tools.files import create_document
    result = create_document(content, filename, format)
    if result["error"]:
        return {"success": False, "error": result["error"]}
    return {
        "success": True,
        "filename": result["filename"],
        "download_url": f"/download/{result['filename']}",
        "message": f"Document created: {result['filename']}",
    }


# ==================== IMAGE GENERATION ====================

@tool(
    name="generate_image",
    description="Generate an image from a text description using AI (SDXL Turbo). Creates high-quality images in seconds.",
    parameters={
        "prompt": "string - detailed description of the image to generate",
        "width": "int - image width in pixels (default 1024, max 1536)",
        "height": "int - image height in pixels (default 1024, max 1536)",
    },
)
async def generate_image_tool(prompt: str, width: int = 1024, height: int = 1024) -> dict:
    from integrations.comfyui.client import generate_image

    # Clamp dimensions
    width = min(max(width, 512), 1536)
    height = min(max(height, 512), 1536)

    result = await generate_image(prompt, width, height)

    if not result["success"]:
        return {"success": False, "error": result["error"]}

    return {
        "success": True,
        "message": f"Image generated successfully",
        "filename": result["filename"],
        "download_url": result["download_url"],
        "base64": result["base64"],
    }


# ==================== KNOWLEDGE BASE (LightRAG) ====================

@tool(
    name="search_knowledge",
    description="Search the knowledge base for information from uploaded documents, contracts, emails, and notes. Use this when the user asks about something that might be in their documents.",
    parameters={
        "query": "string - the question or search query",
        "top_k": "int - number of results to return (default 5)",
    },
)
async def search_knowledge(query: str, top_k: int = 5) -> dict:
    from integrations.lightrag.client import query_context
    result = await query_context(query, top_k=top_k)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "context": result["result"]}


@tool(
    name="ask_knowledge",
    description="Ask a question and get an answer from the knowledge base with full LLM reasoning. Use for complex questions requiring synthesis across multiple documents.",
    parameters={
        "question": "string - the question to answer",
        "mode": "string - search mode: 'hybrid' (default), 'local', 'global', or 'naive'",
    },
)
async def ask_knowledge(question: str, mode: str = "hybrid") -> dict:
    from integrations.lightrag.client import query
    result = await query(question, mode=mode, only_need_context=False)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "answer": result["result"]}


@tool(
    name="store_to_knowledge",
    description="Store text content in the knowledge base for future retrieval. Use for important information, notes, or summaries that should be remembered.",
    parameters={
        "text": "string - the content to store",
        "description": "string - brief description of what this content is about",
    },
)
async def store_to_knowledge(text: str, description: str = "") -> dict:
    from integrations.lightrag.client import upload_text
    result = await upload_text(text, description)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "message": "Content stored in knowledge base"}


@tool(
    name="list_knowledge_documents",
    description="List documents stored in the knowledge base.",
    parameters={"limit": "int - max documents to return (default 20)"},
)
async def list_knowledge_documents(limit: int = 20) -> dict:
    from integrations.lightrag.client import list_documents, get_document_status
    # Try to get document status first (more reliable)
    status = await get_document_status()
    if status.get("success"):
        doc_counts = status.get("status", {})
        result = await list_documents(limit=limit)
        if result.get("success"):
            return {"success": True, "documents": result["documents"], "counts": doc_counts}
        # If list fails, return just the counts
        return {"success": True, "counts": doc_counts, "note": "Document listing unavailable, showing counts only"}
    return {"success": False, "error": "Could not retrieve document status"}


@tool(
    name="list_knowledge_entities",
    description="List entities in the knowledge graph. Shows what concepts, people, systems, and topics Alfred knows about.",
    parameters={
        "limit": "int - max entities to return (default 50)",
        "search": "string - optional filter to search entity names (optional)",
    },
)
async def list_knowledge_entities(limit: int = 50, search: str = None) -> dict:
    """List entities from the knowledge graph."""
    from integrations.lightrag.client import get_popular_entities, search_graph

    if search:
        # Search for specific entities
        result = await search_graph(search)
        if result.get("success"):
            entities = result.get("entities", [])
            return {
                "success": True,
                "search_term": search,
                "entities": entities[:limit] if isinstance(entities, list) else entities,
                "count": len(entities) if isinstance(entities, list) else "unknown",
            }
        return {"success": False, "error": result.get("error", "Search failed")}

    # Get popular entities (most connected in graph)
    result = await get_popular_entities(limit=limit)
    if result.get("success"):
        return {
            "success": True,
            "entities": result.get("entities", []),
            "note": "Showing most connected entities in knowledge graph",
        }
    return {"success": False, "error": result.get("error", "Failed to get entities")}


@tool(
    name="knowledge_graph_stats",
    description="Get statistics about the knowledge graph - how many documents, entities, and what topics are covered.",
    parameters={},
)
async def knowledge_graph_stats() -> dict:
    """Get knowledge graph statistics."""
    from integrations.lightrag.client import get_document_status, get_popular_entities, health_check

    stats = {"success": True}

    # Get health info
    health = await health_check()
    if health.get("healthy"):
        details = health.get("details", {})
        stats["version"] = details.get("core_version", "unknown")
        stats["api_version"] = details.get("api_version", "unknown")

    # Get document counts
    doc_status = await get_document_status()
    if doc_status.get("success"):
        counts = doc_status.get("status", {}).get("status_counts", {})
        stats["documents"] = {
            "total": counts.get("all", 0),
            "processed": counts.get("processed", 0),
            "pending": counts.get("pending", 0),
            "failed": counts.get("failed", 0),
        }

    # Get top entities to show coverage
    entities = await get_popular_entities(limit=30)
    if entities.get("success"):
        stats["top_entities"] = entities.get("entities", [])
        stats["note"] = "Top entities represent the most connected concepts in the knowledge graph"

    return stats


# ==================== GOOGLE DRIVE ====================

@tool(
    name="drive_list_files",
    description="List files in Google Drive or a specific folder.",
    parameters={"folder_id": "str (optional)", "query": "str - search term (optional)", "file_type": "str - folder/document/spreadsheet/presentation/pdf/image (optional)"},
)
def drive_list_files(folder_id: str = None, query: str = None, file_type: str = None) -> list[dict]:
    from integrations.google_drive.client import list_files
    return list_files(folder_id=folder_id, query=query, file_type=file_type)


@tool(
    name="drive_search",
    description="Search for files in Google Drive by name or content.",
    parameters={"query": "str - search query"},
)
def drive_search(query: str) -> list[dict]:
    from integrations.google_drive.client import search_files
    return search_files(query)


@tool(
    name="drive_create_folder",
    description="Create a folder in Google Drive.",
    parameters={"name": "str - folder name", "parent_id": "str - parent folder ID (optional)"},
)
def drive_create_folder(name: str, parent_id: str = None) -> dict:
    from integrations.google_drive.client import create_folder
    return create_folder(name, parent_id)


@tool(
    name="drive_upload",
    description="Upload a file to Google Drive.",
    parameters={"local_path": "str - path to local file", "name": "str - name in Drive (optional)", "folder_id": "str - destination folder (optional)"},
)
def drive_upload(local_path: str, name: str = None, folder_id: str = None) -> dict:
    from integrations.google_drive.client import upload_file
    return upload_file(local_path, name, folder_id)


@tool(
    name="drive_download",
    description="Download a file from Google Drive.",
    parameters={"file_id": "str - Drive file ID", "local_path": "str - where to save locally"},
)
def drive_download(file_id: str, local_path: str) -> dict:
    from integrations.google_drive.client import download_file
    return download_file(file_id, local_path)


@tool(
    name="drive_share",
    description="Share a file with someone.",
    parameters={"file_id": "str - file ID", "email": "str - email to share with", "role": "str - reader/writer/commenter (default reader)"},
)
def drive_share(file_id: str, email: str, role: str = "reader") -> dict:
    from integrations.google_drive.client import share_file
    return share_file(file_id, email, role)


@tool(
    name="drive_delete",
    description="Move a file to trash.",
    parameters={"file_id": "str - file ID to delete"},
)
def drive_delete(file_id: str) -> dict:
    from integrations.google_drive.client import delete_file
    return delete_file(file_id)


# ==================== GOOGLE DOCS ====================

@tool(
    name="docs_create",
    description="Create a new Google Doc.",
    parameters={"title": "str - document title", "content": "str - initial text (optional)", "folder_id": "str - folder to create in (optional)"},
)
def docs_create(title: str, content: str = None, folder_id: str = None) -> dict:
    from integrations.google_docs.client import create_document
    return create_document(title, content, folder_id)


@tool(
    name="docs_read",
    description="Read the text content of a Google Doc.",
    parameters={"document_id": "str - the document ID"},
)
def docs_read(document_id: str) -> str:
    from integrations.google_docs.client import read_document
    return read_document(document_id)


@tool(
    name="docs_append",
    description="Append text to the end of a Google Doc.",
    parameters={"document_id": "str - the document ID", "text": "str - text to append"},
)
def docs_append(document_id: str, text: str) -> dict:
    from integrations.google_docs.client import append_text
    return append_text(document_id, text)


@tool(
    name="docs_replace",
    description="Find and replace text in a Google Doc.",
    parameters={"document_id": "str - the document ID", "find": "str - text to find", "replace": "str - replacement text"},
)
def docs_replace(document_id: str, find: str, replace: str) -> dict:
    from integrations.google_docs.client import replace_text
    return replace_text(document_id, find, replace)


@tool(
    name="docs_list",
    description="List Google Docs.",
    parameters={"max_results": "int (default 20)"},
)
def docs_list(max_results: int = 20) -> list[dict]:
    from integrations.google_docs.client import list_documents
    return list_documents(max_results)


# ==================== GOOGLE SHEETS ====================

@tool(
    name="sheets_create",
    description="Create a new Google Spreadsheet.",
    parameters={"title": "str - spreadsheet title", "sheet_names": "list of str - sheet names (optional)", "folder_id": "str - folder (optional)"},
)
def sheets_create(title: str, sheet_names: list = None, folder_id: str = None) -> dict:
    from integrations.google_sheets.client import create_spreadsheet
    return create_spreadsheet(title, sheet_names, folder_id)


@tool(
    name="sheets_read",
    description="Read data from a Google Sheet range.",
    parameters={"spreadsheet_id": "str", "range_notation": "str - A1 notation like 'Sheet1!A1:D10'"},
)
def sheets_read(spreadsheet_id: str, range_notation: str) -> list[list]:
    from integrations.google_sheets.client import read_range
    return read_range(spreadsheet_id, range_notation)


@tool(
    name="sheets_write",
    description="Write data to a Google Sheet range.",
    parameters={"spreadsheet_id": "str", "range_notation": "str - A1 notation", "values": "2D list of values"},
)
def sheets_write(spreadsheet_id: str, range_notation: str, values: list[list]) -> dict:
    from integrations.google_sheets.client import write_range
    return write_range(spreadsheet_id, range_notation, values)


@tool(
    name="sheets_append_row",
    description="Append a row to a Google Sheet.",
    parameters={"spreadsheet_id": "str", "values": "list of values for the row", "sheet_name": "str (default Sheet1)"},
)
def sheets_append_row(spreadsheet_id: str, values: list, sheet_name: str = "Sheet1") -> dict:
    from integrations.google_sheets.client import append_row
    return append_row(spreadsheet_id, values, sheet_name)


@tool(
    name="sheets_list",
    description="List Google Spreadsheets.",
    parameters={"max_results": "int (default 20)"},
)
def sheets_list(max_results: int = 20) -> list[dict]:
    from integrations.google_sheets.client import list_spreadsheets
    return list_spreadsheets(max_results)


@tool(
    name="sheets_get",
    description="Get spreadsheet metadata including sheet names.",
    parameters={"spreadsheet_id": "str"},
)
def sheets_get(spreadsheet_id: str) -> dict:
    from integrations.google_sheets.client import get_spreadsheet
    return get_spreadsheet(spreadsheet_id)


# ==================== GOOGLE SLIDES ====================

@tool(
    name="slides_create",
    description="Create a new Google Slides presentation.",
    parameters={"title": "str - presentation title", "folder_id": "str - folder (optional)"},
)
def slides_create(title: str, folder_id: str = None) -> dict:
    from integrations.google_slides.client import create_presentation
    return create_presentation(title, folder_id)


@tool(
    name="slides_get",
    description="Get presentation metadata and slide info.",
    parameters={"presentation_id": "str"},
)
def slides_get(presentation_id: str) -> dict:
    from integrations.google_slides.client import get_presentation
    return get_presentation(presentation_id)


@tool(
    name="slides_add_slide",
    description="Add a slide to a presentation.",
    parameters={"presentation_id": "str", "layout": "str - BLANK/TITLE/TITLE_AND_BODY (default BLANK)"},
)
def slides_add_slide(presentation_id: str, layout: str = "BLANK") -> dict:
    from integrations.google_slides.client import add_slide
    return add_slide(presentation_id, layout)


@tool(
    name="slides_add_text",
    description="Add text to a slide.",
    parameters={"presentation_id": "str", "slide_id": "str", "text": "str", "x": "float (default 100)", "y": "float (default 100)"},
)
def slides_add_text(presentation_id: str, slide_id: str, text: str, x: float = 100, y: float = 100) -> dict:
    from integrations.google_slides.client import add_text_to_slide
    return add_text_to_slide(presentation_id, slide_id, text, x, y)


@tool(
    name="slides_list",
    description="List Google Slides presentations.",
    parameters={"max_results": "int (default 20)"},
)
def slides_list(max_results: int = 20) -> list[dict]:
    from integrations.google_slides.client import list_presentations
    return list_presentations(max_results)


# ==================== N8N WORKFLOW AUTOMATION ====================

@tool(
    name="n8n_list_workflows",
    description="List all automation workflows in n8n.",
    parameters={"limit": "int (default 50)", "active_only": "bool - only show active workflows (default false)"},
)
def n8n_list_workflows(limit: int = 50, active_only: bool = False) -> list[dict]:
    from integrations.n8n.client import list_workflows
    return list_workflows(limit, active_only)


@tool(
    name="n8n_get_workflow",
    description="Get details of a specific workflow including its nodes and structure.",
    parameters={"workflow_id": "string - the workflow ID"},
)
def n8n_get_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import get_workflow_summary
    return get_workflow_summary(workflow_id)


@tool(
    name="n8n_create_workflow",
    description="Create a new automation workflow from a description. Describe what you want the workflow to do and it will generate the appropriate structure.",
    parameters={
        "name": "string - workflow name",
        "description": "string - describe what the workflow should do (e.g., 'send a Slack message every day at 9am')",
    },
)
def n8n_create_workflow_from_desc(name: str, description: str) -> dict:
    from integrations.n8n.client import create_workflow_from_description, create_workflow
    spec = create_workflow_from_description(name, description)
    result = create_workflow(spec["name"], spec["nodes"], spec["connections"])
    return {
        "success": True,
        "workflow": result,
        "message": f"Workflow '{name}' created. Use n8n_activate_workflow to enable it.",
        "nodes_created": len(spec["nodes"]),
    }


@tool(
    name="n8n_create_webhook_slack_workflow",
    description="Create a workflow that listens for webhooks and sends notifications to Slack.",
    parameters={
        "name": "string - workflow name",
        "webhook_path": "string - URL path for the webhook (e.g., 'my-webhook')",
        "slack_channel": "string - Slack channel (e.g., '#general')",
        "message_template": "string - message to send (can include {{$json.field}} placeholders)",
    },
)
def n8n_create_webhook_slack(name: str, webhook_path: str, slack_channel: str, message_template: str) -> dict:
    from integrations.n8n.client import create_webhook_to_slack_workflow
    result = create_webhook_to_slack_workflow(name, webhook_path, slack_channel, message_template)
    return {"success": True, "workflow": result}


@tool(
    name="n8n_create_scheduled_workflow",
    description="Create a workflow that runs on a schedule and makes an HTTP request.",
    parameters={
        "name": "string - workflow name",
        "cron": "string - cron expression (e.g., '0 9 * * *' for 9 AM daily)",
        "url": "string - URL to call",
        "method": "string - HTTP method (GET, POST, etc.)",
    },
)
def n8n_create_scheduled(name: str, cron: str, url: str, method: str = "GET") -> dict:
    from integrations.n8n.client import create_scheduled_http_workflow
    result = create_scheduled_http_workflow(name, cron, url, method)
    return {"success": True, "workflow": result}


@tool(
    name="n8n_activate_workflow",
    description="Activate a workflow so it starts running.",
    parameters={"workflow_id": "string - the workflow ID to activate"},
)
def n8n_activate_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import activate_workflow
    result = activate_workflow(workflow_id)
    return {"success": True, "workflow": result, "message": "Workflow activated"}


@tool(
    name="n8n_deactivate_workflow",
    description="Deactivate a workflow to stop it from running.",
    parameters={"workflow_id": "string - the workflow ID to deactivate"},
)
def n8n_deactivate_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import deactivate_workflow
    result = deactivate_workflow(workflow_id)
    return {"success": True, "workflow": result, "message": "Workflow deactivated"}


@tool(
    name="n8n_delete_workflow",
    description="Delete a workflow permanently.",
    parameters={"workflow_id": "string - the workflow ID to delete"},
)
def n8n_delete_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import delete_workflow
    return delete_workflow(workflow_id)


@tool(
    name="n8n_execute_workflow",
    description="Execute/run a workflow manually, optionally with input data.",
    parameters={
        "workflow_id": "string - the workflow ID to execute",
        "data": "dict - optional input data to pass to the workflow",
    },
)
def n8n_execute_workflow(workflow_id: str, data: dict = None) -> dict:
    from integrations.n8n.client import execute_workflow
    return execute_workflow(workflow_id, data)


@tool(
    name="n8n_get_executions",
    description="Get execution history for workflows.",
    parameters={
        "workflow_id": "string - optional workflow ID to filter by",
        "limit": "int - max executions to return (default 20)",
    },
)
def n8n_get_executions(workflow_id: str = "", limit: int = 20) -> list[dict]:
    from integrations.n8n.client import get_executions
    return get_executions(workflow_id if workflow_id else None, limit)


# ==================== NEXTCLOUD ====================

@tool(
    name="nextcloud_list_files",
    description="List files and folders in Nextcloud at a given path.",
    parameters={"path": "string - folder path (default '/')"},
)
def nextcloud_list_files(path: str = "/") -> list[dict]:
    from integrations.nextcloud.client import list_files
    return list_files(path)


@tool(
    name="nextcloud_search_files",
    description="Search for files by name in Nextcloud.",
    parameters={"query": "string - search query", "path": "string - folder to search in (default '/')"},
)
def nextcloud_search_files(query: str, path: str = "/") -> list[dict]:
    from integrations.nextcloud.client import search_files
    return search_files(query, path)


@tool(
    name="nextcloud_read_file",
    description="Read the contents of a text file from Nextcloud.",
    parameters={"path": "string - file path"},
)
def nextcloud_read_file(path: str) -> dict:
    from integrations.nextcloud.client import download_file_text
    try:
        content = download_file_text(path)
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="nextcloud_upload_file",
    description="Upload or create a text file in Nextcloud.",
    parameters={"path": "string - destination path in Nextcloud", "content": "string - file content (text)"},
)
def nextcloud_upload_file(path: str, content: str) -> dict:
    from integrations.nextcloud.client import upload_file
    return upload_file(path, content)


@tool(
    name="nextcloud_upload_attached_file",
    description="Upload an attached file to Nextcloud. The local_file_path is provided in the message context. IMPORTANT: If user doesn't specify a folder, ASK them where they want the file uploaded before calling this tool. Common folders: /Photos, /Documents, /Ruck Talk, /Projects.",
    parameters={
        "local_file_path": "string - the local server path of the attached file (from context)",
        "destination_path": "string - FULL path in Nextcloud including filename (e.g., '/Ruck Talk/image.png', '/Photos/vacation.jpg'). Ask user if not specified.",
    },
)
def nextcloud_upload_attached_file(local_file_path: str, destination_path: str) -> dict:
    """Upload a local file to Nextcloud."""
    import mimetypes
    from pathlib import Path
    from integrations.nextcloud.client import upload_file

    try:
        file_path = Path(local_file_path)
        if not file_path.exists():
            return {"error": f"File not found: {local_file_path}"}

        # Read the file
        file_bytes = file_path.read_bytes()

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"

        result = upload_file(destination_path, file_bytes, content_type)
        return result
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


@tool(
    name="nextcloud_create_folder",
    description="Create a folder in Nextcloud.",
    parameters={"path": "string - folder path to create"},
)
def nextcloud_create_folder(path: str) -> dict:
    from integrations.nextcloud.client import create_folder
    return create_folder(path)


@tool(
    name="nextcloud_delete_file",
    description="Delete a file or folder from Nextcloud.",
    parameters={"path": "string - path to delete"},
)
def nextcloud_delete_file(path: str) -> dict:
    from integrations.nextcloud.client import delete_file
    return delete_file(path)


@tool(
    name="nextcloud_move_file",
    description="Move or rename a file/folder in Nextcloud.",
    parameters={"source": "string - current path", "destination": "string - new path"},
)
def nextcloud_move_file(source: str, destination: str) -> dict:
    from integrations.nextcloud.client import move_file
    return move_file(source, destination)


@tool(
    name="nextcloud_storage_info",
    description="Get Nextcloud storage quota usage.",
    parameters={},
)
def nextcloud_storage_info() -> dict:
    from integrations.nextcloud.client import get_storage_info
    return get_storage_info()


# Nextcloud Notes
@tool(
    name="nextcloud_list_notes",
    description="List all notes in Nextcloud Notes app.",
    parameters={},
)
def nextcloud_list_notes() -> list[dict]:
    from integrations.nextcloud.client import list_notes
    return list_notes()


@tool(
    name="nextcloud_get_note",
    description="Get a specific note from Nextcloud.",
    parameters={"note_id": "int - note ID"},
)
def nextcloud_get_note(note_id: int) -> dict:
    from integrations.nextcloud.client import get_note
    return get_note(note_id)


@tool(
    name="nextcloud_create_note",
    description="Create a new note in Nextcloud.",
    parameters={"title": "string", "content": "string (optional)", "category": "string (optional)"},
)
def nextcloud_create_note(title: str, content: str = "", category: str = "") -> dict:
    from integrations.nextcloud.client import create_note
    return create_note(title, content, category)


@tool(
    name="nextcloud_update_note",
    description="Update a note in Nextcloud.",
    parameters={"note_id": "int", "title": "string (optional)", "content": "string (optional)"},
)
def nextcloud_update_note(note_id: int, title: str = None, content: str = None) -> dict:
    from integrations.nextcloud.client import update_note
    return update_note(note_id, title, content)


@tool(
    name="nextcloud_delete_note",
    description="Delete a note from Nextcloud.",
    parameters={"note_id": "int"},
)
def nextcloud_delete_note(note_id: int) -> dict:
    from integrations.nextcloud.client import delete_note
    return delete_note(note_id)


# Nextcloud Talk
@tool(
    name="nextcloud_list_conversations",
    description="List all Nextcloud Talk conversations/chats.",
    parameters={},
)
def nextcloud_list_conversations() -> list[dict]:
    from integrations.nextcloud.client import list_conversations
    return list_conversations()


@tool(
    name="nextcloud_get_messages",
    description="Get messages from a Nextcloud Talk conversation.",
    parameters={"token": "string - conversation token", "limit": "int (default 50)"},
)
def nextcloud_get_messages(token: str, limit: int = 50) -> list[dict]:
    from integrations.nextcloud.client import get_messages
    return get_messages(token, limit)


@tool(
    name="nextcloud_send_message",
    description="Send a message to a Nextcloud Talk conversation.",
    parameters={"token": "string - conversation token", "message": "string - message to send"},
)
def nextcloud_send_message(token: str, message: str) -> dict:
    from integrations.nextcloud.client import send_message
    return send_message(token, message)


@tool(
    name="nextcloud_create_conversation",
    description="Create a new Nextcloud Talk group conversation.",
    parameters={"name": "string - conversation name", "invite_users": "list of user IDs to invite (optional)"},
)
def nextcloud_create_conversation(name: str, invite_users: list[str] = None) -> dict:
    from integrations.nextcloud.client import create_conversation
    return create_conversation(name, 2, invite_users)


@tool(
    name="nextcloud_add_participant",
    description="Add a user to a Nextcloud Talk conversation.",
    parameters={"token": "string - conversation token", "user_id": "string - user to add"},
)
def nextcloud_add_participant(token: str, user_id: str) -> dict:
    from integrations.nextcloud.client import add_participant
    return add_participant(token, user_id)


# Nextcloud User Management
@tool(
    name="nextcloud_list_users",
    description="List users in Nextcloud (requires admin).",
    parameters={"search": "string - search query (optional)", "limit": "int (default 50)"},
)
def nextcloud_list_users(search: str = "", limit: int = 50) -> list[dict]:
    from integrations.nextcloud.client import list_users
    return list_users(search, limit)


@tool(
    name="nextcloud_get_user",
    description="Get details about a Nextcloud user.",
    parameters={"user_id": "string - username"},
)
def nextcloud_get_user(user_id: str) -> dict:
    from integrations.nextcloud.client import get_user
    return get_user(user_id)


@tool(
    name="nextcloud_create_user",
    description="Create a new Nextcloud user (requires admin).",
    parameters={
        "user_id": "string - username",
        "password": "string - initial password",
        "email": "string (optional)",
        "display_name": "string (optional)",
        "groups": "list of group names (optional)",
    },
)
def nextcloud_create_user(user_id: str, password: str, email: str = "",
                          display_name: str = "", groups: list[str] = None) -> dict:
    from integrations.nextcloud.client import create_user
    return create_user(user_id, password, email, display_name, groups)


@tool(
    name="nextcloud_delete_user",
    description="Delete a Nextcloud user (requires admin).",
    parameters={"user_id": "string - username to delete"},
)
def nextcloud_delete_user(user_id: str) -> dict:
    from integrations.nextcloud.client import delete_user
    return delete_user(user_id)


@tool(
    name="nextcloud_enable_user",
    description="Enable a disabled Nextcloud user.",
    parameters={"user_id": "string - username"},
)
def nextcloud_enable_user(user_id: str) -> dict:
    from integrations.nextcloud.client import enable_user
    return enable_user(user_id)


@tool(
    name="nextcloud_disable_user",
    description="Disable a Nextcloud user.",
    parameters={"user_id": "string - username"},
)
def nextcloud_disable_user(user_id: str) -> dict:
    from integrations.nextcloud.client import disable_user
    return disable_user(user_id)


@tool(
    name="nextcloud_list_groups",
    description="List all Nextcloud groups.",
    parameters={},
)
def nextcloud_list_groups() -> list[dict]:
    from integrations.nextcloud.client import list_groups
    return list_groups()


@tool(
    name="nextcloud_add_user_to_group",
    description="Add a user to a Nextcloud group.",
    parameters={"user_id": "string - username", "group_id": "string - group name"},
)
def nextcloud_add_user_to_group(user_id: str, group_id: str) -> dict:
    from integrations.nextcloud.client import add_user_to_group
    return add_user_to_group(user_id, group_id)


# Nextcloud Calendar
@tool(
    name="nextcloud_list_calendars",
    description="List Nextcloud calendars.",
    parameters={},
)
def nextcloud_list_calendars() -> list[dict]:
    from integrations.nextcloud.client import list_calendars
    return list_calendars()


@tool(
    name="nextcloud_get_calendar_events",
    description="Get events from a Nextcloud calendar.",
    parameters={"calendar_id": "string - calendar ID", "days_ahead": "int - days to look ahead (default 30)"},
)
def nextcloud_get_calendar_events(calendar_id: str, days_ahead: int = 30) -> list[dict]:
    from integrations.nextcloud.client import get_calendar_events
    return get_calendar_events(calendar_id, days_ahead)


# Nextcloud Tasks
@tool(
    name="nextcloud_get_tasks",
    description="Get tasks from a Nextcloud task list.",
    parameters={"list_id": "string - task list/calendar ID"},
)
def nextcloud_get_tasks(list_id: str) -> list[dict]:
    from integrations.nextcloud.client import get_tasks
    return get_tasks(list_id)


# ==================== STRIPE ====================

@tool(
    name="stripe_get_balance",
    description="Get Stripe account balance. Returns amounts in DOLLARS (already converted from cents). Example: amount=398.0 means $398.00 USD, NOT $3.98.",
    parameters={},
)
def stripe_get_balance() -> dict:
    from integrations.stripe.client import get_balance
    result = get_balance()
    # Add explicit formatting hint for LLM
    for bal_type in ["available", "pending"]:
        for item in result.get(bal_type, []):
            item["formatted"] = f"${item['amount']:.2f} {item['currency']}"
    return result


@tool(
    name="stripe_list_payouts",
    description="List recent Stripe payouts to your bank account.",
    parameters={"limit": "int (default 20)"},
)
def stripe_list_payouts(limit: int = 20) -> list[dict]:
    from integrations.stripe.client import list_payouts
    return list_payouts(limit)


@tool(
    name="stripe_revenue_summary",
    description="Get a summary of recent revenue, active subscriptions, and estimated MRR.",
    parameters={},
)
def stripe_revenue_summary() -> dict:
    from integrations.stripe.client import get_revenue_summary
    return get_revenue_summary()


# Payments
@tool(
    name="stripe_list_payments",
    description="List recent Stripe payments/charges.",
    parameters={"limit": "int (default 20)", "customer": "string - customer ID (optional)"},
)
def stripe_list_payments(limit: int = 20, customer: str = None) -> list[dict]:
    from integrations.stripe.client import list_payments
    return list_payments(limit, customer)


@tool(
    name="stripe_get_payment",
    description="Get details of a specific Stripe payment/charge.",
    parameters={"charge_id": "string - charge ID (ch_...)"},
)
def stripe_get_payment(charge_id: str) -> dict:
    from integrations.stripe.client import get_payment
    return get_payment(charge_id)


@tool(
    name="stripe_search_payments",
    description="Search Stripe payments. Query examples: 'amount>1000', 'status:succeeded', 'customer:cus_xxx'.",
    parameters={"query": "string - Stripe search query", "limit": "int (default 20)"},
)
def stripe_search_payments(query: str, limit: int = 20) -> list[dict]:
    from integrations.stripe.client import search_payments
    return search_payments(query, limit)


@tool(
    name="stripe_refund_payment",
    description="Issue a refund for a Stripe payment. Omit amount for full refund.",
    parameters={
        "charge_id": "string - charge ID to refund",
        "amount": "float - amount in dollars (optional, omit for full refund)",
        "reason": "string - 'duplicate', 'fraudulent', or 'requested_by_customer' (optional)",
    },
)
def stripe_refund_payment(charge_id: str, amount: float = None, reason: str = None) -> dict:
    from integrations.stripe.client import refund_payment
    return refund_payment(charge_id, amount, reason)


@tool(
    name="stripe_list_refunds",
    description="List Stripe refunds.",
    parameters={"limit": "int (default 20)", "charge": "string - charge ID to filter by (optional)"},
)
def stripe_list_refunds(limit: int = 20, charge: str = None) -> list[dict]:
    from integrations.stripe.client import list_refunds
    return list_refunds(limit, charge)


# Customers
@tool(
    name="stripe_list_customers",
    description="List Stripe customers.",
    parameters={"limit": "int (default 20)", "email": "string - filter by email (optional)"},
)
def stripe_list_customers(limit: int = 20, email: str = None) -> list[dict]:
    from integrations.stripe.client import list_customers
    return list_customers(limit, email)


@tool(
    name="stripe_search_customers",
    description="Search Stripe customers. Query examples: 'email:john@example.com', 'name:John'.",
    parameters={"query": "string - Stripe search query", "limit": "int (default 20)"},
)
def stripe_search_customers(query: str, limit: int = 20) -> list[dict]:
    from integrations.stripe.client import search_customers
    return search_customers(query, limit)


@tool(
    name="stripe_get_customer",
    description="Get details of a specific Stripe customer.",
    parameters={"customer_id": "string - customer ID (cus_...)"},
)
def stripe_get_customer(customer_id: str) -> dict:
    from integrations.stripe.client import get_customer
    return get_customer(customer_id)


@tool(
    name="stripe_create_customer",
    description="Create a new Stripe customer.",
    parameters={
        "email": "string - customer email",
        "name": "string (optional)",
        "phone": "string (optional)",
        "description": "string (optional)",
    },
)
def stripe_create_customer(email: str, name: str = None, phone: str = None, description: str = None) -> dict:
    from integrations.stripe.client import create_customer
    return create_customer(email, name, phone, description)


@tool(
    name="stripe_update_customer",
    description="Update a Stripe customer.",
    parameters={
        "customer_id": "string - customer ID",
        "email": "string (optional)",
        "name": "string (optional)",
        "phone": "string (optional)",
    },
)
def stripe_update_customer(customer_id: str, email: str = None, name: str = None, phone: str = None) -> dict:
    from integrations.stripe.client import update_customer
    return update_customer(customer_id, email, name, phone)


@tool(
    name="stripe_delete_customer",
    description="Delete a Stripe customer.",
    parameters={"customer_id": "string - customer ID to delete"},
)
def stripe_delete_customer(customer_id: str) -> dict:
    from integrations.stripe.client import delete_customer
    return delete_customer(customer_id)


# Subscriptions
@tool(
    name="stripe_list_subscriptions",
    description="List Stripe subscriptions.",
    parameters={
        "limit": "int (default 20)",
        "customer": "string - customer ID (optional)",
        "status": "string - 'active', 'past_due', 'canceled', 'all' (optional)",
    },
)
def stripe_list_subscriptions(limit: int = 20, customer: str = None, status: str = None) -> list[dict]:
    from integrations.stripe.client import list_subscriptions
    return list_subscriptions(limit, customer, status)


@tool(
    name="stripe_get_subscription",
    description="Get details of a Stripe subscription.",
    parameters={"subscription_id": "string - subscription ID (sub_...)"},
)
def stripe_get_subscription(subscription_id: str) -> dict:
    from integrations.stripe.client import get_subscription
    return get_subscription(subscription_id)


@tool(
    name="stripe_create_subscription",
    description="Create a subscription for a customer.",
    parameters={
        "customer_id": "string - customer ID",
        "price_id": "string - price ID (price_...)",
        "quantity": "int (default 1)",
    },
)
def stripe_create_subscription(customer_id: str, price_id: str, quantity: int = 1) -> dict:
    from integrations.stripe.client import create_subscription
    return create_subscription(customer_id, price_id, quantity)


@tool(
    name="stripe_cancel_subscription",
    description="Cancel a Stripe subscription.",
    parameters={
        "subscription_id": "string - subscription ID",
        "at_period_end": "bool - if true, cancel at end of billing period (default true)",
    },
)
def stripe_cancel_subscription(subscription_id: str, at_period_end: bool = True) -> dict:
    from integrations.stripe.client import cancel_subscription
    return cancel_subscription(subscription_id, at_period_end)


@tool(
    name="stripe_resume_subscription",
    description="Resume a subscription that was scheduled for cancellation.",
    parameters={"subscription_id": "string - subscription ID"},
)
def stripe_resume_subscription(subscription_id: str) -> dict:
    from integrations.stripe.client import resume_subscription
    return resume_subscription(subscription_id)


# Products & Prices
@tool(
    name="stripe_list_products",
    description="List Stripe products.",
    parameters={"limit": "int (default 20)", "active": "bool - filter by active status (optional)"},
)
def stripe_list_products(limit: int = 20, active: bool = None) -> list[dict]:
    from integrations.stripe.client import list_products
    return list_products(limit, active)


@tool(
    name="stripe_get_product",
    description="Get details of a Stripe product.",
    parameters={"product_id": "string - product ID (prod_...)"},
)
def stripe_get_product(product_id: str) -> dict:
    from integrations.stripe.client import get_product
    return get_product(product_id)


@tool(
    name="stripe_create_product",
    description="Create a new Stripe product.",
    parameters={
        "name": "string - product name",
        "description": "string (optional)",
        "active": "bool (default true)",
    },
)
def stripe_create_product(name: str, description: str = None, active: bool = True) -> dict:
    from integrations.stripe.client import create_product
    return create_product(name, description, active)


@tool(
    name="stripe_list_prices",
    description="List Stripe prices.",
    parameters={
        "limit": "int (default 20)",
        "product": "string - product ID (optional)",
        "active": "bool (optional)",
    },
)
def stripe_list_prices(limit: int = 20, product: str = None, active: bool = None) -> list[dict]:
    from integrations.stripe.client import list_prices
    return list_prices(limit, product, active)


@tool(
    name="stripe_create_price",
    description="Create a price for a product.",
    parameters={
        "product_id": "string - product ID",
        "unit_amount": "float - price in dollars",
        "currency": "string (default 'usd')",
        "recurring_interval": "string - 'month' or 'year' for subscriptions (optional)",
    },
)
def stripe_create_price(product_id: str, unit_amount: float, currency: str = "usd",
                        recurring_interval: str = None) -> dict:
    from integrations.stripe.client import create_price
    return create_price(product_id, unit_amount, currency, recurring_interval)


# Invoices
@tool(
    name="stripe_list_invoices",
    description="List Stripe invoices.",
    parameters={
        "limit": "int (default 20)",
        "customer": "string - customer ID (optional)",
        "status": "string - 'draft', 'open', 'paid', 'void' (optional)",
    },
)
def stripe_list_invoices(limit: int = 20, customer: str = None, status: str = None) -> list[dict]:
    from integrations.stripe.client import list_invoices
    return list_invoices(limit, customer, status)


@tool(
    name="stripe_get_invoice",
    description="Get details of a Stripe invoice.",
    parameters={"invoice_id": "string - invoice ID (in_...)"},
)
def stripe_get_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import get_invoice
    return get_invoice(invoice_id)


@tool(
    name="stripe_create_invoice",
    description="Create a draft invoice for a customer.",
    parameters={
        "customer_id": "string - customer ID",
        "description": "string (optional)",
        "days_until_due": "int (default 30)",
    },
)
def stripe_create_invoice(customer_id: str, description: str = None, days_until_due: int = 30) -> dict:
    from integrations.stripe.client import create_invoice
    return create_invoice(customer_id, description, days_until_due)


@tool(
    name="stripe_add_invoice_item",
    description="Add a line item to a draft invoice.",
    parameters={
        "invoice_id": "string - invoice ID",
        "description": "string - item description",
        "amount": "float - amount in dollars",
        "quantity": "int (default 1)",
    },
)
def stripe_add_invoice_item(invoice_id: str, description: str, amount: float, quantity: int = 1) -> dict:
    from integrations.stripe.client import add_invoice_item
    return add_invoice_item(invoice_id, description, amount, quantity)


@tool(
    name="stripe_finalize_invoice",
    description="Finalize a draft invoice (locks it for payment).",
    parameters={"invoice_id": "string - invoice ID"},
)
def stripe_finalize_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import finalize_invoice
    return finalize_invoice(invoice_id)


@tool(
    name="stripe_send_invoice",
    description="Send an invoice to the customer via email.",
    parameters={"invoice_id": "string - invoice ID"},
)
def stripe_send_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import send_invoice
    return send_invoice(invoice_id)


@tool(
    name="stripe_void_invoice",
    description="Void an invoice.",
    parameters={"invoice_id": "string - invoice ID"},
)
def stripe_void_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import void_invoice
    return void_invoice(invoice_id)


# Payment Links
@tool(
    name="stripe_list_payment_links",
    description="List Stripe payment links.",
    parameters={"limit": "int (default 20)", "active": "bool (optional)"},
)
def stripe_list_payment_links(limit: int = 20, active: bool = None) -> list[dict]:
    from integrations.stripe.client import list_payment_links
    return list_payment_links(limit, active)


@tool(
    name="stripe_create_payment_link",
    description="Create a Stripe payment link for a price.",
    parameters={"price_id": "string - price ID", "quantity": "int (default 1)"},
)
def stripe_create_payment_link(price_id: str, quantity: int = 1) -> dict:
    from integrations.stripe.client import create_payment_link
    return create_payment_link(price_id, quantity)


@tool(
    name="stripe_deactivate_payment_link",
    description="Deactivate a Stripe payment link.",
    parameters={"payment_link_id": "string - payment link ID"},
)
def stripe_deactivate_payment_link(payment_link_id: str) -> dict:
    from integrations.stripe.client import deactivate_payment_link
    return deactivate_payment_link(payment_link_id)


# ==================== UI CONTROL ====================

@tool(
    name="toggle_auto_speak",
    description="Turn auto-speak on or off. Use this when user asks to enable/disable auto-speak, mute, unmute, or stop/start speaking.",
    parameters={"enabled": "bool - true to enable, false to disable"},
)
def toggle_auto_speak(enabled: bool) -> dict:
    action = "enable" if enabled else "disable"
    return {
        "ui_action": "set_auto_speak",
        "value": enabled,
        "message": f"Auto-speak {'enabled' if enabled else 'disabled'}, sir."
    }


@tool(
    name="toggle_hands_free",
    description="Turn hands-free mode on or off. Use this when user asks to enable/disable hands-free or start/stop listening.",
    parameters={"enabled": "bool - true to enable, false to disable"},
)
def toggle_hands_free(enabled: bool) -> dict:
    return {
        "ui_action": "set_hands_free",
        "value": enabled,
        "message": f"Hands-free mode {'enabled' if enabled else 'disabled'}, sir."
    }


# ==================== AZURACAST RADIO TOOLS ====================

@tool(
    name="radio_list_stations",
    description="List all available radio stations. Use this to see station names and IDs for other radio commands.",
    parameters={},
)
def radio_list_stations() -> list[dict]:
    from integrations.azuracast.client import list_stations
    return list_stations()


@tool(
    name="radio_now_playing",
    description="Get what's currently playing on a radio station, including listener count, current song artist/title, and playlist info.",
    parameters={"station": "string (optional) - station name, shortcode, or ID. Defaults to first station if not specified."},
)
def radio_now_playing(station: str = None) -> dict:
    from integrations.azuracast.client import get_now_playing, get_station_id
    station_id = get_station_id(station)
    return get_now_playing(station_id)


@tool(
    name="radio_song_history",
    description="Get recently played songs on a radio station.",
    parameters={
        "station": "string (optional) - station name, shortcode, or ID",
        "limit": "int (default 10) - number of recent songs to return",
    },
)
def radio_song_history(station: str = None, limit: int = 10) -> list[dict]:
    from integrations.azuracast.client import get_song_history, get_station_id
    station_id = get_station_id(station)
    return get_song_history(station_id, limit=limit)


@tool(
    name="radio_playlists",
    description="List all playlists on a radio station with their song counts and enabled status.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_playlists(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_playlists, get_station_id
    station_id = get_station_id(station)
    return list_playlists(station_id)


@tool(
    name="radio_toggle_playlist",
    description="Enable or disable a radio playlist by its ID.",
    parameters={
        "playlist_id": "int - the playlist ID to toggle",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_toggle_playlist(playlist_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import toggle_playlist, get_station_id
    station_id = get_station_id(station)
    return toggle_playlist(station_id, playlist_id)


@tool(
    name="radio_queue",
    description="Get the upcoming song queue on a radio station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_queue(station: str = None) -> list[dict]:
    from integrations.azuracast.client import get_queue, get_station_id
    station_id = get_station_id(station)
    return get_queue(station_id)


@tool(
    name="radio_listeners",
    description="Get current listener details and count for a radio station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_listeners(station: str = None) -> dict:
    from integrations.azuracast.client import get_listener_report, get_station_id
    station_id = get_station_id(station)
    return get_listener_report(station_id)


@tool(
    name="radio_restart",
    description="Restart a radio station's broadcasting services.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_restart(station: str = None) -> dict:
    from integrations.azuracast.client import restart_station, get_station_id
    station_id = get_station_id(station)
    return restart_station(station_id)


@tool(
    name="radio_search_media",
    description="Search for songs in a radio station's media library by artist, title, or album.",
    parameters={
        "query": "string - search term for artist, title, or album",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_search_media(query: str, station: str = None) -> list[dict]:
    from integrations.azuracast.client import search_media, get_station_id
    station_id = get_station_id(station)
    return search_media(station_id, query)


# ==================== RADIO ADMIN: STATION MANAGEMENT ====================

@tool(
    name="radio_create_station",
    description="Create a new radio station.",
    parameters={
        "name": "string - the station name",
        "shortcode": "string (optional) - URL-friendly short name",
        "description": "string (optional) - station description",
    },
)
def radio_create_station(name: str, shortcode: str = None, description: str = "") -> dict:
    from integrations.azuracast.client import admin_create_station
    return admin_create_station(name, shortcode, description)


@tool(
    name="radio_update_station",
    description="Update station settings like name, description, or enabled status.",
    parameters={
        "station": "string - station name, shortcode, or ID",
        "name": "string (optional) - new name",
        "description": "string (optional) - new description",
        "is_enabled": "boolean (optional) - enable/disable station",
    },
)
def radio_update_station(station: str, name: str = None, description: str = None,
                         is_enabled: bool = None) -> dict:
    from integrations.azuracast.client import admin_update_station, get_station_id
    station_id = get_station_id(station)
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if description is not None:
        kwargs["description"] = description
    if is_enabled is not None:
        kwargs["is_enabled"] = is_enabled
    return admin_update_station(station_id, **kwargs)


@tool(
    name="radio_delete_station",
    description="Permanently delete a radio station. Use with caution!",
    parameters={"station": "string - station name, shortcode, or ID to delete"},
)
def radio_delete_station(station: str) -> dict:
    from integrations.azuracast.client import admin_delete_station, get_station_id
    station_id = get_station_id(station)
    return admin_delete_station(station_id)


@tool(
    name="radio_clone_station",
    description="Clone an existing station with all its settings, playlists, and media.",
    parameters={
        "station": "string - source station to clone",
        "name": "string - name for the new station",
        "shortcode": "string (optional) - shortcode for new station",
    },
)
def radio_clone_station(station: str, name: str, shortcode: str = None) -> dict:
    from integrations.azuracast.client import admin_clone_station, get_station_id
    station_id = get_station_id(station)
    return admin_clone_station(station_id, name, shortcode)


# ==================== RADIO ADMIN: USER MANAGEMENT ====================

@tool(
    name="radio_list_users",
    description="List all AzuraCast user accounts.",
    parameters={},
)
def radio_list_users() -> list[dict]:
    from integrations.azuracast.client import admin_list_users
    return admin_list_users()


@tool(
    name="radio_create_user",
    description="Create a new AzuraCast user account.",
    parameters={
        "email": "string - user's email address (used for login)",
        "name": "string - display name",
        "password": "string - account password",
    },
)
def radio_create_user(email: str, name: str, password: str) -> dict:
    from integrations.azuracast.client import admin_create_user
    return admin_create_user(email, name, password)


@tool(
    name="radio_update_user",
    description="Update a user's details.",
    parameters={
        "user_id": "int - the user ID",
        "email": "string (optional) - new email",
        "name": "string (optional) - new name",
        "password": "string (optional) - new password",
        "is_enabled": "boolean (optional) - enable/disable account",
    },
)
def radio_update_user(user_id: int, email: str = None, name: str = None,
                      password: str = None, is_enabled: bool = None) -> dict:
    from integrations.azuracast.client import admin_update_user
    kwargs = {}
    if email is not None:
        kwargs["email"] = email
    if name is not None:
        kwargs["name"] = name
    if password is not None:
        kwargs["password"] = password
    if is_enabled is not None:
        kwargs["is_enabled"] = is_enabled
    return admin_update_user(user_id, **kwargs)


@tool(
    name="radio_delete_user",
    description="Delete an AzuraCast user account.",
    parameters={"user_id": "int - the user ID to delete"},
)
def radio_delete_user(user_id: int) -> dict:
    from integrations.azuracast.client import admin_delete_user
    return admin_delete_user(user_id)


@tool(
    name="radio_list_roles",
    description="List all roles/permission groups in AzuraCast.",
    parameters={},
)
def radio_list_roles() -> list[dict]:
    from integrations.azuracast.client import admin_list_roles
    return admin_list_roles()


# ==================== RADIO ADMIN: STORAGE ====================

@tool(
    name="radio_storage_locations",
    description="List all storage locations and their usage/quotas.",
    parameters={},
)
def radio_storage_locations() -> list[dict]:
    from integrations.azuracast.client import admin_list_storage_locations
    return admin_list_storage_locations()


@tool(
    name="radio_update_storage_quota",
    description="Update storage quota for a storage location.",
    parameters={
        "location_id": "int - storage location ID",
        "quota_gb": "float - quota in gigabytes (0 for unlimited)",
    },
)
def radio_update_storage_quota(location_id: int, quota_gb: float) -> dict:
    from integrations.azuracast.client import admin_update_storage_quota
    quota_bytes = int(quota_gb * 1024 * 1024 * 1024)
    return admin_update_storage_quota(location_id, quota_bytes)


@tool(
    name="radio_station_quota",
    description="Get storage quota usage for a specific station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_station_quota(station: str = None) -> dict:
    from integrations.azuracast.client import get_station_quota, get_station_id
    station_id = get_station_id(station)
    return get_station_quota(station_id)


# ==================== RADIO: DJ/STREAMER MANAGEMENT ====================

@tool(
    name="radio_list_djs",
    description="List all DJ/streamer accounts for a station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_list_djs(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_streamers, get_station_id
    station_id = get_station_id(station)
    return list_streamers(station_id)


@tool(
    name="radio_create_dj",
    description="Create a new DJ/streamer account for live broadcasting.",
    parameters={
        "username": "string - login username for the DJ",
        "password": "string - password for streaming",
        "display_name": "string (optional) - name shown when live",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_dj(username: str, password: str, display_name: str = None,
                    station: str = None) -> dict:
    from integrations.azuracast.client import create_streamer, get_station_id
    station_id = get_station_id(station)
    return create_streamer(station_id, username, password, display_name)


@tool(
    name="radio_update_dj",
    description="Update a DJ/streamer account.",
    parameters={
        "dj_id": "int - the streamer/DJ ID",
        "username": "string (optional) - new username",
        "password": "string (optional) - new password",
        "display_name": "string (optional) - new display name",
        "is_active": "boolean (optional) - enable/disable account",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_update_dj(dj_id: int, username: str = None, password: str = None,
                    display_name: str = None, is_active: bool = None,
                    station: str = None) -> dict:
    from integrations.azuracast.client import update_streamer, get_station_id
    station_id = get_station_id(station)
    kwargs = {}
    if username is not None:
        kwargs["username"] = username
    if password is not None:
        kwargs["password"] = password
    if display_name is not None:
        kwargs["display_name"] = display_name
    if is_active is not None:
        kwargs["is_active"] = is_active
    return update_streamer(station_id, dj_id, **kwargs)


@tool(
    name="radio_delete_dj",
    description="Delete a DJ/streamer account.",
    parameters={
        "dj_id": "int - the streamer/DJ ID to delete",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_dj(dj_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_streamer, get_station_id
    station_id = get_station_id(station)
    return delete_streamer(station_id, dj_id)


# ==================== RADIO: MEDIA MANAGEMENT ====================

@tool(
    name="radio_upload_song",
    description="Upload a song file to the station's media library.",
    parameters={
        "file_path": "string - local path to the audio file",
        "folder": "string (optional) - destination folder in library",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_upload_song(file_path: str, folder: str = "", station: str = None) -> dict:
    from integrations.azuracast.client import upload_media, get_station_id
    station_id = get_station_id(station)
    return upload_media(station_id, file_path, folder)


@tool(
    name="radio_update_song",
    description="Update song metadata (artist, title, album, etc.).",
    parameters={
        "media_id": "int - the media file ID",
        "artist": "string (optional) - artist name",
        "title": "string (optional) - song title",
        "album": "string (optional) - album name",
        "genre": "string (optional) - genre",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_update_song(media_id: int, artist: str = None, title: str = None,
                      album: str = None, genre: str = None, station: str = None) -> dict:
    from integrations.azuracast.client import update_media, get_station_id
    station_id = get_station_id(station)
    kwargs = {}
    if artist is not None:
        kwargs["artist"] = artist
    if title is not None:
        kwargs["title"] = title
    if album is not None:
        kwargs["album"] = album
    if genre is not None:
        kwargs["genre"] = genre
    return update_media(station_id, media_id, **kwargs)


@tool(
    name="radio_delete_song",
    description="Delete a song from the media library.",
    parameters={
        "media_id": "int - the media file ID to delete",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_song(media_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_media, get_station_id
    station_id = get_station_id(station)
    return delete_media(station_id, media_id)


@tool(
    name="radio_create_folder",
    description="Create a folder in the media library.",
    parameters={
        "folder_path": "string - path for the new folder (e.g., 'Rock/Classic')",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_folder(folder_path: str, station: str = None) -> dict:
    from integrations.azuracast.client import create_media_folder, get_station_id
    station_id = get_station_id(station)
    return create_media_folder(station_id, folder_path)


# ==================== RADIO: QUEUE & REQUESTS ====================

@tool(
    name="radio_clear_queue",
    description="Clear all songs from the upcoming queue.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_clear_queue(station: str = None) -> dict:
    from integrations.azuracast.client import clear_queue, get_station_id
    station_id = get_station_id(station)
    return clear_queue(station_id)


@tool(
    name="radio_skip_song",
    description="Skip the currently playing song and move to next.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_skip_song(station: str = None) -> dict:
    from integrations.azuracast.client import skip_song, get_station_id
    station_id = get_station_id(station)
    return skip_song(station_id)


@tool(
    name="radio_request_song",
    description="Submit a song request to be played.",
    parameters={
        "query": "string - search for the song to request",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_request_song(query: str, station: str = None) -> dict:
    from integrations.azuracast.client import search_requestable_songs, submit_request, get_station_id
    station_id = get_station_id(station)
    # Search for the song
    results = search_requestable_songs(station_id, query)
    if not results:
        return {"error": f"No requestable song found matching '{query}'"}
    # Request the first matching song
    song = results[0]
    result = submit_request(station_id, song["request_id"])
    result["requested_song"] = f"{song['artist']} - {song['title']}"
    return result


# ==================== RADIO: MOUNT POINTS ====================

@tool(
    name="radio_list_mounts",
    description="List all mount points/stream URLs for a station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_list_mounts(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_mounts, get_station_id
    station_id = get_station_id(station)
    return list_mounts(station_id)


@tool(
    name="radio_create_mount",
    description="Create a new mount point/stream for a station.",
    parameters={
        "name": "string - mount point name (e.g., '/radio.mp3')",
        "display_name": "string (optional) - friendly display name",
        "format": "string (optional) - audio format: mp3, ogg, opus, aac, flac (default: mp3)",
        "bitrate": "int (optional) - stream bitrate in kbps (default: 128)",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_mount(name: str, display_name: str = None, format: str = "mp3",
                       bitrate: int = 128, station: str = None) -> dict:
    from integrations.azuracast.client import create_mount, get_station_id
    station_id = get_station_id(station)
    return create_mount(station_id, name, display_name, False, format, bitrate)


@tool(
    name="radio_delete_mount",
    description="Delete a mount point/stream.",
    parameters={
        "mount_id": "int - the mount point ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_mount(mount_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_mount, get_station_id
    station_id = get_station_id(station)
    return delete_mount(station_id, mount_id)


# ==================== RADIO: PLAYLISTS (EXPANDED) ====================

@tool(
    name="radio_create_playlist",
    description="Create a new playlist on a station.",
    parameters={
        "name": "string - playlist name",
        "type": "string (optional) - playlist type: default, once_per_x_songs, once_per_x_minutes, once_per_hour, once_per_day, advanced (default: default)",
        "weight": "int (optional) - playlist weight/priority 1-25 (default: 3)",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_playlist(name: str, type: str = "default", weight: int = 3,
                          station: str = None) -> dict:
    from integrations.azuracast.client import create_playlist, get_station_id
    station_id = get_station_id(station)
    return create_playlist(station_id, name, type, weight)


@tool(
    name="radio_delete_playlist",
    description="Delete a playlist from a station.",
    parameters={
        "playlist_id": "int - the playlist ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_playlist(playlist_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_playlist, get_station_id
    station_id = get_station_id(station)
    return delete_playlist(station_id, playlist_id)


@tool(
    name="radio_reshuffle_playlist",
    description="Reshuffle a playlist's playback order.",
    parameters={
        "playlist_id": "int - the playlist ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_reshuffle_playlist(playlist_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import reshuffle_playlist, get_station_id
    station_id = get_station_id(station)
    return reshuffle_playlist(station_id, playlist_id)


# ==================== RADIO: WEBHOOKS ====================

@tool(
    name="radio_list_webhooks",
    description="List all webhooks for a station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_list_webhooks(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_webhooks, get_station_id
    station_id = get_station_id(station)
    return list_webhooks(station_id)


@tool(
    name="radio_toggle_webhook",
    description="Toggle a webhook on/off.",
    parameters={
        "webhook_id": "int - the webhook ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_toggle_webhook(webhook_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import toggle_webhook, get_station_id
    station_id = get_station_id(station)
    return toggle_webhook(station_id, webhook_id)


# ==================== RADIO: SYSTEM ====================

@tool(
    name="radio_system_status",
    description="Get AzuraCast server system status (CPU, memory, disk).",
    parameters={},
)
def radio_system_status() -> dict:
    from integrations.azuracast.client import get_system_status
    return get_system_status()


@tool(
    name="radio_services_status",
    description="Get status of all AzuraCast services.",
    parameters={},
)
def radio_services_status() -> list[dict]:
    from integrations.azuracast.client import get_services_status
    return get_services_status()


# ==================== META ADS TOOLS ====================

@tool(
    name="meta_ads_account",
    description="Get Meta (Facebook/Instagram) Ads account info and status.",
    parameters={},
)
def meta_ads_account() -> dict:
    from integrations.meta_ads.client import get_ad_account_info
    return get_ad_account_info()


@tool(
    name="meta_ads_summary",
    description="Get a quick summary of Meta Ads performance (last 7 days).",
    parameters={},
)
def meta_ads_summary() -> dict:
    from integrations.meta_ads.client import get_meta_ads_summary
    return get_meta_ads_summary()


@tool(
    name="meta_ads_performance",
    description="Get ACCOUNT-LEVEL aggregate Meta Ads metrics (total across all campaigns). For campaign-specific metrics, use meta_ads_campaign_insights instead.",
    parameters={
        "period": "string (optional) - time period: today, yesterday, last_7d, last_14d, last_30d, this_month, last_month (default: last_7d)",
    },
)
def meta_ads_performance(period: str = "last_7d") -> dict:
    from integrations.meta_ads.client import get_account_insights
    return get_account_insights(period)


@tool(
    name="meta_ads_insights",
    description="Get performance insights for Meta Ads. Returns ad-level performance data: impressions, clicks, CTR, CPC, spend for each ad.",
    parameters={
        "period": "string (optional) - time period: last_7d, last_14d, last_30d (default: last_7d)",
    },
)
def meta_ads_insights(period: str = "last_7d", date_preset: str = None, time_range: dict = None) -> list[dict]:
    from integrations.meta_ads.client import get_ad_insights, _normalize_period
    # Handle various parameter names the LLM might use
    actual_period = period or date_preset or "last_7d"
    if time_range:
        actual_period = "last_7d"  # Default if LLM sends weird format
    return get_ad_insights(date_preset=_normalize_period(actual_period))


@tool(
    name="meta_ads_campaigns",
    description="List all Meta Ads campaigns with their status and budgets. Does NOT include performance metrics - use meta_ads_campaign_insights for clicks, spend, CPA, etc.",
    parameters={
        "status_filter": "string (optional) - filter by status: ACTIVE, PAUSED, DELETED, ARCHIVED, or omit for all",
    },
)
def meta_ads_campaigns(status_filter: str = None) -> list[dict]:
    from integrations.meta_ads.client import list_campaigns
    return list_campaigns(status_filter=status_filter)


@tool(
    name="meta_ads_campaign_insights",
    description="Get CAMPAIGN performance metrics: impressions, reach, clicks, CTR, CPC, spend, conversions, CPA, ROAS. USE THIS for campaign performance reports.",
    parameters={
        "campaign_id": "string (optional) - specific campaign ID, or omit for all campaigns",
        "period": "string (optional) - time period: last_7d, last_14d, last_30d, etc. (default: last_7d)",
    },
)
def meta_ads_campaign_insights(campaign_id: str = None, period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_campaign_insights
    return get_campaign_insights(campaign_id, period)


@tool(
    name="meta_ads_ad_sets",
    description="List ad sets (audiences/targeting groups) in Meta Ads.",
    parameters={
        "campaign_id": "string (optional) - filter by campaign ID",
    },
)
def meta_ads_ad_sets(campaign_id: str = None) -> list[dict]:
    from integrations.meta_ads.client import list_ad_sets
    return list_ad_sets(campaign_id)


@tool(
    name="meta_ads_ad_set_insights",
    description="Get AD SET performance: impressions, clicks, CTR, CPC, spend, conversions. USE THIS to compare which ad set is performing best.",
    parameters={
        "campaign_id": "string (optional) - filter by campaign ID",
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_ad_set_insights(campaign_id: str = None, period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_ad_set_insights
    return get_ad_set_insights(campaign_id, period)


@tool(
    name="meta_ads_ads",
    description="List individual ADS (creatives) within an ad set. Use this to count ads or see ad names/status in an ad set.",
    parameters={
        "ad_set_id": "string (optional) - ad set ID to filter ads, or omit for all ads in account",
    },
)
def meta_ads_ads(ad_set_id: str = None) -> list[dict]:
    from integrations.meta_ads.client import list_ads
    return list_ads(ad_set_id)


@tool(
    name="meta_ads_ad_insights",
    description="Get performance metrics for individual Meta ads.",
    parameters={
        "ad_set_id": "string (optional) - specific ad set ID",
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_ad_insights(ad_set_id: str = None, period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_ad_insights
    return get_ad_insights(ad_set_id, period)


@tool(
    name="meta_ads_audience",
    description="Get audience demographic breakdown (age, gender, platform, device).",
    parameters={
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_audience(period: str = "last_7d") -> dict:
    from integrations.meta_ads.client import get_audience_insights
    return get_audience_insights(period)


@tool(
    name="meta_ads_placements",
    description="Get performance by ad placement (feed, stories, reels, etc.).",
    parameters={
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_placements(period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_placement_insights
    return get_placement_insights(period)


@tool(
    name="meta_ads_issues",
    description="Check for Meta Ads delivery issues, disapproved ads, or policy violations.",
    parameters={},
)
def meta_ads_issues() -> list[dict]:
    from integrations.meta_ads.client import get_delivery_issues
    return get_delivery_issues()


@tool(
    name="meta_ads_daily_spend",
    description="Get daily spend breakdown for Meta Ads.",
    parameters={
        "days": "int (optional) - number of days to show (default: 7)",
    },
)
def meta_ads_daily_spend(days: int = 7) -> list[dict]:
    from integrations.meta_ads.client import get_spend_by_day
    return get_spend_by_day(days)


# ==================== Meta Ads Write Operations ====================

@tool(
    name="meta_ads_pause_ad",
    description="Pause a specific ad. Use this to stop an underperforming ad.",
    parameters={
        "ad_id": {"type": "string", "description": "The ad ID to pause", "required": True},
    },
)
def meta_ads_pause_ad(ad_id: str) -> dict:
    from integrations.meta_ads.client import pause_ad
    return pause_ad(ad_id)


@tool(
    name="meta_ads_enable_ad",
    description="Enable (unpause) a specific ad.",
    parameters={
        "ad_id": {"type": "string", "description": "The ad ID to enable", "required": True},
    },
)
def meta_ads_enable_ad(ad_id: str) -> dict:
    from integrations.meta_ads.client import enable_ad
    return enable_ad(ad_id)


@tool(
    name="meta_ads_pause_ad_set",
    description="Pause a specific ad set.",
    parameters={
        "ad_set_id": {"type": "string", "description": "The ad set ID to pause", "required": True},
    },
)
def meta_ads_pause_ad_set(ad_set_id: str) -> dict:
    from integrations.meta_ads.client import pause_ad_set
    return pause_ad_set(ad_set_id)


@tool(
    name="meta_ads_enable_ad_set",
    description="Enable (unpause) a specific ad set.",
    parameters={
        "ad_set_id": {"type": "string", "description": "The ad set ID to enable", "required": True},
    },
)
def meta_ads_enable_ad_set(ad_set_id: str) -> dict:
    from integrations.meta_ads.client import enable_ad_set
    return enable_ad_set(ad_set_id)


@tool(
    name="meta_ads_pause_campaign",
    description="Pause a specific campaign.",
    parameters={
        "campaign_id": {"type": "string", "description": "The campaign ID to pause", "required": True},
    },
)
def meta_ads_pause_campaign(campaign_id: str) -> dict:
    from integrations.meta_ads.client import pause_campaign
    return pause_campaign(campaign_id)


@tool(
    name="meta_ads_enable_campaign",
    description="Enable (unpause) a specific campaign.",
    parameters={
        "campaign_id": {"type": "string", "description": "The campaign ID to enable", "required": True},
    },
)
def meta_ads_enable_campaign(campaign_id: str) -> dict:
    from integrations.meta_ads.client import enable_campaign
    return enable_campaign(campaign_id)


@tool(
    name="meta_ads_update_ad_set_budget",
    description="Update an ad set's daily or lifetime budget. Amounts in dollars.",
    parameters={
        "ad_set_id": {"type": "string", "description": "The ad set ID", "required": True},
        "daily_budget": {"type": "number", "description": "New daily budget in dollars (e.g., 25.00)"},
        "lifetime_budget": {"type": "number", "description": "New lifetime budget in dollars"},
    },
)
def meta_ads_update_ad_set_budget(ad_set_id: str, daily_budget: float = None, lifetime_budget: float = None) -> dict:
    from integrations.meta_ads.client import update_ad_set_budget
    return update_ad_set_budget(ad_set_id, daily_budget, lifetime_budget)


@tool(
    name="meta_ads_update_campaign_budget",
    description="Update a campaign's daily or lifetime budget. Amounts in dollars.",
    parameters={
        "campaign_id": {"type": "string", "description": "The campaign ID", "required": True},
        "daily_budget": {"type": "number", "description": "New daily budget in dollars (e.g., 50.00)"},
        "lifetime_budget": {"type": "number", "description": "New lifetime budget in dollars"},
    },
)
def meta_ads_update_campaign_budget(campaign_id: str, daily_budget: float = None, lifetime_budget: float = None) -> dict:
    from integrations.meta_ads.client import update_campaign_budget
    return update_campaign_budget(campaign_id, daily_budget, lifetime_budget)


# ==================== GOOGLE ANALYTICS ====================

@tool(
    name="ga_list_properties",
    description="List all configured Google Analytics properties.",
    parameters={},
)
def ga_list_properties() -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.list_properties()


@tool(
    name="ga_traffic_summary",
    description="Get traffic summary for a Google Analytics property (users, sessions, page views, bounce rate).",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID (e.g., 'LensSniper', 'LoovaCast', 'Rod Wave')", "required": True},
        "period": {"type": "string", "description": "Time period: today, yesterday, last_7_days, last_30_days, last_90_days (default: last_7_days)"},
    },
)
def ga_traffic_summary(property_name: str, period: str = "last_7_days") -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_traffic_summary(property_name, period)


@tool(
    name="ga_realtime",
    description="Get real-time active users for a property right now.",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID", "required": True},
    },
)
def ga_realtime(property_name: str) -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_realtime(property_name)


@tool(
    name="ga_traffic_sources",
    description="Get traffic sources breakdown (where visitors come from).",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID", "required": True},
        "period": {"type": "string", "description": "Time period (default: last_7_days)"},
        "limit": {"type": "integer", "description": "Max number of sources (default: 10)"},
    },
)
def ga_traffic_sources(property_name: str, period: str = "last_7_days", limit: int = 10) -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_traffic_sources(property_name, period, limit)


@tool(
    name="ga_top_pages",
    description="Get top pages by views.",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID", "required": True},
        "period": {"type": "string", "description": "Time period (default: last_7_days)"},
        "limit": {"type": "integer", "description": "Max number of pages (default: 10)"},
    },
)
def ga_top_pages(property_name: str, period: str = "last_7_days", limit: int = 10) -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_top_pages(property_name, period, limit)


@tool(
    name="ga_devices",
    description="Get device breakdown (mobile, desktop, tablet).",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID", "required": True},
        "period": {"type": "string", "description": "Time period (default: last_7_days)"},
    },
)
def ga_devices(property_name: str, period: str = "last_7_days") -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_devices(property_name, period)


@tool(
    name="ga_countries",
    description="Get geographic breakdown by country.",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID", "required": True},
        "period": {"type": "string", "description": "Time period (default: last_7_days)"},
        "limit": {"type": "integer", "description": "Max number of countries (default: 10)"},
    },
)
def ga_countries(property_name: str, period: str = "last_7_days", limit: int = 10) -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_countries(property_name, period, limit)


@tool(
    name="ga_daily_traffic",
    description="Get daily traffic breakdown over time.",
    parameters={
        "property_name": {"type": "string", "description": "Property name or ID", "required": True},
        "period": {"type": "string", "description": "Time period (default: last_30_days)"},
    },
)
def ga_daily_traffic(property_name: str, period: str = "last_30_days") -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_daily_traffic(property_name, period)


@tool(
    name="ga_all_properties",
    description="Get traffic summary for ALL configured properties at once.",
    parameters={
        "period": {"type": "string", "description": "Time period (default: last_7_days)"},
    },
)
def ga_all_properties(period: str = "last_7_days") -> dict:
    from integrations.google_analytics.client import ga_client
    return ga_client.get_all_properties_summary(period)


# ==================== Agent Orchestration Tools ====================

@tool(
    name="spawn_agent",
    description="Spawn a specialized agent to work on a task. Use this to delegate work to specialist agents (coder, researcher, analyst, writer, planner). Returns a task ID to track progress.",
    parameters={
        "goal": {"type": "string", "description": "What the agent should accomplish"},
        "agent_type": {"type": "string", "description": "Type: coder, researcher, analyst, writer, planner, executor, general"},
        "context": {"type": "string", "description": "Additional context to help the agent (optional)"},
    },
)
async def spawn_agent_tool(goal: str, agent_type: str = "general", context: str = "") -> dict:
    from core.orchestration.agents import get_agent_pool, AgentType

    try:
        atype = AgentType(agent_type)
    except ValueError:
        atype = AgentType.GENERAL

    pool = get_agent_pool()
    task_id = await pool.spawn_agent(goal, atype, context)
    return {
        "success": True,
        "task_id": task_id,
        "agent_type": atype.value,
        "message": f"Spawned {atype.value} agent. Task ID: {task_id}",
    }


@tool(
    name="check_agent_task",
    description="Check the status of a spawned agent task. Use this to see if an agent has completed its work.",
    parameters={
        "task_id": {"type": "string", "description": "The task ID returned from spawn_agent"},
        "wait": {"type": "boolean", "description": "Wait for completion (default: false)"},
    },
)
async def check_agent_task(task_id: str, wait: bool = False) -> dict:
    from core.orchestration.agents import get_agent_pool

    pool = get_agent_pool()
    result = await pool.get_task_result(task_id, wait=wait, timeout=30)
    if result:
        return {"success": True, "task": result}
    return {"success": False, "error": "Task not found"}


@tool(
    name="list_agent_tasks",
    description="List all agent tasks and their status. Use to see what agents are working on.",
    parameters={
        "status": {"type": "string", "description": "Filter by status: pending, running, completed, failed (optional)"},
    },
)
async def list_agent_tasks(status: str = None) -> dict:
    from core.orchestration.agents import get_agent_pool, AgentStatus

    pool = get_agent_pool()
    status_filter = AgentStatus(status) if status else None
    tasks = pool.list_tasks(status_filter)
    return {"success": True, "tasks": tasks, "count": len(tasks)}


@tool(
    name="cancel_agent_task",
    description="Cancel a pending or running agent task.",
    parameters={
        "task_id": {"type": "string", "description": "The task ID to cancel"},
    },
)
async def cancel_agent_task(task_id: str) -> dict:
    from core.orchestration.agents import get_agent_pool

    pool = get_agent_pool()
    cancelled = await pool.cancel_task(task_id)
    return {
        "success": cancelled,
        "message": "Task cancelled" if cancelled else "Task could not be cancelled",
    }


# ==================== Daily Briefing ====================

@tool(
    name="daily_briefing",
    description="Generate a personalized daily briefing with calendar events, emails, CRM tasks, server status, revenue, and weather. Use this when the user asks for their morning briefing or wants a summary of their day.",
    parameters={
        "include_weather": {"type": "boolean", "description": "Include weather forecast (default: true)"},
        "weather_location": {"type": "string", "description": "Location for weather (default: Atlanta, GA)"},
    },
)
async def daily_briefing_tool(
    include_weather: bool = True,
    weather_location: str = "Atlanta, GA",
) -> dict:
    from core.briefing.daily import generate_briefing

    briefing = await generate_briefing(
        include_weather=include_weather,
        weather_location=weather_location,
    )
    return briefing.to_dict()


@tool(
    name="quick_briefing",
    description="Generate a quick text briefing suitable for voice output. Use when user wants a spoken summary of their day.",
    parameters={},
)
async def quick_briefing_tool() -> str:
    from core.briefing.daily import generate_quick_briefing
    return await generate_quick_briefing()


# ==================== TWILIO SMS/VOICE TOOLS ====================

@tool(
    name="send_sms",
    description="Send an SMS text message to a phone number. Use this to text someone on Mike's behalf.",
    parameters={
        "to": {"type": "string", "description": "Phone number to send to (e.g., +14045551234 or 404-555-1234)", "required": True},
        "message": {"type": "string", "description": "The text message to send", "required": True},
    },
)
def send_sms_tool(to: str, message: str) -> dict:
    from integrations.twilio.client import send_sms
    return send_sms(to, message)


@tool(
    name="make_call",
    description="Make an outbound phone call and speak a message. Use this to call someone on Mike's behalf.",
    parameters={
        "to": {"type": "string", "description": "Phone number to call (e.g., +14045551234)", "required": True},
        "message": {"type": "string", "description": "The message to speak when they answer", "required": True},
    },
)
def make_call_tool(to: str, message: str) -> dict:
    from integrations.twilio.client import make_call
    return make_call(to, message)


@tool(
    name="get_sms_history",
    description="Get recent SMS messages sent or received on the Twilio number.",
    parameters={
        "limit": {"type": "integer", "description": "Maximum number of messages to return (default 10)"},
    },
)
def get_sms_history_tool(limit: int = 10) -> dict:
    from integrations.twilio.client import get_messages
    return get_messages(limit=limit)


# ============ WordPress Multi-Site Management ============

@tool(
    name="wp_list_sites",
    description="List all configured WordPress sites Alfred can manage.",
    parameters={},
)
def wp_list_sites() -> list[dict]:
    from integrations.wordpress.client import list_sites
    return list_sites()


@tool(
    name="wp_add_site",
    description="Add a new WordPress site to Alfred's roster. Requires the site URL, username, and application password. The user must first create an Application Password in WordPress (Users > Profile > Application Passwords).",
    parameters={
        "name": "string - short name for the site (lowercase, no spaces, e.g. 'mysite')",
        "url": "string - full URL of the WordPress site (e.g. 'https://example.com')",
        "username": "string - WordPress username with admin access",
        "password": "string - WordPress Application Password (NOT the regular login password)",
    },
)
def wp_add_site(name: str, url: str, username: str, password: str) -> dict:
    """Add a new WordPress site to Alfred's configuration."""
    import os
    import re

    env_path = "/home/aialfred/alfred/config/.env"

    # Validate inputs
    name = name.lower().strip()
    if not re.match(r'^[a-z0-9_]+$', name):
        return {"success": False, "error": "Site name must be lowercase letters, numbers, and underscores only"}

    url = url.rstrip("/")
    if not url.startswith("http"):
        url = f"https://{url}"

    # Read current .env file
    try:
        with open(env_path, "r") as f:
            content = f.read()
    except Exception as e:
        return {"success": False, "error": f"Could not read .env file: {e}"}

    # Check if site already exists
    name_upper = name.upper()
    if f"WP_SITE_{name_upper}_URL" in content:
        return {"success": False, "error": f"Site '{name}' already exists in configuration"}

    # Update WP_SITES list
    wp_sites_match = re.search(r'^WP_SITES=(.*)$', content, re.MULTILINE)
    if wp_sites_match:
        current_sites = wp_sites_match.group(1)
        if current_sites:
            new_sites = f"{current_sites},{name}"
        else:
            new_sites = name
        content = content.replace(f"WP_SITES={current_sites}", f"WP_SITES={new_sites}")
    else:
        # WP_SITES doesn't exist, add it
        content += f"\n# WordPress Sites\nWP_SITES={name}\n"

    # Add the new site's credentials
    new_site_config = f"""
# {name.title()} WordPress Site
WP_SITE_{name_upper}_URL={url}
WP_SITE_{name_upper}_USER={username}
WP_SITE_{name_upper}_PASS={password}
"""
    content += new_site_config

    # Write updated .env file
    try:
        with open(env_path, "w") as f:
            f.write(content)
    except Exception as e:
        return {"success": False, "error": f"Could not write .env file: {e}"}

    # Reload the WordPress client configuration
    try:
        from integrations.wordpress import client as wp_client
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        wp_client._load_sites()  # Reload sites from environment
    except Exception as e:
        return {"success": True, "warning": f"Site added but reload failed: {e}. Restart Alfred to apply."}

    # Test the connection
    try:
        from integrations.wordpress.client import test_connection
        test_result = test_connection(name)
        if test_result.get("success"):
            return {
                "success": True,
                "message": f"WordPress site '{name}' added and connected successfully!",
                "site": name,
                "url": url,
                "connected_as": test_result.get("connected_as", username),
            }
        else:
            return {
                "success": True,
                "warning": f"Site added but connection test failed: {test_result.get('error')}. Please verify credentials.",
                "site": name,
                "url": url,
            }
    except Exception as e:
        return {
            "success": True,
            "warning": f"Site added but could not test connection: {e}",
            "site": name,
            "url": url,
        }


@tool(
    name="wp_remove_site",
    description="Remove a WordPress site from Alfred's roster. This only removes the configuration, not the actual WordPress site.",
    parameters={
        "name": "string - the site name to remove (e.g. 'mysite')",
    },
)
def wp_remove_site(name: str) -> dict:
    """Remove a WordPress site from Alfred's configuration."""
    import re

    env_path = "/home/aialfred/alfred/config/.env"
    name = name.lower().strip()
    name_upper = name.upper()

    # Read current .env file
    try:
        with open(env_path, "r") as f:
            content = f.read()
    except Exception as e:
        return {"success": False, "error": f"Could not read .env file: {e}"}

    # Check if site exists
    if f"WP_SITE_{name_upper}_URL" not in content:
        return {"success": False, "error": f"Site '{name}' not found in configuration"}

    # Remove from WP_SITES list
    wp_sites_match = re.search(r'^WP_SITES=(.*)$', content, re.MULTILINE)
    if wp_sites_match:
        current_sites = wp_sites_match.group(1)
        sites_list = [s.strip() for s in current_sites.split(",")]
        sites_list = [s for s in sites_list if s.lower() != name]
        new_sites = ",".join(sites_list)
        content = content.replace(f"WP_SITES={current_sites}", f"WP_SITES={new_sites}")

    # Remove the site's credentials (URL, USER, PASS lines and any comment above)
    content = re.sub(rf'\n# {name.title()} WordPress Site\n', '\n', content, flags=re.IGNORECASE)
    content = re.sub(rf'^WP_SITE_{name_upper}_URL=.*\n', '', content, flags=re.MULTILINE)
    content = re.sub(rf'^WP_SITE_{name_upper}_USER=.*\n', '', content, flags=re.MULTILINE)
    content = re.sub(rf'^WP_SITE_{name_upper}_PASS=.*\n', '', content, flags=re.MULTILINE)

    # Write updated .env file
    try:
        with open(env_path, "w") as f:
            f.write(content)
    except Exception as e:
        return {"success": False, "error": f"Could not write .env file: {e}"}

    # Reload the WordPress client configuration
    try:
        from integrations.wordpress import client as wp_client
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        wp_client._load_sites()
    except Exception as e:
        return {"success": True, "warning": f"Site removed but reload failed: {e}. Restart Alfred to apply."}

    return {
        "success": True,
        "message": f"WordPress site '{name}' removed from Alfred's roster.",
        "site": name,
    }


@tool(
    name="wp_test_connection",
    description="Test connection to a WordPress site.",
    parameters={"site": "string - site name (groundrush, loovacast, rucktalk, nightlife, lumabot, myhandscarwash)"},
)
def wp_test_connection(site: str) -> dict:
    from integrations.wordpress.client import test_connection
    return test_connection(site)


@tool(
    name="wp_get_posts",
    description="Get posts from a WordPress site.",
    parameters={
        "site": "string - site name",
        "per_page": "int - posts per page (default 10)",
        "status": "string - post status: publish, draft, any (default any)",
        "search": "string - search term (optional)",
    },
)
def wp_get_posts(site: str, per_page: int = 10, status: str = "any", search: str = None) -> list[dict]:
    from integrations.wordpress.client import get_posts
    return get_posts(site, per_page=per_page, status=status, search=search)


@tool(
    name="wp_get_post",
    description="Get a single post by ID from a WordPress site.",
    parameters={"site": "string - site name", "post_id": "int - post ID"},
)
def wp_get_post(site: str, post_id: int) -> dict:
    from integrations.wordpress.client import get_post
    return get_post(site, post_id)


@tool(
    name="wp_create_post",
    description="Create a new post on a WordPress site.",
    parameters={
        "site": "string - site name",
        "title": "string - post title",
        "content": "string - post content (HTML)",
        "status": "string - publish or draft (default draft)",
    },
)
def wp_create_post(site: str, title: str, content: str, status: str = "draft") -> dict:
    from integrations.wordpress.client import create_post
    return create_post(site, title, content, status)


@tool(
    name="wp_update_post",
    description="Update an existing post on a WordPress site.",
    parameters={
        "site": "string - site name",
        "post_id": "int - post ID",
        "title": "string - new title (optional)",
        "content": "string - new content (optional)",
        "status": "string - new status (optional)",
    },
)
def wp_update_post(site: str, post_id: int, title: str = None, content: str = None, status: str = None) -> dict:
    from integrations.wordpress.client import update_post
    return update_post(site, post_id, title, content, status)


@tool(
    name="wp_get_pages",
    description="Get pages from a WordPress site.",
    parameters={
        "site": "string - site name",
        "per_page": "int - pages per page (default 10)",
        "search": "string - search term (optional)",
    },
)
def wp_get_pages(site: str, per_page: int = 10, search: str = None) -> list[dict]:
    from integrations.wordpress.client import get_pages
    return get_pages(site, per_page=per_page, search=search)


@tool(
    name="wp_create_page",
    description="Create a new page on a WordPress site.",
    parameters={
        "site": "string - site name",
        "title": "string - page title",
        "content": "string - page content (HTML)",
        "status": "string - publish or draft (default draft)",
    },
)
def wp_create_page(site: str, title: str, content: str, status: str = "draft") -> dict:
    from integrations.wordpress.client import create_page
    return create_page(site, title, content, status)


@tool(
    name="wp_get_seo",
    description="Get RankMath SEO metadata for a post or page.",
    parameters={
        "site": "string - site name",
        "post_id": "int - post or page ID",
        "post_type": "string - 'post' or 'page' (default post)",
    },
)
def wp_get_seo(site: str, post_id: int, post_type: str = "post") -> dict:
    from integrations.wordpress.client import get_seo_meta
    return get_seo_meta(site, post_id, post_type)


@tool(
    name="wp_update_seo",
    description="Update RankMath SEO metadata for a post or page.",
    parameters={
        "site": "string - site name",
        "post_id": "int - post or page ID",
        "post_type": "string - 'post' or 'page' (default post)",
        "seo_title": "string - SEO title (optional)",
        "seo_description": "string - meta description (optional)",
        "focus_keyword": "string - focus keyword (optional)",
    },
)
def wp_update_seo(site: str, post_id: int, post_type: str = "post", seo_title: str = None, seo_description: str = None, focus_keyword: str = None) -> dict:
    from integrations.wordpress.client import update_seo_meta
    return update_seo_meta(site, post_id, post_type, seo_title, seo_description, focus_keyword)


@tool(
    name="wp_get_plugins",
    description="Get list of plugins installed on a WordPress site.",
    parameters={"site": "string - site name"},
)
def wp_get_plugins(site: str) -> list[dict]:
    from integrations.wordpress.client import get_plugins
    return get_plugins(site)


@tool(
    name="wp_activate_plugin",
    description="Activate a plugin on a WordPress site.",
    parameters={"site": "string - site name", "plugin_slug": "string - plugin slug (e.g., 'akismet/akismet.php')"},
)
def wp_activate_plugin(site: str, plugin_slug: str) -> dict:
    from integrations.wordpress.client import activate_plugin
    return activate_plugin(site, plugin_slug)


@tool(
    name="wp_deactivate_plugin",
    description="Deactivate a plugin on a WordPress site.",
    parameters={"site": "string - site name", "plugin_slug": "string - plugin slug"},
)
def wp_deactivate_plugin(site: str, plugin_slug: str) -> dict:
    from integrations.wordpress.client import deactivate_plugin
    return deactivate_plugin(site, plugin_slug)


@tool(
    name="wp_get_themes",
    description="Get list of themes installed on a WordPress site.",
    parameters={"site": "string - site name"},
)
def wp_get_themes(site: str) -> list[dict]:
    from integrations.wordpress.client import get_themes
    return get_themes(site)


@tool(
    name="wp_upload_media",
    description="Upload an image or file to a WordPress site's media library.",
    parameters={
        "site": "string - site name",
        "file_path": "string - local path to the file",
        "title": "string - media title (optional)",
        "alt_text": "string - alt text for images (optional)",
    },
)
def wp_upload_media(site: str, file_path: str, title: str = None, alt_text: str = None) -> dict:
    from integrations.wordpress.client import upload_media
    return upload_media(site, file_path, title, alt_text)


@tool(
    name="wp_upload_media_base64",
    description="Upload an image or file to WordPress from base64-encoded data. Use this when user pastes/uploads an image directly in chat.",
    parameters={
        "site": "string - site name",
        "base64_data": "string - base64-encoded file content (can include data URI prefix)",
        "filename": "string - desired filename with extension (e.g., 'banner.jpg')",
        "title": "string - media title (optional)",
        "alt_text": "string - alt text for images (optional)",
    },
)
def wp_upload_media_base64(site: str, base64_data: str, filename: str, title: str = None, alt_text: str = None) -> dict:
    from integrations.wordpress.client import upload_media_base64
    return upload_media_base64(site, base64_data, filename, title, alt_text)


@tool(
    name="wp_get_media",
    description="Get details of a specific media item from WordPress library.",
    parameters={
        "site": "string - site name",
        "media_id": "integer - media ID",
    },
)
def wp_get_media(site: str, media_id: int) -> dict:
    from integrations.wordpress.client import get_media_item
    return get_media_item(site, media_id)


@tool(
    name="wp_delete_media",
    description="Delete a media item from WordPress library.",
    parameters={
        "site": "string - site name",
        "media_id": "integer - media ID to delete",
        "force": "boolean - permanently delete (true) or trash (false), default true",
    },
)
def wp_delete_media(site: str, media_id: int, force: bool = True) -> dict:
    from integrations.wordpress.client import delete_media
    return delete_media(site, media_id, force)


@tool(
    name="wp_site_health",
    description="Get site health overview for a WordPress site.",
    parameters={"site": "string - site name"},
)
def wp_site_health(site: str) -> dict:
    from integrations.wordpress.client import get_site_health
    return get_site_health(site)


@tool(
    name="wp_test_all_sites",
    description="Test connections to all configured WordPress sites.",
    parameters={},
)
def wp_test_all_sites() -> list[dict]:
    from integrations.wordpress.client import test_all_connections
    return test_all_connections()


# ============ Tracking & Analytics ============

@tool(
    name="wp_analyze_pixel",
    description="Analyze Meta Pixel setup on a WordPress site. Checks if pixel is installed, what events are configured, and identifies issues.",
    parameters={
        "site": "string - site identifier (e.g., 'myhandscarwash', 'groundrush')",
    },
)
def wp_analyze_pixel(site: str) -> dict:
    from integrations.wordpress.client import get_meta_pixel_events
    return get_meta_pixel_events(site)


@tool(
    name="wp_fix_pixel_tracking",
    description="Add Meta Pixel conversion event tracking to a WordPress site. Adds JavaScript to track form submissions as Lead events and phone clicks as Contact events.",
    parameters={
        "site": "string - site identifier",
        "track_forms": "bool - track form submissions as Lead (default True)",
        "track_phone_clicks": "bool - track tel: clicks as Contact (default True)",
        "track_buttons": "bool - track button clicks (default False)",
    },
)
def wp_fix_pixel_tracking(
    site: str,
    track_forms: bool = True,
    track_phone_clicks: bool = True,
    track_buttons: bool = False,
) -> dict:
    from integrations.wordpress.client import add_meta_pixel_events
    return add_meta_pixel_events(site, None, track_forms, track_phone_clicks, track_buttons)


@tool(
    name="wp_add_tracking_code",
    description="Add custom tracking code (JavaScript) to a WordPress site header or footer.",
    parameters={
        "site": "string - site identifier",
        "name": "string - name for this tracking code",
        "code": "string - JavaScript code to add",
        "location": "string - 'header' or 'footer' (default 'header')",
    },
)
def wp_add_tracking_code(site: str, name: str, code: str, location: str = "header") -> dict:
    from integrations.wordpress.client import add_tracking_script
    return add_tracking_script(site, name, code, location)


@tool(
    name="wp_get_snippets",
    description="List all code snippets on a WordPress site (requires WPCode plugin).",
    parameters={
        "site": "string - site identifier",
    },
)
def wp_get_snippets(site: str) -> list[dict]:
    from integrations.wordpress.client import get_wpcode_snippets
    return get_wpcode_snippets(site)


# ============ Elementor Page Design ============

@tool(
    name="wp_design_hero_section",
    description="Design a hero section for a WordPress page. Returns Elementor JSON that can be used with wp_create_elementor_page.",
    parameters={
        "headline": "string - main headline text",
        "subheadline": "string - supporting text (optional)",
        "cta_text": "string - button text (default 'Get Started')",
        "cta_link": "string - button link (default '#')",
        "background_color": "string - hex color (optional)",
        "background_image": "string - image URL for background (optional)",
        "layout": "string - 'centered', 'left', or 'right' (default centered)",
    },
)
def wp_design_hero_section(
    headline: str,
    subheadline: str = "",
    cta_text: str = "Get Started",
    cta_link: str = "#",
    background_color: str = None,
    background_image: str = None,
    layout: str = "centered",
) -> dict:
    from integrations.wordpress.elementor import section_hero, background_color as bg_color, background_image as bg_image

    bg = None
    if background_image:
        bg = bg_image(background_image, overlay_color="#000000", overlay_opacity=0.5)
    elif background_color:
        bg = bg_color(background_color)

    section = section_hero(
        headline=headline,
        subheadline=subheadline,
        cta_text=cta_text,
        cta_link=cta_link,
        background=bg,
        layout=layout,
    )
    return {"success": True, "section": section, "type": "hero"}


@tool(
    name="wp_design_features_section",
    description="Design a features grid section. Returns Elementor JSON.",
    parameters={
        "features": "list - features [{'icon': 'fa fa-check', 'title': 'Feature', 'description': '...'}]",
        "headline": "string - section headline (optional)",
        "columns": "int - 2, 3, or 4 columns (default 3)",
    },
)
def wp_design_features_section(
    features: list,
    headline: str = None,
    columns: int = 3,
) -> dict:
    from integrations.wordpress.elementor import section_features
    section = section_features(features=features, headline=headline, columns=columns)
    return {"success": True, "section": section, "type": "features"}


@tool(
    name="wp_design_cta_section",
    description="Design a call-to-action section. Returns Elementor JSON.",
    parameters={
        "headline": "string - CTA headline",
        "description": "string - supporting text (optional)",
        "button_text": "string - button text",
        "button_link": "string - button link",
        "background_color": "string - hex color (optional)",
    },
)
def wp_design_cta_section(
    headline: str,
    description: str = "",
    button_text: str = "Contact Us",
    button_link: str = "#",
    background_color: str = None,
) -> dict:
    from integrations.wordpress.elementor import section_cta, background_color as bg_color
    bg = bg_color(background_color) if background_color else None
    section = section_cta(
        headline=headline,
        description=description,
        button_text=button_text,
        button_link=button_link,
        background=bg,
    )
    return {"success": True, "section": section, "type": "cta"}


@tool(
    name="wp_design_testimonials_section",
    description="Design a testimonials section. Returns Elementor JSON.",
    parameters={
        "testimonials": "list - [{'content': 'Quote...', 'name': 'John Doe', 'title': 'CEO', 'image_url': '...'}]",
        "headline": "string - section headline (optional)",
    },
)
def wp_design_testimonials_section(
    testimonials: list,
    headline: str = None,
) -> dict:
    from integrations.wordpress.elementor import section_testimonials
    section = section_testimonials(testimonials=testimonials, headline=headline)
    return {"success": True, "section": section, "type": "testimonials"}


@tool(
    name="wp_design_pricing_section",
    description="Design a pricing table section. Returns Elementor JSON.",
    parameters={
        "plans": "list - [{'name': 'Basic', 'price': '$9/mo', 'features': ['Feature 1', ...], 'cta_text': 'Buy', 'cta_link': '#'}]",
        "headline": "string - section headline (optional)",
        "highlighted_plan": "int - index of featured plan (default 1, middle)",
    },
)
def wp_design_pricing_section(
    plans: list,
    headline: str = None,
    highlighted_plan: int = 1,
) -> dict:
    from integrations.wordpress.elementor import section_pricing
    section = section_pricing(plans=plans, headline=headline, highlighted_plan=highlighted_plan)
    return {"success": True, "section": section, "type": "pricing"}


@tool(
    name="wp_design_stats_section",
    description="Design a statistics/numbers section. Returns Elementor JSON.",
    parameters={
        "stats": "list - [{'number': 100, 'label': 'Happy Clients', 'suffix': '+'}]",
        "background_color": "string - hex color (optional)",
    },
)
def wp_design_stats_section(
    stats: list,
    background_color: str = None,
) -> dict:
    from integrations.wordpress.elementor import section_stats, background_color as bg_color
    bg = bg_color(background_color) if background_color else None
    section = section_stats(stats=stats, background=bg)
    return {"success": True, "section": section, "type": "stats"}


@tool(
    name="wp_design_faq_section",
    description="Design an FAQ accordion section. Returns Elementor JSON.",
    parameters={
        "questions": "list - [{'question': 'What is...?', 'answer': 'It is...'}]",
        "headline": "string - section headline (default 'Frequently Asked Questions')",
    },
)
def wp_design_faq_section(
    questions: list,
    headline: str = "Frequently Asked Questions",
) -> dict:
    from integrations.wordpress.elementor import section_faq
    section = section_faq(questions=questions, headline=headline)
    return {"success": True, "section": section, "type": "faq"}


@tool(
    name="wp_design_contact_section",
    description="Design a contact section with form and info. Returns Elementor JSON.",
    parameters={
        "headline": "string - section headline (default 'Contact Us')",
        "address": "string - business address (optional)",
        "phone": "string - phone number (optional)",
        "email": "string - email address (optional)",
        "show_map": "bool - show Google Map (default True)",
    },
)
def wp_design_contact_section(
    headline: str = "Contact Us",
    address: str = None,
    phone: str = None,
    email: str = None,
    show_map: bool = True,
) -> dict:
    from integrations.wordpress.elementor import section_contact
    section = section_contact(
        headline=headline,
        address=address,
        phone=phone,
        email=email,
        show_map=show_map,
    )
    return {"success": True, "section": section, "type": "contact"}


@tool(
    name="wp_design_team_section",
    description="Design a team members section. Returns Elementor JSON.",
    parameters={
        "members": "list - [{'name': 'John', 'title': 'CEO', 'image': 'url', 'bio': '...'}]",
        "headline": "string - section headline (optional)",
        "columns": "int - 2, 3, or 4 columns (default 4)",
    },
)
def wp_design_team_section(
    members: list,
    headline: str = None,
    columns: int = 4,
) -> dict:
    from integrations.wordpress.elementor import section_team
    section = section_team(members=members, headline=headline, columns=columns)
    return {"success": True, "section": section, "type": "team"}


@tool(
    name="wp_create_elementor_page",
    description="Create a new WordPress page with Elementor design. Combine sections from wp_design_* tools.",
    parameters={
        "site": "string - site name (groundrush, loovacast, etc.)",
        "title": "string - page title",
        "sections": "list - list of Elementor section objects from wp_design_* tools",
        "status": "string - 'draft' or 'publish' (default draft)",
    },
)
def wp_create_elementor_page(site: str, title: str, sections: list, status: str = "draft") -> dict:
    from integrations.wordpress.client import create_elementor_page
    return create_elementor_page(site, title, sections, status)


@tool(
    name="wp_update_elementor_page",
    description="Update an existing page with new Elementor design.",
    parameters={
        "site": "string - site name",
        "page_id": "int - page ID to update",
        "sections": "list - list of Elementor section objects",
    },
)
def wp_update_elementor_page(site: str, page_id: int, sections: list) -> dict:
    from integrations.wordpress.client import save_elementor_data
    return save_elementor_data(site, page_id, sections, post_type="page")


@tool(
    name="wp_get_elementor_data",
    description="Get Elementor design data from an existing page.",
    parameters={
        "site": "string - site name",
        "page_id": "int - page ID",
    },
)
def wp_get_elementor_data(site: str, page_id: int) -> dict:
    from integrations.wordpress.client import get_elementor_data
    return get_elementor_data(site, page_id)


@tool(
    name="wp_design_full_landing_page",
    description="Design a complete landing page with hero, features, testimonials, CTA, and contact sections.",
    parameters={
        "headline": "string - main hero headline",
        "subheadline": "string - hero subheadline",
        "features": "list - [{'icon': 'fa fa-check', 'title': '...', 'description': '...'}]",
        "testimonials": "list - [{'content': '...', 'name': '...', 'title': '...'}] (optional)",
        "cta_headline": "string - CTA section headline",
        "cta_button_text": "string - CTA button text",
        "contact_email": "string - contact email (optional)",
        "contact_phone": "string - contact phone (optional)",
        "primary_color": "string - hex color for primary elements (default #3b82f6)",
    },
)
def wp_design_full_landing_page(
    headline: str,
    subheadline: str,
    features: list,
    testimonials: list = None,
    cta_headline: str = "Ready to Get Started?",
    cta_button_text: str = "Contact Us",
    contact_email: str = None,
    contact_phone: str = None,
    primary_color: str = "#3b82f6",
) -> dict:
    from integrations.wordpress.elementor import (
        section_hero, section_features, section_testimonials,
        section_cta, section_contact, background_color, apply_color_scheme
    )

    sections = []

    # Hero
    sections.append(section_hero(
        headline=headline,
        subheadline=subheadline,
        cta_text="Learn More",
        cta_link="#features",
        background=background_color(primary_color),
    ))

    # Features
    sections.append(section_features(
        features=features,
        headline="What We Offer",
        columns=3 if len(features) >= 3 else len(features),
    ))

    # Testimonials (if provided)
    if testimonials:
        sections.append(section_testimonials(
            testimonials=testimonials,
            headline="What Our Clients Say",
        ))

    # CTA
    sections.append(section_cta(
        headline=cta_headline,
        button_text=cta_button_text,
        button_link="#contact",
        background=background_color(primary_color),
    ))

    # Contact
    sections.append(section_contact(
        headline="Get In Touch",
        email=contact_email,
        phone=contact_phone,
    ))

    return {"success": True, "sections": sections, "section_count": len(sections)}


# ============ Firecrawl Web Scraping ============

@tool(
    name="scrape_url",
    description="Scrape a single webpage and get its content as markdown. Use this to read any webpage.",
    parameters={
        "url": "string - the URL to scrape",
        "only_main_content": "bool - remove headers/footers/navs (default True)",
        "wait_for": "int - milliseconds to wait for JS to load (default 0, use for dynamic sites)",
    },
)
def firecrawl_scrape_url(url: str, only_main_content: bool = True, wait_for: int = 0) -> dict:
    from integrations.firecrawl.client import scrape_url
    return scrape_url(url, only_main_content=only_main_content, wait_for=wait_for)


@tool(
    name="crawl_website",
    description="Crawl an entire website starting from a URL. Gets content from multiple pages.",
    parameters={
        "url": "string - starting URL",
        "max_depth": "int - how deep to crawl (default 2)",
        "limit": "int - max pages to crawl (default 10)",
        "include_paths": "list - only crawl paths matching these patterns (optional)",
        "exclude_paths": "list - skip paths matching these patterns (optional)",
    },
)
def firecrawl_crawl_website(
    url: str,
    max_depth: int = 2,
    limit: int = 10,
    include_paths: list = None,
    exclude_paths: list = None,
) -> dict:
    from integrations.firecrawl.client import crawl_website
    return crawl_website(url, max_depth, limit, include_paths, exclude_paths)


@tool(
    name="search_and_scrape",
    description="Search Google and scrape the result pages. Great for research on any topic.",
    parameters={
        "query": "string - search query",
        "limit": "int - number of results (default 5)",
        "scrape_results": "bool - also scrape content of each result (default True)",
    },
)
def firecrawl_search_google(query: str, limit: int = 5, scrape_results: bool = True) -> dict:
    from integrations.firecrawl.client import search_google
    return search_google(query, limit, scrape_results)


@tool(
    name="extract_structured_data",
    description="Extract structured data from a webpage using AI. Define a schema and get JSON back.",
    parameters={
        "url": "string - URL to extract from",
        "schema": "dict - JSON schema defining what to extract",
        "prompt": "string - optional prompt to guide extraction",
    },
)
def firecrawl_extract_data(url: str, schema: dict, prompt: str = None) -> dict:
    from integrations.firecrawl.client import extract_data
    return extract_data(url, schema, prompt)


@tool(
    name="scrape_to_knowledge",
    description="Scrape a webpage and add it directly to Alfred's LightRAG knowledge base.",
    parameters={
        "url": "string - URL to scrape and learn",
        "description": "string - description of the content (optional)",
    },
)
def firecrawl_scrape_to_lightrag(url: str, description: str = "") -> dict:
    from integrations.firecrawl.client import scrape_to_lightrag
    return scrape_to_lightrag(url, description)


@tool(
    name="crawl_to_knowledge",
    description="Crawl a website and add all pages to Alfred's LightRAG knowledge base. Great for learning entire documentation sites.",
    parameters={
        "url": "string - starting URL",
        "max_depth": "int - crawl depth (default 2)",
        "limit": "int - max pages (default 10)",
        "description": "string - description of the content (optional)",
    },
)
def firecrawl_crawl_to_lightrag(
    url: str,
    max_depth: int = 2,
    limit: int = 10,
    description: str = "",
) -> dict:
    from integrations.firecrawl.client import crawl_to_lightrag
    return crawl_to_lightrag(url, max_depth, limit, description)


@tool(
    name="get_crawl_status",
    description="Check the status of a running crawl job.",
    parameters={
        "crawl_id": "string - the crawl job ID",
    },
)
def firecrawl_get_crawl_status(crawl_id: str) -> dict:
    from integrations.firecrawl.client import get_crawl_status
    return get_crawl_status(crawl_id)


# ==================== SYSTEMS CHECK ====================

@tool(
    name="systems_check",
    description="Run a full systems check on all Alfred integrations to verify they are operational. Returns status of Home Assistant, Email, WordPress, Servers, and other integrations.",
    parameters={},
)
def systems_check() -> dict:
    """Run comprehensive systems check on all integrations."""
    results = {}

    # Home Assistant
    try:
        from integrations.homeassistant.client import is_configured, list_devices
        if is_configured():
            devices = list_devices()
            if isinstance(devices, dict) and "error" not in devices:
                results["home_assistant"] = {
                    "status": "OK",
                    "lights": len(devices.get("lights", [])),
                    "climate": len(devices.get("climate", [])),
                    "switches": len(devices.get("switches", [])),
                }
            else:
                results["home_assistant"] = {"status": "ERROR", "error": str(devices)}
        else:
            results["home_assistant"] = {"status": "NOT_CONFIGURED"}
    except Exception as e:
        results["home_assistant"] = {"status": "ERROR", "error": str(e)}

    # Gmail
    try:
        from integrations.gmail.client import get_unread_count
        count = get_unread_count()
        results["gmail"] = {"status": "OK", "unread": count}
    except Exception as e:
        results["gmail"] = {"status": "ERROR", "error": str(e)}

    # Multi-Account Email
    try:
        from integrations.email.client import email_client
        accounts = email_client.list_accounts()
        results["email_accounts"] = {"status": "OK", "count": len(accounts.get("accounts", []))}
    except Exception as e:
        results["email_accounts"] = {"status": "ERROR", "error": str(e)}

    # WordPress
    try:
        from integrations.wordpress.client import list_sites
        sites = list_sites()
        results["wordpress"] = {"status": "OK", "sites": len(sites)}
    except Exception as e:
        results["wordpress"] = {"status": "ERROR", "error": str(e)}

    # Servers
    try:
        from integrations.servers.manager import list_servers
        servers = list_servers()
        results["servers"] = {"status": "OK", "count": len(servers)}
    except Exception as e:
        results["servers"] = {"status": "ERROR", "error": str(e)}

    # LightRAG
    try:
        from integrations.lightrag.client import is_connected
        import asyncio
        connected = asyncio.run(is_connected())
        results["lightrag"] = {"status": "OK" if connected else "ERROR"}
    except Exception as e:
        results["lightrag"] = {"status": "ERROR", "error": str(e)}

    # Twenty CRM
    try:
        from integrations.base_crm.client import list_people
        people = list_people(limit=1)
        results["twenty_crm"] = {"status": "OK"}
    except Exception as e:
        results["twenty_crm"] = {"status": "ERROR", "error": str(e)}

    # n8n Automations
    try:
        from integrations.n8n.client import list_workflows
        workflows = list_workflows()
        if isinstance(workflows, list):
            results["n8n"] = {"status": "OK", "workflows": len(workflows)}
        else:
            results["n8n"] = {"status": "OK"}
    except Exception as e:
        results["n8n"] = {"status": "ERROR", "error": str(e)}

    # Nextcloud
    try:
        from integrations.nextcloud.client import get_storage_info
        info = get_storage_info()
        if "error" not in str(info):
            results["nextcloud"] = {"status": "OK"}
        else:
            results["nextcloud"] = {"status": "ERROR", "error": str(info)}
    except Exception as e:
        results["nextcloud"] = {"status": "ERROR", "error": str(e)}

    # Stripe
    try:
        from integrations.stripe.client import get_balance
        balance = get_balance()
        if "error" not in str(balance):
            results["stripe"] = {"status": "OK"}
        else:
            results["stripe"] = {"status": "ERROR", "error": str(balance)}
    except Exception as e:
        results["stripe"] = {"status": "ERROR", "error": str(e)}

    # Twilio
    try:
        import os
        phone = os.getenv("TWILIO_PHONE_NUMBER")
        if phone:
            results["twilio"] = {"status": "OK", "phone": phone}
        else:
            results["twilio"] = {"status": "NOT_CONFIGURED"}
    except Exception as e:
        results["twilio"] = {"status": "ERROR", "error": str(e)}

    # Summary
    ok_count = sum(1 for v in results.values() if v.get("status") == "OK")
    total = len(results)

    return {
        "summary": f"{ok_count}/{total} integrations operational",
        "integrations": results
    }


@tool(
    name="alfred_self_diagnostic",
    description="Run a comprehensive self-diagnostic on Alfred, checking services, logs, and configuration. Can also log issues to LightRAG for permanent record.",
    parameters={
        "log_to_lightrag": "bool - whether to store diagnostic results in LightRAG (default True)",
    },
)
def alfred_self_diagnostic(log_to_lightrag: bool = True) -> dict:
    """Run comprehensive self-diagnostic on Alfred."""
    import subprocess
    import os
    from datetime import datetime

    results = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "issues": [],
        "recommendations": [],
    }

    # Check Alfred service status
    try:
        status = subprocess.run(
            ["systemctl", "status", "alfred", "--no-pager"],
            capture_output=True, text=True, timeout=10
        )
        is_running = "active (running)" in status.stdout
        results["checks"]["alfred_service"] = {
            "status": "OK" if is_running else "ERROR",
            "running": is_running,
        }
        if not is_running:
            results["issues"].append("Alfred service is not running")
            results["recommendations"].append("Run: sudo systemctl restart alfred")
    except Exception as e:
        results["checks"]["alfred_service"] = {"status": "ERROR", "error": str(e)}

    # Check memory usage
    try:
        mem = subprocess.run(
            ["free", "-h"], capture_output=True, text=True, timeout=5
        )
        results["checks"]["memory"] = {"status": "OK", "output": mem.stdout.strip()}
    except Exception as e:
        results["checks"]["memory"] = {"status": "ERROR", "error": str(e)}

    # Check disk space
    try:
        disk = subprocess.run(
            ["df", "-h", "/home/aialfred"], capture_output=True, text=True, timeout=5
        )
        results["checks"]["disk"] = {"status": "OK", "output": disk.stdout.strip()}
    except Exception as e:
        results["checks"]["disk"] = {"status": "ERROR", "error": str(e)}

    # Check recent logs for errors
    try:
        logs = subprocess.run(
            ["journalctl", "-u", "alfred", "-n", "50", "--no-pager"],
            capture_output=True, text=True, timeout=10
        )
        error_lines = [l for l in logs.stdout.split('\n') if 'error' in l.lower() or 'exception' in l.lower()]
        results["checks"]["recent_errors"] = {
            "status": "OK" if len(error_lines) == 0 else "WARNING",
            "error_count": len(error_lines),
            "recent_errors": error_lines[-5:] if error_lines else [],
        }
        if error_lines:
            results["issues"].append(f"Found {len(error_lines)} errors in recent logs")
    except Exception as e:
        results["checks"]["recent_errors"] = {"status": "ERROR", "error": str(e)}

    # Check key files exist
    key_files = [
        "/home/aialfred/alfred/core/api/main.py",
        "/home/aialfred/alfred/core/tools/definitions.py",
        "/home/aialfred/alfred/core/tools/registry.py",
        "/home/aialfred/alfred/config/.env",
    ]
    for f in key_files:
        exists = os.path.exists(f)
        results["checks"][f"file_{os.path.basename(f)}"] = {
            "status": "OK" if exists else "ERROR",
            "exists": exists,
        }
        if not exists:
            results["issues"].append(f"Missing critical file: {f}")

    # Run systems_check for integrations
    try:
        integration_results = systems_check()
        results["checks"]["integrations"] = integration_results
    except Exception as e:
        results["checks"]["integrations"] = {"status": "ERROR", "error": str(e)}

    # Add self-modification info
    results["self_modification"] = {
        "ui_file": "/home/aialfred/alfred/core/api/main.py",
        "tools_file": "/home/aialfred/alfred/core/tools/definitions.py",
        "config_file": "/home/aialfred/alfred/config/.env",
        "restart_command": "sudo systemctl restart alfred",
    }

    # Log to LightRAG if requested
    if log_to_lightrag and results["issues"]:
        try:
            import asyncio
            from integrations.lightrag.client import upload_text

            log_entry = f"""
# Alfred Diagnostic Log - {results['timestamp']}

## Issues Found
{chr(10).join('- ' + issue for issue in results['issues']) if results['issues'] else 'No issues found'}

## Recommendations
{chr(10).join('- ' + rec for rec in results['recommendations']) if results['recommendations'] else 'None'}

## Integration Status
{results['checks'].get('integrations', {}).get('summary', 'Unknown')}
"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(upload_text(log_entry, f"Alfred diagnostic log {results['timestamp']}"))
            finally:
                loop.close()
            results["logged_to_lightrag"] = True
        except Exception as e:
            results["logged_to_lightrag"] = False
            results["lightrag_error"] = str(e)

    return results


@tool(
    name="alfred_read_own_code",
    description="Read Alfred's own source code files. Use this when asked to modify Alfred's UI, add features, or fix issues.",
    parameters={
        "file": "string - which file to read: 'ui' (main.py), 'tools' (definitions.py), 'registry', 'config', 'router'",
        "section": "string - optional section to focus on (e.g., 'login', 'css', 'html')",
    },
)
def alfred_read_own_code(file: str, section: str = None) -> dict:
    """Read Alfred's source code for self-modification."""
    file_map = {
        "ui": "/home/aialfred/alfred/core/api/main.py",
        "main": "/home/aialfred/alfred/core/api/main.py",
        "tools": "/home/aialfred/alfred/core/tools/definitions.py",
        "definitions": "/home/aialfred/alfred/core/tools/definitions.py",
        "registry": "/home/aialfred/alfred/core/tools/registry.py",
        "config": "/home/aialfred/alfred/config/.env",
        "router": "/home/aialfred/alfred/core/brain/router.py",
    }

    file_path = file_map.get(file.lower())
    if not file_path:
        return {"error": f"Unknown file: {file}. Valid options: {list(file_map.keys())}"}

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # If section specified, try to extract relevant portion
        if section:
            section_lower = section.lower()
            lines = content.split('\n')
            relevant_lines = []
            in_section = False

            for i, line in enumerate(lines):
                if section_lower in line.lower():
                    # Get context around the match
                    start = max(0, i - 5)
                    end = min(len(lines), i + 50)
                    relevant_lines.extend(lines[start:end])
                    relevant_lines.append(f"... (line {i+1})")
                    break

            if relevant_lines:
                content = '\n'.join(relevant_lines)
            else:
                return {"error": f"Section '{section}' not found in {file}"}

        # Limit content length for response
        if len(content) > 10000:
            content = content[:10000] + "\n\n... [truncated, file is very long]"

        return {
            "file": file_path,
            "content": content,
            "note": "Alfred can modify this file to change his own behavior/appearance",
        }
    except Exception as e:
        return {"error": str(e)}


def register_all():
    """Import this module to register all tools."""
    pass
