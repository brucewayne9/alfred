"""Nextcloud API client - WebDAV for files, OCS API for notes and more."""

import logging
from typing import Any
from xml.etree import ElementTree as ET
from pathlib import Path
from datetime import datetime

import requests
from requests.auth import HTTPBasicAuth

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = settings.nextcloud_url.rstrip("/") if settings.nextcloud_url else ""
USERNAME = settings.nextcloud_username
PASSWORD = settings.nextcloud_password


def _auth() -> HTTPBasicAuth:
    return HTTPBasicAuth(USERNAME, PASSWORD)


def _headers(extra: dict = None) -> dict:
    h = {"OCS-APIRequest": "true"}
    if extra:
        h.update(extra)
    return h


# ==================== Connection Check ====================

def is_connected() -> bool:
    if not BASE_URL or not USERNAME or not PASSWORD:
        return False
    try:
        resp = requests.get(
            f"{BASE_URL}/ocs/v1.php/cloud/capabilities",
            auth=_auth(),
            headers=_headers({"Accept": "application/json"}),
            timeout=10
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Nextcloud connection check failed: {e}")
        return False


def get_user_info() -> dict:
    """Get current user info."""
    resp = requests.get(
        f"{BASE_URL}/ocs/v1.php/cloud/user",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    user_data = data.get("ocs", {}).get("data", {})
    return {
        "id": user_data.get("id"),
        "display_name": user_data.get("display-name") or user_data.get("displayname"),
        "email": user_data.get("email"),
        "quota_used": user_data.get("quota", {}).get("used"),
        "quota_total": user_data.get("quota", {}).get("total"),
    }


# ==================== WebDAV File Operations ====================

def _webdav_url(path: str = "") -> str:
    """Build WebDAV URL for a path."""
    path = path.lstrip("/")
    return f"{BASE_URL}/remote.php/dav/files/{USERNAME}/{path}"


def _parse_propfind(xml_content: bytes) -> list[dict]:
    """Parse PROPFIND XML response."""
    ns = {
        "d": "DAV:",
        "oc": "http://owncloud.org/ns",
        "nc": "http://nextcloud.org/ns",
    }
    root = ET.fromstring(xml_content)
    items = []

    for response in root.findall("d:response", ns):
        href = response.find("d:href", ns)
        if href is None:
            continue

        propstat = response.find("d:propstat", ns)
        if propstat is None:
            continue

        prop = propstat.find("d:prop", ns)
        if prop is None:
            continue

        # Extract path from href
        href_text = href.text or ""
        # Remove the WebDAV prefix to get relative path
        path = href_text.split(f"/remote.php/dav/files/{USERNAME}/")[-1]
        path = path.rstrip("/")

        # Check if it's a collection (folder)
        resource_type = prop.find("d:resourcetype", ns)
        is_folder = resource_type is not None and resource_type.find("d:collection", ns) is not None

        # Get properties
        size_el = prop.find("d:getcontentlength", ns)
        modified_el = prop.find("d:getlastmodified", ns)
        etag_el = prop.find("d:getetag", ns)
        content_type_el = prop.find("d:getcontenttype", ns)

        items.append({
            "path": "/" + path if path else "/",
            "name": Path(path).name if path else "/",
            "is_folder": is_folder,
            "size": int(size_el.text) if size_el is not None and size_el.text else 0,
            "modified": modified_el.text if modified_el is not None else None,
            "etag": etag_el.text.strip('"') if etag_el is not None and etag_el.text else None,
            "content_type": content_type_el.text if content_type_el is not None else None,
        })

    return items


def list_files(path: str = "/", depth: int = 1) -> list[dict]:
    """List files and folders at a path."""
    propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
    <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
        <d:prop>
            <d:resourcetype/>
            <d:getcontentlength/>
            <d:getlastmodified/>
            <d:getetag/>
            <d:getcontenttype/>
        </d:prop>
    </d:propfind>"""

    resp = requests.request(
        "PROPFIND",
        _webdav_url(path),
        auth=_auth(),
        headers={"Depth": str(depth), "Content-Type": "application/xml"},
        data=propfind_body,
        timeout=30
    )
    resp.raise_for_status()

    items = _parse_propfind(resp.content)
    # Filter out the parent directory itself
    items = [i for i in items if i["path"].rstrip("/") != path.rstrip("/")]
    return items


def get_file_info(path: str) -> dict | None:
    """Get info about a specific file or folder."""
    items = list_files(path, depth=0)
    # The response includes the item itself
    propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
    <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
        <d:prop>
            <d:resourcetype/>
            <d:getcontentlength/>
            <d:getlastmodified/>
            <d:getetag/>
            <d:getcontenttype/>
        </d:prop>
    </d:propfind>"""

    resp = requests.request(
        "PROPFIND",
        _webdav_url(path),
        auth=_auth(),
        headers={"Depth": "0", "Content-Type": "application/xml"},
        data=propfind_body,
        timeout=30
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    items = _parse_propfind(resp.content)
    return items[0] if items else None


def download_file(path: str) -> bytes:
    """Download a file's contents."""
    resp = requests.get(
        _webdav_url(path),
        auth=_auth(),
        timeout=60
    )
    resp.raise_for_status()
    return resp.content


def download_file_text(path: str) -> str:
    """Download a text file's contents."""
    content = download_file(path)
    return content.decode("utf-8")


def upload_file(path: str, content: bytes | str, content_type: str = None) -> dict:
    """Upload a file."""
    if isinstance(content, str):
        content = content.encode("utf-8")

    headers = {}
    if content_type:
        headers["Content-Type"] = content_type

    resp = requests.put(
        _webdav_url(path),
        auth=_auth(),
        headers=headers,
        data=content,
        timeout=60
    )
    resp.raise_for_status()

    return {"success": True, "path": path}


def create_folder(path: str) -> dict:
    """Create a folder."""
    resp = requests.request(
        "MKCOL",
        _webdav_url(path),
        auth=_auth(),
        timeout=30
    )
    resp.raise_for_status()
    return {"success": True, "path": path}


def delete_file(path: str) -> dict:
    """Delete a file or folder."""
    resp = requests.delete(
        _webdav_url(path),
        auth=_auth(),
        timeout=30
    )
    resp.raise_for_status()
    return {"success": True, "deleted": path}


def move_file(source: str, destination: str) -> dict:
    """Move or rename a file/folder."""
    resp = requests.request(
        "MOVE",
        _webdav_url(source),
        auth=_auth(),
        headers={"Destination": _webdav_url(destination)},
        timeout=30
    )
    resp.raise_for_status()
    return {"success": True, "from": source, "to": destination}


def copy_file(source: str, destination: str) -> dict:
    """Copy a file/folder."""
    resp = requests.request(
        "COPY",
        _webdav_url(source),
        auth=_auth(),
        headers={"Destination": _webdav_url(destination)},
        timeout=30
    )
    resp.raise_for_status()
    return {"success": True, "from": source, "to": destination}


def search_files(query: str, path: str = "/") -> list[dict]:
    """Search for files by name."""
    # Use WebDAV SEARCH or fall back to listing and filtering
    # Nextcloud supports SEARCH but let's use a simple recursive approach
    search_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <d:searchrequest xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
        <d:basicsearch>
            <d:select>
                <d:prop>
                    <d:resourcetype/>
                    <d:getcontentlength/>
                    <d:getlastmodified/>
                    <d:getcontenttype/>
                </d:prop>
            </d:select>
            <d:from>
                <d:scope>
                    <d:href>/files/{USERNAME}{path}</d:href>
                    <d:depth>infinity</d:depth>
                </d:scope>
            </d:from>
            <d:where>
                <d:like>
                    <d:prop><d:displayname/></d:prop>
                    <d:literal>%{query}%</d:literal>
                </d:like>
            </d:where>
            <d:limit><d:nresults>50</d:nresults></d:limit>
        </d:basicsearch>
    </d:searchrequest>"""

    try:
        resp = requests.request(
            "SEARCH",
            f"{BASE_URL}/remote.php/dav",
            auth=_auth(),
            headers={"Content-Type": "application/xml"},
            data=search_body,
            timeout=30
        )
        if resp.status_code == 207:
            return _parse_propfind(resp.content)
    except Exception as e:
        logger.warning(f"WebDAV search failed, falling back: {e}")

    # Fallback: list root and filter
    all_files = list_files(path, depth=1)
    query_lower = query.lower()
    return [f for f in all_files if query_lower in f["name"].lower()]


# ==================== Notes API ====================

def list_notes() -> list[dict]:
    """List all notes."""
    resp = requests.get(
        f"{BASE_URL}/index.php/apps/notes/api/v1/notes",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=15
    )
    resp.raise_for_status()
    notes = resp.json()
    return [{
        "id": n.get("id"),
        "title": n.get("title", ""),
        "category": n.get("category", ""),
        "modified": n.get("modified"),
        "favorite": n.get("favorite", False),
        "content_preview": (n.get("content") or "")[:200],
    } for n in notes]


def get_note(note_id: int) -> dict:
    """Get a note by ID."""
    resp = requests.get(
        f"{BASE_URL}/index.php/apps/notes/api/v1/notes/{note_id}",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=15
    )
    resp.raise_for_status()
    n = resp.json()
    return {
        "id": n.get("id"),
        "title": n.get("title", ""),
        "content": n.get("content", ""),
        "category": n.get("category", ""),
        "modified": n.get("modified"),
        "favorite": n.get("favorite", False),
    }


def create_note(title: str, content: str = "", category: str = "") -> dict:
    """Create a new note."""
    body = {"title": title, "content": content}
    if category:
        body["category"] = category

    resp = requests.post(
        f"{BASE_URL}/index.php/apps/notes/api/v1/notes",
        auth=_auth(),
        headers=_headers({"Accept": "application/json", "Content-Type": "application/json"}),
        json=body,
        timeout=15
    )
    resp.raise_for_status()
    n = resp.json()
    return {
        "id": n.get("id"),
        "title": n.get("title", ""),
        "content": n.get("content", ""),
        "category": n.get("category", ""),
    }


def update_note(note_id: int, title: str = None, content: str = None, category: str = None) -> dict:
    """Update a note."""
    body = {}
    if title is not None:
        body["title"] = title
    if content is not None:
        body["content"] = content
    if category is not None:
        body["category"] = category

    resp = requests.put(
        f"{BASE_URL}/index.php/apps/notes/api/v1/notes/{note_id}",
        auth=_auth(),
        headers=_headers({"Accept": "application/json", "Content-Type": "application/json"}),
        json=body,
        timeout=15
    )
    resp.raise_for_status()
    n = resp.json()
    return {
        "id": n.get("id"),
        "title": n.get("title", ""),
        "content": n.get("content", ""),
        "category": n.get("category", ""),
    }


def delete_note(note_id: int) -> dict:
    """Delete a note."""
    resp = requests.delete(
        f"{BASE_URL}/index.php/apps/notes/api/v1/notes/{note_id}",
        auth=_auth(),
        headers=_headers(),
        timeout=15
    )
    resp.raise_for_status()
    return {"deleted": True, "id": note_id}


# ==================== Storage Info ====================

def get_storage_info() -> dict:
    """Get storage quota info."""
    user = get_user_info()
    used = user.get("quota_used") or 0
    total = user.get("quota_total") or 0

    def format_bytes(b):
        if b < 1024:
            return f"{b} B"
        elif b < 1024**2:
            return f"{b/1024:.1f} KB"
        elif b < 1024**3:
            return f"{b/1024**2:.1f} MB"
        else:
            return f"{b/1024**3:.2f} GB"

    return {
        "used": used,
        "total": total,
        "used_formatted": format_bytes(used),
        "total_formatted": format_bytes(total) if total > 0 else "Unlimited",
        "percent_used": round(used / total * 100, 1) if total > 0 else 0,
    }


# ==================== Nextcloud Talk ====================

def list_conversations() -> list[dict]:
    """List all Talk conversations."""
    resp = requests.get(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v4/room",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    rooms = data.get("ocs", {}).get("data", [])
    return [{
        "token": r.get("token"),
        "name": r.get("displayName") or r.get("name", ""),
        "type": r.get("type"),  # 1=one-on-one, 2=group, 3=public, 4=changelog
        "type_name": {1: "one-on-one", 2: "group", 3: "public", 4: "changelog"}.get(r.get("type"), "unknown"),
        "unread_messages": r.get("unreadMessages", 0),
        "last_message": r.get("lastMessage", {}).get("message") if r.get("lastMessage") else None,
        "participants_count": r.get("participantCount", 0),
    } for r in rooms]


def get_conversation(token: str) -> dict:
    """Get conversation details."""
    resp = requests.get(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v4/room/{token}",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=15
    )
    resp.raise_for_status()
    r = resp.json().get("ocs", {}).get("data", {})
    return {
        "token": r.get("token"),
        "name": r.get("displayName") or r.get("name", ""),
        "type": r.get("type"),
        "description": r.get("description", ""),
        "participants_count": r.get("participantCount", 0),
        "unread_messages": r.get("unreadMessages", 0),
    }


def create_conversation(name: str, conversation_type: int = 2, invite_users: list[str] = None) -> dict:
    """Create a new conversation. Type: 2=group, 3=public."""
    body = {"roomType": conversation_type, "roomName": name}
    if invite_users:
        body["invite"] = invite_users[0]  # Initial invite

    resp = requests.post(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v4/room",
        auth=_auth(),
        headers=_headers({"Accept": "application/json", "Content-Type": "application/json"}),
        json=body,
        timeout=15
    )
    resp.raise_for_status()
    r = resp.json().get("ocs", {}).get("data", {})

    # Add additional users if provided
    if invite_users and len(invite_users) > 1:
        for user in invite_users[1:]:
            add_participant(r.get("token"), user)

    return {
        "token": r.get("token"),
        "name": r.get("displayName") or r.get("name", ""),
        "type": r.get("type"),
    }


def add_participant(token: str, user_id: str) -> dict:
    """Add a participant to a conversation."""
    resp = requests.post(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v4/room/{token}/participants",
        auth=_auth(),
        headers=_headers({"Accept": "application/json", "Content-Type": "application/json"}),
        json={"newParticipant": user_id, "source": "users"},
        timeout=15
    )
    resp.raise_for_status()
    return {"success": True, "added": user_id}


def send_message(token: str, message: str) -> dict:
    """Send a message to a conversation."""
    resp = requests.post(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v1/chat/{token}",
        auth=_auth(),
        headers=_headers({"Accept": "application/json", "Content-Type": "application/json"}),
        json={"message": message},
        timeout=15
    )
    resp.raise_for_status()
    m = resp.json().get("ocs", {}).get("data", {})
    return {
        "id": m.get("id"),
        "message": m.get("message"),
        "timestamp": m.get("timestamp"),
        "actor": m.get("actorDisplayName"),
    }


def get_messages(token: str, limit: int = 50) -> list[dict]:
    """Get messages from a conversation."""
    resp = requests.get(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v1/chat/{token}",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        params={"limit": limit, "lookIntoFuture": 0},
        timeout=15
    )
    resp.raise_for_status()
    messages = resp.json().get("ocs", {}).get("data", [])
    return [{
        "id": m.get("id"),
        "message": m.get("message"),
        "timestamp": m.get("timestamp"),
        "actor": m.get("actorDisplayName"),
        "actor_id": m.get("actorId"),
        "type": m.get("messageType"),
    } for m in messages]


def delete_conversation(token: str) -> dict:
    """Delete a conversation."""
    resp = requests.delete(
        f"{BASE_URL}/ocs/v2.php/apps/spreed/api/v4/room/{token}",
        auth=_auth(),
        headers=_headers(),
        timeout=15
    )
    resp.raise_for_status()
    return {"deleted": True, "token": token}


# ==================== User Management (Admin) ====================

def list_users(search: str = "", limit: int = 50) -> list[dict]:
    """List users (requires admin privileges)."""
    params = {"limit": limit}
    if search:
        params["search"] = search

    resp = requests.get(
        f"{BASE_URL}/ocs/v1.php/cloud/users",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        params=params,
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    users = data.get("ocs", {}).get("data", {}).get("users", [])
    return [{"user_id": u} for u in users]


def get_user(user_id: str) -> dict:
    """Get user details."""
    resp = requests.get(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    u = data.get("ocs", {}).get("data", {})
    return {
        "id": u.get("id"),
        "display_name": u.get("displayname") or u.get("display-name"),
        "email": u.get("email"),
        "enabled": u.get("enabled"),
        "groups": u.get("groups", []),
        "quota_used": u.get("quota", {}).get("used"),
        "quota_total": u.get("quota", {}).get("total"),
        "last_login": u.get("lastLogin"),
    }


def create_user(user_id: str, password: str, email: str = "", display_name: str = "",
                groups: list[str] = None, quota: str = None) -> dict:
    """Create a new user (requires admin privileges)."""
    body = {
        "userid": user_id,
        "password": password,
    }
    if email:
        body["email"] = email
    if display_name:
        body["displayName"] = display_name
    if groups:
        body["groups"] = groups
    if quota:
        body["quota"] = quota

    resp = requests.post(
        f"{BASE_URL}/ocs/v1.php/cloud/users",
        auth=_auth(),
        headers=_headers({"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}),
        data=body,
        timeout=15
    )
    resp.raise_for_status()
    return {"success": True, "user_id": user_id}


def update_user(user_id: str, key: str, value: str) -> dict:
    """Update a user attribute. Keys: email, displayname, password, quota."""
    resp = requests.put(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}",
        auth=_auth(),
        headers=_headers({"Content-Type": "application/x-www-form-urlencoded"}),
        data={"key": key, "value": value},
        timeout=15
    )
    resp.raise_for_status()
    return {"success": True, "user_id": user_id, "updated": key}


def delete_user(user_id: str) -> dict:
    """Delete a user (requires admin privileges)."""
    resp = requests.delete(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}",
        auth=_auth(),
        headers=_headers(),
        timeout=15
    )
    resp.raise_for_status()
    return {"deleted": True, "user_id": user_id}


def enable_user(user_id: str) -> dict:
    """Enable a user."""
    resp = requests.put(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}/enable",
        auth=_auth(),
        headers=_headers(),
        timeout=15
    )
    resp.raise_for_status()
    return {"enabled": True, "user_id": user_id}


def disable_user(user_id: str) -> dict:
    """Disable a user."""
    resp = requests.put(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}/disable",
        auth=_auth(),
        headers=_headers(),
        timeout=15
    )
    resp.raise_for_status()
    return {"disabled": True, "user_id": user_id}


def add_user_to_group(user_id: str, group_id: str) -> dict:
    """Add user to a group."""
    resp = requests.post(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}/groups",
        auth=_auth(),
        headers=_headers({"Content-Type": "application/x-www-form-urlencoded"}),
        data={"groupid": group_id},
        timeout=15
    )
    resp.raise_for_status()
    return {"success": True, "user_id": user_id, "group": group_id}


def remove_user_from_group(user_id: str, group_id: str) -> dict:
    """Remove user from a group."""
    resp = requests.delete(
        f"{BASE_URL}/ocs/v1.php/cloud/users/{user_id}/groups",
        auth=_auth(),
        headers=_headers({"Content-Type": "application/x-www-form-urlencoded"}),
        data={"groupid": group_id},
        timeout=15
    )
    resp.raise_for_status()
    return {"success": True, "user_id": user_id, "removed_from": group_id}


def list_groups() -> list[dict]:
    """List all groups."""
    resp = requests.get(
        f"{BASE_URL}/ocs/v1.php/cloud/groups",
        auth=_auth(),
        headers=_headers({"Accept": "application/json"}),
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    groups = data.get("ocs", {}).get("data", {}).get("groups", [])
    return [{"group_id": g} for g in groups]


def create_group(group_id: str) -> dict:
    """Create a new group."""
    resp = requests.post(
        f"{BASE_URL}/ocs/v1.php/cloud/groups",
        auth=_auth(),
        headers=_headers({"Content-Type": "application/x-www-form-urlencoded"}),
        data={"groupid": group_id},
        timeout=15
    )
    resp.raise_for_status()
    return {"success": True, "group_id": group_id}


# ==================== Calendar (CalDAV) ====================

def list_calendars() -> list[dict]:
    """List all calendars."""
    propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
    <d:propfind xmlns:d="DAV:" xmlns:cs="http://calendarserver.org/ns/" xmlns:c="urn:ietf:params:xml:ns:caldav">
        <d:prop>
            <d:displayname/>
            <cs:getctag/>
            <c:supported-calendar-component-set/>
        </d:prop>
    </d:propfind>"""

    resp = requests.request(
        "PROPFIND",
        f"{BASE_URL}/remote.php/dav/calendars/{USERNAME}/",
        auth=_auth(),
        headers={"Depth": "1", "Content-Type": "application/xml"},
        data=propfind_body,
        timeout=30
    )
    resp.raise_for_status()

    ns = {"d": "DAV:", "cs": "http://calendarserver.org/ns/", "c": "urn:ietf:params:xml:ns:caldav"}
    root = ET.fromstring(resp.content)
    calendars = []

    for response in root.findall("d:response", ns):
        href = response.find("d:href", ns)
        if href is None:
            continue

        href_text = href.text or ""
        # Skip the parent directory
        if href_text.rstrip("/").endswith(f"/calendars/{USERNAME}"):
            continue

        propstat = response.find("d:propstat", ns)
        if propstat is None:
            continue

        prop = propstat.find("d:prop", ns)
        if prop is None:
            continue

        name_el = prop.find("d:displayname", ns)
        cal_id = href_text.rstrip("/").split("/")[-1]

        calendars.append({
            "id": cal_id,
            "name": name_el.text if name_el is not None else cal_id,
            "href": href_text,
        })

    return calendars


def get_calendar_events(calendar_id: str, days_ahead: int = 30) -> list[dict]:
    """Get events from a calendar."""
    from datetime import datetime, timedelta

    start = datetime.utcnow()
    end = start + timedelta(days=days_ahead)

    report_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
        <d:prop>
            <d:getetag/>
            <c:calendar-data/>
        </d:prop>
        <c:filter>
            <c:comp-filter name="VCALENDAR">
                <c:comp-filter name="VEVENT">
                    <c:time-range start="{start.strftime('%Y%m%dT%H%M%SZ')}" end="{end.strftime('%Y%m%dT%H%M%SZ')}"/>
                </c:comp-filter>
            </c:comp-filter>
        </c:filter>
    </c:calendar-query>"""

    resp = requests.request(
        "REPORT",
        f"{BASE_URL}/remote.php/dav/calendars/{USERNAME}/{calendar_id}/",
        auth=_auth(),
        headers={"Depth": "1", "Content-Type": "application/xml"},
        data=report_body,
        timeout=30
    )
    resp.raise_for_status()

    ns = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:caldav"}
    root = ET.fromstring(resp.content)
    events = []

    for response in root.findall("d:response", ns):
        propstat = response.find("d:propstat", ns)
        if propstat is None:
            continue

        prop = propstat.find("d:prop", ns)
        if prop is None:
            continue

        cal_data = prop.find("c:calendar-data", ns)
        if cal_data is None or not cal_data.text:
            continue

        # Parse iCal data (basic parsing)
        ical = cal_data.text
        event = {"raw": ical}

        for line in ical.split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY:"):
                event["summary"] = line[8:]
            elif line.startswith("DTSTART"):
                event["start"] = line.split(":")[-1]
            elif line.startswith("DTEND"):
                event["end"] = line.split(":")[-1]
            elif line.startswith("LOCATION:"):
                event["location"] = line[9:]
            elif line.startswith("DESCRIPTION:"):
                event["description"] = line[12:]

        if "summary" in event:
            events.append(event)

    return events


# ==================== Tasks (if Tasks app installed) ====================

def list_task_lists() -> list[dict]:
    """List task lists (calendars with VTODO support)."""
    calendars = list_calendars()
    # Filter to task-capable calendars - typically all CalDAV calendars can have tasks
    return calendars


def get_tasks(list_id: str) -> list[dict]:
    """Get tasks from a task list."""
    report_body = """<?xml version="1.0" encoding="UTF-8"?>
    <c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
        <d:prop>
            <d:getetag/>
            <c:calendar-data/>
        </d:prop>
        <c:filter>
            <c:comp-filter name="VCALENDAR">
                <c:comp-filter name="VTODO"/>
            </c:comp-filter>
        </c:filter>
    </c:calendar-query>"""

    resp = requests.request(
        "REPORT",
        f"{BASE_URL}/remote.php/dav/calendars/{USERNAME}/{list_id}/",
        auth=_auth(),
        headers={"Depth": "1", "Content-Type": "application/xml"},
        data=report_body,
        timeout=30
    )
    resp.raise_for_status()

    ns = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:caldav"}
    root = ET.fromstring(resp.content)
    tasks = []

    for response in root.findall("d:response", ns):
        href = response.find("d:href", ns)
        propstat = response.find("d:propstat", ns)
        if propstat is None:
            continue

        prop = propstat.find("d:prop", ns)
        if prop is None:
            continue

        cal_data = prop.find("c:calendar-data", ns)
        if cal_data is None or not cal_data.text:
            continue

        ical = cal_data.text
        task = {"href": href.text if href is not None else None}

        for line in ical.split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY:"):
                task["summary"] = line[8:]
            elif line.startswith("STATUS:"):
                task["status"] = line[7:]
            elif line.startswith("PRIORITY:"):
                task["priority"] = int(line[9:])
            elif line.startswith("DUE"):
                task["due"] = line.split(":")[-1]
            elif line.startswith("PERCENT-COMPLETE:"):
                task["percent_complete"] = int(line[17:])

        if "summary" in task:
            tasks.append(task)

    return tasks
