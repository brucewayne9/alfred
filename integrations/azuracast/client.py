"""AzuraCast Radio API client. Manages stations, playlists, media, and streaming."""

import logging
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

BASE_URL = getattr(settings, 'azuracast_url', 'https://studiob.loovacast.com').rstrip("/")
API_KEY = getattr(settings, 'azuracast_api_key', '')


def _headers() -> dict:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _get(path: str, params: dict | None = None) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict = None, files: dict = None) -> Any:
    if files:
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(f"{BASE_URL}{path}", headers=headers, data=body, files=files, timeout=60)
    else:
        resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _put(path: str, body: dict) -> Any:
    resp = requests.put(f"{BASE_URL}{path}", headers=_headers(), json=body, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> Any:
    resp = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json() if resp.content else {"success": True}


# ==================== Station Resolution ====================

_station_cache: dict = {}


def get_station_id(station_name: str = None) -> int:
    """Resolve a station name to its ID. Returns first station if no name given."""
    global _station_cache

    # Refresh cache if empty
    if not _station_cache:
        try:
            stations = list_stations()
            for s in stations:
                _station_cache[s["id"]] = s
                _station_cache[s["name"].lower()] = s
                _station_cache[s["shortcode"].lower()] = s
        except Exception:
            pass

    # No name specified - return first station
    if not station_name:
        for key, val in _station_cache.items():
            if isinstance(key, int):
                return key
        return 22  # Default fallback

    # Try to find by name or shortcode
    name_lower = station_name.lower()
    for key, val in _station_cache.items():
        if isinstance(key, str) and name_lower in key:
            return val["id"]

    # Try as direct ID
    try:
        return int(station_name)
    except ValueError:
        return 22  # Default fallback


# ==================== Now Playing ====================

def get_now_playing(station_id: int | str = None) -> dict | list:
    """Get now playing info for all stations or a specific station."""
    if station_id:
        data = _get(f"/api/nowplaying/{station_id}")
        return _format_now_playing(data)
    else:
        data = _get("/api/nowplaying")
        return [_format_now_playing(s) for s in data]


def _format_now_playing(np: dict) -> dict:
    station = np.get("station", {})
    listeners = np.get("listeners", {})
    now = np.get("now_playing", {})
    song = now.get("song", {})
    next_song = np.get("playing_next", {}).get("song", {})

    return {
        "station_id": station.get("id"),
        "station_name": station.get("name"),
        "listeners": listeners.get("current", 0),
        "is_live": np.get("live", {}).get("is_live", False),
        "dj_name": np.get("live", {}).get("streamer_name", ""),
        "current_song": {
            "artist": song.get("artist", ""),
            "title": song.get("title", ""),
            "album": song.get("album", ""),
            "art": song.get("art", ""),
            "elapsed": now.get("elapsed", 0),
            "duration": now.get("duration", 0),
            "remaining": now.get("remaining", 0),
        },
        "playlist": now.get("playlist", ""),
        "next_song": {
            "artist": next_song.get("artist", ""),
            "title": next_song.get("title", ""),
        } if next_song else None,
        "listen_url": station.get("listen_url", ""),
        "is_online": np.get("is_online", False),
    }


def get_song_history(station_id: int | str, limit: int = 10) -> list[dict]:
    """Get recent song history for a station."""
    np = _get(f"/api/nowplaying/{station_id}")
    history = np.get("song_history", [])[:limit]
    return [{
        "artist": h.get("song", {}).get("artist", ""),
        "title": h.get("song", {}).get("title", ""),
        "played_at": h.get("played_at"),
        "playlist": h.get("playlist", ""),
    } for h in history]


# ==================== Stations ====================

def list_stations() -> list[dict]:
    """List all stations."""
    data = _get("/api/stations")
    return [{
        "id": s.get("id"),
        "name": s.get("name"),
        "shortcode": s.get("shortcode"),
        "description": s.get("description"),
        "listen_url": s.get("listen_url"),
        "is_public": s.get("is_public"),
        "requests_enabled": s.get("requests_enabled"),
    } for s in data]


def get_station(station_id: int | str) -> dict:
    """Get station details."""
    return _get(f"/api/station/{station_id}")


def restart_station(station_id: int | str) -> dict:
    """Restart a station's broadcasting services."""
    return _post(f"/api/station/{station_id}/restart")


def get_station_status(station_id: int | str) -> dict:
    """Get station backend/frontend status."""
    return _get(f"/api/station/{station_id}/status")


# ==================== Playlists ====================

def list_playlists(station_id: int | str) -> list[dict]:
    """List all playlists for a station."""
    data = _get(f"/api/station/{station_id}/playlists")
    return [{
        "id": p.get("id"),
        "name": p.get("name"),
        "type": p.get("type"),
        "source": p.get("source"),
        "is_enabled": p.get("is_enabled"),
        "weight": p.get("weight"),
        "num_songs": p.get("num_songs", 0),
        "total_length": p.get("total_length", 0),
    } for p in data]


def get_playlist(station_id: int | str, playlist_id: int) -> dict:
    """Get playlist details."""
    return _get(f"/api/station/{station_id}/playlist/{playlist_id}")


def toggle_playlist(station_id: int | str, playlist_id: int) -> dict:
    """Toggle a playlist on/off."""
    return _put(f"/api/station/{station_id}/playlist/{playlist_id}/toggle", {})


def reshuffle_playlist(station_id: int | str, playlist_id: int) -> dict:
    """Reshuffle a playlist's playback order."""
    return _put(f"/api/station/{station_id}/playlist/{playlist_id}/reshuffle", {})


def create_playlist(station_id: int | str, name: str, type: str = "default",
                    weight: int = 3, is_enabled: bool = True) -> dict:
    """Create a new playlist."""
    body = {
        "name": name,
        "type": type,
        "weight": weight,
        "is_enabled": is_enabled,
    }
    return _post(f"/api/station/{station_id}/playlists", body)


def delete_playlist(station_id: int | str, playlist_id: int) -> dict:
    """Delete a playlist."""
    return _delete(f"/api/station/{station_id}/playlist/{playlist_id}")


# ==================== Media/Files ====================

def list_media(station_id: int | str, limit: int = 50) -> list[dict]:
    """List media files for a station."""
    data = _get(f"/api/station/{station_id}/files", {"per_page": limit})
    files = data if isinstance(data, list) else data.get("data", [])
    return [{
        "id": f.get("id"),
        "unique_id": f.get("unique_id"),
        "artist": f.get("artist"),
        "title": f.get("title"),
        "album": f.get("album"),
        "path": f.get("path"),
        "length": f.get("length"),
        "length_text": f.get("length_text"),
    } for f in files[:limit]]


def search_media(station_id: int | str, query: str) -> list[dict]:
    """Search media files by artist, title, or album."""
    data = _get(f"/api/station/{station_id}/files", {"searchPhrase": query})
    files = data if isinstance(data, list) else data.get("data", [])
    return [{
        "id": f.get("id"),
        "artist": f.get("artist"),
        "title": f.get("title"),
        "album": f.get("album"),
        "path": f.get("path"),
    } for f in files]


def get_media(station_id: int | str, media_id: int) -> dict:
    """Get media file details."""
    return _get(f"/api/station/{station_id}/file/{media_id}")


def delete_media(station_id: int | str, media_id: int) -> dict:
    """Delete a media file."""
    return _delete(f"/api/station/{station_id}/file/{media_id}")


# ==================== Requests ====================

def list_requests(station_id: int | str) -> list[dict]:
    """List pending song requests."""
    try:
        data = _get(f"/api/station/{station_id}/requests")
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_requestable_songs(station_id: int | str) -> list[dict]:
    """Get list of songs available for request."""
    try:
        data = _get(f"/api/station/{station_id}/requests")
        return [{
            "id": s.get("request_id"),
            "artist": s.get("song", {}).get("artist"),
            "title": s.get("song", {}).get("title"),
        } for s in data]
    except Exception:
        return []


# ==================== Queue ====================

def get_queue(station_id: int | str) -> list[dict]:
    """Get upcoming queue for a station."""
    try:
        data = _get(f"/api/station/{station_id}/queue")
        return [{
            "cued_at": q.get("cued_at"),
            "artist": q.get("song", {}).get("artist"),
            "title": q.get("song", {}).get("title"),
            "playlist": q.get("playlist"),
            "is_request": q.get("is_request", False),
        } for q in data]
    except Exception:
        return []


def clear_queue(station_id: int | str) -> dict:
    """Clear the upcoming queue."""
    return _delete(f"/api/station/{station_id}/queue")


# ==================== Streamers/DJs ====================

def list_streamers(station_id: int | str) -> list[dict]:
    """List DJ/streamer accounts for a station."""
    try:
        data = _get(f"/api/station/{station_id}/streamers")
        return [{
            "id": s.get("id"),
            "username": s.get("streamer_username"),
            "display_name": s.get("display_name"),
            "is_active": s.get("is_active"),
        } for s in data]
    except Exception:
        return []


# ==================== Listeners ====================

def get_listeners(station_id: int | str) -> list[dict]:
    """Get current listener details."""
    try:
        data = _get(f"/api/station/{station_id}/listeners")
        return [{
            "ip": l.get("ip"),
            "user_agent": l.get("user_agent"),
            "connected_time": l.get("connected_time"),
            "location": l.get("location", {}).get("description", "Unknown"),
        } for l in data]
    except Exception:
        return []


# ==================== Reports ====================

def get_listener_report(station_id: int | str) -> dict:
    """Get listener statistics overview."""
    try:
        # Get current now playing for live stats
        np = get_now_playing(station_id)
        listeners = get_listeners(station_id)
        return {
            "current_listeners": np.get("listeners", 0),
            "listener_details": listeners,
            "is_online": np.get("is_online", False),
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== Skip/Control ====================

def skip_song(station_id: int | str) -> dict:
    """Skip to the next song (requires backend restart or specific endpoint)."""
    # Note: AzuraCast may require specific permissions or methods for this
    try:
        return _post(f"/api/station/{station_id}/backend/skip")
    except Exception:
        # Try alternative method
        try:
            return _post(f"/api/station/{station_id}/restart")
        except Exception as e:
            return {"error": str(e), "message": "Skip may require manual intervention"}


# ==================== Connection Check ====================

def is_connected() -> bool:
    """Check if AzuraCast API is accessible."""
    if not API_KEY:
        return False
    try:
        resp = requests.get(f"{BASE_URL}/api/nowplaying", headers=_headers(), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_radio_summary() -> dict:
    """Get a quick summary of radio status."""
    try:
        stations = list_stations()
        now_playing = get_now_playing()

        return {
            "connected": True,
            "station_count": len(stations),
            "stations": [{
                "name": np.get("station_name"),
                "listeners": np.get("listeners"),
                "current_song": f"{np.get('current_song', {}).get('artist')} - {np.get('current_song', {}).get('title')}",
                "is_online": np.get("is_online"),
            } for np in now_playing],
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


# ==================== Admin: Station Management ====================

def admin_list_stations() -> list[dict]:
    """List all stations with admin details."""
    data = _get("/api/admin/stations")
    return [{
        "id": s.get("id"),
        "name": s.get("name"),
        "shortcode": s.get("short_name") or s.get("shortcode"),
        "description": s.get("description"),
        "is_enabled": s.get("is_enabled"),
        "frontend_type": s.get("frontend_type"),
        "backend_type": s.get("backend_type"),
        "media_storage_location": s.get("media_storage_location"),
    } for s in data]


def admin_create_station(name: str, shortcode: str = None, description: str = "",
                         is_enabled: bool = True, frontend_type: str = "icecast",
                         backend_type: str = "liquidsoap") -> dict:
    """Create a new radio station."""
    body = {
        "name": name,
        "short_name": shortcode or name.lower().replace(" ", "_"),
        "description": description,
        "is_enabled": is_enabled,
        "frontend_type": frontend_type,
        "backend_type": backend_type,
        "enable_requests": True,
        "enable_streamers": True,
        "enable_public_page": True,
        "enable_on_demand": False,
    }
    result = _post("/api/admin/stations", body)
    # Clear station cache to pick up new station
    global _station_cache
    _station_cache = {}
    return result


def admin_update_station(station_id: int, **kwargs) -> dict:
    """Update station settings. Accepts: name, description, is_enabled, etc."""
    # Get current station data first
    current = _get(f"/api/admin/station/{station_id}")
    # Merge updates
    for key, value in kwargs.items():
        if key == "shortcode":
            current["short_name"] = value
        else:
            current[key] = value
    result = _put(f"/api/admin/station/{station_id}", current)
    # Clear cache
    global _station_cache
    _station_cache = {}
    return result


def admin_delete_station(station_id: int) -> dict:
    """Delete a station permanently."""
    result = _delete(f"/api/admin/station/{station_id}")
    global _station_cache
    _station_cache = {}
    return result


def admin_clone_station(station_id: int, name: str, shortcode: str = None) -> dict:
    """Clone an existing station with a new name."""
    body = {
        "name": name,
        "short_name": shortcode or name.lower().replace(" ", "_"),
        "clone_media": True,
        "clone_playlists": True,
        "clone_streamers": True,
        "clone_permissions": True,
        "clone_webhooks": True,
    }
    return _post(f"/api/admin/station/{station_id}/clone", body)


# ==================== Admin: User Management ====================

def admin_list_users() -> list[dict]:
    """List all AzuraCast users."""
    data = _get("/api/admin/users")
    return [{
        "id": u.get("id"),
        "email": u.get("email"),
        "name": u.get("name"),
        "is_enabled": not u.get("is_disabled", False),
        "created_at": u.get("created_at"),
        "roles": [r.get("name") for r in u.get("roles", [])],
    } for u in data]


def admin_get_user(user_id: int) -> dict:
    """Get details for a specific user."""
    return _get(f"/api/admin/user/{user_id}")


def admin_create_user(email: str, name: str, password: str, roles: list[int] = None) -> dict:
    """Create a new AzuraCast user account."""
    body = {
        "email": email,
        "name": name,
        "auth_password": password,
        "roles": roles or [],
    }
    return _post("/api/admin/users", body)


def admin_update_user(user_id: int, **kwargs) -> dict:
    """Update user details. Accepts: email, name, auth_password, roles, is_disabled."""
    current = _get(f"/api/admin/user/{user_id}")
    for key, value in kwargs.items():
        if key == "password":
            current["auth_password"] = value
        elif key == "is_enabled":
            current["is_disabled"] = not value
        else:
            current[key] = value
    return _put(f"/api/admin/user/{user_id}", current)


def admin_delete_user(user_id: int) -> dict:
    """Delete a user account."""
    return _delete(f"/api/admin/user/{user_id}")


def admin_list_roles() -> list[dict]:
    """List all roles/permissions."""
    data = _get("/api/admin/roles")
    return [{
        "id": r.get("id"),
        "name": r.get("name"),
        "permissions": r.get("permissions", {}).get("global", []),
    } for r in data]


def admin_create_role(name: str, permissions: list[str] = None) -> dict:
    """Create a new role with specified permissions."""
    body = {
        "name": name,
        "permissions": {
            "global": permissions or [],
            "station": {},
        },
    }
    return _post("/api/admin/roles", body)


# ==================== Admin: Storage Locations ====================

def admin_list_storage_locations() -> list[dict]:
    """List all storage locations."""
    data = _get("/api/admin/storage_locations")
    return [{
        "id": s.get("id"),
        "type": s.get("type"),
        "adapter": s.get("adapter"),
        "path": s.get("path"),
        "storage_quota": s.get("storage_quota"),
        "storage_used": s.get("storage_used"),
        "storage_available": s.get("storage_available"),
        "is_full": s.get("is_full", False),
        "stations": [st.get("name") for st in s.get("stations", [])],
    } for s in data]


def admin_get_storage_location(location_id: int) -> dict:
    """Get storage location details."""
    return _get(f"/api/admin/storage_location/{location_id}")


def admin_update_storage_quota(location_id: int, quota_bytes: int) -> dict:
    """Update storage quota for a location. Set to 0 for unlimited."""
    current = _get(f"/api/admin/storage_location/{location_id}")
    current["storage_quota"] = quota_bytes
    return _put(f"/api/admin/storage_location/{location_id}", current)


# ==================== Station: Storage Quota ====================

def get_station_quota(station_id: int | str) -> dict:
    """Get storage quota usage for a station."""
    try:
        data = _get(f"/api/station/{station_id}/quota")
        return {
            "media": data.get("station_media", {}),
            "recordings": data.get("station_recordings", {}),
            "podcasts": data.get("station_podcasts", {}),
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== DJ/Streamer Management (Expanded) ====================

def create_streamer(station_id: int | str, username: str, password: str,
                    display_name: str = None, is_active: bool = True) -> dict:
    """Create a new DJ/streamer account for a station."""
    body = {
        "streamer_username": username,
        "streamer_password": password,
        "display_name": display_name or username,
        "is_active": is_active,
        "enforce_schedule": False,
    }
    return _post(f"/api/station/{station_id}/streamers", body)


def get_streamer(station_id: int | str, streamer_id: int) -> dict:
    """Get streamer details."""
    return _get(f"/api/station/{station_id}/streamer/{streamer_id}")


def update_streamer(station_id: int | str, streamer_id: int, **kwargs) -> dict:
    """Update streamer. Accepts: username, password, display_name, is_active."""
    current = _get(f"/api/station/{station_id}/streamer/{streamer_id}")
    for key, value in kwargs.items():
        if key == "username":
            current["streamer_username"] = value
        elif key == "password":
            current["streamer_password"] = value
        else:
            current[key] = value
    return _put(f"/api/station/{station_id}/streamer/{streamer_id}", current)


def delete_streamer(station_id: int | str, streamer_id: int) -> dict:
    """Delete a DJ/streamer account."""
    return _delete(f"/api/station/{station_id}/streamer/{streamer_id}")


# ==================== Media Upload & Management ====================

def upload_media(station_id: int | str, file_path: str, folder: str = "") -> dict:
    """Upload a media file to the station library."""
    import os
    filename = os.path.basename(file_path)
    target_path = f"{folder}/{filename}" if folder else filename

    with open(file_path, 'rb') as f:
        files = {'file': (filename, f)}
        data = {'path': target_path}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(
            f"{BASE_URL}/api/station/{station_id}/files",
            headers=headers,
            data=data,
            files=files,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()


def update_media(station_id: int | str, media_id: int, **kwargs) -> dict:
    """Update media metadata. Accepts: artist, title, album, genre, lyrics, etc."""
    current = _get(f"/api/station/{station_id}/file/{media_id}")
    for key, value in kwargs.items():
        current[key] = value
    return _put(f"/api/station/{station_id}/file/{media_id}", current)


def create_media_folder(station_id: int | str, folder_path: str) -> dict:
    """Create a folder in the media library."""
    body = {"path": folder_path}
    return _post(f"/api/station/{station_id}/files/mkdir", body)


def move_media(station_id: int | str, file_path: str, new_path: str) -> dict:
    """Move/rename a media file or folder."""
    body = {"from": file_path, "to": new_path}
    return _put(f"/api/station/{station_id}/files/rename", body)


def batch_media_operation(station_id: int | str, operation: str, files: list[str],
                          destination: str = None) -> dict:
    """Batch operation on media files. Operations: move, copy, delete."""
    body = {
        "do": operation,
        "files": files,
    }
    if destination:
        body["to"] = destination
    return _put(f"/api/station/{station_id}/files/batch", body)


# ==================== Song Requests ====================

def submit_request(station_id: int | str, request_id: str) -> dict:
    """Submit a song request by its request ID."""
    return _post(f"/api/station/{station_id}/request/{request_id}")


def search_requestable_songs(station_id: int | str, query: str) -> list[dict]:
    """Search for requestable songs."""
    try:
        data = _get(f"/api/station/{station_id}/requests", {"searchPhrase": query})
        return [{
            "request_id": s.get("request_id"),
            "artist": s.get("song", {}).get("artist"),
            "title": s.get("song", {}).get("title"),
            "album": s.get("song", {}).get("album"),
        } for s in data]
    except Exception:
        return []


# ==================== Mount Points ====================

def list_mounts(station_id: int | str) -> list[dict]:
    """List all mount points/streams for a station."""
    data = _get(f"/api/station/{station_id}/mounts")
    return [{
        "id": m.get("id"),
        "name": m.get("name"),
        "display_name": m.get("display_name"),
        "url": m.get("url"),
        "is_default": m.get("is_default"),
        "is_public": m.get("is_public"),
        "max_listener_duration": m.get("max_listener_duration"),
        "fallback_mount": m.get("fallback_mount"),
        "format": m.get("autodj_format"),
        "bitrate": m.get("autodj_bitrate"),
    } for m in data]


def get_mount(station_id: int | str, mount_id: int) -> dict:
    """Get mount point details."""
    return _get(f"/api/station/{station_id}/mount/{mount_id}")


def create_mount(station_id: int | str, name: str, display_name: str = None,
                 is_default: bool = False, autodj_format: str = "mp3",
                 autodj_bitrate: int = 128) -> dict:
    """Create a new mount point/stream."""
    body = {
        "name": name,
        "display_name": display_name or name,
        "is_default": is_default,
        "is_public": True,
        "autodj_format": autodj_format,
        "autodj_bitrate": autodj_bitrate,
        "enable_autodj": True,
    }
    return _post(f"/api/station/{station_id}/mounts", body)


def update_mount(station_id: int | str, mount_id: int, **kwargs) -> dict:
    """Update mount point settings."""
    current = _get(f"/api/station/{station_id}/mount/{mount_id}")
    for key, value in kwargs.items():
        current[key] = value
    return _put(f"/api/station/{station_id}/mount/{mount_id}", current)


def delete_mount(station_id: int | str, mount_id: int) -> dict:
    """Delete a mount point."""
    return _delete(f"/api/station/{station_id}/mount/{mount_id}")


# ==================== HLS Streams ====================

def list_hls_streams(station_id: int | str) -> list[dict]:
    """List HLS streams for a station."""
    try:
        data = _get(f"/api/station/{station_id}/hls_streams")
        return [{
            "id": h.get("id"),
            "name": h.get("name"),
            "format": h.get("format"),
            "bitrate": h.get("bitrate"),
        } for h in data]
    except Exception:
        return []


def create_hls_stream(station_id: int | str, name: str, format: str = "aac",
                      bitrate: int = 128) -> dict:
    """Create a new HLS stream."""
    body = {
        "name": name,
        "format": format,
        "bitrate": bitrate,
    }
    return _post(f"/api/station/{station_id}/hls_streams", body)


# ==================== Webhooks ====================

def list_webhooks(station_id: int | str) -> list[dict]:
    """List webhooks for a station."""
    try:
        data = _get(f"/api/station/{station_id}/webhooks")
        return [{
            "id": w.get("id"),
            "name": w.get("name"),
            "type": w.get("type"),
            "is_enabled": w.get("is_enabled"),
            "triggers": w.get("triggers", []),
        } for w in data]
    except Exception:
        return []


def create_webhook(station_id: int | str, name: str, type: str, config: dict,
                   triggers: list[str] = None) -> dict:
    """Create a webhook. Types: discord, telegram, twitter, tunein, etc."""
    body = {
        "name": name,
        "type": type,
        "is_enabled": True,
        "triggers": triggers or ["song_changed"],
        "config": config,
    }
    return _post(f"/api/station/{station_id}/webhooks", body)


def toggle_webhook(station_id: int | str, webhook_id: int) -> dict:
    """Toggle a webhook on/off."""
    current = _get(f"/api/station/{station_id}/webhook/{webhook_id}")
    current["is_enabled"] = not current.get("is_enabled", False)
    return _put(f"/api/station/{station_id}/webhook/{webhook_id}", current)


def delete_webhook(station_id: int | str, webhook_id: int) -> dict:
    """Delete a webhook."""
    return _delete(f"/api/station/{station_id}/webhook/{webhook_id}")


# ==================== Fallback Track ====================

def get_fallback_track(station_id: int | str) -> dict:
    """Get the station's fallback track info."""
    try:
        return _get(f"/api/station/{station_id}/fallback")
    except Exception:
        return {"error": "No fallback track set"}


def set_fallback_track(station_id: int | str, file_path: str) -> dict:
    """Set or update the fallback track from a local file."""
    import os
    filename = os.path.basename(file_path)
    with open(file_path, 'rb') as f:
        files = {'file': (filename, f)}
        headers = {"X-API-Key": API_KEY}
        resp = requests.post(
            f"{BASE_URL}/api/station/{station_id}/fallback",
            headers=headers,
            files=files,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {"success": True}


def delete_fallback_track(station_id: int | str) -> dict:
    """Remove the fallback track."""
    return _delete(f"/api/station/{station_id}/fallback")


# ==================== Station Playback History ====================

def get_station_history(station_id: int | str, start: str = None, end: str = None) -> list[dict]:
    """Get detailed playback history. Dates in ISO format (YYYY-MM-DD)."""
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    try:
        data = _get(f"/api/station/{station_id}/history", params)
        return [{
            "played_at": h.get("played_at"),
            "artist": h.get("song", {}).get("artist"),
            "title": h.get("song", {}).get("title"),
            "playlist": h.get("playlist"),
            "streamer": h.get("streamer"),
            "is_request": h.get("is_request", False),
            "listeners_start": h.get("listeners_start"),
            "listeners_end": h.get("listeners_end"),
        } for h in data]
    except Exception:
        return []


# ==================== System Status ====================

def get_system_status() -> dict:
    """Get overall system status and server stats."""
    try:
        stats = _get("/api/admin/server/stats")
        return {
            "cpu_total": stats.get("cpu", {}).get("total"),
            "memory_used": stats.get("memory", {}).get("used"),
            "memory_total": stats.get("memory", {}).get("total"),
            "disk_used": stats.get("disk", {}).get("used"),
            "disk_total": stats.get("disk", {}).get("total"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_services_status() -> list[dict]:
    """Get status of all system services."""
    try:
        data = _get("/api/admin/services")
        return [{
            "name": s.get("name"),
            "description": s.get("description"),
            "running": s.get("running"),
        } for s in data]
    except Exception:
        return []


def restart_service(service_name: str) -> dict:
    """Restart a specific service (e.g., 'nginx', 'mariadb')."""
    return _post(f"/api/admin/services/restart/{service_name}")
