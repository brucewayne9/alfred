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
