"""Home Assistant API client for smart home control.

Controls lights, switches, media players, climate, and other devices
through the Home Assistant REST API.
"""

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")

logger = logging.getLogger(__name__)

HA_URL = os.getenv("HA_URL", "https://home.groundrushlabs.com")
HA_TOKEN = os.getenv("HA_API_TOKEN", "")


def _headers() -> dict:
    """Get API headers."""
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _api_get(endpoint: str) -> dict | list:
    """Make GET request to Home Assistant API."""
    try:
        resp = requests.get(f"{HA_URL}/api/{endpoint}", headers=_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Home Assistant API error: {e}")
        return {"error": str(e)}


def _api_post(endpoint: str, data: dict = None) -> dict:
    """Make POST request to Home Assistant API."""
    try:
        resp = requests.post(
            f"{HA_URL}/api/{endpoint}",
            headers=_headers(),
            json=data or {},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {"success": True}
    except Exception as e:
        logger.error(f"Home Assistant API error: {e}")
        return {"error": str(e)}


def is_configured() -> bool:
    """Check if Home Assistant is configured."""
    return bool(HA_TOKEN)


def get_status() -> dict:
    """Get Home Assistant status and basic info."""
    if not is_configured():
        return {"error": "Home Assistant not configured - missing HA_API_TOKEN"}

    config = _api_get("config")
    if "error" in config:
        return config

    return {
        "url": HA_URL,
        "location_name": config.get("location_name", "Home"),
        "version": config.get("version", "unknown"),
        "unit_system": config.get("unit_system", {}),
        "time_zone": config.get("time_zone", ""),
    }


def get_all_states() -> list[dict]:
    """Get all entity states."""
    return _api_get("states")


def get_state(entity_id: str) -> dict:
    """Get state of a specific entity."""
    return _api_get(f"states/{entity_id}")


def get_entities_by_domain(domain: str) -> list[dict]:
    """Get all entities for a domain (light, switch, media_player, etc.)."""
    states = get_all_states()
    if isinstance(states, dict) and "error" in states:
        return states

    return [s for s in states if s["entity_id"].startswith(f"{domain}.")]


def list_devices() -> dict:
    """List all devices grouped by type."""
    states = get_all_states()
    if isinstance(states, dict) and "error" in states:
        return states

    devices = {
        "lights": [],
        "switches": [],
        "media_players": [],
        "climate": [],
        "sensors": [],
        "binary_sensors": [],
        "other": [],
    }

    for state in states:
        entity_id = state["entity_id"]
        domain = entity_id.split(".")[0]
        name = state.get("attributes", {}).get("friendly_name", entity_id)

        entry = {
            "entity_id": entity_id,
            "name": name,
            "state": state["state"],
        }

        if domain == "light":
            devices["lights"].append(entry)
        elif domain == "switch":
            devices["switches"].append(entry)
        elif domain == "media_player":
            devices["media_players"].append(entry)
        elif domain == "climate":
            devices["climate"].append(entry)
        elif domain == "sensor":
            # Only include interesting sensors
            if any(x in entity_id for x in ["temperature", "humidity", "battery", "power", "energy"]):
                devices["sensors"].append(entry)
        elif domain == "binary_sensor":
            if any(x in entity_id for x in ["door", "window", "motion", "presence"]):
                devices["binary_sensors"].append(entry)

    # Remove empty categories
    return {k: v for k, v in devices.items() if v}


# ==================== Service Calls ====================

def call_service(domain: str, service: str, data: dict = None) -> dict:
    """Call a Home Assistant service."""
    return _api_post(f"services/{domain}/{service}", data)


def turn_on(entity_id: str, **kwargs) -> dict:
    """Turn on a device."""
    domain = entity_id.split(".")[0]
    data = {"entity_id": entity_id, **kwargs}
    return call_service(domain, "turn_on", data)


def turn_off(entity_id: str) -> dict:
    """Turn off a device."""
    domain = entity_id.split(".")[0]
    return call_service(domain, "turn_off", {"entity_id": entity_id})


def toggle(entity_id: str) -> dict:
    """Toggle a device."""
    domain = entity_id.split(".")[0]
    return call_service(domain, "toggle", {"entity_id": entity_id})


# ==================== Lights ====================

def set_light(entity_id: str, brightness: int = None, color: str = None, temperature: int = None) -> dict:
    """Control a light.

    Args:
        entity_id: Light entity ID
        brightness: 0-255
        color: Color name or hex (e.g., "red", "#FF0000")
        temperature: Color temperature in mireds (lower = cooler)
    """
    data = {"entity_id": entity_id}

    if brightness is not None:
        data["brightness"] = min(255, max(0, brightness))

    if color:
        if color.startswith("#"):
            # Convert hex to RGB
            hex_color = color.lstrip("#")
            data["rgb_color"] = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        else:
            data["color_name"] = color

    if temperature is not None:
        data["color_temp"] = temperature

    return call_service("light", "turn_on", data)


def get_lights() -> list[dict]:
    """Get all lights with their states."""
    return get_entities_by_domain("light")


# ==================== Climate ====================

def set_thermostat(entity_id: str, temperature: float = None, hvac_mode: str = None) -> dict:
    """Control a thermostat.

    Args:
        entity_id: Climate entity ID
        temperature: Target temperature
        hvac_mode: 'heat', 'cool', 'heat_cool', 'off', 'auto'
    """
    if hvac_mode:
        call_service("climate", "set_hvac_mode", {
            "entity_id": entity_id,
            "hvac_mode": hvac_mode
        })

    if temperature:
        return call_service("climate", "set_temperature", {
            "entity_id": entity_id,
            "temperature": temperature
        })

    return {"success": True}


def get_climate() -> list[dict]:
    """Get all climate devices."""
    return get_entities_by_domain("climate")


# ==================== Media Players ====================

def media_play(entity_id: str) -> dict:
    """Play media."""
    return call_service("media_player", "media_play", {"entity_id": entity_id})


def media_pause(entity_id: str) -> dict:
    """Pause media."""
    return call_service("media_player", "media_pause", {"entity_id": entity_id})


def media_stop(entity_id: str) -> dict:
    """Stop media."""
    return call_service("media_player", "media_stop", {"entity_id": entity_id})


def media_next(entity_id: str) -> dict:
    """Next track."""
    return call_service("media_player", "media_next_track", {"entity_id": entity_id})


def media_previous(entity_id: str) -> dict:
    """Previous track."""
    return call_service("media_player", "media_previous_track", {"entity_id": entity_id})


def set_volume(entity_id: str, volume: float) -> dict:
    """Set volume (0.0 - 1.0)."""
    return call_service("media_player", "volume_set", {
        "entity_id": entity_id,
        "volume_level": min(1.0, max(0.0, volume))
    })


def play_media(entity_id: str, media_url: str, media_type: str = "music") -> dict:
    """Play media from URL."""
    return call_service("media_player", "play_media", {
        "entity_id": entity_id,
        "media_content_id": media_url,
        "media_content_type": media_type
    })


def get_media_players() -> list[dict]:
    """Get all media players."""
    players = get_entities_by_domain("media_player")
    if isinstance(players, dict) and "error" in players:
        return players

    result = []
    for p in players:
        attrs = p.get("attributes", {})
        result.append({
            "entity_id": p["entity_id"],
            "name": attrs.get("friendly_name", p["entity_id"]),
            "state": p["state"],
            "volume": attrs.get("volume_level"),
            "media_title": attrs.get("media_title"),
            "media_artist": attrs.get("media_artist"),
            "source": attrs.get("source"),
        })
    return result


# ==================== Scenes & Scripts ====================

def activate_scene(scene_id: str) -> dict:
    """Activate a scene."""
    entity_id = scene_id if scene_id.startswith("scene.") else f"scene.{scene_id}"
    return call_service("scene", "turn_on", {"entity_id": entity_id})


def run_script(script_id: str) -> dict:
    """Run a script."""
    entity_id = script_id if script_id.startswith("script.") else f"script.{script_id}"
    return call_service("script", "turn_on", {"entity_id": entity_id})


def get_scenes() -> list[dict]:
    """Get all scenes."""
    return get_entities_by_domain("scene")


def get_scripts() -> list[dict]:
    """Get all scripts."""
    return get_entities_by_domain("script")


# ==================== Weather ====================

def get_weather() -> dict:
    """Get weather forecast."""
    states = get_entities_by_domain("weather")
    if isinstance(states, dict) and "error" in states:
        return states

    if not states:
        return {"error": "No weather entity found"}

    weather = states[0]
    attrs = weather.get("attributes", {})

    return {
        "condition": weather["state"],
        "temperature": attrs.get("temperature"),
        "humidity": attrs.get("humidity"),
        "wind_speed": attrs.get("wind_speed"),
        "forecast": attrs.get("forecast", [])[:5],  # Next 5 periods
    }


# ==================== Convenience Functions ====================

def find_entity(name: str) -> str | None:
    """Find an entity by friendly name (case-insensitive partial match)."""
    states = get_all_states()
    if isinstance(states, dict) and "error" in states:
        return None

    name_lower = name.lower()

    for state in states:
        entity_id = state["entity_id"]
        friendly_name = state.get("attributes", {}).get("friendly_name", "").lower()

        if name_lower in friendly_name or name_lower in entity_id.lower():
            return entity_id

    return None


def control_by_name(name: str, action: str, **kwargs) -> dict:
    """Control a device by its friendly name.

    Args:
        name: Device name (e.g., "kitchen lights", "living room")
        action: 'on', 'off', 'toggle'
        **kwargs: Additional parameters (brightness, temperature, etc.)
    """
    entity_id = find_entity(name)
    if not entity_id:
        return {"error": f"Device '{name}' not found"}

    if action == "on":
        return turn_on(entity_id, **kwargs)
    elif action == "off":
        return turn_off(entity_id)
    elif action == "toggle":
        return toggle(entity_id)
    else:
        return {"error": f"Unknown action: {action}"}
