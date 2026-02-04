"""Google Calendar integration - read, create, and manage events."""

import logging
from datetime import datetime, timedelta, timezone

from googleapiclient.discovery import build

from core.security.google_oauth import get_credentials

logger = logging.getLogger(__name__)


def _get_service():
    creds = get_credentials()
    if not creds:
        raise RuntimeError("Google not connected. Visit /auth/google to authorize.")
    return build("calendar", "v3", credentials=creds)


def get_upcoming_events(max_results: int = 10, days_ahead: int = 7) -> list[dict]:
    """Get upcoming events from primary calendar."""
    service = _get_service()
    now = datetime.now(timezone.utc).isoformat()
    end = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()

    results = service.events().list(
        calendarId="primary",
        timeMin=now,
        timeMax=end,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = []
    for event in results.get("items", []):
        start = event.get("start", {})
        end_time = event.get("end", {})
        events.append({
            "id": event["id"],
            "summary": event.get("summary", "No title"),
            "start": start.get("dateTime", start.get("date", "")),
            "end": end_time.get("dateTime", end_time.get("date", "")),
            "location": event.get("location", ""),
            "description": event.get("description", ""),
            "status": event.get("status", ""),
            "attendees": [a.get("email") for a in event.get("attendees", [])],
        })

    return events


def get_today_events() -> list[dict]:
    """Get today's events."""
    return get_upcoming_events(max_results=20, days_ahead=1)


def create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    location: str = "",
    attendees: list[str] | None = None,
) -> dict:
    """Create a calendar event.

    Args:
        summary: Event title
        start_time: ISO format datetime (e.g. 2026-01-28T10:00:00-05:00)
        end_time: ISO format datetime
        description: Event description
        location: Event location
        attendees: List of email addresses
    """
    service = _get_service()

    event_body = {
        "summary": summary,
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }

    if description:
        event_body["description"] = description
    if location:
        event_body["location"] = location
    if attendees:
        event_body["attendees"] = [{"email": e} for e in attendees]

    event = service.events().insert(
        calendarId="primary", body=event_body
    ).execute()

    logger.info(f"Event created: {summary} at {start_time}")
    return {
        "id": event["id"],
        "summary": event.get("summary"),
        "link": event.get("htmlLink"),
        "status": "created",
    }


def update_event(
    event_id: str,
    summary: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
) -> dict:
    """Update an existing calendar event.

    Args:
        event_id: The event ID to update
        summary: New event title (optional)
        start_time: New start time in ISO format (optional)
        end_time: New end time in ISO format (optional)
        description: New description (optional)
        location: New location (optional)
        attendees: New list of attendee emails (optional, replaces existing)
    """
    service = _get_service()

    # Get existing event
    event = service.events().get(calendarId="primary", eventId=event_id).execute()

    # Update fields if provided
    if summary is not None:
        event["summary"] = summary
    if start_time is not None:
        event["start"] = {"dateTime": start_time}
    if end_time is not None:
        event["end"] = {"dateTime": end_time}
    if description is not None:
        event["description"] = description
    if location is not None:
        event["location"] = location
    if attendees is not None:
        event["attendees"] = [{"email": e} for e in attendees]

    updated = service.events().update(
        calendarId="primary", eventId=event_id, body=event
    ).execute()

    logger.info(f"Event updated: {event_id}")
    return {
        "id": updated["id"],
        "summary": updated.get("summary"),
        "link": updated.get("htmlLink"),
        "status": "updated",
    }


def delete_event(event_id: str) -> dict:
    """Delete a calendar event.

    Args:
        event_id: The event ID to delete
    """
    service = _get_service()

    # Get event details before deleting for confirmation
    try:
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        summary = event.get("summary", "Unknown")
    except Exception:
        summary = "Unknown"

    service.events().delete(calendarId="primary", eventId=event_id).execute()

    logger.info(f"Event deleted: {event_id}")
    return {
        "id": event_id,
        "summary": summary,
        "status": "deleted",
    }


def add_attendees(event_id: str, attendees: list[str]) -> dict:
    """Add attendees to an existing calendar event.

    Args:
        event_id: The event ID
        attendees: List of email addresses to add
    """
    service = _get_service()

    # Get existing event
    event = service.events().get(calendarId="primary", eventId=event_id).execute()

    # Merge existing and new attendees
    existing = [a.get("email") for a in event.get("attendees", [])]
    all_attendees = list(set(existing + attendees))
    event["attendees"] = [{"email": e} for e in all_attendees]

    updated = service.events().update(
        calendarId="primary", eventId=event_id, body=event, sendUpdates="all"
    ).execute()

    logger.info(f"Added attendees to event {event_id}: {attendees}")
    return {
        "id": updated["id"],
        "summary": updated.get("summary"),
        "attendees": [a.get("email") for a in updated.get("attendees", [])],
        "status": "attendees_added",
    }


def find_free_time(date: str, duration_minutes: int = 60) -> list[dict]:
    """Find available time slots on a given date.

    Args:
        date: Date string (YYYY-MM-DD)
        duration_minutes: Desired meeting duration
    """
    service = _get_service()

    # Business hours: 9 AM to 6 PM
    day_start = f"{date}T09:00:00"
    day_end = f"{date}T18:00:00"

    events = service.events().list(
        calendarId="primary",
        timeMin=day_start + "-05:00",
        timeMax=day_end + "-05:00",
        singleEvents=True,
        orderBy="startTime",
    ).execute().get("items", [])

    # Find gaps
    busy_times = []
    for event in events:
        start = event["start"].get("dateTime", "")
        end = event["end"].get("dateTime", "")
        if start and end:
            busy_times.append((start, end))

    free_slots = []
    current = day_start + "-05:00"
    for busy_start, busy_end in busy_times:
        if current < busy_start:
            free_slots.append({"start": current, "end": busy_start})
        current = max(current, busy_end)
    if current < day_end + "-05:00":
        free_slots.append({"start": current, "end": day_end + "-05:00"})

    return free_slots
