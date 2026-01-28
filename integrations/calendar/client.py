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
