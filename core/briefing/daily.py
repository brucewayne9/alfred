"""Daily briefing system - aggregates information for personalized morning briefings.

This module collects data from various sources (calendar, email, CRM, weather, news)
and generates a structured briefing that can be presented via voice or text.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BriefingSection:
    """A section of the daily briefing."""
    title: str
    content: str
    priority: int = 5  # 1 = highest, 10 = lowest
    source: str = ""
    items: list[dict] = field(default_factory=list)


@dataclass
class DailyBriefing:
    """Complete daily briefing with all sections."""
    date: str = ""
    greeting: str = ""
    sections: list[BriefingSection] = field(default_factory=list)
    generated_at: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "greeting": self.greeting,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "priority": s.priority,
                    "source": s.source,
                    "items": s.items,
                }
                for s in sorted(self.sections, key=lambda x: x.priority)
            ],
            "generated_at": self.generated_at,
            "summary": self.summary,
        }

    def to_text(self) -> str:
        """Generate a text version for TTS or display."""
        lines = [self.greeting, ""]

        for section in sorted(self.sections, key=lambda x: x.priority):
            lines.append(f"**{section.title}**")
            lines.append(section.content)
            lines.append("")

        if self.summary:
            lines.append(self.summary)

        return "\n".join(lines)


async def get_calendar_briefing() -> BriefingSection | None:
    """Get today's calendar events."""
    try:
        from core.tools.definitions import today_schedule

        result = await asyncio.get_event_loop().run_in_executor(None, today_schedule)

        if "error" in result:
            return None

        events = result.get("events", [])
        if not events:
            return BriefingSection(
                title="Calendar",
                content="You have no scheduled events today.",
                priority=2,
                source="google_calendar",
            )

        # Format events
        event_lines = []
        for event in events[:5]:  # Limit to 5 events
            time_str = event.get("start", {}).get("dateTime", "")[:16].split("T")[-1] if "dateTime" in event.get("start", {}) else "All day"
            event_lines.append(f"- {time_str}: {event.get('summary', 'Untitled')}")

        return BriefingSection(
            title="Calendar",
            content=f"You have {len(events)} event{'s' if len(events) != 1 else ''} today:\n" + "\n".join(event_lines),
            priority=2,
            source="google_calendar",
            items=events[:5],
        )
    except Exception as e:
        logger.warning(f"Calendar briefing failed: {e}")
        return None


async def get_email_briefing() -> BriefingSection | None:
    """Get email summary."""
    try:
        from core.tools.definitions import unread_email_count, check_email

        # Get unread count
        count_result = await asyncio.get_event_loop().run_in_executor(None, unread_email_count)
        unread = count_result.get("count", 0)

        if unread == 0:
            return BriefingSection(
                title="Email",
                content="Your inbox is clear. No unread emails.",
                priority=3,
                source="gmail",
            )

        # Get recent emails
        emails_result = await asyncio.get_event_loop().run_in_executor(None, check_email)
        emails = emails_result.get("emails", [])[:3]

        email_lines = []
        for email in emails:
            sender = email.get("from", "Unknown")
            subject = email.get("subject", "No subject")[:50]
            email_lines.append(f"- From {sender}: {subject}")

        content = f"You have {unread} unread email{'s' if unread != 1 else ''}."
        if email_lines:
            content += "\n\nRecent messages:\n" + "\n".join(email_lines)

        return BriefingSection(
            title="Email",
            content=content,
            priority=3,
            source="gmail",
            items=emails,
        )
    except Exception as e:
        logger.warning(f"Email briefing failed: {e}")
        return None


async def get_crm_briefing() -> BriefingSection | None:
    """Get CRM summary - tasks due, opportunities, etc."""
    try:
        from core.tools.definitions import crm_list_tasks, crm_pipeline_summary

        # Get tasks due today
        tasks_result = await asyncio.get_event_loop().run_in_executor(None, crm_list_tasks)
        tasks = tasks_result.get("tasks", [])

        # Filter for due today or overdue
        today = datetime.now(timezone.utc).date()
        urgent_tasks = []
        for task in tasks:
            if task.get("dueAt"):
                try:
                    due_date = datetime.fromisoformat(task["dueAt"].replace("Z", "+00:00")).date()
                    if due_date <= today:
                        urgent_tasks.append(task)
                except (ValueError, TypeError):
                    pass

        # Get pipeline summary
        pipeline = await asyncio.get_event_loop().run_in_executor(None, crm_pipeline_summary)

        content_parts = []

        if urgent_tasks:
            task_lines = [f"- {t.get('title', 'Task')}" for t in urgent_tasks[:3]]
            content_parts.append(f"You have {len(urgent_tasks)} task{'s' if len(urgent_tasks) != 1 else ''} due:\n" + "\n".join(task_lines))

        if pipeline.get("total_value"):
            content_parts.append(f"Pipeline value: ${pipeline.get('total_value', 0):,.0f} across {pipeline.get('total_deals', 0)} deals.")

        if not content_parts:
            return BriefingSection(
                title="CRM",
                content="No urgent CRM tasks or updates.",
                priority=4,
                source="twenty_crm",
            )

        return BriefingSection(
            title="CRM",
            content="\n\n".join(content_parts),
            priority=4,
            source="twenty_crm",
            items=urgent_tasks[:3],
        )
    except Exception as e:
        logger.warning(f"CRM briefing failed: {e}")
        return None


async def get_server_briefing() -> BriefingSection | None:
    """Get server status summary."""
    try:
        from integrations.servers.manager import list_servers, check_server_status

        servers = await asyncio.get_event_loop().run_in_executor(None, list_servers)
        if not servers:
            return None

        statuses = []
        issues = []

        for server in servers:
            try:
                status = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=server: check_server_status(s["name"])
                )
                if status.get("status") != "online":
                    issues.append(f"- {server['name']}: {status.get('status', 'unknown')}")
                statuses.append({
                    "name": server["name"],
                    "status": status.get("status", "unknown"),
                })
            except Exception:
                issues.append(f"- {server['name']}: unreachable")

        if issues:
            return BriefingSection(
                title="Servers",
                content=f"Server issues detected:\n" + "\n".join(issues),
                priority=1,  # High priority for server issues
                source="servers",
                items=statuses,
            )

        return BriefingSection(
            title="Servers",
            content=f"All {len(servers)} server{'s' if len(servers) != 1 else ''} online and healthy.",
            priority=6,
            source="servers",
            items=statuses,
        )
    except Exception as e:
        logger.warning(f"Server briefing failed: {e}")
        return None


async def get_stripe_briefing() -> BriefingSection | None:
    """Get Stripe revenue summary."""
    try:
        from core.tools.definitions import stripe_revenue_summary

        result = await asyncio.get_event_loop().run_in_executor(None, stripe_revenue_summary)

        if "error" in result:
            return None

        today_revenue = result.get("today", {}).get("amount", 0) / 100
        month_revenue = result.get("month", {}).get("amount", 0) / 100

        content = f"Today's revenue: ${today_revenue:,.2f}\nThis month: ${month_revenue:,.2f}"

        if result.get("pending_payouts"):
            pending = result["pending_payouts"].get("amount", 0) / 100
            content += f"\nPending payouts: ${pending:,.2f}"

        return BriefingSection(
            title="Revenue",
            content=content,
            priority=5,
            source="stripe",
            items=[result],
        )
    except Exception as e:
        logger.warning(f"Stripe briefing failed: {e}")
        return None


async def get_weather_briefing(location: str = "New York") -> BriefingSection | None:
    """Get weather forecast."""
    try:
        import httpx

        # Use Open-Meteo API (free, no key required)
        # First get coordinates
        async with httpx.AsyncClient() as client:
            geo_resp = await client.get(
                f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
            )
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return None

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            place = geo_data["results"][0].get("name", location)

            # Get weather
            weather_resp = await client.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,weather_code,wind_speed_10m"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                f"&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
            )
            weather = weather_resp.json()

            current = weather.get("current", {})
            daily = weather.get("daily", {})

            temp = current.get("temperature_2m", "?")
            high = daily.get("temperature_2m_max", [None])[0]
            low = daily.get("temperature_2m_min", [None])[0]
            precip = daily.get("precipitation_probability_max", [None])[0]

            weather_codes = {
                0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
                75: "heavy snow", 80: "rain showers", 81: "heavy showers", 95: "thunderstorm",
            }
            condition = weather_codes.get(current.get("weather_code", 0), "")

            content = f"Currently {temp}°F and {condition} in {place}."
            if high and low:
                content += f" High of {high}°F, low of {low}°F."
            if precip and precip > 20:
                content += f" {precip}% chance of precipitation."

            return BriefingSection(
                title="Weather",
                content=content,
                priority=7,
                source="open_meteo",
            )
    except Exception as e:
        logger.warning(f"Weather briefing failed: {e}")
        return None


def get_greeting() -> str:
    """Generate time-appropriate greeting."""
    hour = datetime.now().hour

    if hour < 12:
        time_greeting = "Good morning"
    elif hour < 17:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    today = datetime.now().strftime("%A, %B %d")
    return f"{time_greeting}! Here's your briefing for {today}."


async def generate_briefing(
    include_calendar: bool = True,
    include_email: bool = True,
    include_crm: bool = True,
    include_servers: bool = True,
    include_revenue: bool = True,
    include_weather: bool = True,
    weather_location: str = "Atlanta, GA",
) -> DailyBriefing:
    """Generate a complete daily briefing.

    Args:
        include_calendar: Include calendar events
        include_email: Include email summary
        include_crm: Include CRM tasks and pipeline
        include_servers: Include server status
        include_revenue: Include Stripe revenue
        include_weather: Include weather forecast
        weather_location: Location for weather (default Atlanta)

    Returns:
        DailyBriefing object with all sections
    """
    briefing = DailyBriefing(
        date=datetime.now().strftime("%Y-%m-%d"),
        greeting=get_greeting(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Gather all sections in parallel
    tasks = []

    if include_calendar:
        tasks.append(("calendar", get_calendar_briefing()))
    if include_email:
        tasks.append(("email", get_email_briefing()))
    if include_crm:
        tasks.append(("crm", get_crm_briefing()))
    if include_servers:
        tasks.append(("servers", get_server_briefing()))
    if include_revenue:
        tasks.append(("revenue", get_stripe_briefing()))
    if include_weather:
        tasks.append(("weather", get_weather_briefing(weather_location)))

    # Execute all tasks concurrently
    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    for (name, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.warning(f"Briefing section {name} failed: {result}")
            continue
        if result:
            briefing.sections.append(result)

    # Generate summary
    high_priority = [s for s in briefing.sections if s.priority <= 3]
    if high_priority:
        summaries = [f"{s.title}: {s.content.split('.')[0]}." for s in high_priority[:3]]
        briefing.summary = "Key highlights: " + " ".join(summaries)

    return briefing


async def generate_quick_briefing() -> str:
    """Generate a quick text briefing for voice output."""
    briefing = await generate_briefing()
    return briefing.to_text()
