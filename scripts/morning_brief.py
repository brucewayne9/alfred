#!/usr/bin/env python3
"""Alfred Morning Brief — Beautiful HTML newsletter delivered daily at 6:30 AM.

Sections:
1. Greeting + Date + Weather
2. Stocks (portfolio watchlist)
3. News: 3 Political, 3 Motivational/Business, 3 Tech stories
4. Server Performance (actual numbers)
5. Daily Task Checklist (from evening ping)
6. CRM Pipeline Quick Stats
7. Calendar Today

Cron: 30 6 * * * cd /home/aialfred/alfred && /home/aialfred/.pyenv/versions/3.11.11/bin/python3 scripts/morning_brief.py >> /tmp/morning_brief.log 2>&1
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

import httpx
import requests

from integrations.email.client import EmailClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("morning_brief")

TASK_FILE = Path("/home/aialfred/alfred/data/daily_tasks.json")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = "7582976864"
MIKE_EMAIL = "mjohnson@groundrushinc.com"

email_client = EmailClient()

# Stocks to track
WATCHLIST = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "SPY"]

# Server list
SERVERS = [
    {"name": "Alfred Labs", "host": "75.43.156.105", "port": 8400, "check": "/health"},
    {"name": "Alfred Claw", "host": "75.43.156.101", "port": 2222, "check": "ssh"},
    {"name": "Lonewolf (Dokploy)", "host": "75.43.156.117", "port": 443, "check": "https"},
    {"name": "Home Assistant", "host": "75.43.156.104", "port": 8123, "check": "http"},
]


# ── Data Fetchers ─────────────────────────────────────────────

async def fetch_weather():
    """Get Atlanta weather."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            geo = await client.get("https://geocoding-api.open-meteo.com/v1/search?name=Atlanta,GA&count=1")
            geo_data = geo.json()
            if not geo_data.get("results"):
                return None
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]

            w = await client.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                f"&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=America/New_York"
            )
            data = w.json()
            current = data.get("current", {})
            daily = data.get("daily", {})

            codes = {
                0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
                45: "Foggy", 48: "Foggy", 51: "Light Drizzle", 53: "Drizzle",
                61: "Light Rain", 63: "Rain", 65: "Heavy Rain", 80: "Showers",
                95: "Thunderstorm",
            }
            return {
                "temp": round(current.get("temperature_2m", 0)),
                "condition": codes.get(current.get("weather_code", 0), "Unknown"),
                "wind": round(current.get("wind_speed_10m", 0)),
                "humidity": current.get("relative_humidity_2m", 0),
                "high": round(daily.get("temperature_2m_max", [0])[0]),
                "low": round(daily.get("temperature_2m_min", [0])[0]),
                "precip": daily.get("precipitation_probability_max", [0])[0],
            }
    except Exception as e:
        log.error(f"Weather fetch failed: {e}")
        return None


async def fetch_stocks():
    """Get stock quotes using Yahoo Finance."""
    results = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for symbol in WATCHLIST:
                try:
                    resp = await client.get(
                        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                        params={"interval": "1d", "range": "2d"},
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    data = resp.json()
                    meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                    price = meta.get("regularMarketPrice", 0)
                    prev = meta.get("chartPreviousClose", 0) or meta.get("previousClose", 0)
                    change = price - prev if prev else 0
                    pct = (change / prev * 100) if prev else 0
                    results.append({
                        "symbol": symbol,
                        "price": round(price, 2),
                        "change": round(change, 2),
                        "pct": round(pct, 2),
                    })
                except Exception:
                    results.append({"symbol": symbol, "price": 0, "change": 0, "pct": 0})
    except Exception as e:
        log.error(f"Stock fetch failed: {e}")
    return results


async def fetch_news_rss(category: str, feed_url: str, count: int = 3):
    """Fetch news from RSS feed."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            root = ET.fromstring(resp.text)

            items = []
            for item in root.findall(".//item")[:count]:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                desc = item.findtext("description", "")
                # Clean HTML from description
                desc = re.sub(r"<[^>]+>", "", desc)[:150]
                if title:
                    items.append({"title": title, "link": link, "desc": desc})
            return items
    except Exception as e:
        log.error(f"RSS fetch failed for {category}: {e}")
        return []


async def fetch_all_news():
    """Fetch news from all categories."""
    feeds = {
        "political": "https://news.google.com/rss/search?q=politics+US&hl=en-US&gl=US&ceid=US:en",
        "motivational": "https://news.google.com/rss/search?q=entrepreneur+success+motivation&hl=en-US&gl=US&ceid=US:en",
        "tech": "https://news.google.com/rss/search?q=technology+AI+startup&hl=en-US&gl=US&ceid=US:en",
    }
    results = {}
    tasks = [fetch_news_rss(cat, url, 3) for cat, url in feeds.items()]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for (cat, _), result in zip(feeds.items(), fetched):
        results[cat] = result if not isinstance(result, Exception) else []
    return results


async def check_servers():
    """Check server status and get basic metrics."""
    results = []
    async with httpx.AsyncClient(timeout=5, verify=False) as client:
        for server in SERVERS:
            status = {"name": server["name"], "host": server["host"], "online": False, "latency_ms": 0}
            try:
                start = datetime.now()
                if server["check"] == "ssh":
                    # SSH check via subprocess
                    proc = await asyncio.create_subprocess_exec(
                        "ssh", "-p", str(server["port"]), "-o", "ConnectTimeout=3",
                        "-o", "StrictHostKeyChecking=no",
                        f"brucewayne9@{server['host']}", "uptime",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                    status["online"] = proc.returncode == 0
                    status["uptime"] = stdout.decode().strip() if stdout else ""
                elif server["check"].startswith("http"):
                    proto = server["check"]
                    url = f"{proto}://{server['host']}:{server['port']}"
                    resp = await client.get(url)
                    status["online"] = resp.status_code < 500
                    status["status_code"] = resp.status_code
                else:
                    url = f"http://{server['host']}:{server['port']}{server['check']}"
                    resp = await client.get(url)
                    status["online"] = resp.status_code == 200
                    try:
                        status["details"] = resp.json()
                    except Exception:
                        pass
                latency = (datetime.now() - start).total_seconds() * 1000
                status["latency_ms"] = round(latency)
            except Exception as e:
                status["error"] = str(e)[:80]
            results.append(status)
    return results


def get_daily_tasks():
    """Get today's task list from the tasks DB (reflects web app check-offs).
    Falls back to daily_tasks.json if DB is unavailable."""
    try:
        from core.tasks.manager import get_tasks_for_date, get_outstanding_tasks
        today = datetime.now().strftime("%Y-%m-%d")
        tasks = get_tasks_for_date(today)
        outstanding = get_outstanding_tasks()
        # Mark outstanding tasks so the brief can highlight them
        for t in outstanding:
            t["overdue"] = True
        # Today's tasks first, then overdue from previous days
        all_tasks = tasks + outstanding
        return all_tasks  # Return even if empty — DB is authoritative
    except Exception as e:
        log.warning(f"Tasks DB unavailable, falling back to JSON: {e}")

    # Fallback to JSON file
    if TASK_FILE.exists():
        try:
            data = json.loads(TASK_FILE.read_text())
            today = datetime.now().strftime("%Y-%m-%d")
            tasks = data.get(today, data.get("tasks", []))
            return tasks if isinstance(tasks, list) else []
        except Exception:
            return []
    return []


async def get_crm_stats():
    """Get CRM pipeline quick stats."""
    try:
        from integrations.base_crm.client import pipeline_summary, list_tasks
        pipeline = pipeline_summary()
        tasks = list_tasks(limit=50)

        # Count urgent tasks (due today or overdue)
        today = datetime.now(timezone.utc).date()
        urgent = 0
        for t in tasks:
            due = t.get("due_date")
            if due:
                try:
                    due_date = datetime.fromisoformat(due.replace("Z", "+00:00")).date()
                    if due_date <= today:
                        urgent += 1
                except Exception:
                    pass

        return {
            "total_deals": pipeline.get("total_deals", 0),
            "total_value": pipeline.get("total_value", 0),
            "stages": pipeline.get("stages", {}),
            "urgent_tasks": urgent,
        }
    except Exception as e:
        log.error(f"CRM stats failed: {e}")
        return None


async def get_calendar_today():
    """Get today's calendar events."""
    try:
        from core.tools.definitions import today_schedule
        result = await asyncio.get_event_loop().run_in_executor(None, today_schedule)
        events = result.get("events", [])
        # Normalize — events may be a list of dicts or list of other
        if isinstance(events, list):
            return [e if isinstance(e, dict) else {"summary": str(e)} for e in events]
        return []
    except Exception as e:
        log.error(f"Calendar fetch failed: {e}")
        return []


async def generate_ai_summary(weather, stocks, servers, crm, calendar, tasks):
    """Generate an AI executive summary of the day's data."""
    try:
        # Build a data snapshot for the LLM
        parts = []
        if weather:
            parts.append(f"Weather: {weather['temp']}°F, {weather['condition']}, high {weather['high']}°, {weather['precip']}% rain chance")
        if crm:
            stages = crm.get("stages", {})
            stage_str = ", ".join(f"{k}: {v}" for k, v in stages.items()) if stages else "no stage breakdown"
            parts.append(f"CRM: {crm['total_deals']} active deals, ${crm['total_value']:,.0f} pipeline, {crm['urgent_tasks']} tasks due. Stages: {stage_str}")
        if calendar:
            event_strs = []
            for e in calendar[:5]:
                s = e.get("start", {})
                t = s.get("dateTime", "all day")
                if "T" in str(t):
                    try:
                        t = datetime.fromisoformat(t.replace("Z", "+00:00")).strftime("%-I:%M %p")
                    except Exception:
                        pass
                event_strs.append(f"{t}: {e.get('summary', 'Untitled')}")
            parts.append(f"Calendar: {', '.join(event_strs)}")
        else:
            parts.append("Calendar: No events today")
        if tasks:
            task_strs = [t if isinstance(t, str) else t.get("text", str(t)) for t in tasks]
            parts.append(f"Tasks: {'; '.join(task_strs[:5])}")
        server_down = [s["name"] for s in (servers or []) if not s.get("online")]
        if server_down:
            parts.append(f"SERVERS DOWN: {', '.join(server_down)}")
        else:
            parts.append("All servers online")

        data_snapshot = "\n".join(parts)

        prompt = f"""You are Alfred, executive AI assistant to Mike Johnson (Groundrush Inc CEO).
Write a 3-4 sentence executive morning summary. Be direct, actionable, highlight what matters most.
Focus on: urgent items, overdue tasks, server issues, key meetings, deal movements.
Do NOT just restate numbers — provide INSIGHT and PRIORITIES.

Today's data:
{data_snapshot}"""

        for model in ["minimax-m2:cloud", "gpt-oss:120b-cloud"]:
            try:
                resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {"temperature": 0.6, "num_predict": 300},
                    },
                    timeout=30,
                )
                data = resp.json()
                summary = data.get("message", {}).get("content", "").strip()
                if summary and len(summary) > 30:
                    log.info(f"AI summary generated via {model}")
                    return summary
            except Exception as e:
                log.warning(f"AI summary via {model} failed: {e}")
                continue

        return None
    except Exception as e:
        log.error(f"AI summary generation failed: {e}")
        return None


# ── HTML Template ─────────────────────────────────────────────

def build_html(weather, stocks, news, servers, tasks, crm, calendar, ai_summary=None):
    """Build beautiful HTML newsletter."""
    today = datetime.now()
    date_str = today.strftime("%A, %B %d, %Y")
    hour = today.hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    # Stock rows
    stock_rows = ""
    for s in stocks:
        color = "#22c55e" if s["change"] >= 0 else "#ef4444"
        arrow = "&#9650;" if s["change"] >= 0 else "&#9660;"
        stock_rows += f"""
        <tr>
            <td style="padding:8px 12px;font-weight:600;color:#f5f5f5;">{s['symbol']}</td>
            <td style="padding:8px 12px;text-align:right;color:#f5f5f5;">${s['price']:,.2f}</td>
            <td style="padding:8px 12px;text-align:right;color:{color};">{arrow} ${abs(s['change']):,.2f} ({s['pct']:+.1f}%)</td>
        </tr>"""

    # News sections
    def news_block(title, emoji, items, accent):
        html = f"""
        <div style="margin-bottom:20px;">
            <h3 style="color:{accent};font-size:16px;margin:0 0 10px 0;text-transform:uppercase;letter-spacing:1px;">{emoji} {title}</h3>"""
        for item in items:
            link = item.get("link", "#")
            html += f"""
            <div style="margin-bottom:12px;padding:10px 14px;background:#1a1a1a;border-radius:8px;border-left:3px solid {accent};">
                <a href="{link}" style="color:#f5f5f5;text-decoration:none;font-weight:600;font-size:14px;">{item['title']}</a>
                <p style="color:#999;font-size:12px;margin:4px 0 0 0;">{item.get('desc', '')}</p>
            </div>"""
        if not items:
            html += '<p style="color:#666;font-size:13px;padding-left:14px;">No stories available right now.</p>'
        html += "</div>"
        return html

    political_html = news_block("Political", "&#127463;", news.get("political", []), "#3b82f6")
    motivational_html = news_block("Motivation & Business", "&#128170;", news.get("motivational", []), "#f97316")
    tech_html = news_block("Tech & AI", "&#129302;", news.get("tech", []), "#8b5cf6")

    # Server status
    server_rows = ""
    for s in servers:
        status_dot = "&#x1F7E2;" if s["online"] else "&#x1F534;"
        latency = f"{s['latency_ms']}ms" if s["online"] else "DOWN"
        uptime = s.get("uptime", "")
        extra = ""
        if uptime:
            # Parse load average from uptime
            load_match = re.search(r"load average:\s*([\d.]+)", uptime)
            if load_match:
                extra = f" | Load: {load_match.group(1)}"
        server_rows += f"""
        <tr>
            <td style="padding:6px 12px;color:#f5f5f5;">{status_dot} {s['name']}</td>
            <td style="padding:6px 12px;text-align:center;color:#f5f5f5;">{latency}</td>
            <td style="padding:6px 12px;color:#999;font-size:12px;">{s['host']}{extra}</td>
        </tr>"""

    # Task checklist
    task_html = ""
    if tasks:
        for i, task in enumerate(tasks):
            t = task if isinstance(task, str) else task.get("text", str(task))
            done = task.get("completed", task.get("done", False)) if isinstance(task, dict) else False
            overdue = task.get("overdue", False) if isinstance(task, dict) else False
            days_out = task.get("days_outstanding", 0) if isinstance(task, dict) else 0
            check = "&#9745;" if done else "&#9744;"
            if done:
                style = "text-decoration:line-through;color:#666;"
            elif overdue:
                style = "color:#ef4444;font-weight:600;"
            else:
                style = "color:#f5f5f5;"
            overdue_badge = f' <span style="font-size:11px;color:#ef4444;">({days_out}d overdue)</span>' if overdue and days_out > 0 else ""
            task_html += f'<div style="padding:8px 14px;margin-bottom:4px;background:#1a1a1a;border-radius:6px;font-size:14px;{style}">{check} {t}{overdue_badge}</div>'
    else:
        task_html = '<p style="color:#666;font-size:13px;padding-left:14px;">No tasks set for today. Tell Alfred tonight what you need done tomorrow!</p>'

    # CRM stats
    crm_html = ""
    if crm:
        crm_html = f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
            <div style="flex:1;min-width:120px;background:#1a1a1a;border-radius:8px;padding:12px;text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#f97316;">{crm['total_deals']}</div>
                <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px;">Active Deals</div>
            </div>
            <div style="flex:1;min-width:120px;background:#1a1a1a;border-radius:8px;padding:12px;text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#22c55e;">${crm['total_value']:,.0f}</div>
                <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px;">Pipeline Value</div>
            </div>
            <div style="flex:1;min-width:120px;background:#1a1a1a;border-radius:8px;padding:12px;text-align:center;">
                <div style="font-size:24px;font-weight:700;color:#ef4444;">{crm['urgent_tasks']}</div>
                <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px;">Tasks Due</div>
            </div>
        </div>"""

    # Calendar
    cal_html = ""
    if calendar:
        for event in calendar[:5]:
            start = event.get("start", {})
            time_str = start.get("dateTime", "")
            if time_str:
                try:
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    time_str = dt.strftime("%-I:%M %p")
                except Exception:
                    time_str = time_str[:16].split("T")[-1]
            else:
                time_str = "All Day"
            summary = event.get("summary", "Untitled")
            cal_html += f'<div style="padding:8px 14px;margin-bottom:4px;background:#1a1a1a;border-radius:6px;font-size:14px;color:#f5f5f5;"><span style="color:#f97316;font-weight:600;">{time_str}</span> &mdash; {summary}</div>'
    else:
        cal_html = '<p style="color:#666;font-size:13px;padding-left:14px;">No events scheduled today.</p>'

    # Weather bar
    weather_html = ""
    if weather:
        precip_note = f" | {weather['precip']}% rain" if weather['precip'] > 10 else ""
        weather_html = f"""
        <div style="background:linear-gradient(135deg,#1e3a5f,#0a2540);border-radius:12px;padding:16px 20px;margin-bottom:24px;">
            <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                <div style="font-size:36px;font-weight:700;color:#fff;">{weather['temp']}°F</div>
                <div style="color:#93c5fd;">
                    <div style="font-size:16px;font-weight:600;">{weather['condition']}</div>
                    <div style="font-size:12px;">H: {weather['high']}° L: {weather['low']}° | Wind: {weather['wind']}mph{precip_note}</div>
                </div>
                <div style="margin-left:auto;color:#93c5fd;font-size:12px;">Atlanta, GA</div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#000;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:640px;margin:0 auto;padding:20px;">

    <!-- Header -->
    <div style="text-align:center;padding:24px 0;border-bottom:1px solid #222;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:3px;color:#f97316;margin-bottom:4px;">Alfred Intelligence</div>
        <h1 style="margin:0;font-size:28px;color:#fff;font-weight:300;">{greeting}, Mike</h1>
        <p style="margin:4px 0 0;color:#666;font-size:13px;">{date_str}</p>
    </div>

    <!-- AI Executive Summary -->
    {f'''<div style="margin:20px 0;padding:16px 20px;background:linear-gradient(135deg,#1a0a00,#2a1500);border:1px solid #f97316;border-radius:12px;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#f97316;margin-bottom:8px;">&#9889; Alfred's Take</div>
        <p style="color:#f5f5f5;font-size:14px;line-height:1.6;margin:0;">{ai_summary}</p>
    </div>''' if ai_summary else ''}

    <!-- Weather -->
    <div style="padding-top:20px;">
        {weather_html}
    </div>

    <!-- Daily Tasks -->
    <div style="margin-bottom:28px;">
        <h2 style="color:#f97316;font-size:18px;margin:0 0 12px 0;border-bottom:1px solid #222;padding-bottom:8px;">&#9745; Today's Priorities</h2>
        {task_html}
    </div>

    <!-- Stocks -->
    <div style="margin-bottom:28px;">
        <h2 style="color:#f97316;font-size:18px;margin:0 0 12px 0;border-bottom:1px solid #222;padding-bottom:8px;">&#128200; Markets</h2>
        <table style="width:100%;border-collapse:collapse;background:#111;border-radius:8px;overflow:hidden;">
            <tr style="background:#1a1a1a;">
                <th style="padding:8px 12px;text-align:left;color:#999;font-size:11px;text-transform:uppercase;">Symbol</th>
                <th style="padding:8px 12px;text-align:right;color:#999;font-size:11px;text-transform:uppercase;">Price</th>
                <th style="padding:8px 12px;text-align:right;color:#999;font-size:11px;text-transform:uppercase;">Change</th>
            </tr>
            {stock_rows}
        </table>
    </div>

    <!-- News -->
    <div style="margin-bottom:28px;">
        <h2 style="color:#f97316;font-size:18px;margin:0 0 16px 0;border-bottom:1px solid #222;padding-bottom:8px;">&#128240; Daily Digest</h2>
        {political_html}
        {motivational_html}
        {tech_html}
    </div>

    <!-- Server Performance -->
    <div style="margin-bottom:28px;">
        <h2 style="color:#f97316;font-size:18px;margin:0 0 12px 0;border-bottom:1px solid #222;padding-bottom:8px;">&#128421; Server Performance</h2>
        <table style="width:100%;border-collapse:collapse;background:#111;border-radius:8px;overflow:hidden;">
            <tr style="background:#1a1a1a;">
                <th style="padding:6px 12px;text-align:left;color:#999;font-size:11px;text-transform:uppercase;">Server</th>
                <th style="padding:6px 12px;text-align:center;color:#999;font-size:11px;text-transform:uppercase;">Latency</th>
                <th style="padding:6px 12px;text-align:left;color:#999;font-size:11px;text-transform:uppercase;">Details</th>
            </tr>
            {server_rows}
        </table>
    </div>

    <!-- CRM Pipeline -->
    <div style="margin-bottom:28px;">
        <h2 style="color:#f97316;font-size:18px;margin:0 0 12px 0;border-bottom:1px solid #222;padding-bottom:8px;">&#128188; Business Pipeline</h2>
        {crm_html if crm_html else '<p style="color:#666;">CRM data unavailable.</p>'}
    </div>

    <!-- Calendar -->
    <div style="margin-bottom:28px;">
        <h2 style="color:#f97316;font-size:18px;margin:0 0 12px 0;border-bottom:1px solid #222;padding-bottom:8px;">&#128197; Today's Schedule</h2>
        {cal_html}
    </div>

    <!-- Footer -->
    <div style="text-align:center;padding:20px 0;border-top:1px solid #222;margin-top:20px;">
        <p style="color:#444;font-size:11px;margin:0;">Generated by Alfred Intelligence at {datetime.now().strftime('%-I:%M %p ET')}</p>
        <p style="color:#333;font-size:10px;margin:4px 0 0;">Groundrush Inc &bull; Atlanta, GA &bull; groundrushlabs.com</p>
    </div>

</div>
</body>
</html>"""

    return html


# ── Main ──────────────────────────────────────────────────────

async def generate_and_send():
    """Generate all data and send the morning brief."""
    log.info("Generating morning brief...")

    # Fetch all data concurrently
    weather_task = fetch_weather()
    stocks_task = fetch_stocks()
    news_task = fetch_all_news()
    servers_task = check_servers()
    crm_task = get_crm_stats()
    calendar_task = get_calendar_today()

    results = await asyncio.gather(
        weather_task, stocks_task, news_task, servers_task, crm_task, calendar_task,
        return_exceptions=True,
    )

    weather = results[0] if not isinstance(results[0], Exception) else None
    stocks = results[1] if not isinstance(results[1], Exception) else []
    news = results[2] if not isinstance(results[2], Exception) else {}
    servers = results[3] if not isinstance(results[3], Exception) else []
    crm = results[4] if not isinstance(results[4], Exception) else None
    calendar = results[5] if not isinstance(results[5], Exception) else []

    tasks = get_daily_tasks()

    # Generate AI executive summary
    ai_summary = await generate_ai_summary(weather, stocks, servers, crm, calendar, tasks)

    # Build HTML
    html = build_html(weather, stocks, news, servers, tasks, crm, calendar, ai_summary)

    # Send email
    today_str = datetime.now().strftime("%A, %B %d")
    result = email_client.send_email(
        "alfred",
        MIKE_EMAIL,
        f"Morning Brief — {today_str}",
        html,
        html=True,
    )

    if "error" in result:
        log.error(f"Failed to send morning brief: {result['error']}")
        # Try Telegram as fallback notification
        if TELEGRAM_BOT_TOKEN:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": "Morning brief email failed to send. Check /tmp/morning_brief.log",
                },
                timeout=10,
            )
    else:
        log.info(f"Morning brief sent to {MIKE_EMAIL}")
        # Also notify on Telegram
        if TELEGRAM_BOT_TOKEN:
            server_summary = " | ".join(
                [f"{'✅' if s['online'] else '❌'} {s['name']}" for s in servers]
            )
            task_count = len(tasks)
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": f"☀️ Morning brief sent to your email!\n\n"
                            f"Servers: {server_summary}\n"
                            f"Tasks today: {task_count}\n"
                            f"Deals in pipeline: {crm['total_deals'] if crm else '?'}",
                },
                timeout=10,
            )

    return result


if __name__ == "__main__":
    asyncio.run(generate_and_send())
