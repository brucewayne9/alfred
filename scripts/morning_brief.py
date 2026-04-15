#!/usr/bin/env python3
"""
Mike's Morning Brief — 6:30 AM ET daily
One brief. One time. Everything Mike needs to know.

Sections:
1. Today's Tasks — from evening ping or manual entry
2. Brain Dumps — what Alfred & Mike worked on (last 24h from Grey Matter)
3. Infrastructure — server fleet health + services
4. Stock Portfolio — watchlist with daily changes
5. News — conservative + tech headlines (clickable)
6. Big Picture — strategic context from Grey Matter

Sends as a designed HTML email via alfred@groundrushinc.com (Google Workspace).
"""

import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

from integrations.email.client import EmailClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("morning_brief")

ET = timezone(timedelta(hours=-4))
NOW = datetime.now(ET)
DATE_STR = NOW.strftime("%A, %B %-d, %Y")
DATE_SHORT = NOW.strftime("%Y-%m-%d")
SCRIPTS = os.path.expanduser("~/.openclaw/workspace/scripts/integrations")
TASK_FILE = Path("/home/aialfred/alfred/data/daily_tasks.json")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = "7582976864"

# Stock watchlist — Mike's portfolio
TICKERS = [
    ("NVDA", "NVIDIA"), ("AMZN", "Amazon"), ("GOOGL", "Google"),
    ("VTI", "Total Market"), ("VXUS", "Int'l Stocks"), ("SCHD", "Dividend ETF"),
    ("BBLU", "Bridgebio"), ("FELC", "Fidelity"), ("SCHK", "Schwab"),
    ("JHAC", "John Hancock"), ("BND", "Bonds"), ("UBER", "Uber"),
    ("LOW", "Lowe's"), ("HD", "Home Depot"),
]


def run_script(script, args, timeout=30):
    """Run an integration script and return stdout."""
    try:
        result = subprocess.run(
            ["python3", f"{SCRIPTS}/{script}"] + args,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception as e:
        return f"(unavailable: {e})"


# ─────────────────────────────────────────────────
# DATA SECTIONS — return structured data
# ─────────────────────────────────────────────────

def get_tasks() -> list[dict]:
    """Return list of {text, done} task dicts."""
    if not TASK_FILE.exists():
        return []
    try:
        data = json.loads(TASK_FILE.read_text())
    except Exception:
        return []
    raw = data.get(DATE_SHORT, data.get("tasks", []))
    tasks = []
    for t in raw:
        if isinstance(t, dict):
            tasks.append({"text": t.get("text", str(t)), "done": t.get("done", False)})
        else:
            tasks.append({"text": str(t), "done": False})
    return tasks


def get_brain_dumps() -> str:
    """Return brain dump text from Grey Matter."""
    output = run_script("lightrag_client.py", [
        "recall", "What did Mike and Alfred work on in the last 24 hours? "
        "Include any brain dumps, decisions made, tasks completed, "
        "and important conversations from yesterday and today."
    ], timeout=45)
    if output and "unavailable" not in output and len(output) > 20:
        lines = output.strip().split("\n")
        return "\n".join(lines[:20])
    return ""


def get_git_activity() -> list[str]:
    """Return list of commit one-liners."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--since=midnight", "--all"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.expanduser("~/alfred")
        )
        if result.stdout.strip():
            return result.stdout.strip().split("\n")[:10]
    except Exception:
        pass
    return []


def get_infrastructure() -> tuple[list[dict], list[dict]]:
    """Return (servers, services) as lists of status dicts."""
    SERVERS = [
        ("105", "75.43.156.105", "Labs + Claw"),
        ("104", "75.43.156.104", "Production"),
        ("117", "75.43.156.117", "Dokploy/CRM"),
        ("111", "75.43.156.111", "CasaOS Dev"),
        ("121", "75.43.156.121", "Mailcow"),
        ("098", "75.43.156.98", "LoovaCast Dev"),
        ("100", "75.43.156.100", "LoovaCast Prod"),
    ]

    servers = []
    for sid, ip, role in SERVERS:
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "3", ip],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                m = re.search(r'time=(\d+\.?\d*)', result.stdout)
                ms = f"{float(m.group(1)):.0f}ms" if m else "ok"
                servers.append({"id": sid, "ip": ip, "role": role, "status": "up", "latency": ms})
            else:
                servers.append({"id": sid, "ip": ip, "role": role, "status": "down", "latency": ""})
        except Exception:
            servers.append({"id": sid, "ip": ip, "role": role, "status": "unknown", "latency": ""})

    services = []
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "openclaw-gateway"],
            capture_output=True, text=True, timeout=5
        )
        services.append({"name": "OpenClaw", "status": result.stdout.strip()})
    except Exception:
        services.append({"name": "OpenClaw", "status": "unknown"})

    for name, url in [("ComfyUI", "http://localhost:8188/system_stats"), ("Alfred Labs", "http://localhost:8400/health")]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "brief"})
            with urllib.request.urlopen(req, timeout=3):
                services.append({"name": name, "status": "active"})
        except Exception:
            services.append({"name": name, "status": "down"})

    return servers, services


def get_stocks() -> list[dict]:
    """Return list of {ticker, name, price, change, pct, up} dicts."""
    stocks = []
    for ticker, name in TICKERS:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            meta = data["chart"]["result"][0]["meta"]
            price = meta["regularMarketPrice"]
            prev = meta["chartPreviousClose"]
            change = price - prev
            pct = (change / prev) * 100 if prev else 0
            stocks.append({
                "ticker": ticker, "name": name, "price": price,
                "change": change, "pct": pct, "up": change >= 0,
            })
        except Exception:
            stocks.append({"ticker": ticker, "name": name, "price": None, "change": 0, "pct": 0, "up": True})
    return stocks


def get_news() -> dict:
    """Return {conservative: [{title, url, snippet}], tech: [{title, url, snippet}]}."""
    news = {"conservative": [], "tech": []}

    try:
        result = subprocess.run(
            ["python3", f"{SCRIPTS}/search.py", "batch",
             "conservative news headlines today site:foxnews.com OR site:dailywire.com OR site:breitbart.com",
             "technology AI news today site:techcrunch.com OR site:theverge.com OR site:arstechnica.com"],
            capture_output=True, text=True, timeout=45
        )
        if result.stdout.strip():
            raw = result.stdout.strip()
            current_category = "conservative"

            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("==="):
                    if "technology" in line.lower() or "tech" in line.lower():
                        current_category = "tech"
                    continue

                # Format: "1. [engine] Title — https://url"
                m = re.match(r'^\d+\.\s*\[\w+\]\s*(.+?)\s*—\s*(https?://\S+)', line)
                if m:
                    title = m.group(1).strip()
                    url = m.group(2).strip()
                    news[current_category].append({"title": title, "url": url, "snippet": ""})

    except Exception:
        pass

    # Cap at 5 each, skip homepage-only results
    for cat in news:
        news[cat] = [n for n in news[cat] if len(n.get("title", "")) > 20][:5]

    return news


def get_big_picture() -> str:
    """Return strategic overview text from Grey Matter."""
    output = run_script("lightrag_client.py", [
        "recall", "What are Ground Rush Inc current strategic priorities, "
        "active projects, and what is the team working towards this week? "
        "Include LoovaCast, RuckTalk, and client work status."
    ], timeout=45)
    if output and "unavailable" not in output and len(output) > 20:
        lines = output.strip().split("\n")
        return "\n".join(lines[:15])
    return ""


# ─────────────────────────────────────────────────
# HTML EMAIL BUILDER
# ─────────────────────────────────────────────────

def _e(text):
    """Escape HTML."""
    return (str(text) if text else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_html_brief(tasks, brain, git, servers, services, stocks, news, big_picture):
    """Build the morning brief as a premium designed HTML email."""

    hour = NOW.hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    day_of_week = NOW.strftime("%A").upper()

    # ── Tasks HTML ──
    if tasks:
        tasks_html = ""
        for t in tasks:
            check = "&#10003;" if t["done"] else ""
            bg = "#0d2818" if t["done"] else "#1a1a2e"
            border_color = "#22c55e" if t["done"] else "#334155"
            text_color = "#6b7280" if t["done"] else "#e2e8f0"
            deco = "line-through" if t["done"] else "none"
            tasks_html += f'''<tr><td style="padding:6px 0;">
              <table cellpadding="0" cellspacing="0" width="100%"><tr>
                <td width="32" style="vertical-align:top;padding-top:2px;">
                  <div style="width:20px;height:20px;border-radius:4px;border:2px solid {border_color};background:{bg};text-align:center;line-height:18px;font-size:13px;color:#22c55e;">{check}</div>
                </td>
                <td style="padding-left:10px;font-size:14px;color:{text_color};text-decoration:{deco};">{_e(t["text"])}</td>
              </tr></table>
            </td></tr>'''
    else:
        tasks_html = '<tr><td style="padding:12px 0;color:#64748b;font-size:14px;font-style:italic;">No tasks set. Reply to tonight\'s evening ping to set tomorrow\'s.</td></tr>'

    # ── Brain Dumps HTML ──
    if brain:
        # Clean up markdown headers and formatting for email
        brain_clean = brain.replace("##", "").replace("**", "").replace("---", "")
        brain_lines = [l.strip() for l in brain_clean.split("\n") if l.strip()]
        brain_html = "".join(f'<p style="margin:0 0 8px 0;color:#c9d1d9;font-size:13px;line-height:1.6;">{_e(l)}</p>' for l in brain_lines[:12])
    else:
        brain_html = '<p style="color:#64748b;font-size:13px;font-style:italic;">No recent activity logged.</p>'

    # ── Git HTML ──
    if git:
        git_html = ""
        for commit in git:
            parts = commit.split(" ", 1)
            sha = parts[0] if parts else ""
            msg = parts[1] if len(parts) > 1 else commit
            git_html += f'<tr><td style="padding:4px 0;"><span style="font-family:\'SF Mono\',Menlo,Monaco,Consolas,monospace;font-size:11px;color:#f97316;background:#1c1917;padding:2px 6px;border-radius:3px;">{_e(sha[:7])}</span> <span style="font-size:13px;color:#c9d1d9;margin-left:8px;">{_e(msg)}</span></td></tr>'
    else:
        git_html = '<tr><td style="padding:8px 0;color:#64748b;font-size:13px;font-style:italic;">No commits today.</td></tr>'

    # ── Infrastructure HTML ──
    infra_html = ""
    up_count = sum(1 for s in servers if s["status"] == "up")
    total = len(servers)

    for s in servers:
        if s["status"] == "up":
            dot_color = "#22c55e"
            status_text = s["latency"]
        elif s["status"] == "down":
            dot_color = "#ef4444"
            status_text = "DOWN"
        else:
            dot_color = "#eab308"
            status_text = "?"

        infra_html += f'''<tr>
          <td style="padding:5px 0;width:16px;"><div style="width:8px;height:8px;border-radius:50%;background:{dot_color};"></div></td>
          <td style="padding:5px 8px;font-size:13px;color:#e2e8f0;font-weight:600;">{_e(s["id"])}</td>
          <td style="padding:5px 8px;font-size:12px;color:#94a3b8;">{_e(s["role"])}</td>
          <td style="padding:5px 0;font-size:12px;color:#64748b;text-align:right;">{_e(status_text)}</td>
        </tr>'''

    services_html = ""
    for svc in services:
        active = "active" in svc["status"].lower()
        bg = "#052e16" if active else "#450a0a"
        color = "#22c55e" if active else "#ef4444"
        dot = "&#9679;" if active else "&#9679;"
        services_html += f'<span style="display:inline-block;background:{bg};color:{color};padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;margin:3px 4px;letter-spacing:0.5px;"><span style="font-size:8px;">{dot}</span> {_e(svc["name"])}</span>'

    # ── Stocks HTML ──
    stocks_html = ""
    total_change = sum(s["pct"] for s in stocks if s["price"] is not None) / max(1, sum(1 for s in stocks if s["price"] is not None))
    portfolio_color = "#22c55e" if total_change >= 0 else "#ef4444"
    portfolio_arrow = "&#9650;" if total_change >= 0 else "&#9660;"

    for s in stocks:
        if s["price"] is None:
            stocks_html += f'<tr><td colspan="4" style="padding:6px 0;color:#64748b;font-size:12px;">{_e(s["ticker"])} — unavailable</td></tr>'
            continue
        color = "#22c55e" if s["up"] else "#ef4444"
        arrow = "&#9650;" if s["up"] else "&#9660;"
        bar_width = min(abs(s["pct"]) * 8, 60)  # Visual bar proportional to % change
        stocks_html += f'''<tr style="border-bottom:1px solid #1e293b;">
          <td style="padding:8px 0;width:55px;">
            <span style="font-size:12px;font-weight:700;color:#f1f5f9;letter-spacing:0.5px;">{_e(s["ticker"])}</span>
          </td>
          <td style="padding:8px 0;text-align:right;width:70px;">
            <span style="font-size:13px;color:#e2e8f0;">${s["price"]:.2f}</span>
          </td>
          <td style="padding:8px 12px;text-align:right;width:90px;">
            <span style="font-size:12px;color:{color};">{arrow} {s["change"]:+.2f} ({s["pct"]:+.1f}%)</span>
          </td>
          <td style="padding:8px 0;">
            <div style="height:4px;background:#1e293b;border-radius:2px;overflow:hidden;">
              <div style="height:4px;width:{bar_width}px;background:{color};border-radius:2px;"></div>
            </div>
          </td>
        </tr>'''

    # ── News HTML ──
    def _news_block(items, accent_color):
        if not items:
            return '<p style="color:#64748b;font-size:13px;font-style:italic;">No headlines available.</p>'
        html = ""
        for n in items:
            title = _e(n.get("title", ""))
            url = n.get("url", "")
            snippet = _e(n.get("snippet", ""))
            if url:
                html += f'''<div style="padding:10px 0;border-bottom:1px solid #1e293b;">
                  <a href="{url}" style="color:#e2e8f0;text-decoration:none;font-size:14px;font-weight:500;line-height:1.4;">{title}</a>
                  <div style="font-size:12px;color:#64748b;margin-top:4px;line-height:1.4;">{snippet}</div>
                </div>'''
            else:
                html += f'''<div style="padding:10px 0;border-bottom:1px solid #1e293b;">
                  <div style="color:#e2e8f0;font-size:14px;font-weight:500;line-height:1.4;">{title}</div>
                </div>'''
        return html

    con_news_html = _news_block(news.get("conservative", []), "#ef4444")
    tech_news_html = _news_block(news.get("tech", []), "#3b82f6")

    # ── Big Picture HTML ──
    if big_picture:
        bp_clean = big_picture.replace("##", "").replace("**", "").replace("---", "")
        bp_lines = [l.strip() for l in bp_clean.split("\n") if l.strip()]
        bp_html = "".join(f'<p style="margin:0 0 8px 0;color:#c9d1d9;font-size:13px;line-height:1.6;">{_e(l)}</p>' for l in bp_lines[:12])
    else:
        bp_html = '<p style="color:#64748b;font-size:13px;font-style:italic;">Ask Alfred for a project status update.</p>'

    # ── FULL EMAIL ──
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body style="margin:0;padding:0;background:#090d18;font-family:'DM Sans',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased;">

<table width="100%" cellpadding="0" cellspacing="0" style="background:#090d18;">
<tr><td align="center" style="padding:24px 12px;">

<!-- Container -->
<table width="620" cellpadding="0" cellspacing="0" style="background:#0f1629;border-radius:12px;overflow:hidden;">

  <!-- ═══ HEADER ═══ -->
  <tr><td style="padding:40px 36px 32px 36px;background:linear-gradient(180deg,#141b33 0%,#0f1629 100%);">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="vertical-align:bottom;">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:4px;color:#ef4444;font-weight:700;">ALFRED</div>
        <div style="font-size:32px;font-weight:700;color:#f8fafc;margin-top:2px;line-height:1.1;">Morning Brief</div>
      </td>
      <td style="text-align:right;vertical-align:bottom;">
        <div style="font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#475569;font-weight:600;">{day_of_week}</div>
        <div style="font-size:13px;color:#94a3b8;margin-top:2px;">{NOW.strftime("%B %-d, %Y")}</div>
      </td>
    </tr></table>
    <div style="height:3px;background:linear-gradient(90deg,#ef4444 0%,#f97316 50%,transparent 100%);margin-top:20px;border-radius:2px;"></div>
    <div style="font-size:13px;color:#64748b;margin-top:16px;">{greeting}, sir.</div>
  </td></tr>

  <!-- ═══ TASKS ═══ -->
  <tr><td style="padding:28px 36px;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;padding-bottom:16px;">
        <span style="color:#22c55e;margin-right:6px;">&#9632;</span> Today's Priorities
      </td>
    </tr>
    {tasks_html}
    </table>
  </td></tr>
  <tr><td style="padding:0 36px;"><div style="height:1px;background:#1e293b;"></div></td></tr>

  <!-- ═══ BRAIN DUMPS ═══ -->
  <tr><td style="padding:28px 36px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;margin-bottom:14px;">
      <span style="color:#818cf8;margin-right:6px;">&#9632;</span> What We Worked On <span style="color:#475569;font-weight:400;">(24h)</span>
    </div>
    {brain_html}
  </td></tr>
  <tr><td style="padding:0 36px;"><div style="height:1px;background:#1e293b;"></div></td></tr>

  <!-- ═══ CODE ═══ -->
  <tr><td style="padding:28px 36px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;margin-bottom:14px;">
      <span style="color:#f97316;margin-right:6px;">&#9632;</span> Code Changes
    </div>
    <table cellpadding="0" cellspacing="0">{git_html}</table>
  </td></tr>
  <tr><td style="padding:0 36px;"><div style="height:1px;background:#1e293b;"></div></td></tr>

  <!-- ═══ INFRASTRUCTURE ═══ -->
  <tr><td style="padding:28px 36px;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;padding-bottom:14px;">
        <span style="color:#38bdf8;margin-right:6px;">&#9632;</span> Infrastructure
      </td>
      <td style="text-align:right;font-size:12px;color:#64748b;padding-bottom:14px;">
        <span style="color:#22c55e;font-weight:600;">{up_count}</span>/{total} online
      </td>
    </tr></table>
    <table width="100%" cellpadding="0" cellspacing="0">{infra_html}</table>
    <div style="margin-top:14px;">{services_html}</div>
  </td></tr>
  <tr><td style="padding:0 36px;"><div style="height:1px;background:#1e293b;"></div></td></tr>

  <!-- ═══ STOCKS ═══ -->
  <tr><td style="padding:28px 36px;">
    <table width="100%" cellpadding="0" cellspacing="0"><tr>
      <td style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;padding-bottom:14px;">
        <span style="color:#eab308;margin-right:6px;">&#9632;</span> Portfolio
      </td>
      <td style="text-align:right;padding-bottom:14px;">
        <span style="font-size:13px;font-weight:600;color:{portfolio_color};">{portfolio_arrow} Avg {total_change:+.1f}%</span>
      </td>
    </tr></table>
    <table width="100%" cellpadding="0" cellspacing="0">{stocks_html}</table>
  </td></tr>
  <tr><td style="padding:0 36px;"><div style="height:1px;background:#1e293b;"></div></td></tr>

  <!-- ═══ NEWS ═══ -->
  <tr><td style="padding:28px 36px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;margin-bottom:14px;">
      <span style="color:#ef4444;margin-right:6px;">&#9632;</span> Conservative
    </div>
    {con_news_html}

    <div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;margin-bottom:14px;margin-top:24px;">
      <span style="color:#3b82f6;margin-right:6px;">&#9632;</span> Tech &amp; AI
    </div>
    {tech_news_html}
  </td></tr>
  <tr><td style="padding:0 36px;"><div style="height:1px;background:#1e293b;"></div></td></tr>

  <!-- ═══ BIG PICTURE ═══ -->
  <tr><td style="padding:28px 36px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:2.5px;color:#94a3b8;font-weight:600;margin-bottom:14px;">
      <span style="color:#a78bfa;margin-right:6px;">&#9632;</span> Big Picture
    </div>
    {bp_html}
  </td></tr>

  <!-- ═══ FOOTER ═══ -->
  <tr><td style="padding:24px 36px;background:#0a0e1a;text-align:center;">
    <div style="height:2px;width:40px;background:linear-gradient(90deg,#ef4444,#f97316);border-radius:1px;margin:0 auto 16px auto;"></div>
    <div style="font-size:11px;color:#475569;letter-spacing:1px;">
      ALFRED &middot; {NOW.strftime("%B %-d, %Y")} &middot; Ground Rush Inc
    </div>
  </td></tr>

</table>
<!-- End Container -->

</td></tr>
</table>

</body>
</html>'''
    return html


# ─────────────────────────────────────────────────
# PLAIN TEXT (for logging / fallback)
# ─────────────────────────────────────────────────

def build_plain_brief(tasks, brain, git, servers, services, stocks, news, big_picture):
    """Assemble plain text version for logging."""
    task_lines = "\n".join(f"  {'✅' if t['done'] else '⬜'} {t['text']}" for t in tasks) if tasks else "  No tasks set."
    git_lines = "\n".join(f"  • {c}" for c in git) if git else "  No commits today."
    infra_lines = "\n".join(f"  {'✅' if s['status']=='up' else '❌'} {s['id']} ({s['ip']}) — {s['role']} [{s['latency']}]" for s in servers)
    svc_line = "  Services: " + " | ".join(f"{s['name']}: {s['status']}" for s in services)
    stock_lines = "\n".join(f"  {s['ticker']:<6} ${s['price']:>8.2f}  {'▲' if s['up'] else '▼'} {s['change']:+.2f} ({s['pct']:+.1f}%)  {s['name']}" for s in stocks if s["price"])
    news_lines = ""
    for cat in ["conservative", "tech"]:
        items = news.get(cat, [])
        if items:
            news_lines += f"\n  {cat.upper()}:\n"
            for n in items:
                news_lines += f"    • {n['title']}\n      {n.get('url', '')}\n"

    return f"""
══════════════════════════════════════════════════════════
  MORNING BRIEF — {DATE_STR}
══════════════════════════════════════════════════════════

📋 TODAY'S TASKS
{task_lines}

🧠 BRAIN DUMPS (Last 24h)
  {brain or 'No recent activity.'}

🔧 CODE CHANGES
{git_lines}

🖥️ INFRASTRUCTURE
{infra_lines}
{svc_line}

📈 STOCKS
{stock_lines}

📰 NEWS
{news_lines}

🗺️ BIG PICTURE
  {big_picture or 'Ask Alfred for a status update.'}

══════════════════════════════════════════════════════════
  End of Brief — {DATE_STR}
  Have a great day, sir.
══════════════════════════════════════════════════════════
""".strip()


# ─────────────────────────────────────────────────
# DELIVERY
# ─────────────────────────────────────────────────

def send_email(html_brief):
    """Email the brief to Mike from Alfred's Google Workspace."""
    try:
        client = EmailClient()
        result = client.send_email(
            account="alfred-gw",
            to="mjohnson@groundrushinc.com",
            subject=f"Morning Brief — {NOW.strftime('%B %-d, %Y')}",
            body=html_brief,
            html=True,
        )
        if result.get("status") == "sent":
            log.info("Morning brief sent to mjohnson@groundrushinc.com from alfred@groundrushinc.com")
        else:
            log.error(f"Email failed: {result}")
    except Exception as e:
        log.error(f"Email error: {e}")


def send_telegram_notification():
    """Short Telegram ping that the brief was sent."""
    if not TELEGRAM_BOT_TOKEN:
        return

    message = (
        f"Good morning, sir. Your morning brief for {DATE_STR} "
        f"has been sent to your email."
    )

    try:
        data = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        log.info("Telegram notification sent")
    except Exception as e:
        log.error(f"Telegram notification failed: {e}")


def main():
    log.info(f"Building morning brief for {DATE_STR}...")

    print("  [1/7] Tasks...")
    tasks = get_tasks()
    print("  [2/7] Brain dumps...")
    brain = get_brain_dumps()
    print("  [3/7] Git...")
    git = get_git_activity()
    print("  [4/7] Infrastructure...")
    servers, services = get_infrastructure()
    print("  [5/7] Stocks...")
    stocks = get_stocks()
    print("  [6/7] News...")
    news = get_news()
    print("  [7/7] Big picture...")
    big_picture = get_big_picture()

    plain = build_plain_brief(tasks, brain, git, servers, services, stocks, news, big_picture)
    html = build_html_brief(tasks, brain, git, servers, services, stocks, news, big_picture)
    print(plain)

    log.info("Delivering...")
    send_email(html)
    send_telegram_notification()

    # Save HTML preview
    preview = Path("/home/aialfred/alfred/static/drafts/morning-brief-preview.html")
    preview.write_text(html)

    log.info("Morning brief complete.")


if __name__ == "__main__":
    main()
