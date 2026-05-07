"""Admin /admin/leads dashboard for the AI Savings Audit funnel.

Lists recent audit leads pulled from Twenty CRM, joining Note titles ("AI
Savings Audit — $X/yr (Y fit)") to their linked Person records.

Auth-gated via the existing JWT cookie (require_auth dependency).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse

from core.security.auth import require_auth
from integrations.base_crm import client as crm

logger = logging.getLogger(__name__)

NOTE_TITLE_PREFIX = "AI Savings Audit"
SAVINGS_RE = re.compile(r"\$([\d,]+)/yr.*?\(([a-z]+) fit\)", re.IGNORECASE)


def _fetch_audit_leads(limit: int = 50) -> list[dict]:
    """Pull recent audit Notes from Twenty, hydrate with their linked Person."""
    notes_resp = crm._get("/rest/notes", {
        "limit": 100,
        "order_by": "createdAt[DescNullsLast]",
    })
    notes = notes_resp.get("data", {}).get("notes", [])
    audit_notes = [n for n in notes if (n.get("title") or "").startswith(NOTE_TITLE_PREFIX)]

    leads: list[dict] = []
    for note in audit_notes[:limit]:
        # Match on the title: "AI Savings Audit — $X/yr (Y fit)"
        title = note.get("title", "")
        m = SAVINGS_RE.search(title)
        annual = m.group(1).replace(",", "") if m else "?"
        fit = m.group(2).lower() if m else "?"

        # Find the linked person via noteTargets
        person = None
        try:
            targets = crm._get("/rest/noteTargets", {"filter": f"noteId[eq]:{note['id']}"})
            target_list = targets.get("data", {}).get("noteTargets", [])
            person_id = None
            for t in target_list:
                if t.get("personId"):
                    person_id = t["personId"]
                    break
            if person_id:
                person_data = crm._get(f"/rest/people/{person_id}")
                p = person_data.get("data", {}).get("person", person_data.get("data", {}))
                person = {
                    "id": person_id,
                    "first_name": (p.get("name", {}) or {}).get("firstName", ""),
                    "last_name": (p.get("name", {}) or {}).get("lastName", ""),
                    "email": (p.get("emails", {}) or {}).get("primaryEmail", ""),
                    "phone": (p.get("phones", {}) or {}).get("primaryPhoneNumber", ""),
                }
        except Exception as e:
            logger.warning(f"Failed to hydrate person for note {note.get('id')}: {e}")

        # Pull a couple useful tags out of the body (industry, utm_source)
        body = (note.get("bodyV2") or {}).get("markdown", "") if isinstance(note.get("bodyV2"), dict) else ""
        industry = ""
        utm = ""
        company = ""
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("**Industry:**"):
                industry = line.split(":**", 1)[1].split("(", 1)[0].strip()
            elif line.startswith("**Company:**"):
                company = line.split(":**", 1)[1].strip()
            elif line.startswith("- utm_source:"):
                utm = line.split(":", 1)[1].strip()

        leads.append({
            "note_id": note.get("id"),
            "created_at": note.get("createdAt", ""),
            "annual": annual,
            "fit": fit,
            "industry": industry,
            "company": company,
            "utm_source": utm,
            "person": person,
        })

    return leads


def _format_when(iso: str) -> str:
    if not iso:
        return ""
    try:
        # Twenty returns ISO 8601 with offset; tolerate 'Z'
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        if delta.total_seconds() < 60:
            return "just now"
        if delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if delta.total_seconds() < 86400:
            return f"{int(delta.total_seconds() / 3600)}h ago"
        return dt.strftime("%b %d, %Y")
    except Exception:
        return iso


def _render_html(leads: list[dict], current_user: dict) -> str:
    rows = []
    for lead in leads:
        person = lead.get("person") or {}
        full_name = f"{person.get('first_name', '')} {person.get('last_name', '')}".strip() or "—"
        email = person.get("email", "—")
        phone = person.get("phone", "—")
        company = lead.get("company") or "—"
        annual = lead.get("annual", "?")
        fit = lead.get("fit", "?")
        fit_color = {"high": "#FF6B35", "mid": "#fbbf24", "low": "#737373"}.get(fit, "#737373")
        industry = lead.get("industry") or "—"
        utm = lead.get("utm_source") or "direct"
        when = _format_when(lead.get("created_at", ""))
        twenty_link = f"https://crm.groundrushlabs.com/object/person/{person.get('id', '')}" if person.get("id") else "#"

        rows.append(f"""
        <tr>
          <td class="when">{when}</td>
          <td><strong>{full_name}</strong><br><span class="muted">{company}</span></td>
          <td><a href="mailto:{email}">{email}</a><br><span class="muted">{phone}</span></td>
          <td class="num">${annual}</td>
          <td><span class="fit" style="background:{fit_color};">{fit.upper()}</span></td>
          <td>{industry}</td>
          <td class="muted">{utm}</td>
          <td><a class="action" href="{twenty_link}" target="_blank" rel="noopener">Twenty →</a></td>
        </tr>
        """)

    rows_html = "\n".join(rows) if rows else """
        <tr><td colspan="8" class="empty">No audit leads yet. Run the calculator at
        <a href="/static/ai-savings-calc/index.html">/static/ai-savings-calc/</a>.</td></tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Audit Leads · GroundRush admin</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #0a0a0a; color: #fafafa; font-family: -apple-system, system-ui, 'Inter', sans-serif; }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px; }}
  header {{ display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #222; }}
  header h1 {{ margin: 0; font-size: 22px; font-weight: 800; }}
  header h1 .accent {{ color: #FF6B35; }}
  header .meta {{ color: #737373; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead th {{ text-align: left; padding: 12px 14px; background: #141414; color: #a3a3a3; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #222; }}
  tbody td {{ padding: 14px 14px; border-bottom: 1px solid #161616; vertical-align: top; }}
  tbody tr:hover {{ background: #111; }}
  td.num {{ font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #FF6B35; }}
  td.when {{ color: #737373; font-size: 12px; white-space: nowrap; }}
  .muted {{ color: #737373; font-size: 12px; }}
  .fit {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 10px; font-weight: 700; color: #000; letter-spacing: 0.04em; }}
  .empty {{ text-align: center; color: #737373; padding: 60px 0; font-size: 14px; }}
  a {{ color: #FF6B35; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  a.action {{ font-size: 12px; color: #a3a3a3; }}
  a.action:hover {{ color: #FF6B35; text-decoration: none; }}
  .stats {{ display: flex; gap: 24px; margin-bottom: 24px; }}
  .stat-card {{ flex: 1; padding: 16px 20px; background: #141414; border: 1px solid #222; border-radius: 10px; }}
  .stat-card .lbl {{ font-size: 10px; color: #a3a3a3; text-transform: uppercase; letter-spacing: 0.08em; }}
  .stat-card .val {{ font-size: 22px; font-weight: 800; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }}
  .stat-card .val.orange {{ color: #FF6B35; }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>AI Audit <span class="accent">Leads</span></h1>
      <div class="meta">{len(leads)} recent · signed in as <strong>{current_user.get('username', '?')}</strong></div>
    </header>

    <div class="stats">
      <div class="stat-card">
        <div class="lbl">Total leads (recent)</div>
        <div class="val">{len(leads)}</div>
      </div>
      <div class="stat-card">
        <div class="lbl">High-fit</div>
        <div class="val orange">{sum(1 for l in leads if l.get('fit') == 'high')}</div>
      </div>
      <div class="stat-card">
        <div class="lbl">Mid-fit</div>
        <div class="val">{sum(1 for l in leads if l.get('fit') == 'mid')}</div>
      </div>
      <div class="stat-card">
        <div class="lbl">Low-fit</div>
        <div class="val">{sum(1 for l in leads if l.get('fit') == 'low')}</div>
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th>When</th>
          <th>Person · Company</th>
          <th>Contact</th>
          <th>Annual</th>
          <th>Fit</th>
          <th>Industry</th>
          <th>Source</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def register(app: FastAPI) -> None:
    @app.get("/admin/leads", response_class=HTMLResponse)
    async def admin_leads(request: Request, user: dict = Depends(require_auth)):
        try:
            leads = _fetch_audit_leads(limit=50)
        except Exception as e:
            logger.exception("Failed to fetch audit leads")
            leads = []
        return HTMLResponse(_render_html(leads, user))

    @app.get("/admin/leads.json")
    async def admin_leads_json(user: dict = Depends(require_auth)):
        return {"leads": _fetch_audit_leads(limit=50)}
