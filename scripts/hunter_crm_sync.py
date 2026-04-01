#!/usr/bin/env python3
"""Hunter.io → Twenty CRM sync for Rod Wave Fall 2026 Sponsorship campaign.

Polls Hunter.io leads API, detects status changes, updates CRM opportunities,
and sends Telegram alerts for high-intent signals.

Run via cron once daily at 12 PM EST (16:00 UTC):
  0 16 * * * /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/hunter_crm_sync.py >> /home/aialfred/alfred/data/hunter_sync.log 2>&1
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("hunter_crm_sync")

# --- Config ---
HUNTER_API_KEY = "20180d31f84d8278f18955bf0dde141b0026cbaa"
HUNTER_CAMPAIGN_ID = 801983
CRM_BASE = settings.base_crm_url.rstrip("/")
CRM_KEY = settings.base_crm_api_key
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", settings.telegram_bot_token if hasattr(settings, "telegram_bot_token") else "")
TELEGRAM_CHAT_ID = "7582976864"
STATE_FILE = Path("/home/aialfred/alfred/data/hunter_sync_state.json")

# Hunter sending_status values: pending, sent, opened, clicked, replied, bounced, unsubscribed
STAGE_MAP = {
    "pending": "NEW",
    "sent": "NEW",
    "opened": "SCREENING",
    "clicked": "SCREENING",
    "replied": "MEETING",
    "bounced": "NEW",
    "unsubscribed": "NEW",
}

HIGH_INTENT = {"clicked", "replied"}

# Email → CRM opportunity ID mapping (seeded from creation)
OPP_MAP = {
    "nike.com": "bd0000d1-eadf-459c-9015-c73b5eeedd39",
    "uber.com": "9b306ca3-13c4-4b7d-90ee-3f92d79249b9",
    "nixon.com": "7519a9a1-342f-408b-8f06-f43be67c803e",
    "ubisoft.com": "139f22f4-f994-4272-98c9-ca1ef40b767b",
    "reebok.com": "4094dcaa-3b5f-40fc-8228-ba208bba6283",
    "progressive.com": "c1149c42-3328-4bca-b4a8-2545e6a84eca",
    "diesel.com": "fabe20df-0a8b-4cb0-95e7-1a4b410ffc9e",
    "dtlr.com": "27de258a-b136-464b-b3ba-cb58141fcf2c",
    "robinhood.com": "ac340370-af51-493c-ad07-84fac4ce2bbd",
    "rockstargames.com": "4fbd88f8-803a-4f02-8b9f-4e1097d2a049",
    "ford.com": "876bfa5e-6023-4a2e-9c80-70b3228df7a3",
}

COMPANY_MAP = {
    "nike.com": "Nike",
    "uber.com": "Uber",
    "nixon.com": "Nixon",
    "ubisoft.com": "Ubisoft",
    "reebok.com": "Reebok",
    "progressive.com": "Progressive",
    "diesel.com": "Diesel",
    "dtlr.com": "DTLR",
    "robinhood.com": "Robinhood",
    "rockstargames.com": "Rockstar Games",
    "ford.com": "Ford Motor",
}


def crm_headers():
    return {"Authorization": f"Bearer {CRM_KEY}", "Content-Type": "application/json"}


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_hunter_leads():
    """Fetch all leads from Hunter.io campaign."""
    resp = requests.get(
        f"https://api.hunter.io/v2/leads",
        params={"api_key": HUNTER_API_KEY, "limit": 100},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("data", {}).get("leads", [])


def get_domain(email):
    """Extract domain from email."""
    return email.split("@")[-1].lower() if "@" in email else ""


def update_crm_stage(opp_id, stage):
    """Update opportunity stage in CRM."""
    resp = requests.patch(
        f"{CRM_BASE}/rest/opportunities/{opp_id}",
        headers=crm_headers(),
        json={"stage": stage},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def add_crm_note(opp_id, title, body):
    """Add note to opportunity (create note, then link via noteTargets)."""
    # Step 1: Create the note
    resp = requests.post(
        f"{CRM_BASE}/rest/notes",
        headers=crm_headers(),
        json={"title": title, "bodyV2": {"markdown": body}},
        timeout=15,
    )
    resp.raise_for_status()
    note_data = resp.json().get("data", {}).get("createNote", resp.json().get("data", {}))
    note_id = note_data.get("id")

    # Step 2: Link to opportunity
    if note_id:
        resp2 = requests.post(
            f"{CRM_BASE}/rest/noteTargets",
            headers=crm_headers(),
            json={"noteId": note_id, "opportunityId": opp_id},
            timeout=15,
        )
        resp2.raise_for_status()
    return note_data


def send_telegram(message):
    """Send alert to Mike via Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        log.warning("No Telegram bot token - skipping alert")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.error(f"Telegram alert failed: {e}")


def sync():
    """Sync Hunter leads to CRM silently — no Telegram alerts.

    CRM stages still advance and notes are added, but no individual pings.
    Use weekly_digest() for the Telegram summary.
    """
    state = load_state()
    leads = get_hunter_leads()
    changes = 0
    now = datetime.now(timezone.utc).isoformat()

    for lead in leads:
        email = lead.get("email", "").lower()
        status = lead.get("sending_status", "pending")
        prev_status = state.get(email, {}).get("status", "unknown")

        if status == prev_status:
            continue

        domain = get_domain(email)
        opp_id = OPP_MAP.get(domain)
        company = COMPANY_MAP.get(domain, domain)
        name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()
        title = lead.get("position", "")

        log.info(f"STATUS CHANGE: {email} ({company}) {prev_status} -> {status}")
        changes += 1

        # Update CRM stage (only advance, never go backwards)
        if opp_id:
            new_stage = STAGE_MAP.get(status, "NEW")
            try:
                opp_resp = requests.get(
                    f"{CRM_BASE}/rest/opportunities/{opp_id}",
                    headers=crm_headers(),
                    timeout=15,
                )
                opp_data = opp_resp.json().get("data", {}).get("opportunity", {})
                current = opp_data.get("stage", "NEW")
                stage_order = ["NEW", "SCREENING", "MEETING", "PROPOSAL", "CUSTOMER"]
                if stage_order.index(new_stage) > stage_order.index(current):
                    update_crm_stage(opp_id, new_stage)
                    log.info(f"  CRM stage: {current} -> {new_stage}")
            except Exception as e:
                log.error(f"  CRM stage update failed: {e}")

            # Add CRM note
            try:
                status_labels = {
                    "sent": "Email Sent",
                    "opened": "Email Opened",
                    "clicked": "Link Clicked",
                    "replied": "Email Replied",
                    "bounced": "Email Bounced",
                }
                label = status_labels.get(status, status)
                note_body = (
                    f"**Event:** {label}\n"
                    f"**Contact:** {name} ({title})\n"
                    f"**Email:** {email}\n"
                    f"**Time:** {now}\n"
                    f"**Previous Status:** {prev_status}"
                )
                add_crm_note(opp_id, f"Hunter.io: {label} - {name}", note_body)
                log.info(f"  CRM note added")
            except Exception as e:
                log.error(f"  CRM note failed: {e}")

        # Update state
        state[email] = {
            "status": status,
            "last_updated": now,
            "name": name,
            "company": company,
        }

    save_state(state)

    if changes:
        log.info(f"Sync complete: {changes} status changes detected")
    else:
        log.info("Sync complete: no changes")

    return changes


def weekly_digest():
    """Build and send a weekly Telegram digest of all Hunter.io campaign activity."""
    leads = get_hunter_leads()
    state = load_state()
    now = datetime.now(timezone.utc)

    # Bucket leads by status
    by_status = {}
    by_company = {}
    for lead in leads:
        email = lead.get("email", "").lower()
        status = lead.get("sending_status", "pending")
        domain = get_domain(email)
        company = COMPANY_MAP.get(domain, domain)
        name = f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip()

        by_status.setdefault(status, []).append({"name": name, "company": company, "email": email})
        by_company.setdefault(company, []).append({"name": name, "status": status})

    # Count changes since last digest
    last_digest = state.get("_last_digest", "")
    changes_since = 0
    for email, info in state.items():
        if email.startswith("_"):
            continue
        updated = info.get("last_updated", "")
        if updated > last_digest:
            changes_since += 1

    total = len(leads)
    replied = by_status.get("replied", [])
    clicked = by_status.get("clicked", [])
    opened = by_status.get("opened", [])
    sent = by_status.get("sent", [])
    bounced = by_status.get("bounced", [])
    pending = by_status.get("pending", [])

    # Build message
    msg = f"<b>HUNTER.IO WEEKLY DIGEST</b>\n"
    msg += f"<i>Week of {now.strftime('%b %d, %Y')}</i>\n\n"

    msg += f"<b>Pipeline:</b> {total} leads across {len(by_company)} companies\n"
    if changes_since:
        msg += f"<b>Changes this week:</b> {changes_since}\n"
    msg += "\n"

    # High intent first
    if replied:
        msg += f"<b>REPLIED ({len(replied)}):</b>\n"
        for r in replied:
            msg += f"  - {r['name']} ({r['company']})\n"
        msg += "\n"

    if clicked:
        msg += f"<b>CLICKED ({len(clicked)}):</b>\n"
        for c in clicked:
            msg += f"  - {c['name']} ({c['company']})\n"
        msg += "\n"

    # Summary counts for the rest
    msg += "<b>Status breakdown:</b>\n"
    for status_name, emoji in [("replied", "✅"), ("clicked", "🔥"), ("opened", "👀"), ("sent", "📧"), ("pending", "⏳"), ("bounced", "❌")]:
        count = len(by_status.get(status_name, []))
        if count:
            msg += f"  {emoji} {status_name}: {count}\n"

    # Company summary
    msg += f"\n<b>By company:</b>\n"
    for company in sorted(by_company.keys()):
        contacts = by_company[company]
        best = max(contacts, key=lambda x: ["pending", "sent", "opened", "clicked", "replied"].index(x["status"]) if x["status"] in ["pending", "sent", "opened", "clicked", "replied"] else -1)
        msg += f"  - {company}: {len(contacts)} leads, best: {best['status']}\n"

    send_telegram(msg)
    log.info("Weekly digest sent to Telegram")

    # Save digest timestamp
    state["_last_digest"] = now.isoformat()
    save_state(state)

    return msg


def generate_report():
    """Generate a summary report of all lead statuses for Claw/Alfred to use."""
    leads = get_hunter_leads()
    by_company = {}
    for lead in leads:
        domain = get_domain(lead.get("email", ""))
        company = COMPANY_MAP.get(domain, domain)
        if company not in by_company:
            by_company[company] = []
        by_company[company].append({
            "name": f"{lead.get('first_name', '')} {lead.get('last_name', '')}".strip(),
            "email": lead.get("email"),
            "title": lead.get("position", ""),
            "status": lead.get("sending_status", "pending"),
            "last_activity": lead.get("last_activity_at"),
            "last_contacted": lead.get("last_contacted_at"),
        })

    report = "# Rod Wave Fall 2026 Sponsorship Pipeline Report\n\n"
    report += f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"

    status_counts = {"pending": 0, "sent": 0, "opened": 0, "clicked": 0, "replied": 0, "bounced": 0}
    for company, contacts in sorted(by_company.items()):
        report += f"## {company}\n"
        for c in contacts:
            status_counts[c["status"]] = status_counts.get(c["status"], 0) + 1
            emoji = {"pending": "⏳", "sent": "📧", "opened": "👀", "clicked": "🔥", "replied": "✅", "bounced": "❌"}.get(c["status"], "❓")
            report += f"- {emoji} **{c['name']}** ({c['title']}) - {c['status']}\n"
        report += "\n"

    report += "## Summary\n"
    for status, count in status_counts.items():
        if count > 0:
            report += f"- {status}: {count}\n"
    report += f"- **Total leads:** {sum(status_counts.values())}\n"

    return report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        print(generate_report())
    elif len(sys.argv) > 1 and sys.argv[1] == "digest":
        weekly_digest()
    else:
        sync()
