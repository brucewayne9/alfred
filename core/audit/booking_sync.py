"""Booking → Twenty CRM sync.

Watches Mike's Gmail (`mjohnson@groundrushinc.com`) for Google Calendar
appointment-confirmation emails, parses the invitee's email + scheduled time,
matches the invitee to a recent AI Audit lead in Twenty, and appends a Note
"Booking confirmed for <datetime>" to the matching Person.

Activation requires `EMAIL_PASS_MJOHNSON_GW` (Gmail App Password) in
`config/.env`. Until that's set, the script logs and exits cleanly so the cron
doesn't error.

Run via `scripts/run_booking_sync.py` on the same 10-min cron as the existing
email monitor.
"""
from __future__ import annotations

import imaplib
import logging
import os
import re
import time
from datetime import datetime, timezone
from email import message_from_bytes
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from config.settings import settings
from integrations.base_crm import client as crm

logger = logging.getLogger(__name__)

# Mike's Gmail (the calendar owner — appointment confirmations land here)
MJOHNSON_GMAIL = "mjohnson@groundrushinc.com"
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# Identifiers in Google Calendar appointment-schedule confirmation emails.
GCAL_SENDERS = (
    "calendar-notification@google.com",
    "noreply@google.com",
)
GCAL_SUBJECT_HINTS = (
    "Confirmation:", "Invitation:", "Booking confirmed", "appointment scheduled",
)

# Subject usually looks like:
#   "Confirmation: 15-min audit call - Tue Aug 13, 2026 4:00 PM (EDT) — Jane Smith"
# Body has the invitee email near the top in a "Guests:" or "Who" line.
INVITEE_EMAIL_RE = re.compile(r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b")
WHEN_RE = re.compile(
    r"(\w{3,9}\s+\w{3}\s+\d{1,2},?\s*\d{4}.*?(?:AM|PM|am|pm))",
    re.IGNORECASE,
)

# How far back to consider a lead "recent" enough to attach the booking to.
# (If the booking comes from someone we never audited, we still log it but skip.)
RECENT_LEAD_DAYS = 30


def _imap_password() -> str | None:
    return os.environ.get("EMAIL_PASS_MJOHNSON_GW")


def _send_telegram(text: str) -> None:
    token = settings.telegram_bot_token
    chat = settings.telegram_chat_id
    if not (token and chat):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "Markdown",
                  "disable_web_page_preview": True},
            timeout=12,
        )
    except Exception as e:
        logger.warning(f"Telegram ping failed: {e}")


def _is_gcal_booking(msg) -> bool:
    sender = (msg.get("From") or "").lower()
    subject = (msg.get("Subject") or "").lower()
    if not any(s in sender for s in GCAL_SENDERS):
        return False
    return any(h.lower() in subject for h in GCAL_SUBJECT_HINTS)


def _extract_invitee_and_when(msg) -> tuple[str | None, str | None]:
    """Pull the invitee's email + scheduled time from a Google Calendar booking email."""
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                try:
                    body_parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
                except Exception:
                    continue
    else:
        try:
            body_parts.append(msg.get_payload(decode=True).decode("utf-8", errors="ignore"))
        except Exception:
            pass

    body = "\n".join(body_parts)
    subject = msg.get("Subject", "") or ""

    # Invitee email — find addresses NOT belonging to Mike's domain.
    invitee = None
    for addr in INVITEE_EMAIL_RE.findall(body):
        low = addr.lower()
        if "mjohnson" in low or "alfred" in low or "groundrush" in low or "google.com" in low:
            continue
        invitee = low
        break

    # When — pulled from subject as Google's "Tue Aug 13, 2026 4:00 PM" pattern.
    when = None
    m = WHEN_RE.search(subject) or WHEN_RE.search(body)
    if m:
        when = m.group(1)

    return invitee, when


def _process_booking(invitee_email: str, when: str | None, msg_subject: str) -> dict[str, Any]:
    """Find the matching Twenty lead and attach a 'Booking confirmed' note."""
    # Match the lead. search_people fuzzy-matches but doesn't always find email
    # via full-string. Fall back to direct list-and-filter.
    matches = crm.search_people(invitee_email, limit=10)
    person = None
    for p in matches:
        if p.get("email", "").lower() == invitee_email:
            person = p
            break

    if not person:
        # Direct fetch fallback (recent persons)
        try:
            data = crm._get("/rest/people", {"limit": 100, "order_by": "createdAt[DescNullsLast]"})
            for p in data.get("data", {}).get("people", []):
                em = (p.get("emails", {}) or {}).get("primaryEmail", "").lower()
                if em == invitee_email:
                    fn = (p.get("name", {}) or {}).get("firstName", "")
                    ln = (p.get("name", {}) or {}).get("lastName", "")
                    person = {
                        "id": p.get("id"),
                        "first_name": fn,
                        "last_name": ln,
                        "email": em,
                    }
                    break
        except Exception as e:
            logger.warning(f"Twenty list fallback failed for {invitee_email}: {e}")

    if not person:
        return {"status": "skipped", "reason": "no-twenty-match", "email": invitee_email}

    note_title = "Audit call BOOKED" + (f" — {when}" if when else "")
    note_body = (
        f"**Booked via Google Calendar appointment scheduler**\n\n"
        f"Invitee: {invitee_email}\n"
        f"Subject: {msg_subject}\n"
        f"When: {when or '(parse failed — see raw email)'}\n\n"
        f"_Synced automatically from {MJOHNSON_GMAIL} inbox by core/audit/booking_sync.py_"
    )
    try:
        crm.create_note_for_person(title=note_title, person_id=person["id"], body=note_body)
    except Exception as e:
        logger.exception(f"Twenty note create failed for {invitee_email}: {e}")
        return {"status": "error", "reason": "twenty-note-failed", "email": invitee_email}

    full_name = f"{person.get('first_name', '')} {person.get('last_name', '')}".strip()
    _send_telegram(
        f"🟢 *BOOKING* — {full_name} ({invitee_email}) just booked an audit call.\n\n"
        f"📅 {when or 'time unknown'}\n"
        f"[Open in Twenty](https://crm.groundrushlabs.com/object/person/{person['id']})"
    )

    return {
        "status": "ok",
        "email": invitee_email,
        "person_id": person["id"],
        "when": when,
    }


def run_once() -> dict[str, Any]:
    """Single sweep: scan recent unread Gmail messages and sync any new bookings.

    Returns a summary dict for cron logging.
    """
    pwd = _imap_password()
    if not pwd:
        return {
            "status": "skipped",
            "reason": "EMAIL_PASS_MJOHNSON_GW not set — booking sync inactive. "
                      "Add a Gmail App Password to config/.env to activate.",
        }

    summary = {"status": "ok", "scanned": 0, "matched": 0, "synced": [], "skipped": []}
    try:
        imap = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        imap.login(MJOHNSON_GMAIL, pwd)
        imap.select("INBOX")
        # Look back 1 day's worth, unseen, to keep the work tight per cron tick.
        typ, data = imap.search(None, "UNSEEN", "FROM", "calendar-notification@google.com")
        ids = data[0].split() if data and data[0] else []
        summary["scanned"] = len(ids)

        for msg_id in ids:
            try:
                typ, raw = imap.fetch(msg_id, "(RFC822)")
                if typ != "OK" or not raw or not raw[0]:
                    continue
                msg = message_from_bytes(raw[0][1])
                if not _is_gcal_booking(msg):
                    continue
                invitee, when = _extract_invitee_and_when(msg)
                if not invitee:
                    summary["skipped"].append({"reason": "no-invitee", "subject": msg.get("Subject")})
                    continue
                result = _process_booking(invitee, when, msg.get("Subject", ""))
                if result.get("status") == "ok":
                    summary["matched"] += 1
                    summary["synced"].append(result)
                else:
                    summary["skipped"].append(result)
            except Exception as e:
                logger.exception(f"Booking sync: failed to process message {msg_id}: {e}")
                continue

        imap.logout()
    except Exception as e:
        logger.exception(f"Booking sync IMAP error: {e}")
        return {"status": "error", "error": str(e)}

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_once()
    print(result)
