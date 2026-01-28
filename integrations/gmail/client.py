"""Gmail integration - read, search, and send emails."""

import base64
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from googleapiclient.discovery import build

from core.security.google_oauth import get_credentials

logger = logging.getLogger(__name__)


def _get_service():
    creds = get_credentials()
    if not creds:
        raise RuntimeError("Google not connected. Visit /auth/google to authorize.")
    return build("gmail", "v1", credentials=creds)


def get_inbox(max_results: int = 10, query: str = "") -> list[dict]:
    """Get recent emails from inbox."""
    service = _get_service()
    q = query or "in:inbox"
    results = service.users().messages().list(
        userId="me", q=q, maxResults=max_results
    ).execute()

    messages = []
    for msg_ref in results.get("messages", []):
        msg = service.users().messages().get(
            userId="me", id=msg_ref["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()

        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        messages.append({
            "id": msg["id"],
            "from": headers.get("From", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "unread": "UNREAD" in msg.get("labelIds", []),
        })

    return messages


def read_email(message_id: str) -> dict:
    """Read a full email by ID."""
    service = _get_service()
    msg = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()

    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

    # Extract body
    body = ""
    payload = msg.get("payload", {})
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain":
                data = part.get("body", {}).get("data", "")
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                break
    elif "body" in payload and "data" in payload["body"]:
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    return {
        "id": msg["id"],
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "subject": headers.get("Subject", ""),
        "date": headers.get("Date", ""),
        "body": body,
        "labels": msg.get("labelIds", []),
    }


def send_email(to: str, subject: str, body: str, html: bool = False) -> dict:
    """Send an email."""
    service = _get_service()

    if html:
        message = MIMEMultipart("alternative")
        message.attach(MIMEText(body, "html"))
    else:
        message = MIMEText(body)

    message["to"] = to
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    result = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()

    logger.info(f"Email sent to {to}: {subject}")
    return {"id": result["id"], "status": "sent"}


def search_emails(query: str, max_results: int = 10) -> list[dict]:
    """Search emails with Gmail query syntax."""
    return get_inbox(max_results=max_results, query=query)


def get_unread_count() -> int:
    """Get count of unread emails."""
    service = _get_service()
    results = service.users().messages().list(
        userId="me", q="is:unread in:inbox", maxResults=1
    ).execute()
    return results.get("resultSizeEstimate", 0)
