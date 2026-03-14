#!/usr/bin/env python3
"""Alfred Email Monitor — checks all business email accounts every 10 minutes.

Reads unread emails, determines if they need a response, and either:
- Auto-replies to general questions (Groundrush info, availability, services)
- Forwards urgent/personal items to Mike via Telegram
- Ignores spam/no-reply/marketing emails

Safety guardrails:
- NEVER shares API keys, credentials, CRM access, or internal system details
- Responds to meeting requests with Mike's Google Calendar scheduling link
- NEVER commits Mike to financial decisions
- NEVER shares personal/family information
- Only answers general business questions about Groundrush Inc/Labs/Cloud
- Offers to relay messages to Mike for anything beyond general info

Cron: */10 * * * * cd /home/aialfred/alfred && /home/aialfred/.pyenv/versions/3.11.11/bin/python3 scripts/alfred_email_monitor.py >> /tmp/alfred_email_monitor.log 2>&1
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

from integrations.email.client import EmailClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("email_monitor")

import requests

STATE_FILE = Path("/home/aialfred/alfred/data/email_monitor_state.json")
TASK_FILE = Path("/home/aialfred/alfred/data/daily_tasks.json")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = "7582976864"

# Accounts to monitor (alfred is Alfred's outbound identity)
MONITOR_ACCOUNTS = ["alfred", "groundrush", "groundrush info", "support", "loovacast", "rucktalk"]

# Senders to ignore
IGNORE_PATTERNS = [
    r"no-?reply", r"noreply", r"donotreply", r"mailer-daemon", r"postmaster",
    r"notification", r"newsletter", r"marketing@", r"promo@", r"unsubscribe",
    r"@google\.com$", r"@linkedin\.com$", r"@facebook\.com$", r"@twitter\.com$",
    r"@github\.com$", r"@stripe\.com$", r"@amazon\.com$", r"@paypal\.com$",
    r"@wpmudev\.com$", r"@wordpress\.com$", r"@mailchimp\.com$", r"@hubspot\.com$",
    r"@sendgrid\.com$", r"@constantcontact\.com$", r"contact@", r"info@.*\.com$",
    r"updates@", r"digest@", r"news@", r"team@", r"hello@",
]

# Subjects to ignore
IGNORE_SUBJECTS = [
    "unsubscribe", "newsletter", "weekly digest", "promotion", "sale", "discount",
    "offer expires", "new feature", "product update", "release notes", "what's new",
    "tips and tricks", "getting started", "your weekly", "your monthly",
]

# Keywords that signal someone is asking a question
QUESTION_SIGNALS = [
    "?", "how do", "what is", "can you", "do you", "are you", "does",
    "interested in", "looking for", "inquire", "inquiry", "information",
    "quote", "pricing", "services", "available", "schedule", "meeting",
    "reach", "contact", "speak with", "talk to", "help with",
]

# Things Alfred should NEVER reveal
FORBIDDEN_TOPICS = [
    "api key", "password", "credential", "secret", "token", "database",
    "ssh", "server ip", "internal", "crm access", "admin",
    "social security", "bank account", "credit card",
]

# Groundrush business info Alfred CAN share
BUSINESS_INFO = {
    "company": "Groundrush Inc is a full-service digital agency specializing in marketing, web development, music industry services, and technology solutions.",
    "services": "We offer web development, digital marketing, social media management, SEO, music tour marketing, radio streaming (LoovaCast), and AI-powered business solutions.",
    "brands": "Our brands include Groundrush Labs (technology), Groundrush Cloud (hosting), LoovaCast (radio streaming), and RuckTalk (lifestyle media).",
    "contact": "You can reach us at info@groundrushlabs.com or through our website. For urgent matters, I can relay your message directly to Mike Johnson, our President.",
    "location": "We're based in the Atlanta, Georgia area.",
    "owner": "Mike Johnson is the President and Owner of Groundrush Inc.",
}

email_client = EmailClient()

OLLAMA_URL = "http://localhost:11434"
VISION_MODELS = ["kimi-k2.5:cloud", "gemini-3-pro-preview:latest"]


def analyze_image(image_b64: str, context: str = "Describe this image briefly.") -> str:
    """Analyze an image using Ollama cloud vision models."""
    for model in VISION_MODELS:
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": context, "images": [image_b64]}],
                    "stream": False,
                },
                timeout=60,
            )
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if content:
                return content
        except Exception:
            continue
    return ""


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_ids": {}, "last_run": None}


def save_state(state):
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def should_ignore(from_addr: str, subject: str) -> bool:
    """Check if email should be ignored (spam, marketing, no-reply)."""
    from_lower = from_addr.lower()
    subj_lower = subject.lower()

    # CRITICAL: Never reply to our own addresses (prevents reply loops)
    own_addresses = [
        "alfred@groundrushlabs.com", "oracle@groundrushlabs.com",
        "lumabot@groundrushlabs.com",
        "mjohnson@groundrushlabs.com", "info@groundrushlabs.com",
        "support@loovacast.com", "info@loovacast.com",
        "info@rucktalk.com", "mjohnson@groundrushinc.com",
    ]
    for own in own_addresses:
        if own in from_lower:
            return True

    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, from_lower):
            return True

    # Ignore obvious marketing/product emails
    for ms in IGNORE_SUBJECTS:
        if ms in subj_lower:
            return True

    return False


def is_question(subject: str, body: str) -> bool:
    """Check if the email is asking a question or requesting info."""
    combined = (subject + " " + body).lower()
    return any(signal in combined for signal in QUESTION_SIGNALS)


def contains_forbidden(body: str) -> bool:
    """Check if the email is asking for sensitive info."""
    body_lower = body.lower()
    return any(topic in body_lower for topic in FORBIDDEN_TOPICS)


def classify_email(subject: str, body: str, from_addr: str) -> str:
    """Classify email into: auto_reply, forward_to_mike, ignore."""
    if should_ignore(from_addr, subject):
        return "ignore"

    # Check if from Mike himself
    if "mjohnson@groundrush" in from_addr.lower() or "mjohnson@groundrushinc" in from_addr.lower() or "mike" in from_addr.lower():
        return "ignore"  # Don't auto-reply to Mike

    combined = (subject + " " + body).lower()

    # Urgent/personal → forward to Mike
    urgent_words = ["urgent", "asap", "emergency", "lawsuit", "legal", "invoice", "payment due", "contract"]
    if any(w in combined for w in urgent_words):
        return "forward_to_mike"

    # Asking for sensitive info → forward to Mike
    if contains_forbidden(body):
        return "forward_to_mike"

    # Asking to schedule a meeting → auto reply with calendar link
    if any(w in combined for w in ["schedule a meeting", "set up a call", "book a time", "calendar", "meet with", "meeting with", "schedule a call", "set up a meeting"]):
        return "auto_reply"

    # General question → auto reply
    if is_question(subject, body):
        return "auto_reply"

    # Default: forward anything non-trivial to Mike
    if len(body.strip()) > 50:
        return "forward_to_mike"

    return "ignore"


def generate_auto_reply(subject: str, body: str, from_name: str, account: str) -> str:
    """Generate a contextual auto-reply using LLM + Grey Matter knowledge."""
    # Try LLM-powered reply first, fall back to template
    try:
        return _llm_auto_reply(subject, body, from_name, account)
    except Exception as e:
        log.warning(f"LLM reply failed, using template fallback: {e}")
        return _template_auto_reply(subject, body, from_name, account)


def _llm_auto_reply(subject: str, body: str, from_name: str, account: str) -> str:
    """Generate reply using Ollama LLM with Grey Matter context."""
    # Query Grey Matter for relevant context
    knowledge_context = _query_grey_matter(f"{subject} {body[:200]}")

    first_name = from_name.split()[0] if from_name else "there"
    system_prompt = f"""You are Alfred, the AI assistant for Mike Johnson at Groundrush Inc.
You are replying to an email. Be professional, warm, concise (3-5 sentences max).

BUSINESS INFO:
{chr(10).join(f'- {k}: {v}' for k, v in BUSINESS_INFO.items())}

RULES — STRICTLY FOLLOW:
- NEVER share API keys, passwords, server IPs, internal tools, CRM access, or architecture details
- NEVER commit to meetings, pricing, contracts, or financial decisions on Mike's behalf
- NEVER reveal family info, personal details, or strategy documents
- NEVER make up services, pricing, or capabilities that don't exist
- If they ask about pricing: say you'll connect them with Mike
- If they want to schedule: share the calendar link https://calendar.app.google/MCvRTvdtSULr7Jve6
- If unsure about something: say you'll pass their message to Mike
- Sign off as: Alfred, AI Assistant, Groundrush Inc"""

    if knowledge_context:
        system_prompt += f"\n\nRELEVANT KNOWLEDGE:\n{knowledge_context[:1000]}"

    user_msg = f"Reply to this email from {first_name} ({from_name}):\nSubject: {subject}\n\n{body[:1500]}"

    # Try minimax-m2 first (fast), fall back to gpt-oss
    for model in ["minimax-m2:cloud", "gpt-oss:120b-cloud"]:
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 500},
                },
                timeout=30,
            )
            data = resp.json()
            reply = data.get("message", {}).get("content", "").strip()
            if reply and len(reply) > 20:
                # Safety check: scan for forbidden content
                reply_lower = reply.lower()
                for forbidden in FORBIDDEN_TOPICS:
                    if forbidden in reply_lower:
                        log.warning(f"LLM reply contained forbidden topic '{forbidden}', using template")
                        return _template_auto_reply(subject, body, from_name, account)
                log.info(f"LLM reply generated via {model} ({len(reply)} chars)")
                return reply
        except Exception as e:
            log.warning(f"LLM call to {model} failed: {e}")
            continue

    raise RuntimeError("All LLM models failed")


def _query_grey_matter(query: str) -> str:
    """Query LightRAG/Grey Matter for relevant business context."""
    try:
        import urllib.request
        import base64
        url = "https://greymatter.groundrushlabs.com"
        user = "brucewayne9"
        passwd = "AlwaysGive100%"
        auth = base64.b64encode(f"{user}:{passwd}".encode()).decode()
        payload = json.dumps({
            "query": query[:300],
            "mode": "hybrid",
            "only_need_context": True,
        }).encode()
        req = urllib.request.Request(
            f"{url}/query",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Basic {auth}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "")[:1500]
    except Exception as e:
        log.debug(f"Grey Matter query failed: {e}")
        return ""


def _template_auto_reply(subject: str, body: str, from_name: str, account: str) -> str:
    """Fallback template-based reply when LLM is unavailable."""
    combined = (subject + " " + body).lower()

    reply_parts = [f"Hi {from_name.split()[0] if from_name else 'there'},\n"]
    reply_parts.append("Thank you for reaching out! I'm Alfred, Mike Johnson's AI assistant at Groundrush.\n")

    if any(w in combined for w in ["service", "what do you", "offer", "help with", "looking for"]):
        reply_parts.append(BUSINESS_INFO["services"] + "\n")

    if any(w in combined for w in ["who", "about", "company", "groundrush"]):
        reply_parts.append(BUSINESS_INFO["company"] + "\n")

    if any(w in combined for w in ["brand", "loova", "ruck", "radio"]):
        reply_parts.append(BUSINESS_INFO["brands"] + "\n")

    if any(w in combined for w in ["where", "location", "based", "office"]):
        reply_parts.append(BUSINESS_INFO["location"] + "\n")

    if any(w in combined for w in ["price", "cost", "quote", "rate", "pricing"]):
        reply_parts.append("For pricing and quotes, I'd like to connect you directly with Mike to discuss your specific needs and budget.\n")

    if any(w in combined for w in ["available", "schedule", "meet", "call", "reach mike", "meeting", "book a time", "set up a call"]):
        reply_parts.append("Absolutely! You can schedule a meeting with Mike directly through his calendar here:\nhttps://calendar.app.google/MCvRTvdtSULr7Jve6\n\nJust pick a time that works for you and you'll be all set.\n")

    if len(reply_parts) <= 2:
        reply_parts.append("I've noted your message and will make sure Mike sees it. If there's anything specific I can help with in the meantime, please let me know.\n")

    reply_parts.append("If you'd like to speak with Mike directly, just let me know and I'll pass along your message right away.\n")
    reply_parts.append("Best regards,\nAlfred\nAI Assistant, Groundrush Inc\ninfo@groundrushlabs.com")

    return "\n".join(reply_parts)


def send_telegram(message: str):
    """Send notification to Mike via Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        log.warning("No Telegram bot token configured")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


def process_account(account: str, state: dict):
    """Check one email account for new messages."""
    log.info(f"Checking {account}...")

    try:
        inbox = email_client.get_inbox(account, limit=10)
    except Exception as e:
        log.error(f"  Failed to get inbox for {account}: {e}")
        return 0

    if "error" in inbox:
        log.error(f"  Inbox error for {account}: {inbox['error']}")
        return 0

    messages = inbox.get("messages", [])
    processed = 0
    account_key = account.replace(" ", "_")

    if account_key not in state["processed_ids"]:
        state["processed_ids"][account_key] = []

    for msg in messages:
        msg_id = msg.get("id", "")
        # Create a unique key from account + message id
        unique_key = f"{account_key}:{msg_id}"

        if unique_key in state["processed_ids"][account_key]:
            continue

        from_addr = msg.get("from", "")
        subject = msg.get("subject", "")
        snippet = msg.get("snippet", "")

        # Read the full email for classification
        try:
            full = email_client.read_email(account, msg_id)
            body = full.get("body", snippet)
        except Exception:
            body = snippet

        action = classify_email(subject, body, from_addr)
        log.info(f"  [{action}] From: {from_addr[:40]} | Subject: {subject[:50]}")

        if action == "auto_reply":
            # Extract sender name
            from_name = from_addr.split("<")[0].strip().strip('"')
            reply = generate_auto_reply(subject, body, from_name, account)

            # Extract reply-to email
            to_email = from_addr
            email_match = re.search(r'<(.+?)>', from_addr)
            if email_match:
                to_email = email_match.group(1)

            # Send reply from alfred (Alfred's identity)
            try:
                result = email_client.send_email(
                    "alfred",
                    to_email,
                    f"Re: {subject}",
                    reply,
                )
                if "error" not in result:
                    log.info(f"  Auto-replied to {to_email}")
                    # Also notify Mike
                    send_telegram(
                        f"📧 <b>Alfred Auto-Reply Sent</b>\n\n"
                        f"<b>To:</b> {to_email}\n"
                        f"<b>Subject:</b> {subject[:60]}\n"
                        f"<b>Account:</b> {account}\n"
                        f"<b>Snippet:</b> {snippet[:100]}..."
                    )
                else:
                    log.error(f"  Reply failed: {result['error']}")
            except Exception as e:
                log.error(f"  Reply exception: {e}")

        elif action == "forward_to_mike":
            # Check for image attachments and analyze them
            attachments = full.get("attachments", []) if isinstance(full, dict) else []
            image_descriptions = []
            for att in attachments:
                if att.get("data_b64") and att.get("mimetype", "").startswith("image/"):
                    desc = analyze_image(
                        att["data_b64"],
                        f"This image was attached to an email with subject: '{subject}'. Describe what you see concisely."
                    )
                    if desc:
                        image_descriptions.append(f"📎 <b>{att.get('filename', 'image')}</b>: {desc[:200]}")

            image_section = ""
            if image_descriptions:
                image_section = "\n\n<b>Attachments:</b>\n" + "\n".join(image_descriptions)

            send_telegram(
                f"📬 <b>Email Needs Attention</b>\n\n"
                f"<b>From:</b> {from_addr[:60]}\n"
                f"<b>Subject:</b> {subject[:80]}\n"
                f"<b>Account:</b> {account}\n\n"
                f"<b>Preview:</b>\n{snippet[:300]}..."
                f"{image_section}\n\n"
                f"<i>Reply here if you want me to respond on your behalf.</i>"
            )
            log.info(f"  Forwarded to Mike via Telegram")

        # Mark as processed
        state["processed_ids"][account_key].append(unique_key)
        processed += 1

    # Keep only last 200 processed IDs per account to prevent state bloat
    if len(state["processed_ids"][account_key]) > 200:
        state["processed_ids"][account_key] = state["processed_ids"][account_key][-200:]

    return processed


def run():
    """Main monitor loop."""
    state = load_state()
    total = 0

    for account in MONITOR_ACCOUNTS:
        try:
            count = process_account(account, state)
            total += count
        except Exception as e:
            log.error(f"Account {account} failed: {e}")

    save_state(state)

    if total:
        log.info(f"Processed {total} new emails across {len(MONITOR_ACCOUNTS)} accounts")
    else:
        log.info("No new emails to process")


if __name__ == "__main__":
    run()
