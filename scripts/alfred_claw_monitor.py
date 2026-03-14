#!/usr/bin/env python3
"""
Alfred Claw Health Monitor
Runs on Alfred Labs (105) every 15 minutes via cron.
Monitors Alfred Claw (OpenClaw) LOCALLY on this same machine (105) for:
  - Gateway running
  - Telegram channel OK
  - Sessions not at 100% context
  - No stuck/frozen state

NOTE: As of 2026-03-03, OpenClaw has been migrated from 101 to 105.
All checks are now local (no SSH). The gateway runs as a systemd user
service on this machine.

Flow:
  1. Detect issue → email Mike with "reply FIX IT or LEAVE IT"
  2. Each 15min check: read alfred inbox for Mike's reply
  3. "fix it" → auto-fix → send recovery report
  4. "leave it" → acknowledge, stop alerting, don't touch
  5. No reply after 1 hour → auto-fix anyway (safety net)
  6. On recovery → email Mike what was fixed + log to LightRAG

State file: /home/aialfred/alfred/data/claw_monitor_state.json
"""

import os
import sys
import json
import subprocess
import smtplib
import imaplib
import email as email_lib
import time
import re
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from datetime import datetime, timedelta

# Load env from Alfred Labs config
from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

# Config — OpenClaw is LOCAL on this machine (105) as of 2026-03-03
OPENCLAW_HOME = "/home/aialfred/.openclaw"
OPENCLAW_CLI = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/openclaw"
NODE_BIN = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/node"
STATE_FILE = "/home/aialfred/alfred/data/claw_monitor_state.json"
LOG_FILE = "/home/aialfred/alfred/data/claw_monitor.log"

# Email config
MAIL_SERVER = "mail.doowoprnb.com"
SMTP_PORT = 465
IMAP_PORT = 993
SMTP_LOGIN_EMAIL = "alfred@groundrushlabs.com"    # Auth account
FROM_EMAIL = "alfred@groundrushlabs.com"            # Must match auth (server enforces)
FROM_NAME = "Alfred Claw"                          # Display name shows as Alfred
TO_EMAIL = "mjohnson@groundrushinc.com"
MIKE_EMAILS = ["mjohnson@groundrushinc.com", "mjohnson@groundrushlabs.com"]
EMAIL_PASSWORD = os.environ.get("EMAIL_PASS_ALFRED", "")

# Timing
AUTO_FIX_TIMEOUT_MIN = 60  # Auto-fix if no reply after 60 minutes
CONSECUTIVE_FAILURES_BEFORE_ALERT = 2  # Require 2 consecutive unhealthy checks before alerting


def log(msg):
    """Log to file and stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_state():
    """Load previous monitor state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "unknown",
        "last_check": None,
        "failures": 0,
        "alert_sent": False,
        "awaiting_command": False,
        "command_received": None,
        "down_since": None,
        "issues": [],
        "fix_actions": [],
        "acknowledged": False,
        "processed_escalations": [],
    }


def save_state(state):
    """Save monitor state."""
    state["last_check"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def local_exec(cmd, timeout=30):
    """Execute command locally (OpenClaw is on this machine now)."""
    try:
        env = os.environ.copy()
        env["PATH"] = f"/home/aialfred/.nvm/versions/node/v22.22.0/bin:{env.get('PATH', '')}"
        env["HOME"] = "/home/aialfred"
        # Required for systemctl --user and journalctl --user from cron
        env["XDG_RUNTIME_DIR"] = "/run/user/1000"
        env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/run/user/1000/bus"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timeout"
    except Exception as e:
        return -1, "", str(e)


# Keep ssh_exec as alias for backward compat (but it now runs locally)
ssh_exec = local_exec


# ============================================================
# HEALTH CHECKS
# ============================================================

def check_health():
    """Run health checks on Alfred Claw. Returns (healthy: bool, issues: list, details: dict)."""
    issues = []
    details = {}

    # Check 1: Gateway service running? (local systemd check)
    rc, out, err = local_exec("systemctl --user is-active openclaw-gateway.service")
    gw_status = out.strip()
    # "activating" is a transient systemd state (service starting up).
    # Wait briefly and re-check before declaring it down.
    if gw_status == "activating":
        time.sleep(15)
        rc, out, err = local_exec("systemctl --user is-active openclaw-gateway.service")
        gw_status = out.strip()
    if gw_status != "active":
        issues.append(f"Gateway service not active: {gw_status}")
        details["gateway"] = gw_status
    else:
        details["gateway"] = "active"

    # Check 2: Full status check
    rc, out, err = local_exec("openclaw status --json 2>/dev/null", timeout=30)
    if rc != 0:
        issues.append(f"openclaw status failed: {err}")
        details["status_cmd"] = "FAIL"
    else:
        try:
            status = json.loads(out)
            details["status_cmd"] = "OK"

            # Check Telegram
            channels = status.get("channelSummary", [])
            telegram_ok = any("Telegram" in c for c in channels)
            details["telegram"] = "OK" if telegram_ok else "DOWN"
            if not telegram_ok:
                issues.append("Telegram channel not configured or down")

            # Check sessions
            sessions = status.get("sessions", {})
            recent = sessions.get("recent", [])
            details["session_count"] = len(recent)

            maxed_sessions = []
            for s in recent:
                pct = s.get("percentUsed") or 0
                key = s.get("key", "unknown").replace("agent:main:", "")
                if pct >= 95:
                    maxed_sessions.append(f"{key} at {pct}%")

            if maxed_sessions:
                issues.append(f"Sessions at critical context: {', '.join(maxed_sessions)}")
                details["maxed_sessions"] = maxed_sessions

            # Check for main session specifically
            main_sessions = [s for s in recent if s.get("key", "").endswith(":main")]
            if main_sessions:
                main = main_sessions[0]
                details["main_context_pct"] = main.get("percentUsed") or 0
                details["main_tokens"] = f"{(main.get('totalTokens') or 0)//1000}k/{(main.get('contextTokens') or 0)//1000}k"
            else:
                details["main_context_pct"] = 0
                details["main_tokens"] = "no main session"

        except json.JSONDecodeError:
            issues.append("Failed to parse openclaw status JSON")
            details["status_cmd"] = "PARSE_FAIL"

    # Check 3: Recent errors in logs (last 10 min)
    rc, out, err = local_exec(
        "journalctl --user -u openclaw-gateway --no-pager --since '10 min ago' 2>/dev/null "
        "| grep -c 'HTTP 500\\|timed out\\|FailoverError'",
        timeout=15
    )
    try:
        error_count = int(out.strip())
    except ValueError:
        error_count = 0
    details["recent_errors_10min"] = error_count
    if error_count >= 5:
        issues.append(f"{error_count} errors in last 10 minutes (500s/timeouts)")

    healthy = len(issues) == 0
    return healthy, issues, details


# ============================================================
# AUTO-FIX
# ============================================================

def _restart_gateway_via_systemd():
    """Restart gateway using systemctl (NOT openclaw gateway restart).
    Kills any orphan gateway processes first to prevent port conflicts.
    Returns (success: bool, output: str).
    """
    # Kill any orphan gateway processes not managed by systemd
    local_exec(
        "systemctl --user stop openclaw-gateway.service 2>/dev/null; "
        "sleep 2; "
        "pkill -f 'openclaw-gateway' 2>/dev/null; "
        "sleep 2; "
        "systemctl --user reset-failed openclaw-gateway.service 2>/dev/null",
        timeout=20
    )
    rc, out, err = local_exec(
        "systemctl --user start openclaw-gateway.service 2>&1",
        timeout=30
    )
    return rc == 0, out or err


def attempt_fix(issues, details):
    """Try to auto-fix common issues. Returns list of actions taken."""
    actions = []
    gateway_restarted = False

    # Fix maxed sessions
    if details.get("maxed_sessions"):
        log("AUTO-FIX: Clearing maxed sessions...")
        rc, out, err = local_exec(
            "python3 -c 'import json; "
            "for p in [\"/home/aialfred/.openclaw/agents/main/sessions/sessions.json\","
            "\"/home/aialfred/.openclaw/sessions.json\"]: "
            " f=open(p,\"w\"); json.dump({},f); f.close(); "
            "print(\"cleared\")'",
            timeout=15
        )
        if "cleared" in out:
            actions.append("Cleared all sessions (context overflow)")
            log("AUTO-FIX: Restarting gateway via systemctl after session clear...")
            ok, msg = _restart_gateway_via_systemd()
            gateway_restarted = True
            actions.append("Restarted gateway after clearing sessions" if ok else f"Failed to restart gateway: {msg}")
        else:
            actions.append(f"Failed to clear sessions: {err}")

    # Fix gateway not running
    if details.get("gateway") != "active" and not gateway_restarted:
        log("AUTO-FIX: Restarting gateway via systemctl...")
        ok, msg = _restart_gateway_via_systemd()
        gateway_restarted = True
        actions.append("Restarted gateway service" if ok else f"Failed to restart gateway: {msg}")

    # If lots of recent errors, restart
    if details.get("recent_errors_10min", 0) >= 5 and not gateway_restarted:
        log("AUTO-FIX: High error rate, restarting gateway via systemctl...")
        ok, msg = _restart_gateway_via_systemd()
        actions.append("Restarted gateway (high error rate)" if ok else f"Failed to restart gateway: {msg}")

    if actions:
        log("Waiting 10s for services to stabilize...")
        time.sleep(10)

    return actions


# ============================================================
# EMAIL: SEND
# ============================================================

def send_email(subject, body_html):
    """Send email alert to Mike."""
    if not EMAIL_PASSWORD:
        log("ERROR: No email password configured (EMAIL_PASS_LUMABOT)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{FROM_NAME} <{FROM_EMAIL}>"
    msg["To"] = TO_EMAIL
    msg["Reply-To"] = FROM_EMAIL

    # Plain text version
    plain = body_html.replace("<br>", "\n").replace("<h2>", "\n## ").replace("</h2>", "\n")
    plain = plain.replace("<h3>", "\n### ").replace("</h3>", "\n")
    plain = plain.replace("<strong>", "").replace("</strong>", "")
    plain = plain.replace("<li>", "- ").replace("</li>", "\n")
    plain = plain.replace("<ul>", "").replace("</ul>", "")
    plain = re.sub(r'<[^>]+>', '', plain)

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP_SSL(MAIL_SERVER, SMTP_PORT, timeout=30) as server:
            server.login(SMTP_LOGIN_EMAIL, EMAIL_PASSWORD)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
        log(f"Email sent: {subject}")
        return True
    except Exception as e:
        log(f"Email failed: {e}")
        return False


def send_down_alert(issues, details):
    """Send alert email that Alfred Claw is down — asks Mike to reply."""
    issue_list = "".join(f"<li>{i}</li>" for i in issues)
    subject = "🚨 Alfred Claw is DOWN — Reply FIX IT or LEAVE IT"

    body = f"""
    <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0f; color: #ffffff; padding: 2rem; border-radius: 12px;">
        <h2 style="color: #e94560; margin-bottom: 1rem;">🚨 Alfred Claw is Unresponsive</h2>
        <p style="color: #a0a0b0;">Detected at <strong style="color: #fff;">{datetime.now().strftime('%I:%M %p EST on %B %d, %Y')}</strong></p>

        <div style="background: #1a1a25; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #e94560;">
            <h3 style="color: #e94560; margin: 0 0 0.5rem 0;">Issues Found:</h3>
            <ul style="color: #a0a0b0; margin: 0; padding-left: 1.5rem;">
                {issue_list}
            </ul>
        </div>

        <div style="background: #1a1a25; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h3 style="color: #3498db; margin: 0 0 0.5rem 0;">System Details:</h3>
            <ul style="color: #a0a0b0; margin: 0; padding-left: 1.5rem; list-style: none;">
                <li>Gateway: <strong style="color: #fff;">{details.get('gateway', '?')}</strong></li>
                <li>Telegram: <strong style="color: #fff;">{details.get('telegram', '?')}</strong></li>
                <li>Sessions: <strong style="color: #fff;">{details.get('session_count', '?')}</strong></li>
                <li>Main Context: <strong style="color: #fff;">{details.get('main_tokens', '?')}</strong></li>
                <li>Errors (10min): <strong style="color: #fff;">{details.get('recent_errors_10min', '?')}</strong></li>
            </ul>
        </div>

        <div style="background: rgba(0, 210, 106, 0.1); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 2px solid #00d26a; text-align: center;">
            <h3 style="color: #00d26a; margin: 0 0 0.75rem 0;">Reply to this email:</h3>
            <p style="color: #ffffff; font-size: 1.2rem; margin: 0;">
                <strong>FIX IT</strong> — I will diagnose and auto-repair<br>
                <strong>LEAVE IT</strong> — I will stand down until you say otherwise
            </p>
            <p style="color: #f39c12; font-size: 0.85rem; margin-top: 0.75rem;">
                ⏱ If no reply in 1 hour, I will auto-fix to prevent extended downtime.
            </p>
        </div>

        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: #6a6a7a; font-size: 0.85rem;">
            Alfred Labs Health Monitor • 75.43.156.105 (local)
        </div>
    </div>
    """
    return send_email(subject, body)


def send_recovery_alert(actions, details, downtime_minutes, trigger="auto"):
    """Send recovery email with what was fixed."""
    action_list = "".join(f"<li>{a}</li>" for a in actions) if actions else "<li>Self-recovered (no action needed)</li>"
    trigger_text = {
        "fix_it": 'Your "FIX IT" command',
        "auto_timeout": "Auto-fix (no reply after 1 hour)",
        "auto": "Automatic recovery",
    }.get(trigger, trigger)

    subject = "✅ Alfred Claw is BACK ONLINE"

    body = f"""
    <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0f; color: #ffffff; padding: 2rem; border-radius: 12px;">
        <h2 style="color: #00d26a; margin-bottom: 1rem;">✅ Alfred Claw is Back Online</h2>
        <p style="color: #a0a0b0;">Recovered at <strong style="color: #fff;">{datetime.now().strftime('%I:%M %p EST on %B %d, %Y')}</strong></p>
        <p style="color: #a0a0b0;">Downtime: <strong style="color: #f39c12;">~{downtime_minutes} minutes</strong></p>
        <p style="color: #a0a0b0;">Triggered by: <strong style="color: #3498db;">{trigger_text}</strong></p>

        <div style="background: #1a1a25; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #00d26a;">
            <h3 style="color: #00d26a; margin: 0 0 0.5rem 0;">What Was Fixed:</h3>
            <ul style="color: #a0a0b0; margin: 0; padding-left: 1.5rem;">
                {action_list}
            </ul>
        </div>

        <div style="background: #1a1a25; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h3 style="color: #3498db; margin: 0 0 0.5rem 0;">Current Status:</h3>
            <ul style="color: #a0a0b0; margin: 0; padding-left: 1.5rem; list-style: none;">
                <li>Gateway: <strong style="color: #00d26a;">{details.get('gateway', '?')}</strong></li>
                <li>Telegram: <strong style="color: #00d26a;">{details.get('telegram', '?')}</strong></li>
                <li>Sessions: <strong style="color: #fff;">{details.get('session_count', '?')}</strong></li>
                <li>Main Context: <strong style="color: #fff;">{details.get('main_tokens', '?')}</strong></li>
            </ul>
        </div>

        <p style="color: #a0a0b0; font-size: 0.9rem;">This incident has been logged to Alfred Claw's memory so he can learn from it.</p>

        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: #6a6a7a; font-size: 0.85rem;">
            Alfred Labs Health Monitor • 75.43.156.105 (local)
        </div>
    </div>
    """
    return send_email(subject, body)


def send_acknowledged_alert():
    """Send confirmation that LEAVE IT was received."""
    subject = "⏸ Alfred Claw — Standing Down (LEAVE IT received)"
    body = f"""
    <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0f; color: #ffffff; padding: 2rem; border-radius: 12px;">
        <h2 style="color: #f39c12; margin-bottom: 1rem;">⏸ Standing Down</h2>
        <p style="color: #a0a0b0;">Your <strong style="color: #fff;">LEAVE IT</strong> command was received at <strong style="color: #fff;">{datetime.now().strftime('%I:%M %p EST')}</strong></p>
        <p style="color: #a0a0b0;">I will not attempt any fixes or send further alerts until Alfred Claw recovers on its own or you tell me otherwise.</p>
        <p style="color: #a0a0b0; margin-top: 1rem;">Monitoring continues silently in the background.</p>
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: #6a6a7a; font-size: 0.85rem;">
            Alfred Labs Health Monitor
        </div>
    </div>
    """
    return send_email(subject, body)


# ============================================================
# EMAIL: READ INBOX FOR COMMANDS
# ============================================================

def check_inbox_for_command(since_time):
    """Check alfred inbox for replies from Mike with 'fix it' or 'leave it'.
    Only looks at emails received after since_time (ISO string).
    Returns: 'fix_it', 'leave_it', or None
    """
    if not EMAIL_PASSWORD:
        log("Cannot check inbox: no email password")
        return None

    try:
        mail = imaplib.IMAP4_SSL(MAIL_SERVER, IMAP_PORT, timeout=15)
        mail.login(SMTP_LOGIN_EMAIL, EMAIL_PASSWORD)
        mail.select("INBOX")

        # Search for recent emails from Mike
        since_dt = datetime.fromisoformat(since_time) if since_time else datetime.now() - timedelta(hours=2)
        since_str = since_dt.strftime("%d-%b-%Y")

        status, data = mail.search(None, f'(SINCE "{since_str}")')
        if status != "OK" or not data[0]:
            mail.logout()
            return None

        msg_ids = data[0].split()
        # Check most recent first
        for msg_id in reversed(msg_ids[-20:]):
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue

            msg = email_lib.message_from_bytes(msg_data[0][1])
            from_addr = msg.get("From", "").lower()
            subject = _decode_header(msg.get("Subject", "")).lower()
            date_str = msg.get("Date", "")

            # Must be from Mike
            is_from_mike = any(addr in from_addr for addr in MIKE_EMAILS)
            if not is_from_mike:
                continue

            # Must be a reply to our alert (check subject)
            is_reply = "alfred claw" in subject or "re:" in subject
            if not is_reply:
                continue

            # Get body text
            body = _get_email_body(msg).lower().strip()

            # Check for commands - look in subject + body
            full_text = subject + " " + body

            if "fix it" in full_text:
                log(f"COMMAND RECEIVED: FIX IT (from {from_addr})")
                mail.logout()
                return "fix_it"
            elif "leave it" in full_text:
                log(f"COMMAND RECEIVED: LEAVE IT (from {from_addr})")
                mail.logout()
                return "leave_it"

        mail.logout()
        return None

    except Exception as e:
        log(f"Inbox check error: {e}")
        return None


def _decode_header(value):
    """Decode email header value."""
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for text, charset in parts:
        if isinstance(text, bytes):
            decoded.append(text.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(text)
    return " ".join(decoded)


def _get_email_body(msg):
    """Extract plain text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    return part.get_payload(decode=True).decode("utf-8", errors="replace")
                except Exception:
                    pass
        # Fallback to HTML
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                try:
                    html = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    return re.sub(r'<[^>]+>', ' ', html)
                except Exception:
                    pass
    else:
        try:
            return msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            pass
    return ""


# ============================================================
# LIGHTRAG LOGGING
# ============================================================

def log_to_lightrag(incident_summary):
    """Log incident to LightRAG on Alfred Claw so he can learn from it."""
    memory_entry = f"""## Incident Report - {datetime.now().strftime('%Y-%m-%d %H:%M EST')}

{incident_summary}

### Lesson Learned
- Monitor context usage and request compaction before hitting limits
- Sub-agents with local models can overwhelm system resources
- When errors loop for 10+ minutes, the system needs external intervention
- Auto-compaction (safeguard mode, 40k reserveTokensFloor) should prevent context overflow
"""

    try:
        incidents_path = os.path.join(OPENCLAW_HOME, "workspace", "INCIDENTS.md")
        header = "# Alfred Incident Log\n\nIncidents detected by health monitor.\n"
        if os.path.exists(incidents_path):
            with open(incidents_path) as f:
                content = f.read()
        else:
            content = header
        content += "\n" + memory_entry
        with open(incidents_path, "w") as f:
            f.write(content)
        log("Incident written to INCIDENTS.md")
    except Exception as e:
        log(f"LightRAG logging error: {e}")

    # Also try to add to LightRAG memory index
    local_exec(
        "openclaw memory add --source incidents "
        "--content 'Health monitor detected and auto-fixed an incident. See INCIDENTS.md for details.' 2>/dev/null",
        timeout=15
    )

    log("Incident logged to LightRAG/INCIDENTS.md on Alfred Claw")


# ============================================================
# QUEUE.MD ESCALATION BRIDGE
# ============================================================

QUEUE_PATH = "/home/aialfred/.openclaw/workspace/QUEUE.md"
ESCALATION_STATE_KEY = "processed_escalations"
CLAUDE_CLI = "/home/aialfred/.nvm/versions/node/v20.12.2/bin/claude"
TELEGRAM_CHAT_ID = "7582976864"


def check_queue():
    """Read QUEUE.md from Alfred Claw, return list of ESCALATED items.
    Each item is a dict with: id, title, status, body (full text block).

    Supports two QUEUE.md formats:
    1. Pipe-delimited: Q-ID | timestamp | ACTION: description | STATUS: ESCALATED
    2. Markdown headers: ### [Q-ID] — Title  with  - **Status**: ESCALATED
    """
    try:
        if not os.path.exists(QUEUE_PATH):
            return []
        with open(QUEUE_PATH) as f:
            out = f.read()
        if not out or not out.strip():
            return []
    except Exception as e:
        log(f"Could not read QUEUE.md: {e}")
        return []

    items = []

    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Format 1: Pipe-delimited
        # Q-20260309-002 | 2026-03-09T18:10:00Z | ACTION: ... | STATUS: ESCALATED
        pipe_match = re.match(
            r'^(Q-\d{8}-\d{3})\s*\|\s*(\S+)\s*\|\s*(?:ACTION:\s*)?(.+?)\s*\|\s*STATUS:\s*(\w+)',
            line
        )
        if pipe_match:
            items.append({
                "id": pipe_match.group(1),
                "title": pipe_match.group(3).strip(),
                "status": pipe_match.group(4).strip(),
                "body": line,
            })
            continue

        # Format 2: Markdown header (legacy)
        # ### [Q-20250214-003] — Title
        header_match = re.match(r'^###\s+\[([^\]]+)\]\s*[—–-]\s*(.*)', line)
        if header_match:
            items.append({
                "id": header_match.group(1).strip(),
                "title": header_match.group(2).strip(),
                "status": "",
                "body": "",
            })
            continue

        # For markdown format, extract status from body lines
        if items and not items[-1]["status"]:
            status_match = re.match(r'^-\s+\*\*Status\*\*:\s*(.*)', line)
            if status_match:
                items[-1]["status"] = status_match.group(1).strip()
            items[-1]["body"] += line + "\n"

    # Filter for ESCALATED items only
    escalated = [i for i in items if "ESCALATED" in i.get("status", "").upper()]
    return escalated


def process_escalation(item, state):
    """Process a single ESCALATED queue item using Claude Code CLI."""
    item_id = item["id"]
    title = item["title"]
    body = item["body"]

    log(f"ESCALATION: Processing {item_id} — {title}")

    # Build a prompt for Claude Code
    prompt = f"""You are Alfred Labs (Claude Code) on server 75.43.156.105.
Alfred Claw (OpenClaw) is running locally on this same machine and has escalated this issue via QUEUE.md.

## Escalated Item: {item_id} — {title}

{body}

## Your Task
1. Diagnose the issue described above (everything is local, no SSH needed)
2. Fix it — update scripts, configs, files as needed
3. Return a detailed summary of:
   - What was wrong
   - What you fixed (include file paths and what changed)
   - Any follow-up actions needed

## Important
- Alfred Claw's workspace is at /home/aialfred/.openclaw/workspace/
- Integration scripts are at /home/aialfred/.openclaw/workspace/scripts/integrations/
- OpenClaw config is at /home/aialfred/.openclaw/openclaw.json
- OpenClaw CLI requires Node 22: /home/aialfred/.nvm/versions/node/v22.22.0/bin/openclaw
- Be specific about what files you changed and what the changes were
"""

    # Write prompt to temp file and invoke claude
    prompt_file = f"/tmp/_escalation_{item_id.replace('-', '_')}.md"
    try:
        with open(prompt_file, "w") as f:
            f.write(prompt)
    except Exception as e:
        log(f"Failed to write prompt file: {e}")
        return None

    log(f"Invoking Claude Code for {item_id}...")
    try:
        # Must unset CLAUDECODE env var to allow nested invocation
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE", None)
        result = subprocess.run(
            [CLAUDE_CLI, "-p", "--output-format", "text",
             "--dangerously-skip-permissions",
             prompt],
            capture_output=True, text=True,
            timeout=600,  # 10 minute max per escalation
            cwd="/home/aialfred/alfred",
            env=env,
        )
        response = result.stdout.strip()
        if not response and result.stderr:
            response = f"Claude Code error: {result.stderr.strip()[:1000]}"
        log(f"Claude Code responded ({len(response)} chars) for {item_id}")
    except subprocess.TimeoutExpired:
        response = "Claude Code timed out after 10 minutes working on this issue."
        log(f"Claude Code TIMEOUT for {item_id}")
    except Exception as e:
        response = f"Claude Code invocation failed: {str(e)}"
        log(f"Claude Code FAILED for {item_id}: {e}")
    finally:
        try:
            os.unlink(prompt_file)
        except Exception:
            pass

    return response


def update_queue_resolved(item_id, resolution_text):
    """Update the QUEUE.md item from ESCALATED to DONE with details.
    Supports both pipe-delimited and markdown QUEUE.md formats.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M EST")

    try:
        with open(QUEUE_PATH) as f:
            content = f.read()

        lines = content.split("\n")
        new_lines = []
        updated = False

        for line in lines:
            # Pipe-delimited format: replace STATUS: ESCALATED with STATUS: DONE
            if item_id in line and "STATUS:" in line:
                line = re.sub(r'STATUS:\s*\w+', 'STATUS: DONE', line)
                line += f" | RESOLVED: {timestamp}"
                updated = True
            # Markdown format: replace **Status**: ESCALATED
            elif item_id in line or (not updated and re.match(r'^-\s+\*\*Status\*\*:\s*ESCALATED', line)):
                pass  # keep as-is for id line
            if re.match(r'^-\s+\*\*Status\*\*:\s*ESCALATED', line) and not updated:
                line = f"- **Status**: DONE"
                new_lines.append(line)
                new_lines.append(f"- **Resolved**: {timestamp}")
                updated = True
                continue
            new_lines.append(line)

        with open(QUEUE_PATH, "w") as f:
            f.write("\n".join(new_lines))

        if updated:
            log(f"QUEUE.md updated: {item_id} → DONE")
        else:
            log(f"Could not find {item_id} with ESCALATED status in QUEUE.md")
        return updated

    except Exception as e:
        log(f"Queue update error: {e}")
        return False


def send_escalation_report(item_id, title, resolution_text):
    """Email Mike and trigger Telegram notification about the resolved escalation."""
    # 1. Email Mike
    subject = f"Alfred — {item_id} — Fixed by Claude Code"

    body = f"""
    <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0f; color: #ffffff; padding: 2rem; border-radius: 12px;">
        <h2 style="color: #00d26a; margin-bottom: 1rem;">✅ Escalation Resolved — {item_id}</h2>
        <p style="color: #a0a0b0;">Task: <strong style="color: #fff;">{title}</strong></p>
        <p style="color: #a0a0b0;">Resolved at <strong style="color: #fff;">{datetime.now().strftime('%I:%M %p EST on %B %d, %Y')}</strong></p>

        <div style="background: #1a1a25; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #00d26a;">
            <h3 style="color: #00d26a; margin: 0 0 0.5rem 0;">What Claude Code Did:</h3>
            <pre style="color: #a0a0b0; white-space: pre-wrap; font-family: 'Fira Code', monospace; font-size: 0.85rem; margin: 0; line-height: 1.6;">{resolution_text[:3000]}</pre>
        </div>

        <p style="color: #a0a0b0; font-size: 0.9rem; margin-top: 1rem;">
            Alfred Claw's QUEUE.md has been updated to RESOLVED. Alfred Claw will notify you via Telegram as well.
        </p>

        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: #6a6a7a; font-size: 0.85rem;">
            Alfred Labs Escalation Bridge • Claude Code → Alfred Claw
        </div>
    </div>
    """
    email_sent = send_email(subject, body)

    # 2. Trigger Telegram notification via Alfred Claw
    tg_msg = (
        f"Sir, the issue with {title} - {item_id} has been resolved by Claude Code. "
        f"Full details have been sent to your email."
    )

    # Send via openclaw message send on Alfred Claw
    safe_msg = tg_msg.replace("'", "").replace('"', '').replace('`', '').replace('$', '').replace('\\', '')
    rc, out, err = local_exec(
        f"openclaw message send --channel telegram --target {TELEGRAM_CHAT_ID} -m '{safe_msg}' 2>&1",
        timeout=30
    )

    if rc != 0:
        # Fallback: write to pending notifications file
        log(f"Direct Telegram send failed ({err} {out}), writing to pending notifications...")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M EST")
        pending_path = os.path.join(OPENCLAW_HOME, "workspace", "PENDING_NOTIFICATIONS.md")
        try:
            with open(pending_path, "a") as f:
                f.write(f"\n---\n{ts}\n{safe_msg}\n")
        except Exception as e:
            log(f"Failed to write pending notification: {e}")

    log(f"Escalation report sent for {item_id}: email={'OK' if email_sent else 'FAIL'}, telegram={'OK' if rc == 0 else 'FALLBACK'}")
    return email_sent


def process_all_escalations(state):
    """Check QUEUE.md for ESCALATED items and process them."""
    escalated = check_queue()
    if not escalated:
        return

    # Track which items we've already processed
    processed = set(state.get(ESCALATION_STATE_KEY, []))

    for item in escalated:
        item_id = item["id"]
        if item_id in processed:
            log(f"Skipping already-processed escalation: {item_id}")
            continue

        log(f"Found new escalation: {item_id} — {item['title']}")

        # Process with Claude Code
        resolution = process_escalation(item, state)

        # Check if Claude Code actually succeeded
        failed = (
            not resolution
            or "Claude Code error:" in resolution
            or "Claude Code invocation failed:" in resolution
            or "Claude Code timed out" in resolution
            or len(resolution) < 50
        )
        if failed:
            log(f"Claude Code did not resolve {item_id} (will retry next cycle): {(resolution or 'None')[:100]}")
            continue

        # Update QUEUE.md to RESOLVED
        update_queue_resolved(item_id, resolution)

        # Email Mike + Telegram notification
        send_escalation_report(item_id, item["title"], resolution)

        # Only mark as processed after successful resolution + notification
        processed.add(item_id)
        state[ESCALATION_STATE_KEY] = list(processed)
        save_state(state)

        log(f"Escalation {item_id} fully processed and reported.")


# ============================================================
# MAIN LOGIC
# ============================================================

def do_fix_and_report(state, trigger="auto"):
    """Execute fix, check recovery, send report, log to LightRAG."""
    _, issues_now, details_now = check_health()
    actions = attempt_fix(state.get("issues", []), details_now)
    state["fix_actions"] = actions

    if actions:
        log(f"Fix actions taken: {actions}")
        time.sleep(5)

    # Re-check
    healthy, issues_after, details_after = check_health()

    down_since = state.get("down_since", "")
    downtime_min = 0
    if down_since:
        try:
            dt = datetime.fromisoformat(down_since)
            downtime_min = int((datetime.now() - dt).total_seconds() / 60)
        except Exception:
            pass

    if healthy:
        log(f"RECOVERED after fix! Trigger: {trigger}")
        send_recovery_alert(actions, details_after, downtime_min, trigger=trigger)

        incident = (
            f"Alfred Claw was down for ~{downtime_min} minutes. "
            f"Issues: {'; '.join(state.get('issues', []))}. "
            f"Fix trigger: {trigger}. "
            f"Actions: {'; '.join(actions) if actions else 'self-recovered'}."
        )
        log_to_lightrag(incident)

        state["status"] = "healthy"
        state["failures"] = 0
        state["alert_sent"] = False
        state["awaiting_command"] = False
        state["command_received"] = None
        state["acknowledged"] = False
    else:
        log(f"Still unhealthy after fix attempt: {issues_after}")
        # Send update that fix didn't fully work
        if trigger == "fix_it":
            send_email(
                "⚠️ Alfred Claw — Fix attempted but issues remain",
                f"""<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0f; color: #ffffff; padding: 2rem; border-radius: 12px;">
                <h2 style="color: #f39c12;">⚠️ Fix Attempted — Issues Remain</h2>
                <p style="color: #a0a0b0;">Actions taken: {', '.join(actions) if actions else 'None possible'}</p>
                <p style="color: #a0a0b0;">Remaining issues: {', '.join(issues_after)}</p>
                <p style="color: #a0a0b0;">I'll keep monitoring. This may need manual intervention via Claude Code.</p>
                <div style="margin-top: 1rem; color: #6a6a7a; font-size: 0.85rem;">Alfred Labs Health Monitor</div>
                </div>"""
            )

    return healthy


def main():
    log("=" * 50)
    log("Alfred Claw Health Check starting...")

    state = load_state()
    healthy, issues, details = check_health()

    # ---- HEALTHY ----
    if healthy:
        log(f"HEALTHY — Sessions: {details.get('session_count', '?')}, Main: {details.get('main_tokens', 'n/a')}")

        # Was down before, now self-recovered
        if state.get("status") == "down" and state.get("alert_sent") and not state.get("acknowledged"):
            down_since = state.get("down_since", "")
            downtime_min = 0
            if down_since:
                try:
                    dt = datetime.fromisoformat(down_since)
                    downtime_min = int((datetime.now() - dt).total_seconds() / 60)
                except Exception:
                    pass

            actions = state.get("fix_actions", [])
            send_recovery_alert(actions, details, downtime_min, trigger="auto")

            incident = (
                f"Alfred Claw was down for ~{downtime_min} minutes. "
                f"Issues: {'; '.join(state.get('issues', []))}. "
                f"Self-recovered. Actions: {'; '.join(actions) if actions else 'none'}."
            )
            log_to_lightrag(incident)

        state["status"] = "healthy"
        state["failures"] = 0
        state["alert_sent"] = False
        state["awaiting_command"] = False
        state["command_received"] = None
        state["acknowledged"] = False
        state["issues"] = []
        state["fix_actions"] = []

        # Check QUEUE.md for escalations when system is healthy
        log("Checking QUEUE.md for escalations...")
        try:
            process_all_escalations(state)
        except Exception as e:
            log(f"Escalation check error: {e}")

        save_state(state)
        log("Health check complete.")
        return

    # ---- UNHEALTHY ----
    log(f"UNHEALTHY — Issues: {issues}")
    state["failures"] = state.get("failures", 0) + 1
    state["issues"] = issues

    if state.get("status") != "down":
        state["down_since"] = datetime.now().isoformat()
    state["status"] = "down"

    # If acknowledged (LEAVE IT), just log and don't do anything
    if state.get("acknowledged"):
        log("Status: DOWN but acknowledged (LEAVE IT). Monitoring silently.")
        save_state(state)
        log("Health check complete.")
        return

    # If awaiting command, check inbox
    if state.get("awaiting_command"):
        log("Checking inbox for command...")
        command = check_inbox_for_command(state.get("down_since"))

        if command == "fix_it":
            log("FIX IT received! Proceeding with auto-fix...")
            state["command_received"] = "fix_it"
            state["awaiting_command"] = False
            do_fix_and_report(state, trigger="fix_it")
            save_state(state)
            log("Health check complete.")
            return

        elif command == "leave_it":
            log("LEAVE IT received. Acknowledging and standing down.")
            state["command_received"] = "leave_it"
            state["awaiting_command"] = False
            state["acknowledged"] = True
            send_acknowledged_alert()
            save_state(state)
            log("Health check complete.")
            return

        else:
            # No command yet — check timeout
            down_since = state.get("down_since", "")
            if down_since:
                try:
                    dt = datetime.fromisoformat(down_since)
                    minutes_down = (datetime.now() - dt).total_seconds() / 60
                    if minutes_down >= AUTO_FIX_TIMEOUT_MIN:
                        log(f"No reply after {int(minutes_down)} minutes. Auto-fixing (safety net)...")
                        state["awaiting_command"] = False
                        do_fix_and_report(state, trigger="auto_timeout")
                        save_state(state)
                        log("Health check complete.")
                        return
                    else:
                        remaining = int(AUTO_FIX_TIMEOUT_MIN - minutes_down)
                        log(f"Awaiting command. {remaining} min until auto-fix.")
                except Exception:
                    pass

            save_state(state)
            log("Health check complete.")
            return

    # First detection — require consecutive failures before alerting.
    # This avoids spamming on transient blips (e.g. gateway "activating" for a few seconds).
    if not state.get("alert_sent"):
        if state["failures"] < CONSECUTIVE_FAILURES_BEFORE_ALERT:
            log(f"Unhealthy ({state['failures']}/{CONSECUTIVE_FAILURES_BEFORE_ALERT} before alert). Waiting for next check.")
            save_state(state)
            log("Health check complete.")
            return
        log(f"Consecutive failures ({state['failures']}) reached threshold. Sending DOWN alert to Mike...")
        sent = send_down_alert(issues, details)
        state["alert_sent"] = sent
        state["awaiting_command"] = True
        save_state(state)
        log("Health check complete. Awaiting Mike's reply (fix it / leave it).")
        return

    # Alert was sent but awaiting_command not set (shouldn't happen, but handle it)
    state["awaiting_command"] = True
    save_state(state)
    log("Health check complete.")


if __name__ == "__main__":
    main()
