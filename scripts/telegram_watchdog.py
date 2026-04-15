#!/usr/bin/env python3
"""Smart Telegram Watchdog — detects when Alfred's Telegram session is alive but unresponsive.

The old watchdog only checked if the tmux process was alive and context wasn't full.
This watchdog detects the actual failure mode: Claude finishes a conversation turn,
drops to an idle prompt, and stops processing incoming Telegram messages.

Detection strategy:
  1. Capture tmux pane content
  2. Check if session is at an idle prompt (❯ with no activity spinner)
  3. If idle, check how long it's been idle
  4. If idle for too long (>3 minutes), restart the session
  5. Also handles: context too high (compact), process dead (restart)

This replaces both telegram_autocompact.py and telegram_session.sh as a single
unified watchdog.

Usage: Run via cron every 2 minutes.
  */2 * * * * /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/telegram_watchdog.py >> /home/aialfred/alfred/data/telegram_watchdog.log 2>&1
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

TMUX_SESSION = "claude-telegram"
CLAUDE_BIN = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/claude"
WORKDIR = "/home/aialfred/alfred"
STATE_FILE = Path("/home/aialfred/alfred/data/telegram_watchdog_state.json")

# Thresholds
IDLE_RESTART_MINUTES = 3       # Restart if idle at prompt for this long
COMPACT_THRESHOLD = 70          # Send /compact at this context %
COMPACT_COOLDOWN_MINUTES = 60   # Don't compact more than once per hour
RESTART_COOLDOWN_MINUTES = 5    # Don't restart more than once per 5 min

LOG_PREFIX = "TelegramWatchdog"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {LOG_PREFIX}: {msg}", flush=True)


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def session_exists() -> bool:
    """Check if the tmux session exists."""
    r = subprocess.run(
        ["tmux", "has-session", "-t", TMUX_SESSION],
        capture_output=True, timeout=5,
    )
    return r.returncode == 0


def get_pane_content() -> str:
    """Capture the current visible tmux pane content."""
    try:
        r = subprocess.run(
            ["tmux", "capture-pane", "-t", TMUX_SESSION, "-p", "-S", "-50"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def parse_context_usage(content: str) -> int | None:
    """Extract context usage % from the status line."""
    for line in content.split("\n"):
        m = re.search(r'(\d{1,3})%', line)
        if m:
            pct = int(m.group(1))
            if 0 <= pct <= 100:
                return pct
    return None


def is_at_idle_prompt(content: str) -> bool:
    """Check if the session is sitting at an idle ❯ prompt with no activity.

    Returns True if:
      - There's a ❯ prompt line
      - No activity spinners (✻, ◐, ●, ←) appear AFTER the last ❯
      - The prompt appears to be waiting for input
    """
    lines = content.strip().split("\n")
    if not lines:
        return False

    # Find the last prompt line
    last_prompt_idx = -1
    for i, line in enumerate(lines):
        if "❯" in line:
            last_prompt_idx = i

    if last_prompt_idx == -1:
        return False

    # Check if there's any activity AFTER the prompt
    # Activity indicators: ✻ (thinking), ◐ (working), ● (tool use), ← (incoming message)
    lines_after_prompt = lines[last_prompt_idx + 1:]
    for line in lines_after_prompt:
        stripped = line.strip()
        if not stripped:
            continue
        # Status bar lines are OK (they have %, ⏵, etc.)
        if "%" in stripped or "⏵" in stripped or "bypass" in stripped:
            continue
        # Active processing indicators — session is working, not idle
        if any(indicator in stripped for indicator in ["✻", "◐", "●"]):
            return False
        # Incoming message indicator — session has new messages to process
        if "←" in stripped:
            return False

    return True


def has_unprocessed_messages(content: str) -> bool:
    """Check if there are incoming messages (← telegram) that appeared
    after the last response or tool call from Claude."""
    lines = content.strip().split("\n")

    # Find positions of last Claude activity and last incoming message
    last_claude_activity = -1
    last_incoming_msg = -1

    for i, line in enumerate(lines):
        # Claude doing things: tool calls, responses
        if any(x in line for x in ["●", "✻", "Brewed for", "⎿"]):
            last_claude_activity = i
        # Incoming telegram messages
        if "← telegram" in line:
            last_incoming_msg = i

    # If there's an incoming message AFTER Claude's last activity, it's unprocessed
    return last_incoming_msg > last_claude_activity and last_incoming_msg > 0


def kill_session():
    """Kill the tmux session."""
    try:
        # Try graceful exit first
        subprocess.run(
            ["tmux", "send-keys", "-t", TMUX_SESSION, "/exit", "Enter"],
            capture_output=True, timeout=5,
        )
        time.sleep(3)
    except Exception:
        pass

    try:
        subprocess.run(
            ["tmux", "kill-session", "-t", TMUX_SESSION],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def start_session():
    """Start a fresh Telegram channel session."""
    try:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", WORKDIR,
             f"{CLAUDE_BIN} --dangerously-skip-permissions --channels plugin:telegram@claude-plugins-official"],
            capture_output=True, timeout=10,
        )
        time.sleep(3)
        if session_exists():
            log("Session started successfully")
            return True
        else:
            log("ERROR: Session failed to start")
            return False
    except Exception as e:
        log(f"ERROR starting session: {e}")
        return False


def restart_session(reason: str, state: dict) -> bool:
    """Kill and restart the session with cooldown check."""
    last_restart = state.get("last_restart_time")
    if last_restart:
        last_dt = datetime.fromisoformat(last_restart)
        if datetime.now() - last_dt < timedelta(minutes=RESTART_COOLDOWN_MINUTES):
            elapsed = (datetime.now() - last_dt).total_seconds() / 60
            log(f"Skipping restart — last restart was {elapsed:.1f}m ago (cooldown: {RESTART_COOLDOWN_MINUTES}m)")
            return False

    log(f"RESTARTING session — reason: {reason}")
    kill_session()
    time.sleep(2)
    success = start_session()

    if success:
        state["last_restart_time"] = datetime.now().isoformat()
        state["last_restart_reason"] = reason
        state["restart_count"] = state.get("restart_count", 0) + 1

    return success


def send_compact(state: dict) -> bool:
    """Send /compact with cooldown."""
    last_compact = state.get("last_compact_time")
    if last_compact:
        last_dt = datetime.fromisoformat(last_compact)
        if datetime.now() - last_dt < timedelta(minutes=COMPACT_COOLDOWN_MINUTES):
            elapsed = (datetime.now() - last_dt).total_seconds() / 60
            log(f"Skipping compact — last compact was {elapsed:.0f}m ago")
            return False

    try:
        subprocess.run(
            ["tmux", "send-keys", "-t", TMUX_SESSION, "/compact", "Enter"],
            capture_output=True, timeout=10,
        )
        log("Sent /compact")
        state["last_compact_time"] = datetime.now().isoformat()
        state["compact_count"] = state.get("compact_count", 0) + 1
        return True
    except Exception:
        log("Failed to send /compact")
        return False


def main():
    state = load_state()

    # 1. Check if session exists at all
    if not session_exists():
        log("Session not found — starting fresh")
        restart_session("session_not_found", state)
        save_state(state)
        return

    # 2. Get pane content
    content = get_pane_content()
    if not content:
        log("Could not capture pane content")
        save_state(state)
        return

    # 3. Check context usage
    usage = parse_context_usage(content)
    if usage is not None:
        state["last_context_usage"] = usage
        log(f"Context: {usage}%")

        if usage >= COMPACT_THRESHOLD:
            log(f"Context at {usage}% — compacting")
            send_compact(state)
            save_state(state)
            return  # Let compact finish before checking idle state

    # 4. Check if idle at prompt
    idle = is_at_idle_prompt(content)

    if idle:
        # Track how long we've been idle
        idle_since = state.get("idle_since")
        if idle_since:
            idle_dt = datetime.fromisoformat(idle_since)
            idle_minutes = (datetime.now() - idle_dt).total_seconds() / 60
            log(f"Idle for {idle_minutes:.1f}m")

            if idle_minutes >= IDLE_RESTART_MINUTES:
                # Check if there are unprocessed messages — that's the real problem
                if has_unprocessed_messages(content):
                    log(f"Idle {idle_minutes:.1f}m WITH unprocessed messages — restarting")
                    restart_session(f"idle_{idle_minutes:.0f}m_with_unprocessed_messages", state)
                    state.pop("idle_since", None)
                else:
                    # Idle but no messages waiting — this is fine, just waiting for input
                    # But if idle for a very long time (30+ min), restart anyway as a precaution
                    # because the Telegram plugin may have silently disconnected
                    if idle_minutes >= 30:
                        log(f"Idle {idle_minutes:.1f}m with no visible messages — precautionary restart")
                        restart_session(f"idle_{idle_minutes:.0f}m_precautionary", state)
                        state.pop("idle_since", None)
                    else:
                        log("Idle but no unprocessed messages — OK for now")
        else:
            state["idle_since"] = datetime.now().isoformat()
            log("Session is idle — starting timer")
    else:
        # Not idle — session is active, clear the idle timer
        if state.get("idle_since"):
            log("Session is active again")
        state.pop("idle_since", None)

    state["last_check_time"] = datetime.now().isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
