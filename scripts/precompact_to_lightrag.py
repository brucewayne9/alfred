#!/usr/bin/env python3
"""PreCompact hook: Dump conversation to LightRAG before context compaction.

Runs automatically when Claude Code is about to compact the conversation.
Extracts the messages, formats them as a development session log, and
uploads to LightRAG so Alfred can learn about his own architecture and
how he's being modified.
"""

import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime

# Load LightRAG credentials from Alfred's .env
ENV_PATH = "/home/aialfred/alfred/config/.env"


def load_env():
    """Read key=value pairs from .env file."""
    env = {}
    try:
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def get_token(url, user, password):
    """Authenticate with LightRAG and get a bearer token."""
    data = urllib.parse.urlencode({
        "username": user,
        "password": password,
    }).encode()
    req = urllib.request.Request(f"{url}/login", data=data, method="POST")
    resp = urllib.request.urlopen(req, timeout=15)
    return json.loads(resp.read())["access_token"]


def upload_text(url, token, text, description):
    """Upload text to LightRAG for knowledge graph indexing."""
    payload = json.dumps({"text": text, "description": description}).encode()
    req = urllib.request.Request(
        f"{url}/documents/text",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())


def extract_messages(hook_input):
    """Pull messages out of whatever JSON structure the hook provides."""
    if isinstance(hook_input, list):
        return hook_input

    if isinstance(hook_input, dict):
        # Try common field names
        for key in ("messages", "transcript", "conversation", "content"):
            val = hook_input.get(key)
            if isinstance(val, list) and val:
                return val
        # Maybe it's nested under a data key
        data = hook_input.get("data", {})
        if isinstance(data, dict):
            for key in ("messages", "transcript", "conversation"):
                val = data.get(key)
                if isinstance(val, list) and val:
                    return val

    return []


def format_conversation(messages):
    """Format messages into a readable development session document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"ALFRED DEVELOPMENT SESSION — {now}",
        "=" * 60,
        "",
        "This is a record of a Claude Code session working on the Alfred AI assistant.",
        "It captures changes made, reasoning behind decisions, and architectural knowledge.",
        "",
    ]

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        # Handle content that's a list of blocks (Claude format)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        parts.append(text)
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)

        if not isinstance(content, str) or not content.strip():
            continue

        # Skip very short system noise
        if role == "SYSTEM" and len(content) < 20:
            continue

        # Truncate extremely long messages (tool results, etc.) but keep enough
        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated]"

        lines.append(f"[{role}]")
        lines.append(content.strip())
        lines.append("")

    return "\n".join(lines)


def main():
    try:
        # Read hook input from stdin
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}

        messages = extract_messages(hook_input)

        # If we couldn't parse messages, dump the raw input as-is
        if not messages and raw.strip():
            text = (
                f"ALFRED DEVELOPMENT SESSION — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"{'=' * 60}\n\n"
                f"Raw session data (could not parse structured messages):\n\n"
                f"{raw[:8000]}"
            )
        elif messages:
            text = format_conversation(messages)
        else:
            # Empty input — nothing to save
            print(json.dumps({"continue": True}))
            return

        # Skip if too little content
        if len(text) < 200:
            print(json.dumps({"continue": True}))
            return

        # Load credentials
        env = load_env()
        lightrag_url = env.get("LIGHTRAG_URL", "")
        lightrag_user = env.get("LIGHTRAG_USER", "")
        lightrag_pass = env.get("LIGHTRAG_PASS", "")

        if not all([lightrag_url, lightrag_user, lightrag_pass]):
            sys.stderr.write("PreCompact: LightRAG credentials not found in .env\n")
            print(json.dumps({"continue": True}))
            return

        # Authenticate and upload
        token = get_token(lightrag_url, lightrag_user, lightrag_pass)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        upload_text(
            lightrag_url,
            token,
            text,
            f"Claude Code dev session — {now} — Alfred self-knowledge",
        )
        sys.stderr.write(f"PreCompact: Saved session to LightRAG ({len(text)} chars)\n")

    except Exception as e:
        # Never block compaction — log the error and let it proceed
        sys.stderr.write(f"PreCompact LightRAG upload failed: {e}\n")

    # Always allow compaction to continue
    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
