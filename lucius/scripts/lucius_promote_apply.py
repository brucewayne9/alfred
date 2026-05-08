#!/usr/bin/env python3
"""Read Mike's reply to today's digest; ingest approved entries to Grey Matter.

Runs at 7:30 AM ET (cron) — gives Mike 30 min to reply. Polls Telegram updates
(getUpdates) for any reply to digest_state.tg_message_id. Parses
'1,3,5' or 'none' from the reply text. Approved → POST to LightRAG insert.
Rejected → moved to promote_queue.rejected.jsonl. Approved entries removed.
"""
import json
import os
import re
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

LUCIUS_HOME = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius")))
QUEUE = LUCIUS_HOME / "promote_queue.jsonl"
REJECTED = LUCIUS_HOME / "promote_queue.rejected.jsonl"
DIGEST_STATE = LUCIUS_HOME / "promote_digest_state.json"
ENV_FILE = LUCIUS_HOME / "config" / ".env"


def load_env() -> dict[str, str]:
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_reply_text(token: str, chat_id: str, target_message_id: int) -> str | None:
    url = f"https://api.telegram.org/bot{token}/getUpdates?timeout=2&allowed_updates=%5B%22message%22%5D"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    if not data.get("ok"):
        return None
    for upd in reversed(data.get("result", [])):
        msg = upd.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(chat_id):
            continue
        rt = (msg.get("reply_to_message") or {}).get("message_id")
        if rt == target_message_id:
            return msg.get("text", "").strip()
    return None


def parse_approvals(text: str, n: int) -> list[int]:
    """Parse '1,3,5', '1 3 5', or 'none'. Returns sorted unique indexes in [1, n]."""
    if text.lower().strip() == "none":
        return []
    nums = []
    for tok in re.split(r"[,\s]+", text):
        tok = tok.strip()
        if tok.isdigit():
            i = int(tok)
            if 1 <= i <= n:
                nums.append(i)
    return sorted(set(nums))


def lightrag_insert(host: str, api_key: str, content: str, track_id: str) -> bool:
    url = f"{host.rstrip('/')}/documents/text"
    body = json.dumps({"text": content, "track_id": track_id}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status < 400
    except Exception as e:
        print(f"lightrag insert failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    if not DIGEST_STATE.exists():
        print("no digest state — nothing to apply")
        return 0

    state = json.loads(DIGEST_STATE.read_text())
    entries = state["entries"]
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN_LUCIUS")
    chat_id = env.get("TELEGRAM_CHAT_ID", "7582976864")
    gm_host = env.get("LIGHTRAG_URL") or env.get("LIGHTRAG_HOST", "http://75.43.156.117:9621")
    gm_key = env.get("LIGHTRAG_API_KEY", "")

    reply = get_reply_text(token, chat_id, state["tg_message_id"])
    if reply is None:
        print("no reply yet — leaving state for next run")
        return 0

    approved_idx = parse_approvals(reply, len(entries))
    approved = [entries[i - 1] for i in approved_idx]
    rejected = [e for i, e in enumerate(entries, 1) if i not in approved_idx]

    successes: list[str] = []
    for e in approved:
        ok = lightrag_insert(gm_host, gm_key, e["content"], e["proposed_track_id"])
        if ok:
            successes.append(e["id"])

    if QUEUE.exists():
        remaining = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
        processed_ids = {e["id"] for e in entries}
        remaining = [r for r in remaining if r["id"] not in processed_ids]
        QUEUE.write_text("".join(json.dumps(r) + "\n" for r in remaining))

    with REJECTED.open("a") as f:
        for e in rejected:
            e["rejected_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(e) + "\n")

    summary = f"✅ Promoted {len(successes)} to Grey Matter. Rejected {len(rejected)}."
    urllib.request.urlopen(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode({"chat_id": chat_id, "text": summary}).encode(),
        timeout=10,
    )

    DIGEST_STATE.unlink()
    print(f"approved={len(successes)} rejected={len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
