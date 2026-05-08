#!/usr/bin/env python3
"""Daily 7 AM ET digest of the Lucius promote queue.

Reads ~/.lucius/promote_queue.jsonl, sends a numbered Telegram message
via Lucius bot to Mike. Mike replies with comma-separated indexes (or 'none')
to approve. The companion script `lucius_promote_apply.py` polls for
that reply and executes the approvals.

Caps at 10 entries/day; older entries roll to next day.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.parse

LUCIUS_HOME = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius")))
QUEUE = LUCIUS_HOME / "promote_queue.jsonl"
DIGEST_STATE = LUCIUS_HOME / "promote_digest_state.json"
ENV_FILE = LUCIUS_HOME / "config" / ".env"
CAP_PER_DIGEST = 10


def load_env() -> dict[str, str]:
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def send_telegram(token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).encode()
    with urllib.request.urlopen(url, data=data, timeout=15) as r:
        return json.loads(r.read())


def main() -> int:
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN_LUCIUS")
    chat_id = env.get("TELEGRAM_CHAT_ID", "7582976864")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN_LUCIUS missing", file=sys.stderr)
        return 2

    if not QUEUE.exists() or QUEUE.stat().st_size == 0:
        # Nothing to digest; per CLAUDE.md, "don't message Mike 'nothing new'"
        print("queue empty — silent")
        return 0

    entries = [json.loads(l) for l in QUEUE.read_text().splitlines() if l.strip()]
    today = entries[:CAP_PER_DIGEST]

    lines = ["*Lucius proposes these for long-term memory:*", ""]
    for i, e in enumerate(today, 1):
        snippet = e["content"][:200] + ("…" if len(e["content"]) > 200 else "")
        lines.append(f"{i}. {snippet}")
        lines.append(f"   _{e['reasoning'][:140]}_")
        lines.append("")
    if len(entries) > len(today):
        lines.append(f"Reply with comma-separated numbers (1,3,4) to approve, or `none` to skip all. {len(entries) - len(today)} more queued behind these.")
    else:
        lines.append("Reply with comma-separated numbers (1,3,4) to approve, or `none` to skip all.")

    text = "\n".join(lines)
    resp = send_telegram(token, chat_id, text)
    if not resp.get("ok"):
        print(f"ERROR: telegram send failed: {resp}", file=sys.stderr)
        return 3

    # Stash digest state — apply.py uses it to map index→entry
    state = {
        "digest_id": str(int(datetime.now(timezone.utc).timestamp())),
        "ts": datetime.now(timezone.utc).isoformat(),
        "tg_message_id": resp["result"]["message_id"],
        "entries": today,
    }
    DIGEST_STATE.write_text(json.dumps(state, indent=2))
    print(f"sent digest with {len(today)} entries; message_id={resp['result']['message_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
