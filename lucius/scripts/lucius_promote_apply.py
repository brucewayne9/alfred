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


APPROVALS = LUCIUS_HOME / "promote_approvals.jsonl"


def get_recorded_approval(digest_id: str) -> list[int] | None:
    """Read promote_approvals.jsonl for the most recent entry matching digest_id.

    Returns the approved_indexes list, or None if no matching entry exists yet.
    The applier was originally designed to poll Telegram getUpdates, but that
    races with Hermes' running gateway long-poll (whoever calls first consumes
    the update). The fix: Lucius captures approvals via the
    `memory.record_approval` MCP tool when Mike replies to a digest, writing
    to this file. The applier then reads from here — no Telegram race.
    """
    if not APPROVALS.exists():
        return None
    for line in reversed(APPROVALS.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(entry.get("digest_id")) == str(digest_id):
            return entry.get("approved_indexes", [])
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


def lightrag_insert(content: str, track_id: str) -> bool:
    """Insert via the existing lightrag_client.py CLI (same auth + endpoint as recall).

    The script handles HTTPS auth against greymatter.groundrushlabs.com via
    LIGHTRAG_USER/PASS — we reuse that path rather than duplicating the auth
    handshake. Track_id becomes a prefix on the content for traceability;
    the CLI's insert() function doesn't accept a separate track_id field,
    but a "[Lucius/<track_id>]" prefix on the text gives the same audit signal.
    """
    import subprocess
    script_path = "/home/brucewayne9/.lucius/workspace/scripts/integrations/lightrag_client.py"
    prefixed = f"[Lucius promoted / {track_id}]\n\n{content}"
    try:
        proc = subprocess.run(
            [sys.executable, script_path, "insert", prefixed],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        print("lightrag insert timeout (60s)", file=sys.stderr)
        return False
    if proc.returncode != 0:
        print(f"lightrag insert failed: exit={proc.returncode} stderr={proc.stderr[:300]}", file=sys.stderr)
        return False
    # The CLI prints JSON to stdout; success if it parses with no "error" key
    try:
        result = json.loads(proc.stdout.strip())
        if isinstance(result, dict) and result.get("error"):
            print(f"lightrag insert returned error: {result['error']}", file=sys.stderr)
            return False
    except json.JSONDecodeError:
        # Some success responses may be non-JSON; tolerate
        pass
    return True


def main() -> int:
    if not DIGEST_STATE.exists():
        print("no digest state — nothing to apply")
        return 0

    state = json.loads(DIGEST_STATE.read_text())
    entries = state["entries"]
    digest_id = state.get("digest_id", "")
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN_LUCIUS")
    chat_id = env.get("TELEGRAM_CHAT_ID", "7582976864")
    gm_host = env.get("LIGHTRAG_URL") or env.get("LIGHTRAG_HOST", "http://75.43.156.117:9621")
    gm_key = env.get("LIGHTRAG_API_KEY", "")

    approved_idx = get_recorded_approval(digest_id)
    if approved_idx is None:
        print(f"no recorded approval for digest_id={digest_id} — leaving state for next run")
        return 0

    # Filter to valid range [1, len(entries)] — Lucius's record_approval already
    # parses indexes, but defensive bounds-check here keeps a corrupt approvals
    # file from indexing past the entries list.
    approved_idx = sorted({i for i in approved_idx if 1 <= i <= len(entries)})
    approved = [entries[i - 1] for i in approved_idx]
    rejected = [e for i, e in enumerate(entries, 1) if i not in approved_idx]

    successes: list[str] = []
    for e in approved:
        ok = lightrag_insert(e["content"], e["proposed_track_id"])
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
