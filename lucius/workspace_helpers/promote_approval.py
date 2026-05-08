#!/usr/bin/env python3
"""Record Mike's approval of a promote-queue digest.

Called by Lucius via the `memory.record_approval` MCP tool when Mike replies
to a digest message. Writes one JSONL entry per approval to
~/.lucius/promote_approvals.jsonl. The applier (lucius_promote_apply.py)
reads that file instead of polling Telegram getUpdates — which conflicts
with the running Hermes gateway's own long-poll.

CLI:
    python3 promote_approval.py record [DIGEST_ID] [INDEXES]

DIGEST_ID  — empty string or 'latest' resolves to the current
             ~/.lucius/promote_digest_state.json's digest_id.
INDEXES    — comma-separated digits ('1', '1,3,5') or 'none' for skip-all.
             Empty / missing also treated as 'none'.

Output (stdout, JSON):
    {"recorded": true, "digest_id": "...", "approved_indexes": [...]}
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


LUCIUS_HOME = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius")))
APPROVALS = LUCIUS_HOME / "promote_approvals.jsonl"
DIGEST_STATE = LUCIUS_HOME / "promote_digest_state.json"


def resolve_digest_id(requested: str) -> str:
    """If requested is empty or 'latest', read digest_state.json. Else return as-is."""
    if requested and requested.lower() != "latest":
        return requested
    if not DIGEST_STATE.exists():
        return ""
    try:
        return json.loads(DIGEST_STATE.read_text()).get("digest_id", "")
    except (json.JSONDecodeError, OSError):
        return ""


def parse_indexes(raw: str) -> list[int]:
    """'1,3,5' → [1,3,5];  '1 3 5' → [1,3,5];  'none' → [];  '' → []."""
    s = (raw or "").strip().lower()
    if s in ("", "none"):
        return []
    out = []
    for tok in s.replace(" ", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return sorted(set(out))


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "record":
        print(json.dumps({"error": "usage: promote_approval.py record [DIGEST_ID] [INDEXES]"}))
        return 2

    digest_id_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    indexes_arg = sys.argv[3] if len(sys.argv) > 3 else ""

    digest_id = resolve_digest_id(digest_id_arg)
    if not digest_id:
        print(json.dumps({"error": "no digest_id and no current digest state file"}))
        return 3

    indexes = parse_indexes(indexes_arg)

    APPROVALS.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "digest_id": digest_id,
        "approved_indexes": indexes,
    }
    with APPROVALS.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    print(json.dumps({"recorded": True, "digest_id": digest_id, "approved_indexes": indexes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
