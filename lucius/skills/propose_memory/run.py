#!/usr/bin/env python3
"""propose_memory — append a graduation candidate to ~/.lucius/promote_queue.jsonl."""
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    payload = json.loads(sys.stdin.read())
    content = payload.get("content", "").strip()
    reasoning = payload.get("reasoning", "").strip()
    track_hint = payload.get("track_id_hint", "").strip() or "general"

    if not content:
        print(json.dumps({"error": "content required"}))
        return 2
    if not reasoning:
        print(json.dumps({"error": "reasoning required"}))
        return 2

    queue_path = Path(os.environ.get("LUCIUS_HOME", os.path.expanduser("~/.lucius"))) / "promote_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "reasoning": reasoning,
        "proposed_track_id": f"lucius_{track_hint}",
        "session_id": payload.get("session_id"),
    }
    with queue_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    print(json.dumps({"queued": True, "id": entry["id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
