# core/casting/scheduler.py
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
from core.casting import db
from core.casting import deploy
from core.casting.deploy import DeployError

def apply_due(now_iso: str | None = None) -> int:
    # Default to Eastern Time (Mike's TZ; CLAUDE.md mandates ET), naive seconds-ISO
    # to match how assignments are stored.
    now_iso = now_iso or datetime.now(ZoneInfo("America/New_York")).replace(
        tzinfo=None).isoformat(timespec="seconds")
    applied = 0
    for a in db.due_assignments(now_iso=now_iso):
        dj = db.get_dj(a["dj_id"])
        if not dj or dj["status"] == "draft" or not dj["moods_present"]:
            continue
        try:
            schedule_start, schedule_end = deploy.slot_to_times(a["slot"])
        except ValueError:
            # unparseable slot — leave unapplied, operator must fix the slot
            continue
        try:
            deploy.deploy_dj(dj_id=dj["id"], dj_name=dj["name"],
                             moods=dj["moods_present"], persona_prompt=dj["persona_prompt"],
                             station_id=a["station_id"], schedule_start=schedule_start,
                             schedule_end=schedule_end, enabled=False)
        except DeployError:
            # leave unapplied so the next run retries; alerting handled by caller
            continue
        # demote any currently-live DJ on this station before promoting the new one
        db.demote_live_djs(a["station_id"])
        db.mark_applied(a["id"])
        db.set_status(dj["id"], "live")
        applied += 1
    return applied

if __name__ == "__main__":
    print(f"applied {apply_due()} assignment(s)")
