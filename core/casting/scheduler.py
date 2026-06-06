# core/casting/scheduler.py
from __future__ import annotations
from datetime import datetime
from core.casting import db
from core.casting.deploy import deploy_dj, DeployError

def apply_due(now_iso: str | None = None) -> int:
    now_iso = now_iso or datetime.now().isoformat(timespec="seconds")
    applied = 0
    for a in db.due_assignments(now_iso=now_iso):
        dj = db.get_dj(a["dj_id"])
        if not dj or dj["status"] == "draft" or not dj["moods_present"]:
            continue
        try:
            deploy_dj(dj_id=dj["id"], base_name=dj["name"].replace(" ", "_"),
                      moods=dj["moods_present"], persona_prompt=dj["persona_prompt"],
                      station_id=a["station_id"])
        except DeployError:
            # leave unapplied so the next run retries; alerting handled by caller
            continue
        db.mark_applied(a["id"])
        db.set_status(dj["id"], "live")
        applied += 1
    return applied

if __name__ == "__main__":
    print(f"applied {apply_due()} assignment(s)")
