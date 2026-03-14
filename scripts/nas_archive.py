#!/usr/bin/env python3
"""NAS Archive Script — sweeps generated content to NAS to keep local NVMe lean.

Runs every 6 hours via cron. Moves:
- ComfyUI output images older than 1 day (keeps only today's work local)
- Generated images (data/generated) older than 1 day
- Backup files older than 7 days
Everything stays accessible at /mnt/nas/alfred-archive/ if any service needs it.
TTS audio is disposable and gets deleted (not archived) via separate cron.
NAS mount: /mnt/nas/alfred-archive/
"""

import os
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NAS-ARCHIVE] %(message)s"
)
log = logging.getLogger(__name__)

NAS_BASE = Path("/mnt/nas/alfred-archive")

ARCHIVE_JOBS = [
    {
        "label": "ComfyUI",
        "src": Path("/home/aialfred/ComfyUI/output"),
        "dest": "comfyui-output",
        "max_age_days": 1,
    },
    {
        "label": "Generated Images",
        "src": Path("/home/aialfred/alfred/data/generated"),
        "dest": "generated",
        "max_age_days": 1,
    },
    {
        "label": "Backups",
        "src": Path("/home/aialfred/alfred/data/backups"),
        "dest": "backups",
        "max_age_days": 7,
    },
]


def is_nas_mounted():
    return NAS_BASE.exists() and os.path.ismount("/mnt/nas")


def archive_old_files(src_dir: Path, dest_dir: Path, max_age_days: int, label: str):
    if not src_dir.exists():
        log.info(f"{label}: source {src_dir} does not exist, skipping")
        return 0, 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - (max_age_days * 86400)
    moved = 0
    total_bytes = 0

    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        if f.stat().st_mtime < cutoff:
            dest = dest_dir / f.name
            try:
                shutil.move(str(f), str(dest))
                total_bytes += dest.stat().st_size
                moved += 1
            except Exception as e:
                log.error(f"{label}: failed to move {f.name}: {e}")

    if moved:
        log.info(f"{label}: archived {moved} files ({total_bytes / 1024 / 1024:.1f} MB)")
    else:
        log.info(f"{label}: nothing to archive")
    return moved, total_bytes


def main():
    if not is_nas_mounted():
        log.error("NAS not mounted at /mnt/nas — aborting")
        return

    log.info("Starting NAS archive sweep")

    total_moved = 0
    total_bytes = 0

    for job in ARCHIVE_JOBS:
        m, b = archive_old_files(
            job["src"],
            NAS_BASE / job["dest"],
            job["max_age_days"],
            job["label"],
        )
        total_moved += m
        total_bytes += b

    total_mb = total_bytes / 1024 / 1024
    log.info(f"Sweep complete: {total_moved} files, {total_mb:.1f} MB moved to NAS")


if __name__ == "__main__":
    main()
