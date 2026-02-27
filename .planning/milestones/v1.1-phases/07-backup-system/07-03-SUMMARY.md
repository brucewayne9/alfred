---
phase: 07-backup-system
plan: 03
subsystem: infra
tags: [backup, google-drive, docker, python, cron, ssh]

# Dependency graph
requires:
  - phase: 07-backup-system-01
    provides: drive_manager.py (upload_backup, prune_old_backups), backup_utils.py (SERVERS, DAILY_TARGETS, WEEKLY_EXTRAS, run_cmd, collect_remote_files, pack_tarball, cleanup_staging)
  - phase: 07-backup-system-02
    provides: daily_backup.py pattern (daily collection logic reused in weekly phase 1)

provides:
  - scripts/backup/weekly_backup.py — full weekly backup: daily configs + Docker volume exports + package lists + 30-day retention pruning
  - cron entry: 0 2 * * 0 (Sunday 2 AM) for weekly_backup.py
  - cron update: daily_backup.py changed from * to 1-6 (Mon-Sat) to prevent Sunday overlap

affects: [phase-08, phase-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Docker volume discovery live at backup time via docker volume ls -q (NOT from static inventory)
    - Alpine container for Docker volume export: docker run --rm -v {vol}:/data -v /tmp:/backup alpine tar czf
    - Per-volume 300s timeout prevents hanging on large volumes; volume failures are non-fatal
    - Anonymous volume detection via 64-char hex regex — excludes garbage volumes automatically
    - labsliveserver selective include patterns (_db_, _data_, postgres, mysql, redis, mongo) for 55-container server
    - Weekly script phases: 1=daily targets, 2=weekly extras, 3=Docker volume exports, 4=server-specific data
    - Two-phase retention: prune_old_backups(30) prunes both daily and weekly after every weekly run

key-files:
  created:
    - scripts/backup/weekly_backup.py
  modified: []

key-decisions:
  - "Docker volumes discovered live (docker volume ls -q) at backup time — no dependency on audit.py inventory"
  - "Per-volume timeout of 300s (not global) — prevents one huge volume from blocking all others"
  - "labsliveserver uses allowlist patterns (not blocklist) for volume export — 55 containers needs safe default"
  - "Weekly script collects daily targets inline (DAILY_TARGETS) — no import from daily_backup.py, decoupled"
  - "Retention cleanup runs after all uploads (not before) — never prune before new backup is confirmed"
  - "cron daily changed to 1-6 (Mon-Sat), weekly is 0 (Sunday) — mutually exclusive at 2 AM"

patterns-established:
  - "Two-phase collection in weekly: daily targets first, then extras — explicit ordering prevents confusion"
  - "Docker volume exports go into docker-volumes/ subdirectory within staging — keeps tarball organized"
  - "alfred-labs sqlite DBs (conversations.db, learning.db) and chromadb/ collected as separate weekly artifact"

requirements-completed: [BACKUP-02, BACKUP-03, BACKUP-04]

# Metrics
duration: 3min
completed: 2026-02-26
---

# Phase 7 Plan 03: Weekly Backup Script Summary

**Weekly full backup with live Docker volume discovery, Alpine-based volume exports, alfred-labs SQLite/ChromaDB archival, and 30-day retention pruning across all 7 servers**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-26T19:23:04Z
- **Completed:** 2026-02-26T19:26:02Z
- **Tasks:** 2
- **Files modified:** 1 created

## Accomplishments

- Weekly backup script (577 lines) covering 4 collection phases: daily configs, weekly extras, Docker volume exports, server-specific data (alfred-labs SQLite DBs + ChromaDB)
- Live Docker volume discovery via `docker volume ls -q` with anonymous volume filtering (64-char hex regex) and skip patterns (cache/tmp/log)
- labsliveserver-specific include filter (`_db_`, `_data_`, `postgres`, `mysql`, `redis`, `mongo`) to safely handle 55-container server without exporting everything
- Per-volume 300s timeout with Alpine container export pattern — one volume failure never blocks others
- 30-day retention cleanup via `prune_old_backups(30)` after all uploads — prunes both daily and weekly folders
- Cron updated: daily moved to Mon-Sat (1-6), weekly on Sunday (0) — zero overlap at 2 AM

## Task Commits

Each task was committed atomically:

1. **Task 1: Create weekly full backup script with Docker volume export and retention cleanup** - `39ad9fb` (feat)
2. **Task 2: Install weekly backup cron entry at 2 AM Sunday and verify no conflict with daily** - (cron system change — no git-tracked files modified)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `scripts/backup/weekly_backup.py` — Full weekly backup (577 lines): live Docker volume discovery, Alpine export, alfred-labs data archival, Drive upload, 30-day retention pruning

## Decisions Made

- Docker volumes discovered live at backup time (not from audit.py inventory) — avoids stale data and adds flexibility
- Per-volume 300s timeout (not a global timeout) — one large volume doesn't block the entire server backup
- labsliveserver uses allowlist volume patterns — safer than blocklist for a 55-container server
- Weekly script imports DAILY_TARGETS directly from backup_utils (not from daily_backup.py) — decoupled design
- Retention cleanup runs after all uploads (not before) — always back up first, then prune
- Cron daily schedule changed from `* * *` to `1-6` (Mon-Sat) with weekly covering Sunday

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] daily_backup.py cron was installed as `* * *` (every day) from plan 07-02**
- **Found during:** Task 2 (Install cron entry — reviewing existing crontab)
- **Issue:** Plan 07-03 task 2 requires daily cron to be `1-6` (Mon-Sat) to avoid Sunday overlap with weekly. The cron installed by plan 07-02 used `* * *` (every day).
- **Fix:** Updated daily_backup.py cron from `0 2 * * *` to `0 2 * * 1-6` when installing the weekly entry. Both entries added atomically in a single `crontab /tmp/crontab_new.txt` call.
- **Verification:** `crontab -l | grep -E "daily_backup|weekly_backup"` shows both entries with correct day-of-week fields.
- **Committed in:** Not committed (crontab is system state, not git-tracked)

---

**Total deviations:** 1 auto-fixed (1 blocking — incorrect cron schedule from previous plan)
**Impact on plan:** Fix was necessary — Sunday overlap would have caused both daily and weekly to run simultaneously, duplicating Drive uploads. No scope creep.

## Issues Encountered

None beyond the cron schedule deviation documented above.

## Next Phase Readiness

- Full backup system is operational: daily (Mon-Sat 2 AM) + weekly (Sunday 2 AM) + 30-day retention
- Phase 7 complete: all 3 plans done — backup infrastructure, daily script, and weekly script
- All 7 servers covered: alfred-labs, groundrush-radio, labs-edge, alfred-claw, labsliveserver, lonewolf, cloud-mail
- Drive retention automatically keeps 30 days of both daily and weekly backups

---
*Phase: 07-backup-system*
*Completed: 2026-02-26*
