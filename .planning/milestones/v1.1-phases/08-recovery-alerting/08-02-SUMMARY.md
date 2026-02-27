---
phase: 08-recovery-alerting
plan: 02
subsystem: infra
tags: [backup, restore, documentation, disaster-recovery, postgresql, mysql, sqlite, docker, mailu, azuracast, openclaw, dokploy, traefik]

# Dependency graph
requires:
  - phase: 07-backup-system
    provides: DAILY_TARGETS and WEEKLY_EXTRAS per server, Drive folder hierarchy, backup scripts
provides:
  - Per-server restore documentation for all 7 infrastructure servers
  - Step-by-step rebuild instructions from Google Drive backups
  - Quick reference table with restore order, time estimates, and key gotchas
affects: [future-restore-operations, onboarding]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Restore order: databases first, then application containers, then reverse proxy"
    - "Volume restore pattern: docker volume create → alpine container tar extraction"
    - "Backup extraction: tar -xzf {server}-{date}.tar.gz"

key-files:
  created:
    - data/infrastructure/restore-procedures.md
  modified: []

key-decisions:
  - "Restore document derives from backup_utils.py DAILY_TARGETS and WEEKLY_EXTRAS — single source of truth for what is backed up"
  - "Each server section documents the specific database engine (PostgreSQL, MySQL, SQLite) with engine-appropriate restore commands"
  - "labsliveserver volume restore priority follows allowlist patterns: _db_, _data_, postgres, mysql, redis, mongo"
  - "alfred-claw SSH port 2222 prominently called out — failure to use correct port is the most common restore mistake"
  - "cloud-mail SMTP gotcha documented: auth via lumabot@, not alfred@ (no password on alfred@ account)"

patterns-established:
  - "Restore guide references actual backup script constants (DAILY_TARGETS, WEEKLY_EXTRAS) to stay accurate as backup config evolves"
  - "Post-restore validation section per server — each validates the specific critical services for that server role"

requirements-completed: [RECOV-02]

# Metrics
duration: 3min
completed: 2026-02-26
---

# Phase 8 Plan 02: Server Restore Procedures Summary

**1089-line restore guide covering all 7 servers with engine-specific DB restore commands, Docker volume import steps, and per-server post-restore validation**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-26T20:16:14Z
- **Completed:** 2026-02-26T20:19:09Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created comprehensive restore procedures document at `data/infrastructure/restore-procedures.md`
- All 7 servers covered: alfred-labs, groundrush-radio, labs-edge, alfred-claw, labsliveserver, lonewolf, cloud-mail
- Database restore commands are engine-specific: `pg_dumpall`/`psql` for PostgreSQL, `mysqldump`/`mysql` for MySQL, `cp` for SQLite
- Docker volume import steps included for all servers with volume exports (alfred-labs, labsliveserver, lonewolf, cloud-mail)
- Quick reference table at end: server, critical restore order, estimated time, key gotcha

## Task Commits

1. **Task 1: Create per-server restore procedures document** - `f605be0` (feat)

## Files Created/Modified

- `data/infrastructure/restore-procedures.md` — 1089-line per-server restore guide derived from DAILY_TARGETS, WEEKLY_EXTRAS, and inventory.json

## Decisions Made

- Document derives from `backup_utils.py` DAILY_TARGETS/WEEKLY_EXTRAS as single source of truth — ensures the "what is backed up" section stays accurate as backup configuration evolves
- Each server section follows identical structure: overview → what is backed up → numbered restore steps → post-restore validation
- labsliveserver (55 containers) documented with allowlist volume pattern (`_db_`, `_data_`, `postgres`, `mysql`, `redis`, `mongo`) for clear restore prioritization
- alfred-claw SSH port 2222 prominently documented — the non-standard port is the most likely failure point during restore
- cloud-mail SMTP auth gotcha explicitly called out: `lumabot@` for auth, `alfred@` is display-only with no password

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Restore procedures complete — operators (Mike or Claude) can now follow documented steps to rebuild any server from Drive backups
- Phase 8 Plan 01 (monitoring/alerting) is the companion plan for this phase
- Document should be reviewed and updated when backup targets change in `backup_utils.py`

## Self-Check: PASSED

- `data/infrastructure/restore-procedures.md` — FOUND
- Commit `f605be0` — FOUND

---
*Phase: 08-recovery-alerting*
*Completed: 2026-02-26*
