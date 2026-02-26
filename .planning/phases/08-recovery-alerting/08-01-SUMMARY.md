---
phase: 08-recovery-alerting
plan: 01
subsystem: infra
tags: [backup, alerting, telegram, google-drive, cron, subprocess]

# Dependency graph
requires:
  - phase: 07-backup-system
    provides: daily_backup.py, weekly_backup.py, drive_manager.py with list_date_folders()
  - phase: 06-ssh-keys
    provides: SSH alias "claw" in ~/.ssh/config for Alfred Claw (101)
provides:
  - Telegram alert on any backup failure (daily or weekly) via backup_alerting.py
  - Drive integrity validator for all 7 servers checking daily+weekly recency and file count
  - Machine-readable backup_status.json updated at 5 AM daily for API consumption
  - Cron entry at 5 AM daily running validate_backups.py
affects: [09-ad-intelligence, api-endpoints-using-backup-status]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "send_backup_alert() wraps subprocess.run(ssh claw openclaw message send) in try/except — alert failure never crashes caller"
    - "validate_server() returns status=ok|stale|empty|missing — explicit status enum not boolean"
    - "Atomic JSON write: open(.tmp) -> json.dump -> os.rename to status_file"
    - "list_date_folders() returns sorted YYYY-MM-DD folders; age computed from latest[-1]"

key-files:
  created:
    - scripts/backup/backup_alerting.py
    - scripts/backup/validate_backups.py
  modified:
    - scripts/backup/daily_backup.py
    - scripts/backup/weekly_backup.py

key-decisions:
  - "Telegram alert uses SSH to claw alias + openclaw message send CLI — reuses existing channel, no new bot needed"
  - "Alert failure is non-fatal — try/except wrap means Telegram outage cannot crash backup scripts"
  - "validate_server() status thresholds: daily=26h (timezone drift), weekly=170h (~7d+2h buffer)"
  - "Validation cron at 5 AM (3 hours after 2 AM backup window) — sufficient time for all uploads"
  - "backup_status.json written atomically via tmp+rename — prevents partial reads by API"
  - "validate_backups.py exit code 1 if any issues — enables cron job monitoring"

patterns-established:
  - "Alert module pattern: separate module imported at top level, called in main() after failure detection"
  - "Results list format: list of dicts with success (bool) and error (str) keys — shared by alerting module"

requirements-completed: [RECOV-01, RECOV-03]

# Metrics
duration: 2min
completed: 2026-02-26
---

# Phase 08 Plan 01: Recovery Alerting Summary

**Telegram backup failure alerts via SSH to Alfred Claw + Drive integrity validator for all 7 servers writing machine-readable backup_status.json at 5 AM daily**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T20:16:12Z
- **Completed:** 2026-02-26T20:18:30Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created backup_alerting.py with send_backup_alert() — sends Telegram message via SSH to claw alias, wrapped in try/except so alert failure never crashes backup scripts
- Wired send_backup_alert() into daily_backup.py and weekly_backup.py — called in main() after failure detection
- Created validate_backups.py — checks Drive for latest daily/weekly backup per server, reports ok/stale/empty/missing status
- Atomic JSON write to data/backup_status.json for downstream API consumption
- Installed 5 AM daily cron entry (3h after 2 AM backup window) for automatic validation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create backup alerting module and wire into daily/weekly scripts** - `9ebf312` (feat)
2. **Task 2: Create backup validation script with Drive checks and cron** - `34fa9de` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `scripts/backup/backup_alerting.py` - Telegram failure alerter; send_backup_alert(backup_type, results) filters failures and sends via SSH to claw
- `scripts/backup/validate_backups.py` - Drive integrity validator; validate_all_servers() checks all 7 servers daily+weekly, writes backup_status.json
- `scripts/backup/daily_backup.py` - Added import + call to send_backup_alert("daily", results) when failures exist
- `scripts/backup/weekly_backup.py` - Added import + call to send_backup_alert("weekly", results) when failures exist

## Decisions Made
- Telegram alert uses SSH to claw alias + openclaw message send CLI — reuses Phase 6 SSH config and existing Telegram channel, no new bot auth needed on Alfred Labs
- Alert failure wrapped in try/except — Telegram or SSH outage must never crash the backup script itself
- Validation thresholds: 26h for daily (timezone drift + upload time buffer), 170h for weekly (7 days + 2h buffer)
- 5 AM cron for validation — 3 hours after the 2 AM backup window to ensure all uploads complete
- Atomic write pattern (write to .tmp then os.rename) for backup_status.json prevents partial reads

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 8 Plan 01 complete — backup alerting and validation fully operational
- backup_status.json at data/backup_status.json is ready for API endpoints to query
- Next: Phase 8 Plan 02 (if defined) or subsequent phases

---
*Phase: 08-recovery-alerting*
*Completed: 2026-02-26*
