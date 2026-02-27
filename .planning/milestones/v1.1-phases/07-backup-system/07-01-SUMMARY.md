---
phase: 07-backup-system
plan: 01
subsystem: infra
tags: [backup, google-drive, ssh, python, tar]

# Dependency graph
requires:
  - phase: 06-ssh-access-server-audit
    provides: SSH aliases for all 7 servers, server inventory with database and config locations
  - phase: existing
    provides: integrations/google_drive/client.py — Drive API wrappers (create_folder, upload_file, list_files, delete_file)

provides:
  - scripts/backup/drive_manager.py — Drive folder manager with find-or-create hierarchy, upload, list, and prune operations
  - scripts/backup/backup_utils.py — SERVERS registry, DAILY_TARGETS, WEEKLY_EXTRAS, SSH run_cmd, collect_remote_files, pack_tarball, cleanup_staging
  - scripts/backup/__init__.py — package marker

affects: [07-backup-system-02, 07-backup-system-03]

# Tech tracking
tech-stack:
  added: [google-api-python-client, pydantic-settings]
  patterns:
    - Find-or-create folder pattern with module-level cache for Drive API efficiency
    - alias=None means local execution, alias=string means SSH — consistent across all utilities
    - DAILY_TARGETS and WEEKLY_EXTRAS dicts keyed by server name for O(1) lookup in backup scripts

key-files:
  created:
    - scripts/backup/__init__.py
    - scripts/backup/drive_manager.py
    - scripts/backup/backup_utils.py
  modified: []

key-decisions:
  - "Drive folder hierarchy: Alfred Backups/{server_name}/{backup_type}/YYYY-MM-DD/ — date folder per run enables per-day pruning"
  - "Module-level folder ID cache in drive_manager.py avoids repeated list_files() API calls within a single backup run"
  - "DAILY_TARGETS excludes all labsliveserver container volumes — 55 containers is too heavy for daily; weekly Docker volume inspection only"
  - "WEEKLY_EXTRAS uses common _COMMON_WEEKLY_COMMANDS for dpkg, docker volumes, docker images across all servers"
  - "collect_remote_files() silently skips missing files with log warnings — allows partial success instead of hard failure"

patterns-established:
  - "Backup scripts import from scripts.backup.drive_manager and scripts.backup.backup_utils — no logic duplication"
  - "sys.path.insert(0, project_root) at top of each module for standalone execution"

requirements-completed: [BACKUP-03]

# Metrics
duration: 2min
completed: 2026-02-26
---

# Phase 7 Plan 01: Backup Infrastructure Summary

**Google Drive folder manager + SSH backup utilities providing the shared foundation for all daily and weekly backup scripts across 7 servers**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-26T19:18:12Z
- **Completed:** 2026-02-26T19:20:41Z
- **Tasks:** 2
- **Files modified:** 3 created

## Accomplishments
- Drive folder manager with find-or-create hierarchy, upload, list, and prune operations wrapping existing google_drive client (no duplicated API code)
- Per-server DAILY_TARGETS covering 7 servers with server-specific paths informed by Phase 6 inventory (PostgreSQL dump on 105, OpenClaw workspace on 101, MySQL dumps on 100/104, mail configs on 121, etc.)
- WEEKLY_EXTRAS with dpkg package list, Docker volume inventory, and Docker image list for all 7 servers, plus extended collection for heavy servers (alfred-labs data/, labsliveserver/lonewolf/cloud-mail volume exports)
- SSH helpers (run_cmd, collect_remote_files) matching audit.py alias pattern, plus tar packing and staging cleanup utilities

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Drive folder manager** - `7587ffc` (feat)
2. **Task 2: Create backup utilities with SSH helpers and per-server targets** - `bc70f33` (feat)

**Plan metadata:** (see final commit hash below)

## Files Created/Modified
- `scripts/backup/__init__.py` - Package marker
- `scripts/backup/drive_manager.py` - Drive folder manager (ensure_root_folder, ensure_server_folder, upload_backup, list_date_folders, delete_folder_recursive, prune_old_backups)
- `scripts/backup/backup_utils.py` - Shared utilities (SERVERS, DAILY_TARGETS, WEEKLY_EXTRAS, run_cmd, collect_remote_files, pack_tarball, cleanup_staging)

## Decisions Made
- Drive folder hierarchy uses `Alfred Backups/{server_name}/{backup_type}/YYYY-MM-DD/` — date-folder-per-run enables simple retention pruning by folder name
- Module-level cache in drive_manager.py stores folder IDs to avoid repeated list_files() calls during a single backup run
- labsliveserver daily target excludes Docker volume exports (55 containers too heavy) — weekly only shows volume mount points
- collect_remote_files() skips missing files with warnings rather than aborting — allows partial backup on misconfigured servers

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing google-api-python-client dependency**
- **Found during:** Task 1 verification
- **Issue:** `googleapiclient` module not installed; client.py import failed
- **Fix:** `pip3 install google-api-python-client`
- **Files modified:** System packages (no project file changes)
- **Verification:** Import succeeded after install
- **Committed in:** Not committed (system dependency install)

**2. [Rule 3 - Blocking] Installed missing pydantic-settings dependency**
- **Found during:** Task 1 verification (second import attempt)
- **Issue:** `pydantic_settings` module not installed; config/settings.py import failed
- **Fix:** `pip3 install pydantic-settings`
- **Files modified:** System packages (no project file changes)
- **Verification:** Import succeeded after install
- **Committed in:** Not committed (system dependency install)

---

**Total deviations:** 2 auto-fixed (both blocking — missing system dependencies)
**Impact on plan:** Both required for import chain to resolve. No scope creep. System packages pre-exist in production environment via requirements.txt/virtualenv.

## Issues Encountered
None beyond the missing system packages documented above.

## Next Phase Readiness
- `scripts/backup/` package is ready for daily and weekly backup scripts to import
- SERVERS, DAILY_TARGETS, WEEKLY_EXTRAS provide all configuration needed for plan 02 (daily) and plan 03 (weekly)
- Drive hierarchy will be created on first backup run

---
*Phase: 07-backup-system*
*Completed: 2026-02-26*

## Self-Check: PASSED

- FOUND: scripts/backup/__init__.py
- FOUND: scripts/backup/drive_manager.py
- FOUND: scripts/backup/backup_utils.py
- FOUND: .planning/phases/07-backup-system/07-01-SUMMARY.md
- FOUND commit: 7587ffc (Task 1 — drive_manager.py)
- FOUND commit: bc70f33 (Task 2 — backup_utils.py)
