# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** Milestone v1.1 Infrastructure Resilience — Phase 7: Backup System

## Current Position

Phase: 7 — Backup System
Plan: 03 complete (3 of 3 plans) — PHASE COMPLETE
Status: Phase 7 complete — backup infrastructure + daily script + weekly script all done
Progress: [##--------] 25% (1/4 phases complete)

Last activity: 2026-02-26 — Phase 7 Plan 03 complete (weekly_backup.py — Docker volume exports, 30-day retention pruning, cron Sunday 2 AM)

## Performance Metrics

**v1.0 Summary:**
- 5 phases, 13 plans, 40 commits
- 45 files changed (+7,136 / -94)
- 24 days (2026-01-28 → 2026-02-21)

**v1.1 In Progress:**
- 4 phases defined, 0 complete
- Started: 2026-02-26

## Accumulated Context

### Decisions

- 105 as backup orchestrator — central point, SSH into all servers, upload to Drive
- Google Workspace for Drive uploads — reuse existing google_workspace.py, no new auth setup
- Daily configs + weekly full schedule — balance safety vs storage, 30-day retention
- Phase 9 (Ad Intelligence) is independent — can run in parallel with or after infra phases
- Per-server dedicated key pattern: each server gets its own alfred_{suffix} key pair for fine-grained revocation
- Named SSH aliases used in all scripts: server-98, server-100, claw, server-104, lonewolf, server-121
- Bootstrap deployment via existing default key — all 6 servers were already accessible, no manual user steps needed
- [Phase 06]: Gitignore changed from directory to file-level patterns so audit.py is git-tracked while inventory JSON/MD stay private
- [Phase 06]: Database detection uses exact 'active' string match (not substring) to avoid 'inactive' false positives
- [Phase 07-01]: Drive folder hierarchy: Alfred Backups/{server_name}/{backup_type}/YYYY-MM-DD/ — date-folder-per-run enables simple retention pruning
- [Phase 07-01]: Module-level folder ID cache in drive_manager.py avoids repeated list_files() API calls within a single backup run
- [Phase 07-01]: labsliveserver daily target excludes Docker volume exports (55 containers too heavy) — weekly only
- [Phase 07-02]: DB dump command failures are non-fatal — placeholder error file written to tarball so artifact is visible without blocking config collection
- [Phase 07-02]: Drive upload failure is fatal for that server — no local tarball retention without cloud copy
- [Phase 07-02]: Cron uses venv Python matching alfred_claw_monitor pattern; date_str computed once and passed to all backup_server() calls

### Pending Todos

- Phase 8: Alert system (next phase after backup system complete)

### Decisions

- [Phase 07-03]: Docker volumes discovered live (docker volume ls -q) at backup time — no dependency on audit.py inventory
- [Phase 07-03]: Per-volume 300s timeout prevents one large volume blocking others; volume failures are non-fatal
- [Phase 07-03]: labsliveserver uses allowlist patterns (_db_, _data_, postgres, mysql, redis, mongo) for volume export — 55 containers needs safe default
- [Phase 07-03]: Weekly collects DAILY_TARGETS inline (not importing daily_backup.py) — decoupled design
- [Phase 07-03]: Retention cleanup runs after all uploads — never prune before backup confirmed
- [Phase 07-03]: Cron daily changed to 1-6 (Mon-Sat), weekly is 0 (Sunday) — mutually exclusive

### Blockers/Concerns

- None — Phase 7 complete, all 3 plans executed.

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed 07-03-PLAN.md — weekly backup script (weekly_backup.py + cron Sunday 2 AM + 30-day retention) complete
Resume at: Run `/gsd:execute-phase 8` to execute Phase 8 (next milestone phase)
