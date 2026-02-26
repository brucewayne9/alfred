# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** Milestone v1.1 Infrastructure Resilience — Phase 7: Backup System

## Current Position

Phase: 7 — Backup System
Plan: 01 complete (1 of 3 plans)
Status: Phase 7 in progress — shared backup infrastructure complete
Progress: [----------] 0% (0/4 phases complete)

Last activity: 2026-02-26 — Phase 7 Plan 01 complete (backup infrastructure — Drive folder manager + SSH utilities for 7 servers)

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

### Pending Todos

- Phase 7: Write daily backup script (plan 02) — unblocked, infrastructure complete
- Phase 7: Write weekly backup script (plan 03) — unblocked, infrastructure complete

### Blockers/Concerns

- None — server inventory complete. labsliveserver (104) has 55 containers (highest density), will need careful backup design.

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed 07-01-PLAN.md — shared backup infrastructure (drive_manager.py + backup_utils.py) created
Resume at: Run `/gsd:execute-phase 7` to execute Phase 7 Plan 02 (daily backup script)
