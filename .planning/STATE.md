# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** Milestone v1.1 Infrastructure Resilience — Phase 6: SSH Access & Server Audit

## Current Position

Phase: 6 — SSH Access & Server Audit
Plan: 02 complete, phase done (2 of 2 plans)
Status: Phase 6 complete — SSH setup + server audit done
Progress: [----------] 0% (0/4 phases complete)

Last activity: 2026-02-26 — Phase 6 Plan 02 complete (server audit — 7 servers inventoried in JSON + markdown)

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

### Pending Todos

- Phase 7: Write per-server backup scripts (now unblocked — inventory complete)

### Blockers/Concerns

- None — server inventory complete. labsliveserver (104) has 55 containers (highest density), will need careful backup design.

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed 06-02-PLAN.md — server audit complete, inventory.json + inventory.md generated for all 7 servers
Resume at: Run `/gsd:execute-phase 7` to execute Phase 7 (backup automation)
