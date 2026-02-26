# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** Milestone v1.1 Ad Intelligence — Phase 9: Ad Intelligence

## Current Position

Phase: 9 — Ad Intelligence
Plan: 02 complete (2 of 2 plans) — PHASE COMPLETE
Status: Phase 9 complete — cross-platform ad summary + guardrails (Plan 01) + optimization suggestions engine (Plan 02)
Progress: [#########-] 90% (estimated, v1.1 in progress)

Last activity: 2026-02-26 — Phase 9 Plan 02 complete (ads_optimization_suggestions tool with 7 heuristic rules)

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
- [Phase 09-ad-intelligence]: Guardrail enforcement is programmatic (code-level) — tool body refuses to execute without confirmed=True regardless of LLM description hints
- [Phase 09-ad-intelligence]: Cross-platform summary handles partial failures — if Meta fails, Google data is still returned; combined metrics only include successful platforms
- [Phase 09-ad-intelligence]: ads_cross_platform_summary added to ad_intelligence, meta_ads, and google_ads TOOL_CATEGORIES for maximum discoverability
- [Phase 09-ad-intelligence]: Rule-based heuristics in suggestions engine (not external LLM) — deterministic, low-latency, LLM already provides natural language layer
- [Phase 09-ad-intelligence]: ads_optimization_suggestions CTR normalization: Meta raw CTR is already %, Google raw is decimal — normalized to % for human-readable comparisons
- [Phase 09-ad-intelligence]: ads_optimization_suggestions registered in ad_intelligence, meta_ads, and google_ads categories for maximum discoverability

### Pending Todos

- None — Phase 8 complete (3 plans: alerting + validation, restore procedures, backup status API).

### Decisions

- [Phase 07-03]: Docker volumes discovered live (docker volume ls -q) at backup time — no dependency on audit.py inventory
- [Phase 07-03]: Per-volume 300s timeout prevents one large volume blocking others; volume failures are non-fatal
- [Phase 07-03]: labsliveserver uses allowlist patterns (_db_, _data_, postgres, mysql, redis, mongo) for volume export — 55 containers needs safe default
- [Phase 07-03]: Weekly collects DAILY_TARGETS inline (not importing daily_backup.py) — decoupled design
- [Phase 07-03]: Retention cleanup runs after all uploads — never prune before backup confirmed
- [Phase 07-03]: Cron daily changed to 1-6 (Mon-Sat), weekly is 0 (Sunday) — mutually exclusive
- [Phase 08-01]: Telegram alert uses SSH to claw alias + openclaw message send CLI — reuses Phase 6 SSH config, no new bot auth on Labs
- [Phase 08-01]: Alert failure wrapped in try/except — Telegram or SSH outage must never crash backup scripts
- [Phase 08-01]: validate_server() status thresholds: daily=26h (timezone drift buffer), weekly=170h (~7d+2h)
- [Phase 08-01]: Validation cron at 5 AM (3h after 2 AM backup window) — sufficient time for all uploads
- [Phase 08-01]: backup_status.json written atomically via tmp+rename — prevents partial reads by API

- [Phase 08-02]: Restore document derives from backup_utils.py DAILY_TARGETS/WEEKLY_EXTRAS — single source of truth, stays accurate as backup config evolves
- [Phase 08-02]: labsliveserver volume restore priority follows allowlist patterns (_db_, _data_, postgres, mysql, redis, mongo)
- [Phase 08-02]: alfred-claw SSH port 2222 prominently documented — most common failure point during restore
- [Phase 08-02]: cloud-mail SMTP auth via lumabot@ only — alfred@ has no password set

- [Phase 08-03]: GET /api/backup/status placed after circuit-breaker endpoints — logical system health grouping
- [Phase 08-03]: Endpoint auth required but not admin-only — backup status is operational info any authenticated user should see
- [Phase 08-03]: human_summary truncates to first 3 issues + count — readable in Telegram without truncation
- [Phase 08-03]: Missing backup_status.json returns 200 with status=unknown — avoids special-case error handling in consumers

### Blockers/Concerns

- None — Phase 9 all plans complete.

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed 09-02-PLAN.md — ads_optimization_suggestions tool with 7 heuristic rules for campaign-specific ad optimization suggestions
Resume at: Phase 9 complete. Run `/gsd:execute-phase` for the next phase if any remain.
