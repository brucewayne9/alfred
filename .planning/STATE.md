# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** Phase 2 — Alfred Claw Config Fixes

## Current Position

Phase: 2 of 5 (Alfred Claw Config Fixes)
Plan: 1 of 3 in current phase (02-01 complete)
Status: Phase 2 in progress
Last activity: 2026-02-20 — Plan 02-01 complete (USER.md trimmed 9,199→3,798 chars; HEARTBEAT.md rebuilt 1,961→231 chars)

Progress: [████░░░░░░] 37%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 3.3 min
- Total execution time: ~10 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Infrastructure Repairs | 2 | ~8 min | ~4 min |
| 2. Alfred Claw Config Fixes | 1 (of 3) | ~2 min | ~2 min |

**Recent Trend:**
- Last 5 plans: 01-01 (3 min), 01-02 (5 min), 02-01 (2 min)
- Trend: fast (remote file edits)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Pre-planning]: Fix via SSH from 105 — single Claude Code session manages both servers
- [Pre-planning]: Full ads API integration — Mike needs conversational campaign control for active campaigns
- [Pre-planning]: Fix CRM before building new — reliability of existing tools before new features
- [01-01]: Targeted sed edit of rules.v4 (not iptables-save) to preserve Docker dynamic rules on 117
- [01-01]: Circuit breaker admin endpoints placed in Knowledge Management section of main.py for co-location with LightRAG routes
- [01-01]: Admin role check done inline (not shared dependency) to match existing codebase pattern
- [01-02]: INFRA-03 was already complete — GA4 was already fully working, no code changes needed
- [01-02]: git gc --prune=now used (immediate prune) — safe because all work is committed to main branch
- [02-01]: USER.md: kept both family AND business context, trimmed equally — cut narrative prose, "how we met" story, grandchildren detail, sibling locations
- [02-01]: HEARTBEAT.md: rebuilt as 5-line micro-checklist (was 37-line protocol doc) — protocol detail belongs in SOUL.md/AGENTS.md
- [02-01]: grep -E flag embedded in HEARTBEAT.md checklist to fix CLAW-04 alternation issue with absolute path

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4 prerequisite]: Verify Google Ads developer token approval level before writing budget mutations — sandbox-only token will silently fail against production accounts
- [Phase 5 prerequisite]: Confirm whether current Meta access token in .env is System User (non-expiring) or personal user token before Phase 5
- [Phase 2 decision RESOLVED]: Switched Claw to Ollama nomic-embed-text for embeddings (Plan 02-03 will execute the config change)

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 02-01-PLAN.md (USER.md + HEARTBEAT.md trimmed)
Resume file: None
