# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-20)

**Core value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns and CRM contacts conversationally without touching the ad platforms or CRM directly.
**Current focus:** Phase 2 — Alfred Claw Config Fixes

## Current Position

Phase: 2 of 5 (Alfred Claw Config Fixes)
Plan: 3 of 3 in current phase (02-01, 02-02, 02-03 complete) — Phase 2 COMPLETE
Status: Phase 2 complete, ready for Phase 3
Last activity: 2026-02-20 — Plan 02-03 complete (nomic-embed-text embeddings, log cleanup cron, gateway restart, Phase 2 verified)

Progress: [██████░░░░] 60%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: 4 min
- Total execution time: ~24 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Infrastructure Repairs | 2 | ~8 min | ~4 min |
| 2. Alfred Claw Config Fixes | 3 (of 3) | ~16 min | ~5 min |

**Recent Trend:**
- Last 5 plans: 01-01 (3 min), 01-02 (5 min), 02-01 (2 min), 02-02 (4 min), 02-03 (10 min)
- Trend: fast (remote file edits + human verify checkpoint)

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
- [02-02]: Behavioral fix for Telegram dedup — instruct agent via AGENTS.md + SOUL.md rather than modifying OpenClaw gateway config
- [02-02]: TOOLS.md Telegram section added as section 18 (built-in tool, not Python script) to document correct target format
- [02-02]: email_client.py mark-read: added dispatch branch only (function already existed); use scp for complex multi-line writes to Server 101
- [02-03]: Use provider: openai with Ollama baseUrl (11434/v1/) — Ollama exposes OpenAI-compatible embeddings endpoint
- [02-03]: Configure fallback: openai per locked user decision despite HTTP 401 key — architecture correct, key fix is separate concern
- [02-03]: cron.daily script preferred over logrotate for /tmp/ date-based log cleanup

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4 prerequisite]: Verify Google Ads developer token approval level before writing budget mutations — sandbox-only token will silently fail against production accounts
- [Phase 5 prerequisite]: Confirm whether current Meta access token in .env is System User (non-expiring) or personal user token before Phase 5
- [Phase 2 COMPLETE]: All 6 Claw bugs + 3 infra issues fixed (CLAW-01 through CLAW-06, INFRA-03 through INFRA-05)
- [Phase 3 prerequisite]: OpenAI API key on Server 101 needs replacement for cloud embedding fallback to work

## Session Continuity

Last session: 2026-02-20
Stopped at: Completed 02-03-PLAN.md (nomic-embed-text embeddings + log cleanup cron + gateway restart + Phase 2 verified end-to-end)
Resume file: None
