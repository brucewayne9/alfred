---
phase: 02-alfred-claw-config-fixes
plan: 03
subsystem: infra
tags: [openclaw, ollama, embeddings, nomic-embed-text, cron, log-cleanup, gateway]

# Dependency graph
requires:
  - phase: 02-01
    provides: USER.md and HEARTBEAT.md trimmed within token limits
  - phase: 02-02
    provides: AGENTS.md heartbeat protocol, SOUL.md Telegram target, TOOLS.md tool formats, email mark-read dispatch
provides:
  - Local Ollama nomic-embed-text embeddings active in OpenClaw (replaces dead OpenAI key)
  - Log cleanup cron installed — openclaw logs auto-pruned after 7 days
  - Stale systemd gateway service confirmed absent
  - Gateway restarted — all Phase 2 workspace file changes active
affects: [phase-03, phase-04, phase-05]

# Tech tracking
tech-stack:
  added: [nomic-embed-text (Ollama), /etc/cron.daily/openclaw-log-cleanup]
  patterns: [Ollama OpenAI-compatible /v1/embeddings endpoint for local model access, local-first with cloud fallback pattern for AI services]

key-files:
  created:
    - /etc/cron.daily/openclaw-log-cleanup
  modified:
    - ~/.openclaw/openclaw.json (agents.defaults.memorySearch — Ollama nomic-embed-text config)

key-decisions:
  - "Use provider: openai with Ollama baseUrl (http://127.0.0.1:11434/v1/) — Ollama exposes OpenAI-compatible embeddings endpoint"
  - "Configure fallback: openai per locked user decision (local-first, cloud-fallback architecture) even though current OpenAI key returns HTTP 401"
  - "Use cron.daily script instead of logrotate for /tmp/ date-based log cleanup — logrotate handles /tmp/ glob paths poorly"

patterns-established:
  - "Ollama local-first pattern: provider=openai, baseUrl=localhost:11434/v1/, apiKey=ollama-local"
  - "Gateway restart procedure: pkill -f openclaw, sleep 3, nohup openclaw gateway, sleep 5, verify ps aux"

requirements-completed: [CLAW-06, INFRA-04, INFRA-05]

# Metrics
duration: ~10min
completed: 2026-02-20
---

# Phase 2 Plan 03: Alfred Claw Config Fixes — Embeddings + Log Cleanup + Gateway Restart Summary

**Ollama nomic-embed-text configured as local embedding provider in OpenClaw, log cleanup cron installed, stale systemd gateway absent confirmed, and gateway restarted to activate all Phase 2 workspace file changes**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-02-20T00:00:00Z
- **Completed:** 2026-02-20
- **Tasks:** 2 (1 auto + 1 human-verify checkpoint)
- **Files modified:** 2 (openclaw.json, /etc/cron.daily/openclaw-log-cleanup)

## Accomplishments
- nomic-embed-text pulled from Ollama (274MB, 768-dim vectors), verified working via /api/embeddings
- openclaw.json memorySearch section updated from empty `{}` to full Ollama config (local-first, OpenAI fallback)
- Daily cron cleanup script installed at /etc/cron.daily/openclaw-log-cleanup — removes logs older than 7 days
- No stale openclaw-gateway.service found in systemctl (INFRA-05 confirmed)
- Gateway restarted — USER.md, HEARTBEAT.md, AGENTS.md, SOUL.md, TOOLS.md changes from Plans 01+02 now active
- Human verified: Telegram sends exactly one response with no duplicates (all Phase 2 fixes confirmed working)

## Task Commits

Each task was committed atomically:

1. **Task 1: Pull nomic-embed-text, configure embeddings, log cleanup, restart gateway** - `df4abbf` (feat)
2. **Task 2: Verify all Phase 2 fixes are working** - human-verify checkpoint (approved by user)

**Plan metadata:** (this commit — docs: complete plan)

## Files Created/Modified
- `~/.openclaw/openclaw.json` - memorySearch config updated to Ollama nomic-embed-text
- `/etc/cron.daily/openclaw-log-cleanup` - Daily cron script to prune openclaw logs older than 7 days

## Decisions Made
- Used `provider: "openai"` with Ollama baseUrl — Ollama exposes OpenAI-compatible `/v1/embeddings` endpoint, no custom provider needed
- Configured `fallback: "openai"` per locked user decision despite current OpenAI API key returning HTTP 401 — architecture is correct, key fix is a separate concern
- Used cron.daily script instead of logrotate — logrotate handles `/tmp/` glob paths poorly; date-based file naming makes per-day deletion simpler with `find -mtime +7`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The current OpenAI API key returns HTTP 401 (unauthorized). This means the `fallback: "openai"` in memorySearch config will silently fail if Ollama is unavailable. This is a known pre-existing issue — not caused by this plan. User is aware per the locked decision to maintain the fallback architecture. Resolution: update OPENAI_API_KEY env var on Server 101 when a new key is available.

## User Setup Required

None — no external service configuration required beyond what was executed.

## Next Phase Readiness

- All Phase 2 fixes are active and verified on Server 101
- OpenClaw is running with correct USER.md, HEARTBEAT.md, AGENTS.md, SOUL.md, TOOLS.md content
- Telegram dedup is fixed, embeddings are local, log cleanup is scheduled
- Phase 3 (Alfred Labs React/UI work) can proceed — no blockers from Phase 2
- Outstanding concern: OpenAI API key on 101 needs replacement before cloud embedding fallback will work

---
*Phase: 02-alfred-claw-config-fixes*
*Completed: 2026-02-20*
