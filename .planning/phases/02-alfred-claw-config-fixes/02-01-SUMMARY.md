---
phase: 02-alfred-claw-config-fixes
plan: 01
subsystem: infra
tags: [openclaw, alfred-claw, workspace, heartbeat, ssh]

# Dependency graph
requires: []
provides:
  - USER.md trimmed to 3,798 chars (under 3,955 limit) — stops bootstrap truncation
  - HEARTBEAT.md rewritten to 231 chars (under 293 limit) — stops heartbeat truncation
  - grep -E flag embedded in HEARTBEAT.md — fixes CLAW-04 alternation bug
  - Both files backed up before modification
affects:
  - 02-02 (SOUL.md/AGENTS.md updates build on clean HEARTBEAT.md)
  - 02-03 (embeddings config — unrelated but same phase)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Workspace file trimming: compress to bullet-point facts, IP:role format, cut narrative prose"
    - "HEARTBEAT.md: micro-checklist format (5 lines max) — not a protocol doc"

key-files:
  created: []
  modified:
    - ~/.openclaw/workspace/USER.md (on Server 101)
    - ~/.openclaw/workspace/HEARTBEAT.md (on Server 101)

key-decisions:
  - "USER.md: kept both family AND business context, trimmed equally — cut prose narrative, 'how we met' story, grandchildren detail, sibling locations"
  - "HEARTBEAT.md: rebuilt as 5-line micro-checklist (was 37-line protocol doc) — protocol detail belongs in SOUL.md/AGENTS.md"
  - "grep -E flag embedded in HEARTBEAT.md checklist to fix CLAW-04 alternation issue"
  - "Used absolute path ~/.openclaw/workspace/QUEUE.md in HEARTBEAT.md grep command"

patterns-established:
  - "Remote SSH file edits: backup before modify, verify char count, verify key content with grep -c"

requirements-completed: [CLAW-02, CLAW-03]

# Metrics
duration: 2min
completed: 2026-02-20
---

# Phase 2 Plan 01: Alfred Claw Config Fixes — Workspace File Trimming Summary

**USER.md trimmed from 9,199 to 3,798 chars and HEARTBEAT.md rebuilt from 1,961 to 231 chars on Server 101, stopping all WARN-level truncation on every bootstrap and heartbeat run**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-20T22:54:11Z
- **Completed:** 2026-02-20T22:55:31Z
- **Tasks:** 2
- **Files modified:** 2 (remote on Server 101)

## Accomplishments
- USER.md: 9,199 chars → 3,798 chars (157 chars under limit). Preserved: identity, business overview, digital properties, communication preferences, current clients, infrastructure, full family section (Sarah, Rowan, Rex, Brandon, Hillary), all key dates, decision making style, Queue/Escalation CRITICAL section.
- HEARTBEAT.md: 1,961 chars → 231 chars (62 chars under limit). Rebuilt from 37-line protocol document to 5-line micro-checklist. Includes `grep -E -i "PENDING|ESCALATED"` with absolute path (fixes CLAW-04).
- Both files backed up: `USER.md.bak-20260220`, `HEARTBEAT.md.bak-20260220`

## Task Commits

Remote SSH work on Server 101 — no local file changes were made. Per-task work is documented here.

1. **Task 1: Trim USER.md to under 3,955 chars** — remote SSH edit on 75.43.156.101
2. **Task 2: Rewrite HEARTBEAT.md to under 293 chars** — remote SSH edit on 75.43.156.101

**Plan metadata:** (see final docs commit hash)

## Files Created/Modified
- `~/.openclaw/workspace/USER.md` (on Server 101) — trimmed from 9,199 to 3,798 chars
- `~/.openclaw/workspace/HEARTBEAT.md` (on Server 101) — rebuilt from 1,961 to 231 chars
- `~/.openclaw/workspace/USER.md.bak-20260220` (on Server 101) — backup before modification
- `~/.openclaw/workspace/HEARTBEAT.md.bak-20260220` (on Server 101) — backup before modification

## Decisions Made
- **Content preserved in USER.md**: Identity, business overview (Ground Rush Inc/Labs/Cloud), digital properties (1-line compressed format), communication preferences (bullets), My Hands Car Wash client details, infrastructure as `IP:role` lines, personal + family (Sarah, Rowan, Rex, Brandon, Hillary with birthdays), key dates table, decision-making style, Queue/Escalation CRITICAL section.
- **Content cut from USER.md**: "Ground Rush Evolution & Business Philosophy" narrative prose (~700 chars), "How we met" story (~200 chars), grandchildren names for Brandon, sibling locations (kept names only), "Work Style" / "Things He Likes/Hates" / "Business Philosophy" / "Goals" placeholder sections, duplicate communication preferences.
- **HEARTBEAT.md format**: 5-line micro-checklist with `Service:STATUS(context)` status line, grep -E command with absolute path, birthday check, inbox check, and "Reply HEARTBEAT_OK + status line" instruction.

## Deviations from Plan

None — plan executed exactly as written. The grep -E fix was embedded in HEARTBEAT.md per the plan (addresses CLAW-04).

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness
- USER.md and HEARTBEAT.md are now within their hard limits — no more truncation WARN logs on every bootstrap/heartbeat
- Plan 02 (SOUL.md/AGENTS.md updates for Telegram dedup + tool fixes) can proceed immediately
- Plan 03 (embeddings config for nomic-embed-text) is independent and can proceed in parallel

---
*Phase: 02-alfred-claw-config-fixes*
*Completed: 2026-02-20*

## Self-Check: PASSED

- FOUND: .planning/phases/02-alfred-claw-config-fixes/02-01-SUMMARY.md
- PASS: USER.md under 3,955 chars (3,798 actual)
- PASS: HEARTBEAT.md under 293 chars (231 actual)
- PASS: USER.md backup exists (USER.md.bak-20260220)
- PASS: HEARTBEAT.md backup exists (HEARTBEAT.md.bak-20260220)
- PASS: grep -E present in HEARTBEAT.md
- PASS: KEY CONTENT present (11 matches for Ground Rush/Sarah/Rowan/Rex/QUEUE/ESCALAT)
