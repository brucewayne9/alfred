---
phase: 02-alfred-claw-config-fixes
plan: "04"
subsystem: infra
tags: [openclaw, workspace, USER.md, HEARTBEAT.md, alfred-claw, server-101]

# Dependency graph
requires:
  - phase: 02-alfred-claw-config-fixes
    provides: "Plan 02-01 trimmed USER.md/HEARTBEAT.md to wrong limits — this plan corrects to actual runtime limits"
provides:
  - "USER.md under 2,520 char injection limit (1,798 chars)"
  - "HEARTBEAT.md under 149 char injection limit (140 chars)"
  - "Zero truncation WARNs on OpenClaw bootstrap and heartbeat"
affects: [phase-03, phase-04, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Remote file editing via SSH cat heredoc for complex multi-line files on Server 101"]

key-files:
  created: []
  modified:
    - "~/.openclaw/workspace/USER.md (Server 101)"
    - "~/.openclaw/workspace/HEARTBEAT.md (Server 101)"

key-decisions:
  - "USER.md trimmed to 1,798 chars (720 chars of margin below 2,520 limit) — identity, business overview, digital properties, communication prefs, current clients, infrastructure (1-liner), family (names+birthdays only), Queue/Escalation (verbatim)"
  - "Decision Making section removed entirely — personality info belongs in SOUL.md, not USER.md"
  - "Key Dates table merged into single Family line: 'Wife: Sarah (Jan 1). Kids: Rowan (Oct 21), Rex (Oct 8), Brandon (Dec 28), Hillary (Dec 5).'"
  - "HEARTBEAT.md rewritten to 4-line ultra-compact format: inbox check, grep QUEUE.md, birthday check, HEARTBEAT_OK reply instruction"
  - "HEARTBEAT.md header (#Heartbeat) retained — without header the file still has 4 substantive lines; removing header saves only 12 chars and was not needed"

patterns-established:
  - "OpenClaw injection limits are NOT the same as workspace file size limits in openclaw.json — runtime injection limit (2,520 / 149) is lower than config file-size limit (3,955 / 293)"
  - "When trimming workspace files, target 5-10% below the injection limit as margin, not just under it"

requirements-completed: [CLAW-02, CLAW-03]

# Metrics
duration: 8min
completed: 2026-02-20
---

# Phase 2 Plan 04: USER.md + HEARTBEAT.md Gap Closure Summary

**USER.md trimmed to 1,798 chars and HEARTBEAT.md to 140 chars — both under OpenClaw runtime injection limits (2,520 / 149), zero truncation WARNs in post-restart gateway logs**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-20T23:26:03Z
- **Completed:** 2026-02-20T23:34:00Z
- **Tasks:** 1
- **Files modified:** 2 (on Server 101)

## Accomplishments
- USER.md trimmed from 3,798 to 1,798 chars — 2,000 char reduction, 722-char margin below the 2,520 injection limit
- HEARTBEAT.md trimmed from 231 to 140 chars — 91 char reduction, 9-char margin below the 149 injection limit
- Gateway restarted and verified: zero truncation WARNs in 20+ post-restart log entries (23:27 UTC+)
- All key content preserved: identity, business overview, clients, infrastructure (compact), family names/birthdays, Queue & Escalation section (verbatim)

## Task Commits

Each task was committed atomically:

1. **Task 1: Trim USER.md and HEARTBEAT.md to actual injection limits** - `9465ac6` (fix)

**Plan metadata:** (pending final commit)

## Files Created/Modified
- `~/.openclaw/workspace/USER.md` (Server 101) — Trimmed from 3,798 to 1,798 chars. Removed: Decision Making section, Key Dates table (merged into Family line), verbose family detail (Sarah's parents/siblings/phone/email, Mike's birth/parents/siblings), verbose client KPI detail, multi-line infrastructure block
- `~/.openclaw/workspace/HEARTBEAT.md` (Server 101) — Rewritten from 231 to 140 chars. 4-line ultra-compact format: inbox check, grep QUEUE.md, birthdays within 7d, HEARTBEAT_OK reply instruction
- `~/.openclaw/workspace/USER.md.bak-gap-20260220` (Server 101) — Backup of pre-trim USER.md
- `~/.openclaw/workspace/HEARTBEAT.md.bak-gap-20260220` (Server 101) — Backup of pre-trim HEARTBEAT.md

## Decisions Made
- **USER.md structure**: Kept 7 sections (Identity, Business Overview, Digital Properties, Communication, Current Clients, Infrastructure, Family, Queue & Escalation). Removed Decision Making entirely — that belongs in SOUL.md.
- **Family line format**: Merged Key Dates table into single compact line. Brandon and Hillary adult detail (location, family) removed — only name + birthday kept per plan instructions.
- **HEARTBEAT.md**: Retained the `# Heartbeat` header. With header, file is 140 chars (under 149). The header helps OpenClaw identify file context.
- **Infrastructure format**: Compressed 7 multi-line IP:role entries to single comma-separated line "Servers: 101(Claw/SSH:2222), 104(Prod), 105(Labs), 98(Loovacast Dev), 100(Loovacast Prod), 117(Dokploy/CRM), 121(Mailcow)" — saves ~130 chars.

## Deviations from Plan

None - plan executed exactly as written.

Note: The plan's verification criterion `grep -c -E "Ground Rush|Sarah|Rowan|Rex|QUEUE|ESCALAT" USER.md >= 6` returns 4 in the trimmed file (not 6). This is because grep -c counts matching LINES not total matches — Sarah, Rowan, and Rex all appear on the same single family line. All 6 patterns ARE present in the file. The actual content requirement from the must_haves truths is fully satisfied: identity, family names/birthdays, business context, and Queue/Escalation are all present.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CLAW-02 (USER.md char limit) and CLAW-03 (HEARTBEAT.md char limit) are now fully satisfied
- Phase 2 gap closure is complete for char limit issues (plans 02-04 covers the last two open gaps from the VERIFICATION.md)
- CLAW-06 (embedding batches never complete) remains open — addressed in plan 02-05
- Phase 3 ready once CLAW-06 is resolved

---
*Phase: 02-alfred-claw-config-fixes*
*Completed: 2026-02-20*
