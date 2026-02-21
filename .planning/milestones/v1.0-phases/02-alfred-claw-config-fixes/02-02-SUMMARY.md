---
phase: 02-alfred-claw-config-fixes
plan: 02
subsystem: infra
tags: [openclaw, telegram, heartbeat, email, bash, python]

# Dependency graph
requires:
  - phase: 01-infrastructure-repairs
    provides: Stable Alfred Labs backend (connectivity prerequisite)
provides:
  - AGENTS.md with explicit HEARTBEAT_OK plain-text reply protocol (no messaging tool)
  - AGENTS.md with corrected QUEUE.md grep using -E flag and absolute path
  - SOUL.md with Telegram numeric chat ID format (7582976864)
  - TOOLS.md with Telegram numeric chat ID documentation and Telegram tool section
  - email_client.py with working mark-read command dispatch
affects:
  - 02-03 (USER.md trimming, HEARTBEAT.md trimming, embeddings config)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "HEARTBEAT_OK: reply as plain text only — heartbeat framework delivers; tool sends cause duplicates"
    - "Telegram target: numeric chat ID only (7582976864) — no display names"
    - "QUEUE.md grep: always use -E flag and absolute path for reliable alternation match"
    - "email_client.py dispatch: elif chain gates on cmd value and len(sys.argv)"

key-files:
  created: []
  modified:
    - ~/.openclaw/workspace/AGENTS.md (on Server 101)
    - ~/.openclaw/workspace/SOUL.md (on Server 101)
    - ~/.openclaw/workspace/TOOLS.md (on Server 101)
    - ~/.openclaw/workspace/scripts/integrations/email_client.py (on Server 101)

key-decisions:
  - "Behavioral fix for Telegram dedup: instruct agent via AGENTS.md + SOUL.md rather than modifying OpenClaw gateway config"
  - "TOOLS.md Telegram section added as section 18 (built-in tool, not Python script) to document correct target format"
  - "email_client.py mark-read: added dispatch branch only (function already existed at line 233)"

patterns-established:
  - "All remote changes on Server 101 made via SSH from 105 using Python scripts via scp to avoid shell escaping issues"
  - "Changes tracked via empty/planning commits in 105 git repo with detailed remote-change documentation"

requirements-completed: [CLAW-01, CLAW-04, CLAW-05]

# Metrics
duration: 4min
completed: 2026-02-20
---

# Phase 2 Plan 02: Alfred Claw Config Fixes (Telegram + Tools) Summary

**Telegram dedup fixed via HEARTBEAT_OK plain-text protocol in AGENTS.md, QUEUE.md grep corrected with -E flag, and email mark-read dispatch added to email_client.py**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-20T22:54:10Z
- **Completed:** 2026-02-20T22:58:14Z
- **Tasks:** 2
- **Files modified:** 4 (on Server 101 via SSH)

## Accomplishments

- Fixed Telegram duplicate message bug (CLAW-01): AGENTS.md now explicitly instructs the agent to reply HEARTBEAT_OK as plain text and NOT use the messaging tool — the heartbeat framework handles delivery
- Fixed QUEUE.md grep syntax (CLAW-04): AGENTS.md now shows `grep -E -i "PENDING|ESCALATED" ~/.openclaw/workspace/QUEUE.md` with -E flag and absolute path
- Fixed email mark-read dispatch (CLAW-05): email_client.py elif chain now includes mark-read branch — `mark_read()` function existed but was unreachable
- Fixed Telegram target format (CLAW-05): SOUL.md and TOOLS.md now document numeric chat ID only (7582976864), no display name format

## Task Commits

1. **Task 1: Fix AGENTS.md + SOUL.md** - `0f3d648` (feat)
2. **Task 2: Fix TOOLS.md + email_client.py** - `b442203` (feat)

**Plan metadata:** (final docs commit, recorded below)

## Files Created/Modified

All files on Server 101 (brucewayne9@75.43.156.101 port 2222):

- `~/.openclaw/workspace/AGENTS.md` - Added Heartbeat Protocol section with HEARTBEAT_OK plain-text mandate, QUEUE.md grep with -E flag + absolute path. Size: 6267 chars.
- `~/.openclaw/workspace/SOUL.md` - Added Telegram Target Format section before Escalation Protocol. Numeric chat ID 7582976864 documented. Size: 8011 chars.
- `~/.openclaw/workspace/TOOLS.md` - Added section 18 (Telegram built-in tool) with numeric chat ID format. No python33 typo. Size: 6458 chars (under 20,000 limit).
- `~/.openclaw/workspace/scripts/integrations/email_client.py` - Added `elif cmd == "mark-read"` dispatch branch between search and mark-unread branches.

Backups created on Server 101:
- `~/.openclaw/workspace/AGENTS.md.bak-20260220`
- `~/.openclaw/workspace/SOUL.md.bak-20260220`
- `~/.openclaw/workspace/TOOLS.md.bak-20260220`
- `~/.openclaw/workspace/scripts/integrations/email_client.py.bak-20260220`

## Decisions Made

- **Behavioral fix approach for Telegram dedup:** Modified AGENTS.md + SOUL.md instruction files rather than changing OpenClaw gateway config. The root cause is the agent using its messaging tool to send HEARTBEAT_OK — the fix is to tell the agent not to do this. No gateway restart needed.
- **TOOLS.md Telegram section:** Added as section 18 ("Telegram Built-in OpenClaw Tool") since the messaging tool is not a Python script like the other 17 tools. This is the right location for the target format documentation.
- **email_client.py mark-read:** Added dispatch branch only. The mark_read() function was already implemented at line 233 with correct IMAP +FLAGS \\Seen logic — only the elif routing was missing.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SOUL.md Telegram section had shell-interpolated empty values**
- **Found during:** Task 1 (SOUL.md Telegram target section)
- **Issue:** First write attempt used SSH inline Python with triple-backtick code blocks; shell interpreted backtick content as command substitution, leaving Mike's chat ID and format examples as empty strings
- **Fix:** Used scp to transfer Python script to Server 101, ran locally to avoid shell escaping issues. Replaced broken section with correct content.
- **Files modified:** ~/.openclaw/workspace/SOUL.md
- **Verification:** `grep '7582976864' SOUL.md` returns 4 matches
- **Committed in:** 0f3d648 (Task 1 commit)

**2. [Rule 1 - Bug] AGENTS.md Heartbeat Protocol code block corrupted by shell**
- **Found during:** Task 1 (AGENTS.md QUEUE.md grep command)
- **Issue:** Triple-backtick code block in Python string sent via SSH got merged into an existing ESCALATION template block in AGENTS.md — grep command text replaced by escalation bullet points
- **Fix:** Used scp to transfer fix script to Server 101. Replaced corrupted section with correctly indented grep command (4-space indent instead of code fence).
- **Files modified:** ~/.openclaw/workspace/AGENTS.md
- **Verification:** `grep '\.openclaw/workspace/QUEUE\.md' AGENTS.md` matches once with correct grep -E command
- **Committed in:** 0f3d648 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bug, caused by shell escaping during first write attempt)
**Impact on plan:** Both auto-fixes resolved immediately. Pattern established: use scp + local Python execution for complex multi-line file writes to Server 101.

## Issues Encountered

- Shell escaping challenges when writing multi-line Python strings with backtick content via SSH inline Python. Resolved by always using `scp` to transfer Python fix scripts to Server 101 and executing locally.

## User Setup Required

None - all changes are to instruction files and Python scripts already in place. No environment variables or external service configuration required. Changes take effect on next OpenClaw session/heartbeat (workspace files are read at session bootstrap time).

## Next Phase Readiness

- Plan 02-03 (USER.md trim, HEARTBEAT.md rebuild, embeddings config) can proceed
- CLAW-01, CLAW-04, CLAW-05 requirements are complete
- Remaining: CLAW-02 (USER.md 9199->3955 chars), CLAW-03 (HEARTBEAT.md 1961->293 chars), CLAW-06 (Ollama nomic-embed-text embeddings)

---
*Phase: 02-alfred-claw-config-fixes*
*Completed: 2026-02-20*
