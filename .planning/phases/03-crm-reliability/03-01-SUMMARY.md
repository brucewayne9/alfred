---
phase: 03-crm-reliability
plan: 01
subsystem: api
tags: [crm, twenty-crm, python, cli, graphql, rest-api]

# Dependency graph
requires: []
provides:
  - "create_linked_note() in crm.py: two-step atomic note+noteTarget creation with rollback"
  - "create_linked_task() in crm.py: two-step atomic task+taskTarget creation with rollback"
  - "search_people() cap increased from first:50 to first:500 with truncation flag"
  - "search-people CLI: numbered list disambiguation for multiple matches"
  - "search-people CLI: create-new-contact offer on zero results"
  - "TOOLS.md: create-linked-note and create-linked-task commands documented"
  - "create_note() fixed to use bodyV2.markdown format (was silently failing)"
affects:
  - "Alfred Claw agent behavior for CRM note/task creation"
  - "Alfred Claw contact search quality for broad queries"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-step atomic creation with immediate rollback (no retry): create record -> link record; if link fails, DELETE record immediately"
    - "CLI output layer separates human-readable output from function return values"
    - "SCP + heredoc for deploying multi-line Python files to remote Server 101"

key-files:
  created: []
  modified:
    - "/home/brucewayne9/.openclaw/workspace/scripts/integrations/crm.py"
    - "/home/brucewayne9/.openclaw/workspace/TOOLS.md"

key-decisions:
  - "Immediate rollback (no retry) on step-2 failure — HTTP 400/500 unlikely to succeed on retry"
  - "first:500 cap for search_people (not unlimited) — covers all realistic name searches; truncated flag added when count==500"
  - "Numbered list disambiguation lives in CLI output layer (_print_search_results), not in function return value"
  - "Zero-result create offer is CLI message only — agent behavior, not script data structure"
  - "search-people command alias added alongside existing 'search' command for clarity"

patterns-established:
  - "Pattern: Two-step atomic linked creation with rollback — use for any future linked record creation in Twenty CRM"

requirements-completed:
  - CRM-01
  - CRM-02

# Metrics
duration: 25min
completed: 2026-02-21
---

# Phase 3 Plan 01: CRM Reliability Summary

**Atomic two-step note/task creation with rollback, 500-result search cap, numbered-list disambiguation, and zero-result create offer — all deployed to Alfred Claw crm.py and TOOLS.md**

## Performance

- **Duration:** 25 min
- **Started:** 2026-02-21T00:53:56Z
- **Completed:** 2026-02-21T01:18:00Z
- **Tasks:** 3
- **Files modified:** 2 (on Server 101)

## Accomplishments
- Added `create_linked_note()` with bodyV2 format + rollback on noteTarget link failure (no orphaned notes)
- Added `create_linked_task()` with rollback on taskTarget link failure (no orphaned tasks)
- Fixed broken `create_note()` to use `bodyV2.markdown` format (was silently failing with old `body` field)
- Increased `search_people()` GQL cap from `first: 50` to `first: 500` (114 contacts now returned for "son" vs. 50 before)
- Added `_print_search_results()` CLI formatter: numbered list for multiple matches, create offer for zero matches
- Updated TOOLS.md with create-linked-note, create-linked-task docs and search-people disambiguation note
- Restarted OpenClaw gateway with updated workspace files

## Task Commits

Each task was committed atomically:

1. **Task 1: create_linked_note() and create_linked_task() with rollback** - `c7b8ca8` (feat)
2. **Task 2: Fix search cap + update TOOLS.md** - included in `c7b8ca8` (same deployment)
3. **Task 3: Numbered list disambiguation + zero-result offer** - included in `c7b8ca8` (same deployment)

**Plan metadata:** TBD (docs: complete plan)

_Note: All three tasks modified the same two remote files (crm.py, TOOLS.md) and were deployed in a single scp pass. Single atomic commit captures all task changes._

## Files Created/Modified
- `/home/brucewayne9/.openclaw/workspace/scripts/integrations/crm.py` (Server 101) - Added create_linked_note(), create_linked_task(), _print_search_results(); fixed create_note() bodyV2; increased search cap to 500
- `/home/brucewayne9/.openclaw/workspace/TOOLS.md` (Server 101) - Added create-linked-note, create-linked-task entries; updated search-people description with 500-result cap and disambiguation behavior
- `data/backups/crm_updated.py` (local) - Implementation record / staging file
- `data/backups/TOOLS_updated.md` (local) - Implementation record / staging file

## Decisions Made
- Immediate rollback on step-2 failure (no retry) — HTTP 400/500 errors are deterministic, retrying would not help and extends the window of partial state
- `first: 500` cap chosen (not unlimited) — covers all realistic name searches; `truncated: True` flag added when results hit the cap
- Numbered list disambiguation in `_print_search_results()` CLI layer, not in function return value — keeps function API clean for programmatic use
- `search-people` added as command alias alongside existing `search` command for clarity in TOOLS.md

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered
- Shell escaping issues when running `nohup openclaw gateway > /dev/null 2>&1 &;` via SSH with semicolon after `&` — resolved by using bash heredoc for the gateway restart command

## User Setup Required
None — no external service configuration required.

## Next Phase Readiness
- CRM note/task creation is now reliable and atomic — Alfred Claw can be instructed to use create-linked-note and create-linked-task for all contact note/task operations
- Contact search now returns full result sets for realistic name queries
- Phase 3 Plan 01 complete — all CRM reliability requirements (CRM-01, CRM-02) satisfied

---
*Phase: 03-crm-reliability*
*Completed: 2026-02-21*
