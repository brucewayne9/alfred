---
phase: 04-google-ads-budget-control
plan: 02
subsystem: api
tags: [google-ads, tool-definitions, tool-registry, llm-tools, budget-control, ad-groups]

# Dependency graph
requires:
  - phase: 04-google-ads-budget-control
    plan: 01
    provides: "update_campaign_budget, set_ad_group_status backend functions in integrations/google_ads/client.py"
provides:
  - "gads_update_campaign_budget LLM-callable tool with confirmation threshold guidance (>$100/day or >2x current)"
  - "gads_pause_ad_group LLM-callable tool with bulk operation warning behavior"
  - "gads_enable_ad_group LLM-callable tool"
  - "gads_set_campaign_status description updated with before/after confirmation guidance"
  - "TOOL_CATEGORIES[google_ads] updated with three new write tools"
  - "CATEGORY_KEYWORDS[google_ads] updated with budget/daily budget/pause-group/enable-group/unpause triggers"
  - "TOOLS.md section 6 updated with budget, pause-group, enable-group commands"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Confirmation threshold in tool description: LLM reads description to know when to ask Mike before calling"
    - "Lazy import pattern: tool function body does 'from integrations.x.client import fn' at call time"
    - "Parameters as dict (not string) for tools that need type info: {type, description, required}"

key-files:
  created: []
  modified:
    - "core/tools/definitions.py — added gads_update_campaign_budget, gads_pause_ad_group, gads_enable_ad_group; updated gads_set_campaign_status description"
    - "core/tools/registry.py — added three new tools to TOOL_CATEGORIES[google_ads]; added 5 keywords to CATEGORY_KEYWORDS[google_ads]"
    - "TOOLS.md — added budget, pause-group, enable-group commands to section 6 (Google Ads)"

key-decisions:
  - "Confirmation threshold lives in tool description: LLM sees the >$100/day and >2x rule as natural language in the description field — no code logic needed in the tool layer"
  - "Parameters as dict (not string shorthand) for new tools: allows type annotation and required flag, consistent with other write tools in definitions.py"
  - "Budget keyword shared with meta_ads: intentional — budget queries correctly surface both Meta and Google Ads tools"

patterns-established:
  - "Pattern: Write tool descriptions include CONFIRMATION RULES section for operations with financial impact"
  - "Pattern: Tool descriptions reference specific thresholds (dollar amounts, multipliers) so LLM can make rule-based decisions"

requirements-completed: [GADS-01, GADS-02, GADS-03]

# Metrics
duration: 2min
completed: 2026-02-21
---

# Phase 4 Plan 02: Google Ads Tool Registration Summary

**Three Google Ads write tools wired into the LLM tool system (gads_update_campaign_budget with $100/2x confirmation threshold, gads_pause_ad_group, gads_enable_ad_group) with keyword routing and TOOLS.md documentation**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-21T02:24:18Z
- **Completed:** 2026-02-21T02:25:48Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `gads_update_campaign_budget` tool with confirmation threshold guidance in description: asks Mike to confirm if budget >$100/day or >2x current; warns on shared budgets
- Added `gads_pause_ad_group` and `gads_enable_ad_group` tools routing to `set_ad_group_status()` with PAUSED/ENABLED status; pause warns if last active group in campaign
- Updated `gads_set_campaign_status` description to include before/after confirmation guidance
- Added all three new tools to `TOOL_CATEGORIES["google_ads"]` and 5 new keywords to `CATEGORY_KEYWORDS["google_ads"]`
- Updated TOOLS.md section 6 with budget, pause-group, enable-group commands (file remains at 6,065 chars, under 20,000 limit)

## Task Commits

Each task was committed atomically:

1. **Task 1: Register new Google Ads write tools in definitions.py and registry.py** - `27b2875` (feat)
2. **Task 2: Update TOOLS.md Google Ads section with new write commands** - `26d9457` (feat)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified
- `core/tools/definitions.py` — added gads_update_campaign_budget, gads_pause_ad_group, gads_enable_ad_group; updated gads_set_campaign_status description with confirmation guidance
- `core/tools/registry.py` — added three new tool names to TOOL_CATEGORIES[google_ads]; added budget/daily budget/pause ad group/enable ad group/unpause to CATEGORY_KEYWORDS[google_ads]
- `TOOLS.md` — added budget, pause-group, enable-group commands to Google Ads section 6

## Decisions Made
- Confirmation threshold lives in tool description (natural language): LLM reads ">$100/day OR >2x current budget" from the description field — no code logic needed at the tool layer, matches how gads_set_campaign_status and meta_ads write tools handle guidance
- Parameters formatted as dicts with type/description/required fields for new tools (not the string shorthand used by older tools) — consistent with other write tools like meta_ads_update_campaign_budget
- "budget" keyword shared with meta_ads CATEGORY_KEYWORDS is intentional — budget-related queries correctly surface both Meta and Google Ads tools

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None. Note: `get_tools()` alone returns empty list because `_tools` dict is populated when definitions.py is imported; `get_relevant_tools()` works correctly because router.py imports definitions first. This is existing behavior, not a new issue.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 4 complete: all Google Ads write operations (budget, ad group pause/enable, campaign pause/enable) are now callable by the LLM with appropriate confirmation guidance
- Alfred Claw on Server 101 can use new TOOLS.md commands: `google_ads.py budget <camp_id> <daily_dollars>`, `google_ads.py pause-group <ag_id>`, `google_ads.py enable-group <ag_id>`
- Blocker from STATE.md remains: verify Google Ads developer token access level (Explorer or Standard tier) in Google Ads console before live testing budget mutations — Test Account tokens will fail against production accounts

---
*Phase: 04-google-ads-budget-control*
*Completed: 2026-02-21*
