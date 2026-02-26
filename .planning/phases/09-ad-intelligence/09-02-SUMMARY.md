---
phase: 09-ad-intelligence
plan: 02
subsystem: api
tags: [meta-ads, google-ads, ad-intelligence, optimization, suggestions, heuristics]

# Dependency graph
requires:
  - phase: 09-01
    provides: Cross-platform ad summary tool (ads_cross_platform_summary) and get_cross_platform_summary() function
provides:
  - AI-powered optimization suggestions tool (ads_optimization_suggestions) covering Meta + Google in one response
  - 7 rule-based heuristics: high CPC outlier, zero conversions, low CTR, strong performer underfunded, paused good metrics, budget imbalance, declining performance
  - integrations/ad_intelligence/suggestions.py engine
affects: [09-ad-intelligence]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rule-based heuristic engine: each rule is a separate _check_*() function, all collected into sorted suggestions list"
    - "Priority sort: suggestions sorted by high/medium/low priority using _PRIORITY_ORDER dict key"
    - "Safe division helper: _safe_div() used throughout to avoid ZeroDivisionError"
    - "Lazy API fetching: Rule g only fetches 7-day summary if days >= 14, avoids unnecessary API calls"
    - "CTR normalization: Meta stores CTR as percentage (1.5 = 1.5%), Google as decimal (0.015 = 1.5%)"

key-files:
  created:
    - integrations/ad_intelligence/suggestions.py
  modified:
    - core/tools/definitions.py
    - core/tools/registry.py

key-decisions:
  - "Rule-based heuristics not external LLM — the LLM calling this tool already has context; we analyze data and return structured suggestions it can relay"
  - "Rule g (declining performance) lazy-fetches a 7-day summary for comparison only when days >= 14 — avoids extra API calls for default 7-day queries"
  - "CTR normalization applied per-platform: Meta raw CTR is already %, Google raw CTR is decimal — normalized to % for human-readable comparisons"
  - "ads_optimization_suggestions added to ad_intelligence, meta_ads, and google_ads categories matching ads_cross_platform_summary pattern"
  - "Optimization keywords added to ad_intelligence CATEGORY_KEYWORDS so queries like 'what should I do with my ads' route correctly"

patterns-established:
  - "Heuristic engine pattern: separate _check_*() functions per rule, all return list[dict], collected and sorted centrally"
  - "Suggestion schema: priority, type, platform, campaign_name, campaign_id, reason, metric_detail, suggested_action"

requirements-completed: [ADS-01]

# Metrics
duration: 12min
completed: 2026-02-26
---

# Phase 9 Plan 02: Ad Intelligence — Optimization Suggestions Engine Summary

**Rule-based ad optimization suggestion engine returning campaign-specific pause/budget/investigate recommendations across Meta and Google in a single `ads_optimization_suggestions` tool call**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-02-26T21:07:17Z
- **Completed:** 2026-02-26T21:19:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `integrations/ad_intelligence/suggestions.py` with `generate_suggestions(days)` implementing 7 heuristic rules
- Suggestions are structured dicts with priority, type, campaign name, plain-English reason, metric_detail, and suggested_action
- `ads_optimization_suggestions` tool registered and discoverable via optimization-related keyword queries
- Both Plan 01 tools (`ads_cross_platform_summary`) and Plan 02 tools (`ads_optimization_suggestions`) coexist and surface together for ad queries

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ad optimization suggestions engine** - `6063996` (feat)
2. **Task 2: Register optimization suggestions tool and update keyword routing** - `341e1a4` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `integrations/ad_intelligence/suggestions.py` - `generate_suggestions(days)` — 7 heuristic rules, sorts by priority, returns structured suggestions + plain-English summary
- `core/tools/definitions.py` - Added `ads_optimization_suggestions` tool after `ads_cross_platform_summary` in AD INTELLIGENCE section
- `core/tools/registry.py` - Added `ads_optimization_suggestions` to ad_intelligence, meta_ads, google_ads TOOL_CATEGORIES; added optimization keywords to CATEGORY_KEYWORDS

## Decisions Made

- Rule-based heuristics rather than external LLM: The tool runs server-side and returns structured data — the LLM calling it already provides the natural language layer. This keeps latency low and results deterministic.
- Rule g (declining performance) is lazy: only fetches a second 7-day API call when `days >= 14`. For the default 7-day case, no extra API calls happen.
- CTR normalization per platform: Meta returns CTR as a percentage value (1.5 = 1.5%), Google returns as decimal (0.015 = 1.5%). The check functions normalize to percentage for human-readable output while keeping raw values for platform comparisons.
- Matched ads_cross_platform_summary registration pattern: added to all three categories (ad_intelligence, meta_ads, google_ads) so it surfaces regardless of how Mike phrases the question.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 9 complete: ads_cross_platform_summary (Plan 01) and ads_optimization_suggestions (Plan 02) are both live
- Mike can ask "what should I do with my ad budget?" and get campaign-specific, metric-backed suggestions
- Both tools coexist in three TOOL_CATEGORIES for maximum discoverability

---
*Phase: 09-ad-intelligence*
*Completed: 2026-02-26*
