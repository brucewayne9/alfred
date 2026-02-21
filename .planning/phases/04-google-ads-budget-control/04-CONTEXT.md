# Phase 4: Google Ads Budget Control - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Mike can update Google Ads campaign budgets and pause/enable ad groups conversationally — mirroring the capability that already exists for Meta Ads. This phase also aligns the existing `gads_set_campaign_status` tool and applies consistent safety guardrails to both Google and Meta Ads write operations.

Requirements: GADS-01, GADS-02, GADS-03

</domain>

<decisions>
## Implementation Decisions

### Budget update behavior
- Confirmation threshold: confirm if new budget > $100/day OR more than 2x current budget (whichever triggers first)
- Below threshold: apply immediately and confirm
- Handle both shared and individual campaign budgets — if shared, warn Mike which campaigns are affected before applying
- Always show before/after comparison: "Campaign X budget: $30/day → $50/day — done."

### Ad group controls
- Support bulk operations (e.g., "Pause all ad groups in campaign X") but always list affected ad groups and confirm before applying
- Single ad group changes follow the same confirmation pattern as budgets (threshold-based)
- Always verify by reading back the ad group status from the API after mutation
- Warn if pausing the last active ad group in a campaign ("This is the last active ad group — pausing it effectively stops the campaign. Continue?")
- Align existing `gads_set_campaign_status` tool to match new patterns (verification read-back, before/after display, confirmation)

### Safety guardrails
- No hard budget cap — the confirmation threshold is sufficient protection
- Audit log: append-only JSONL file in `data/` recording every write operation (timestamp, old value, new value, requester, platform)
- Apply the same guardrail patterns (confirmation threshold, audit logging, verify read-back) to existing Meta Ads write tools too — platform consistency
- When ambiguous requests come in, ask to clarify by listing options ("Which campaign? You have: Campaign A ($30/day), Campaign B ($50/day)")

### Conversational style
- Proactively surface relevant context alongside confirmations (e.g., recent spend, trends, warnings)
- Error messages in plain English only — no API error codes shown to user
- Alfred should ask to clarify when requests are ambiguous, listing available options

### Claude's Discretion
- Response verbosity — match tone to context (terse for quick changes, more detail for significant ones)
- Exact audit log schema and file naming
- How to detect shared vs individual budgets from the API
- Loading/progress indicators for multi-step operations

</decisions>

<specifics>
## Specific Ideas

- Pattern should mirror what already exists for Meta Ads (same tool naming convention: `gads_update_campaign_budget`, `gads_pause_ad_group`, `gads_enable_ad_group`)
- Existing read tools (`gads_campaigns`, `gads_ad_groups`, etc.) already work — this phase adds write operations
- The `gads_set_campaign_status` tool already exists but needs alignment with the new verification/confirmation patterns

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-google-ads-budget-control*
*Context gathered: 2026-02-20*
