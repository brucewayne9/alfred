# Phase 5: Ad Workflow Validation & Hardening - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate all 18 Meta Ads tools against live campaigns (Rod Wave + One Music Festival), add read-after-write verification for all write operations, upgrade Meta Graph API from v21.0 to v22.0, and confirm the access token is a non-expiring System User token. No new tool creation — this hardens what exists.

</domain>

<decisions>
## Implementation Decisions

### Live testing strategy
- Validate against BOTH Rod Wave and One Music Festival campaigns for broader coverage
- Batch by type: validate all read/query tools first, then move to write/mutation tools
- Budget mutation testing: make a small change (e.g. +$1), verify it took via read-back, then revert to original value
- On tool failure: log the failure, continue to the next tool, report all failures at the end (don't halt)

### Read-after-write verification
- Applies to ALL write operations, not just budget mutations (includes pause/enable, status changes, etc.)
- Short delay (1-2 seconds) between write and read-back to allow Meta backend propagation
- On mismatch: retry the mutation once, then send Telegram alert to Mike if still mismatched
- Conversational response: normal success response by default, add a warning only if verification failed (no explicit "confirmed" on success)

### Claude's Discretion
- API migration approach (v21→v22 upgrade strategy — all-at-once vs gradual)
- Token verification method and where to document it in config
- Exact retry timing and delay values
- Validation report format

</decisions>

<specifics>
## Specific Ideas

- Change-and-revert pattern for budget testing: small increment, verify, restore original — minimizes risk to live campaigns
- Reads-first ordering gives confidence the API connection works before attempting any mutations

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-ad-workflow-validation-hardening*
*Context gathered: 2026-02-21*
