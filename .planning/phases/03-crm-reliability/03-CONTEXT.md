# Phase 3: CRM Reliability - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

CRM note and task creation never leaves orphaned records (rollback on second-step failure), and contact search returns results from the full contact database using server-side filtering instead of the 100-record Python fuzzy match. These are fixes to existing CRM integration scripts on Alfred Claw (101) that talk to Twenty CRM.

</domain>

<decisions>
## Implementation Decisions

### Contact search matching
- Use partial match (contains) — searching "Sarah" should find "Sarah Johnson", "Sarah Smith", etc.
- When multiple contacts match, Alfred asks the user to pick from a numbered list rather than auto-selecting
- If zero results, Alfred offers to create a new contact rather than just reporting "not found"

### Contact search fields
- Claude's discretion on whether to search first/last name fields separately or combined — based on what Twenty CRM's API supports best

### Claude's Discretion
- Rollback strategy: how many retries before cleanup, immediate vs delayed rollback
- Failure messaging: detail level, tone, and format of CRM error messages to Mike via Telegram
- Error visibility: whether rollback actions are mentioned in failure messages
- Search field implementation: first+last separate vs full name combined — whatever Twenty CRM API supports best

</decisions>

<specifics>
## Specific Ideas

- Contact search should feel natural — partial name matching like you'd expect from any modern search
- When multiple matches come back, present them clearly so Mike can pick the right one
- Zero-result case should be helpful, not a dead end — offer to create the contact

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-crm-reliability*
*Context gathered: 2026-02-20*
