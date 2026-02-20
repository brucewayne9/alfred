# Phase 1: Infrastructure Repairs - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Restore shared services on Labs (105) and clean up maintenance debt so downstream phases build on a stable environment. This phase is Labs-only — INFRA-04 (log rotation) and INFRA-05 (stale gateway) have been moved to Phase 2 since they require SSH to Server 101.

Remaining requirements: INFRA-01 (LightRAG), INFRA-02 (circuit breaker reset), INFRA-03 (GA4 sync), INFRA-06 (git gc).

</domain>

<decisions>
## Implementation Decisions

### LightRAG restoration
- Claude decides whether to restore existing data or start fresh (assess what's recoverable)
- LightRAG runs on Labs (105) alongside the FastAPI backend
- LightRAG is best-effort enrichment, NOT a hard dependency — Alfred answers without RAG if LightRAG is unavailable
- When LightRAG has no relevant context for a query, Alfred answers using LLM knowledge alone (no error shown to user)
- LightRAG should be tackled first before other Phase 1 work

### Circuit breaker reset
- Exposed as an HTTP endpoint (POST) on the Labs API
- Admin-only auth required (bruce/admin role)
- Alfred (the AI) can self-reset stuck circuit breakers — it has permission to call the reset endpoint
- When all breakers are already healthy, the endpoint returns 200 with a "all breakers healthy" message (no-op success, safe to call anytime)

### Work ordering
- LightRAG restoration first, then circuit breaker, GA4 sync, and git gc can be parallelized

### Claude's Discretion
- LightRAG data recovery vs fresh start (based on what's actually recoverable)
- Circuit breaker endpoint path and response format
- GA4 property ID storage approach
- Git gc strategy
- Exact error handling patterns

</decisions>

<specifics>
## Specific Ideas

- Phase 1 is Labs (105) only — no SSH to 101. All Server 101 infrastructure work bundled into Phase 2.
- ROADMAP.md to be updated to reflect INFRA-04/INFRA-05 moving from Phase 1 to Phase 2.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-infrastructure-repairs*
*Context gathered: 2026-02-20*
