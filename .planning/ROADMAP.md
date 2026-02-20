# Roadmap: Alfred Platform Stabilization & Ad Management

## Overview

This roadmap repairs a mature dual-server AI assistant platform so it becomes a reliable daily operations tool. Work proceeds in dependency order: fix blocked infrastructure first, isolate all Alfred Claw (101) SSH changes in one phase, harden CRM reliability, add the one missing Google Ads write capability, then validate and harden the full ad management workflow end-to-end.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Infrastructure Repairs** - Restore LightRAG, sync GA4, add circuit breaker reset, clean up Labs git repo (Labs-only) (completed 2026-02-20)
- [ ] **Phase 2: Alfred Claw Config Fixes** - Batch all SSH-only changes to Server 101 (Telegram dedup, tool args, size limits, log rotation, stale gateway cleanup)
- [ ] **Phase 3: CRM Reliability** - Fix note/task linking rollback and contact search on Labs-side CRM client
- [ ] **Phase 4: Google Ads Budget Control** - Install SDK and add budget + ad group mutation tools to complete Google Ads write parity
- [ ] **Phase 5: Ad Workflow Validation & Hardening** - Upgrade Meta API version, verify token, add read-after-write, validate end-to-end conversational ad workflow

## Phase Details

### Phase 1: Infrastructure Repairs
**Goal**: All shared services and Labs maintenance issues are resolved so downstream phases can validate against a stable environment
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-06
**Success Criteria** (what must be TRUE):
  1. A test query to Alfred on Labs returns context from LightRAG without empty-context fallback
  2. Calling the circuit breaker reset endpoint clears the open breaker without restarting any process
  3. Alfred on Labs can answer a GA4 analytics question using the correct property IDs
  4. Labs git repo shows no unreachable loose object warnings after gc
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Restore LightRAG network access and add circuit breaker reset endpoints
- [x] 01-02-PLAN.md — Verify GA4 and run git gc

### Phase 2: Alfred Claw Config Fixes
**Goal**: Alfred Claw on Server 101 sends one response per message, all tool calls succeed, context files are within size limits, the escalation bridge grep works correctly, and infrastructure maintenance (log rotation, stale services) is resolved
**Depends on**: Phase 1
**Requirements**: CLAW-01, CLAW-02, CLAW-03, CLAW-04, CLAW-05, CLAW-06, INFRA-04, INFRA-05
**Success Criteria** (what must be TRUE):
  1. Sending a message via Telegram results in exactly one response from Alfred Claw (no duplicates)
  2. Creating a CRM note or task via Telegram chat completes without argument errors
  3. OpenClaw memory embeddings complete successfully (either via restored OpenAI project or Ollama nomic-embed-text fallback)
  4. USER.md and HEARTBEAT.md are within their respective character limits and load without truncation warnings
  5. The QUEUE.md escalation bridge grep command correctly matches multiple patterns using the -E flag
  6. Claw log rotation produces a new daily file at midnight (or on-demand test) and the stale gateway service is gone from systemctl
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Trim USER.md and HEARTBEAT.md to fit OpenClaw char limits
- [ ] 02-02-PLAN.md — Fix Telegram dedup, QUEUE.md grep, and tool argument errors
- [ ] 02-03-PLAN.md — Switch embeddings to Ollama nomic-embed-text, log cleanup, gateway restart + verify

### Phase 3: CRM Reliability
**Goal**: CRM note and task creation never leaves orphaned records, and contact search returns results from the full contact database rather than the first 100 records
**Depends on**: Phase 2
**Requirements**: CRM-01, CRM-02
**Success Criteria** (what must be TRUE):
  1. When the second step of note/task linking fails, the first-step record is deleted and Alfred reports a clean failure (no orphaned notes or tasks in Twenty CRM)
  2. A contact search by name returns the correct contact even when it would have been outside the first 100 records in the old pagination approach
**Plans**: TBD

### Phase 4: Google Ads Budget Control
**Goal**: Mike can update Google Ads campaign budgets and pause/enable ad groups conversationally with the same capability that already exists for Meta Ads
**Depends on**: Phase 1
**Requirements**: GADS-01, GADS-02, GADS-03
**Success Criteria** (what must be TRUE):
  1. Asking Alfred to update a Google Ads campaign budget results in the confirmed new budget visible in the Google Ads console
  2. Asking Alfred to pause or enable a Google Ads ad group changes its status as confirmed by a read-back of the ad group state
  3. The google-ads SDK is installed on Labs (105) and the new tools appear in the registered tool list
**Plans**: TBD

### Phase 5: Ad Workflow Validation & Hardening
**Goal**: The complete conversational ad management workflow is validated against live campaigns, all budget mutations include read-after-write verification, and the Meta Ads integration is upgraded and confirmed reliable
**Depends on**: Phase 4
**Requirements**: META-01, META-02, META-03, META-04
**Success Criteria** (what must be TRUE):
  1. All 18 Meta Ads tools execute successfully against live Rod Wave or One Music Festival campaigns without API errors
  2. A Meta Ads budget mutation returns the confirmed budget value read back from the API (not just the request parameters)
  3. The Meta access token is confirmed as a non-expiring System User token and this is documented in config
  4. Meta Graph API version is v22.0 across all client calls and no deprecation warnings appear in responses
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure Repairs | 2/2 | Complete   | 2026-02-20 |
| 2. Alfred Claw Config Fixes | 1/3 | In progress | - |
| 3. CRM Reliability | 0/TBD | Not started | - |
| 4. Google Ads Budget Control | 0/TBD | Not started | - |
| 5. Ad Workflow Validation & Hardening | 0/TBD | Not started | - |
