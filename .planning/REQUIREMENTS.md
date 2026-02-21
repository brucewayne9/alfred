# Requirements: Alfred Platform

**Defined:** 2026-02-20
**Core Value:** Alfred must be a reliable daily operations tool — every integration works correctly, no duplicate messages, no broken queues, and Mike can manage ad campaigns conversationally.

## v1.0 Requirements

Requirements for milestone v1.0 Ops Ready. Each maps to roadmap phases.

### Infrastructure

- [x] **INFRA-01**: LightRAG server restored and accessible from Labs (105)
- [x] **INFRA-02**: LightRAG circuit breaker reset endpoint added (clearable without process restart)
- [x] **INFRA-03**: GA4 property IDs synced to Labs settings
- [x] **INFRA-04**: Log rotation producing daily log files correctly on Claw (101)
- [x] **INFRA-05**: Stale openclaw-gateway.service cleaned up on Claw (101)
- [x] **INFRA-06**: Labs git repo gc — unreachable loose objects cleaned

### Claw Config

- [x] **CLAW-01**: Telegram duplicate message bug resolved
- [x] **CLAW-02**: USER.md trimmed to fit 3,955 char limit
- [x] **CLAW-03**: HEARTBEAT.md trimmed to fit 293 char limit
- [x] **CLAW-04**: QUEUE.md grep fixed with `-E` flag for alternation
- [x] **CLAW-05**: Tool argument errors fixed (python33→python3, CRM commands, email args, HEARTBEAT_OK)
- [x] **CLAW-06**: OpenAI project unarchived or switched to local embeddings

### CRM

- [x] **CRM-01**: Note/task linking uses rollback on second-step failure (no orphaned records)
- [x] **CRM-02**: Contact search uses server-side filter instead of 100-record Python fuzzy match

### Google Ads

- [x] **GADS-01**: User can update Google Ads campaign budget conversationally
- [x] **GADS-02**: User can pause/enable Google Ads ad groups conversationally
- [x] **GADS-03**: `google-ads` SDK installed on Labs (105)

### Meta Ads

- [x] **META-01**: Meta Ads API updated from v21.0 to v22.0
- [x] **META-02**: Meta access token verified as System User (non-expiring) type
- [x] **META-03**: Budget mutations include read-after-write verification
- [ ] **META-04**: All 18 existing Meta Ads tools validated against live campaigns

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Ad Intelligence

- **ADS-01**: AI-generated performance suggestions for ad campaigns
- **ADS-02**: Cross-platform ad performance summary (Meta + Google combined)
- **ADS-03**: Confirmation guardrail pattern for financial mutations

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New feature development beyond ads | Stabilization focus for v1.0 |
| Mobile app | Web-first |
| CRM migration off Twenty | Fix integration, don't replace |
| OpenClaw version upgrade | Fix within current 2026.2.14 |
| Google OAuth re-authorization flow | Existing scopes should work; revisit if needed |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| INFRA-05 | Phase 1 | Complete |
| INFRA-06 | Phase 1 | Complete |
| CLAW-01 | Phase 2 | Complete |
| CLAW-02 | Phase 2 | Complete |
| CLAW-03 | Phase 2 | Complete |
| CLAW-04 | Phase 2 | Complete |
| CLAW-05 | Phase 2 | Complete |
| CLAW-06 | Phase 2 | Complete |
| CRM-01 | Phase 3 | Complete |
| CRM-02 | Phase 3 | Complete |
| GADS-01 | Phase 4 | Complete |
| GADS-02 | Phase 4 | Complete |
| GADS-03 | Phase 4 | Complete |
| META-01 | Phase 5 | Complete |
| META-02 | Phase 5 | Complete |
| META-03 | Phase 5 | Complete |
| META-04 | Phase 5 | Pending |

**Coverage:**
- v1.0 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0

---
*Requirements defined: 2026-02-20*
*Last updated: 2026-02-20 — traceability complete after roadmap creation*
