# Project Research Summary

**Project:** Alfred Platform Stabilization & Ad Management
**Domain:** Dual-server AI assistant with conversational ad campaign management + CRM integration
**Researched:** 2026-02-20
**Confidence:** HIGH

## Executive Summary

Alfred Labs (105) is a mature FastAPI + React AI assistant with 353 registered tools, a working LLM routing layer, and nearly complete integrations for Meta Ads, Google Ads, Twenty CRM, GA4, and LightRAG. This is a stabilization and completion project — not greenfield. The scaffolding exists; the gaps are one missing package (`google-ads` SDK on Labs), one missing client function (Google Ads budget update), a handful of configuration fixes on Alfred Claw (101), and reliability bugs in CRM note/task linking and LightRAG circuit breaker management. The recommended approach is infrastructure-first: fix the blocked services (LightRAG, OpenAI embeddings, Telegram deduplication) before adding new capabilities.

The single most important dependency insight is that Google Ads budget control requires a `CampaignBudgetService.mutate_campaign_budgets()` call — a separate service from campaign mutations, unlike Meta Ads where budget is a field on the campaign. This architectural difference means Google Ads budget update is a new client function, not a configuration change. Everything else is either a broken config, a wrong path, or an API version number that needs updating. Meta Ads is fully implemented (18 tools, all write operations present). The Meta Graph API base URL must be updated from v21.0 to v22.0 (deprecated September 2025).

The top financial risk is the Meta access token expiry pattern combined with no read-after-write verification on budget mutations. A budget update tool that returns success without reading back the confirmed value cannot detect a double-conversion error (dollars passed as cents). For a system managing live ad spend on Rod Wave and One Music Festival campaigns, this is a P0 correctness requirement. All budget mutations must include a read-after-write confirmation step and explicit dollar/cents parameter documentation.

---

## Key Findings

### Recommended Stack

The existing stack requires exactly one new package: `google-ads==29.1.0` on Labs (105). All dependencies (grpcio 1.76, protobuf 6.31.1, google-auth 2.48.0) are already installed and within the SDK's compatibility range. Everything else is code fixes, config changes, or API version constants. Do not add `facebook-business` SDK — the current raw `requests` approach for Meta Ads is correct and already handles all required operations. Do not add GraphQL clients for Twenty CRM — the REST API is stable for all CRUD operations needed.

**Core technologies (new additions only):**
- `google-ads==29.1.0`: Google Ads API budget and ad group mutations — only missing package, clean install confirmed
- Meta Graph API `v22.0`: Replace hardcoded `v21.0` in `meta_ads/client.py` BASE_URL — one-line fix
- No new packages for CRM, LightRAG, GA4, or OpenAI — all are code/config fixes only

**Supporting libraries already installed and relevant:**
- `requests 2.32.3`: Meta Ads REST calls — correct choice, no SDK needed
- `httpx 0.28.1`: LightRAG async client — correct
- `openai 2.17.0` (pip) / `2.15.0` (requirements.txt): Minor drift — update requirements.txt to match

### Expected Features

**Must have (table stakes — broken or missing today):**
- Google Ads budget update (`gads_update_campaign_budget`) — only missing mutation capability
- LightRAG knowledge graph restored — both systems depend on it for contextual memory
- Telegram deduplication fixed on Claw — currently sends 2-3 duplicate responses per message
- CRM note/task creation working on Claw — tool arg errors break all note/task operations
- OpenAI embeddings resolved on Claw — 401 errors break RAG memory; switch to Ollama `nomic-embed-text` if project cannot be unarchived
- GA4 property IDs synced to Labs — analytics queries broken from Labs
- Infrastructure cleanup (log rotation, grep `-E` flag, stale gateway service, git gc)
- USER.md and HEARTBEAT.md trimmed to OpenClaw size limits (3,955 and 293 chars)
- End-to-end ad workflow validated: pause, enable, budget update confirmed against live campaigns

**Should have (competitive differentiators, v1.x):**
- Cross-platform ad summary — single LLM prompt combining Meta + Google read tools; no new API code needed
- Confirmation guardrail pattern — "show proposed change → await confirm → execute" for all ad mutations
- AI budget recommendations — wire existing `get_campaign_recommendations()` in Meta client + basic Google heuristic

**Defer to v2+:**
- Meta Ads campaign creation from chat — high complexity, many required inputs, large error surface
- Performance anomaly alerting with scheduled polling — requires persistent scheduler; add after core ops stable
- Escalation bridge for ad performance alerts — extend health monitor; after infrastructure monitor confirmed stable
- Google Ads ad group / keyword-level mutations — useful but edge case; campaign-level control first

### Architecture Approach

The architecture is a thin-wrapper pattern: each external service has a standalone client in `integrations/<service>/client.py` handling auth, HTTP calls, and response normalization. Tool definitions in `core/tools/definitions.py` are lazy-import wrappers that expose the client functions to the LLM via the `@tool` decorator. The router uses `TOOL_CATEGORIES` in `registry.py` for smart filtering — injecting only relevant tools (capped at 128) based on detected task type. This keeps LLM context clean across 353 registered tools. No new directories are needed; all new code slots into existing files.

**Major components:**
1. `core/api/main.py` — All HTTP endpoints, auth, WebSocket, chat streaming (~6900 lines)
2. `core/brain/router.py` — LLM routing (Ollama→Cloud→Claude Code), tool call execution loop, task detection
3. `core/tools/registry.py` + `definitions.py` — 353 tools via `@tool` decorator, 128-tool smart filtering
4. `integrations/meta_ads/client.py` — Complete: 18 read/write tools, all write operations present
5. `integrations/google_ads/client.py` — Partial: 10 read functions, budget mutation missing
6. `integrations/base_crm/client.py` — Correct structure, reliability bugs in two-step note/task linking
7. `integrations/lightrag/client.py` — Circuit breaker implemented, server down on 117, breaker open

### Critical Pitfalls

1. **Meta access token expiry silently kills all ad ops** — Use System User token (non-expiring) from Business Manager instead of short-lived Graph API Explorer token. Add expiry date tracking and pre-expiry Telegram alert.

2. **Meta budget double-conversion (dollars vs cents)** — Meta API requires cents (`dollars * 100`). Current code converts correctly but has no read-after-write verification. A future tool change could pass cents to a function expecting dollars, setting 100x the intended budget. Always read back the confirmed budget value after every mutation.

3. **Google Ads field mask silent no-op** — Every Google Ads mutation requires an explicit `update_mask` (FieldMask proto). Without it, the API returns HTTP 200 but changes nothing. Current `set_campaign_status()` handles this correctly; every new mutation function must replicate the pattern exactly. Verify with read-after-write.

4. **LightRAG circuit breaker stays open 1 hour after recovery** — After restoring LightRAG on Lonewolf (117), Labs (105) continues returning empty context for up to 1 hour. Add a `/admin/lightrag/reset` endpoint (5-line fix) before restoring the server. Recovery workflow: verify health → reset circuit breaker → confirm context enriches a test query.

5. **OpenClaw `doctor --fix` destroys all custom config** — Running this command resets Telegram channel config, model routing, sub-agent concurrency, compaction settings, and heartbeat model assignment to defaults. Never run it. Before any Claw SSH session: backup `openclaw.json` to a dated known-good file, apply targeted minimal changes, verify with `openclaw status`.

---

## Implications for Roadmap

Based on combined research, the phase order follows dependency and risk reduction: fix blocked infrastructure first, then fix Claw config in isolation, then harden CRM, then complete Google Ads write operations, then validate the full ad management workflow.

### Phase 1: Infrastructure Repairs
**Rationale:** These fixes are independent of each other and unblock everything downstream. LightRAG restoration is a prerequisite for context-aware conversations on both systems. Without these, subsequent phases are harder to validate.
**Delivers:** LightRAG operational, circuit breaker reset endpoint in place, GA4 analytics working from Labs, git maintenance complete
**Addresses:** LightRAG (table stakes), GA4 property sync (table stakes), Labs maintenance
**Avoids:** Circuit breaker stuck-open pitfall (add reset endpoint before restoring server), LightRAG silent empty context pitfall

### Phase 2: Alfred Claw (101) Config Fixes
**Rationale:** All Claw fixes are SSH-only (no Labs code changes). They are independent of each other and should be done as a batch to minimize SSH sessions. Fixes here (OpenAI 401, Telegram dedup, tool arg errors) have the highest operational impact for Mike's daily Telegram workflow.
**Delivers:** Telegram sends one response per message, CRM notes/tasks work via Telegram, OpenAI embeddings restored or replaced with Ollama fallback, OpenClaw context files within size limits, grep fix for escalation bridge, stale gateway cleaned up
**Addresses:** Telegram deduplication, CRM arg errors, OpenAI 401, USER.md/HEARTBEAT.md trim, log rotation, QUEUE.md grep fix (all table stakes)
**Avoids:** `doctor --fix` destruction (backup before every change), sessions path confusion (correct path: `~/.openclaw/agents/main/sessions/sessions.json`), TOOLS.md 20K truncation (measure before every addition)

### Phase 3: CRM Reliability
**Rationale:** CRM fixes are Labs-side code changes to `integrations/base_crm/client.py` only. No new tools needed. Fix correctness bugs (note/task linking rollback, search limit) before adding any new CRM capabilities. The 100-record search limit is a silent correctness bug that worsens as the CRM grows.
**Delivers:** CRM note/task creation with rollback on linking failure, server-side filter search (no silent truncation at 100 records), correct response unwrapping for all entity types
**Addresses:** CRM reliability (table stakes), CRM search completeness
**Avoids:** Two-step CRM operation without rollback pitfall, 100-record limit silent data loss pitfall

### Phase 4: Google Ads Budget Control
**Rationale:** This is the only missing write capability. Meta Ads is fully implemented. Google Ads read operations work. Adding `gads_update_campaign_budget`, `gads_pause_ad_group`, and `gads_enable_ad_group` completes Google Ads parity with Meta. Requires `google-ads` SDK installed on Labs (Phase 1 prerequisite) and OAuth scope verification (check before writing mutations).
**Delivers:** Full Google Ads write capability (pause, enable, budget update at campaign and ad group level), `gads_update_campaign_budget` tool registered in `TOOL_CATEGORIES`
**Addresses:** Google Ads budget update (table stakes P1), Google Ads parity
**Avoids:** Field mask silent no-op (include `update_mask` in every mutation), OAuth scope overwrite (verify combined scopes before any re-auth), sandbox developer token (verify production approval first), budget micros conversion (dollars × 1,000,000)

### Phase 5: Ad Workflow Validation & Hardening
**Rationale:** After all write capabilities are in place, validate the end-to-end conversational ad workflow against live campaigns. Add the confirmation guardrail pattern and read-after-write verification that financial mutations require. Wire cross-platform summary (no new API code, just prompt pattern). Upgrade Meta Graph API version from v21.0 to v22.0.
**Delivers:** Confirmed working ad management workflow, Meta API upgraded to v22.0, confirmation guardrail for all mutations, read-after-write on all budget changes, cross-platform ad summary prompt pattern, Meta token type verification and expiry tracking
**Addresses:** End-to-end validation (table stakes), cross-platform summary (v1.x), confirmation guardrail (v1.x)
**Avoids:** Meta token expiry silent failure, Meta budget double-conversion, no confirmation before destructive mutations

### Phase Ordering Rationale

- Phase 1 before all others because LightRAG, GA4, and maintenance are prerequisites that have no code dependencies
- Phase 2 (Claw) isolated because all changes are SSH to 101 — batching minimizes session risk and backup overhead
- Phase 3 (CRM) before Google Ads because CRM bugs affect both Labs and Claw and the fix is code-only with no external service dependencies
- Phase 4 (Google Ads writes) after infrastructure is stable so write operations can be tested against live accounts with reliable context
- Phase 5 (validation + hardening) last because it requires all prior phases to be complete before end-to-end flows can be confirmed

### Research Flags

Phases with well-documented patterns (skip `/gsd:research-phase`):
- **Phase 1:** LightRAG server restart and circuit breaker reset — implementation is clear from codebase audit
- **Phase 2:** All Claw config fixes — paths and procedures are documented in MEMORY.md and PITFALLS.md
- **Phase 3:** CRM rollback pattern — standard try/except/delete, no external research needed
- **Phase 5:** Meta API version bump and prompt engineering — straightforward

Phases that may benefit from deeper research during planning:
- **Phase 4:** Google Ads `CampaignBudgetService` — the `update_campaign_budgets()` mutation flow requires careful field mask construction. The existing `set_campaign_status()` is a reference but budgets are separate resources linked by resource name. Verify the exact mutation structure against Google Ads API v23 docs before writing the client function.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified via live environment inspection (`pip list`, `pip install --dry-run`, `docker ps`). One package gap confirmed. |
| Features | HIGH (codebase) / MEDIUM (API gaps) | Full codebase audit performed. API capability gaps verified against official docs. Some conversational UI patterns are LOW confidence (third-party workflow templates). |
| Architecture | HIGH | Direct codebase inspection of all integration clients, registry, router, and tool definitions. Architecture is mature and consistent. |
| Pitfalls | HIGH | Most pitfalls derived from actual code reading + official API documentation. Token expiry and field mask pitfalls verified against Meta/Google official docs. |

**Overall confidence:** HIGH

### Gaps to Address

- **Google Ads developer token approval level:** Research flags this as a prerequisite check. If the token is sandbox-only, production account mutations will silently fail. Verify at Google Ads API Center before Phase 4 work begins.
- **Meta System User token:** Research identified that the current token type is unconfirmed. Determine whether the token in `.env` is a System User token or a personal user token before Phase 5. This affects both security posture and operational reliability.
- **OpenAI project status on Claw:** Whether to unarchive the existing project or switch to Ollama `nomic-embed-text` depends on whether the OpenAI project can be unarchived without generating a new billing cycle. Decide during Phase 2 execution.
- **Twenty CRM `filter` query parameter exact syntax:** STACK.md notes the syntax as `filter=name[like]:%query%`. Verify the exact Twenty v1.14 filter syntax against live API before Phase 3 implementation to avoid silent no-filter behavior.

---

## Sources

### Primary (HIGH confidence)
- Live environment inspection of `/home/aialfred/alfred/` codebase (2026-02-20) — all integration clients, tool registry, router, API credentials
- PyPI: `google-ads==29.1.0` — verified current, dependency ranges checked via dry-run
- HKUDS/LightRAG GitHub README — server API endpoints and auth confirmed
- Meta Graph API deprecation notices — v21.0 deprecated September 2025, v22.0 required
- `docker ps` on Lonewolf (117) — LightRAG container confirmed running, port 9621
- MEMORY.md — system-specific constraints (OpenClaw version, sessions path, TOOLS.md limit)

### Secondary (MEDIUM confidence)
- Google Ads API v23 Release Guide (January 2026) — feature changes and CampaignBudgetService behavior
- Twenty CRM API documentation — REST API structure, filter parameter syntax
- Meta Ads API: Complete Guide for Advertisers and Developers (2025) — token expiry patterns
- Google Ads API: Updates Using Field Masks (official docs) — field mask requirement for mutations
- Google Ads API: Mutate Best Practices (official docs) — silent no-op on missing field mask

### Tertiary (LOW confidence)
- n8n conversational Meta Ads management workflow template — used only for feature pattern reference, not implementation decisions
- Google Ads Advisor conversational AI (official support page) — market comparison only
- FastAPI circuit breaker pattern blog post — pattern confirmed, implementation already in codebase

---
*Research completed: 2026-02-20*
*Ready for roadmap: yes*
