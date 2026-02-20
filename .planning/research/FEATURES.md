# Feature Research

**Domain:** Conversational ad campaign management + CRM reliability + infrastructure stabilization for dual-server AI assistant
**Researched:** 2026-02-20
**Confidence:** HIGH (codebase audit) / MEDIUM (API capability gaps) / LOW (AI suggestion patterns only from web sources)

## Context: Existing vs. New

This is a SUBSEQUENT MILESTONE. The tool scaffolding already exists. Most features are partially built — the gaps are completion of mutation capabilities (Google Ads budget update), reliability of existing integrations (CRM command errors, OpenAI 401, Telegram duplicates), and infrastructure restoration (LightRAG, log rotation). "Table Stakes" here means "users expect Alfred to reliably do what it advertises."

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that Alfred claims to support. Missing or broken = platform is not trustworthy as a daily ops tool.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Meta Ads campaign on/off toggle | Core campaign control — already exposed as tools | LOW | `meta_ads_pause_campaign` / `meta_ads_enable_campaign` exist; need end-to-end reliability test |
| Meta Ads budget update (campaign + ad set) | Primary budget lever for Rod Wave / One Music campaigns | LOW | `meta_ads_update_campaign_budget` + `meta_ads_update_ad_set_budget` exist in client and definitions |
| Meta Ads performance read (spend, CTR, CPA, ROAS) | Mike needs to query performance conversationally | LOW | `meta_ads_campaign_insights`, `meta_ads_performance` exist; test with live data |
| Google Ads campaign on/off toggle | Parity with Meta — `gads_set_campaign_status` exists | LOW | Exists; verify ENABLED/PAUSED enum values work against API v19+ |
| Google Ads budget update | Critical gap — currently read-only for budgets | MEDIUM | `set_campaign_budget()` not in `integrations/google_ads/client.py`; must add + expose as `gads_update_budget` tool |
| Google Ads performance read (spend, CPC, conversions) | `gads_campaign_performance`, `gads_spend` exist | LOW | Test against live Rod Wave / One Music account |
| CRM contact search (people + companies) | Fuzzy search already in `base_crm/client.py` | LOW | Existing — verify search result quality on real contacts |
| CRM note creation (person, company, deal) | `crm_add_note_to_*` tools exist | LOW | Broken on Claw via tool arg errors — fix argument passing, not the logic |
| CRM task creation and update | `crm_create_task`, `crm_update_task` exist | LOW | Same arg error pattern as notes on Claw side |
| LightRAG knowledge graph available | Both systems depend on it for contextual memory | MEDIUM | Currently down with circuit breakers open — restore server on 117, reset circuit breakers |
| No duplicate Telegram messages | Claw sends duplicates currently | LOW | Bug in OpenClaw message handling — fix deduplication/idempotency |
| GA4 analytics accessible on Labs | Already fixed on Claw; Labs needs property ID sync | LOW | Copy property IDs from Claw config to Labs `.env` / `settings.py` |
| OpenAI embeddings working on Claw | 401 errors break RAG and memory features | MEDIUM | Either unarchive OpenAI project or switch to alternative embeddings (Ollama `nomic-embed-text`) |

### Differentiators (Competitive Advantage)

Features that make Alfred's ad management meaningfully better than opening Ads Manager manually.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Cross-platform ad summary ("how are all my campaigns doing?") | Single query returns Meta + Google performance consolidated | MEDIUM | Requires calling both `meta_ads_campaign_insights` and `gads_campaign_performance` and merging — LLM synthesizes narrative; no new tools needed, new prompt pattern |
| AI-driven budget recommendations | Alfred proactively surfaces "Campaign X is underperforming, consider pausing" based on CPA vs. target | HIGH | `meta_ads_campaign_recommendations` already exists in client but not wired to suggestions workflow; Google side needs similar heuristic layer |
| Conversational campaign creation for standard templates | Mike can say "create a Rod Wave concert awareness campaign on Meta, $500/day" | HIGH | Meta Ads API supports campaign creation — `Campaign.create()` exists in facebook-business SDK but not exposed as Alfred tool; defer unless explicitly needed |
| Performance anomaly alerting | Alfred notices when spend spikes or CTR drops and notifies proactively | HIGH | Would require scheduled polling + WebSocket push — foundation (WebSocket, health monitor) exists but ad polling not wired; defer to v1.x |
| Escalation bridge for ad alerts | Alfred Claw pings Mike on Telegram when campaign performance crosses threshold | HIGH | Escalation bridge exists for infrastructure — extending to ad performance requires new monitor script; defer |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Autonomous budget reallocation (no confirmation) | "AI just fixes it" sounds ideal | Real money on the line — unreviewed mutations can drain budgets; Meta/Google both allow large budget changes via API with no guardrails | Always require explicit confirmation before mutation; show proposed change, await "yes/confirm" |
| Campaign creation from scratch via chat | "Create a whole campaign" is tempting | Requires creative assets, targeting definitions, pixel setup — too many inputs to gather reliably in conversation; error surface is large | Scope to status/budget/pause mutations only for v1; creation deferred to v2 |
| Real-time performance dashboard | "Live numbers" sounds better | Polling every 30s hits Meta/Google API rate limits; insights APIs have 15-min minimum freshness; polling adds complexity with minimal practical gain | On-demand queries ("check my campaigns") are sufficient; cache results 15 min |
| Bulk keyword-level control via chat | "Pause all keywords with QS < 4" | Keyword IDs require lookup + iteration; high mutation risk on large accounts; Google Ads API quotas for bulk mutations are tight | Surface keyword performance as read-only (`gads_keywords` exists); manual action for bulk keyword changes |
| Full QUEUE.md / HEARTBEAT.md rewrite | Seems like cleanup | File format is consumed by OpenClaw internals — structural changes can break parsing | Trim content to fit limits (3,955 chars USER.md, 293 chars HEARTBEAT.md), do not change format |

---

## Feature Dependencies

```
[Google Ads budget update tool]
    └──requires──> [gads_update_budget in client.py + definitions.py]
                       └──requires──> [Google Ads API v19 campaign budget mutate service]

[Cross-platform ad summary]
    └──requires──> [Meta Ads read tools working]
    └──requires──> [Google Ads read tools working]

[AI budget recommendations]
    └──requires──> [Meta campaign recommendations client function (exists, unwired)]
    └──requires──> [Google Ads performance read working]
    └──enhances──> [Cross-platform ad summary]

[CRM note/task creation (Claw)]
    └──requires──> [Tool argument errors fixed in openclaw.json / TOOLS.md]
    └──does NOT require──> [New tool logic — base_crm/client.py is correct]

[LightRAG knowledge graph]
    └──requires──> [LightRAG server running on 117:9621]
    └──requires──> [Circuit breakers reset in integrations/lightrag/client.py]
    └──enhances──> [All conversational context on both systems]

[OpenAI embeddings on Claw]
    └──alternative──> [Ollama nomic-embed-text as local fallback]
    └──blocks──> [RAG memory features on Claw]

[Telegram deduplication]
    └──requires──> [Fix in OpenClaw message dispatch on 101]
    └──independent of──> [Labs FastAPI changes]

[Infrastructure fixes (log rotation, grep -E, git gc, stale service)]
    └──independent of each other]
    └──prerequisite for──> [Reliable daily operations]
```

### Dependency Notes

- **Google Ads budget update requires new client function:** `set_campaign_budget()` must be added to `integrations/google_ads/client.py` using the `CampaignBudgetServiceClient.mutate_campaign_budgets()` API before the tool definition can be exposed.
- **CRM reliability does NOT require new logic:** The `base_crm/client.py` fuzzy search and CRUD functions are correct. Breakage is in how Claw invokes tool arguments (wrong parameter names, wrong Python binary `python33` vs `python3`). Fix the invocation layer, not the CRM layer.
- **LightRAG must be restored before AI summary features work well:** Without knowledge graph context, Alfred answers from short-term conversation only. Circuit breaker state in `integrations/lightrag/client.py` must be reset after server is restored.
- **OpenAI 401 on Claw blocks memory embeddings independently of LightRAG:** LightRAG uses its own embedding; OpenAI embeddings are for ChromaDB memory. Separate fix paths.

---

## MVP Definition

### Launch With (v1 — "Ops Ready")

Minimum set to make Alfred a reliable daily operations tool for active ad campaigns.

- [ ] **Google Ads budget update** — Add `gads_update_budget` tool (only missing mutation capability)
- [ ] **LightRAG restored** — Reset server on 117, clear circuit breakers; both systems reconnect
- [ ] **Telegram deduplication fixed** — Stop duplicate messages on Claw
- [ ] **CRM tool arg errors fixed on Claw** — Notes, tasks, workflow commands work via Telegram
- [ ] **OpenAI 401 resolved on Claw** — Either unarchive project or switch embeddings provider
- [ ] **GA4 property IDs synced to Labs** — Analytics queries work from Labs chat
- [ ] **Infrastructure cleanup** — Log rotation, grep `-E` flag, stale gateway service, git gc
- [ ] **USER.md + HEARTBEAT.md trimmed** — Fits within OpenClaw size limits
- [ ] **End-to-end ad workflow validated** — Pause, enable, budget update confirmed against live campaigns

### Add After Validation (v1.x)

Features to add once core reliability is established.

- [ ] **Cross-platform ad summary** — Single LLM prompt pattern combining Meta + Google read tools; no new API code needed
- [ ] **AI budget recommendations** — Wire `get_campaign_recommendations()` in Meta client to a new `meta_ads_ai_suggestions` tool; add basic heuristic layer for Google (CPA vs. threshold)
- [ ] **Confirmation guardrail pattern** — Standardize "show proposed change → await confirm → execute" for all ad mutations; add to LLM system prompt instructions

### Future Consideration (v2+)

Features to defer until v1 is stable and Mike validates the workflow.

- [ ] **Meta Ads campaign creation** — High complexity, many required inputs; wait until ad mutation workflow is proven reliable
- [ ] **Performance anomaly alerting (scheduled polling)** — Requires persistent scheduler on Labs; add after core ops are stable
- [ ] **Escalation bridge for ad alerts** — Extend health monitor pattern to ad performance; after infrastructure monitor is confirmed stable
- [ ] **Google Ads ad group / keyword-level mutations** — Useful but edge case; focus on campaign-level control first

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Google Ads budget update tool | HIGH | LOW (one new client function + tool definition) | P1 |
| LightRAG restoration | HIGH | LOW (server restart + circuit breaker reset) | P1 |
| CRM arg errors fixed on Claw | HIGH | LOW (config/invocation fix, no logic change) | P1 |
| Telegram deduplication | HIGH | LOW (OpenClaw config/code fix on 101) | P1 |
| OpenAI 401 resolved | HIGH | LOW-MEDIUM (unarchive project or swap provider) | P1 |
| GA4 property sync to Labs | MEDIUM | LOW (copy IDs to env/settings) | P1 |
| Infrastructure cleanup (logs, grep, gc) | MEDIUM | LOW (individual shell/config fixes) | P1 |
| USER.md / HEARTBEAT.md trim | MEDIUM | LOW (content editing only) | P1 |
| Cross-platform ad summary | HIGH | LOW (prompt engineering only) | P2 |
| AI budget recommendations | MEDIUM | MEDIUM (wire existing + heuristic layer) | P2 |
| Confirmation guardrail pattern | HIGH | LOW (system prompt + LLM behavior) | P2 |
| Meta campaign creation | MEDIUM | HIGH (many inputs, error surface) | P3 |
| Performance anomaly alerting | MEDIUM | HIGH (persistent scheduler, new monitor) | P3 |
| Escalation bridge for ad alerts | LOW | HIGH (extend monitor, new Claw integration) | P3 |

---

## Competitor Feature Analysis

Included for context on what conversational ad management looks like in the market (tools like Madgicx, n8n + GPT, Google Ads Advisor).

| Feature | Market Pattern | Alfred's Approach |
|---------|----------------|-------------------|
| Performance read | Chat → fetch → LLM narrates results | Same pattern; tools already wired |
| Campaign pause/enable | Confirmation step → API mutation | Tools exist; add explicit confirmation guardrail in system prompt |
| Budget update | Show current → propose new → confirm → mutate | Tools exist (Meta) / one tool gap (Google); add confirm pattern |
| Recommendations | AI analyzes performance, surfaces suggestions | Meta client has `get_campaign_recommendations()`; expose as tool |
| Cross-platform view | Single consolidated report | LLM orchestrates multi-tool call; no new API code needed |

---

## Sources

- Codebase audit: `/home/aialfred/alfred/core/tools/registry.py`, `/home/aialfred/alfred/core/tools/definitions.py`, `/home/aialfred/alfred/integrations/google_ads/client.py`, `/home/aialfred/alfred/integrations/meta_ads/client.py`, `/home/aialfred/alfred/integrations/base_crm/client.py`, `/home/aialfred/alfred/integrations/lightrag/client.py`
- [Google Ads API v23 Release Guide (January 2026)](https://almcorp.com/blog/google-ads-api-v23-complete-guide-2026/) — MEDIUM confidence (WebSearch verified against official Google Ads API docs)
- [Meta Ads API — facebook-python-business-sdk](https://github.com/facebook/facebook-python-business-sdk) — HIGH confidence (official SDK repo)
- [Conversational Meta Ads management pattern (n8n workflow)](https://n8n.io/workflows/7957-conversational-meta-ads-reporting-and-management-with-gpt-5/) — LOW confidence (third-party workflow template)
- [Twenty CRM API documentation](https://twenty.com/developers/section/api-and-webhooks/api) — MEDIUM confidence (official docs, accessed via WebSearch)
- [Google Ads Advisor conversational AI](https://support.google.com/google-ads/answer/14145186?hl=en) — MEDIUM confidence (official Google support page)
- [FastAPI circuit breaker patterns](https://blog.stackademic.com/system-design-1-implementing-the-circuit-breaker-pattern-in-fastapi-e96e8864f342) — MEDIUM confidence (verified with multiple sources)

---

*Feature research for: Alfred Platform Stabilization & Ad Management*
*Researched: 2026-02-20*
