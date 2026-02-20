# Architecture Research

**Domain:** AI assistant platform with conversational ad campaign management + CRM integration
**Researched:** 2026-02-20
**Confidence:** HIGH (based on direct codebase inspection)

---

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Alfred Labs (Server 105)                             │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────────────────────────────────────┐│
│  │  React Frontend  │   │              FastAPI Backend (port 8400)          ││
│  │  (Vite+TS+Zustand│──▶│   core/api/main.py (~6900 lines)                 ││
│  │  port 5173 dev)  │   │                                                   ││
│  └──────────────────┘   │  ┌────────────────────────────────────────────┐  ││
│                          │  │           LLM Router (router.py)           │  ││
│                          │  │  Ollama (local) → Cloud → Claude Code CLI  │  ││
│                          │  └────────────────┬───────────────────────────┘  ││
│                          │                   │ tool calls                    ││
│                          │  ┌────────────────▼───────────────────────────┐  ││
│                          │  │         Tool Registry (registry.py)         │  ││
│                          │  │  353 registered tools via @tool decorator   │  ││
│                          │  │  Smart filtering by CATEGORY_KEYWORDS       │  ││
│                          │  └──┬──────────┬──────────┬────────────────┬──┘  ││
│                          │     │          │          │                │      ││
│                          │  ┌──▼──┐  ┌───▼───┐ ┌────▼────┐ ┌────────▼──┐   ││
│                          │  │ CRM │  │Meta   │ │Google   │ │  Other    │   ││
│                          │  │     │  │Ads    │ │Ads      │ │  Tools    │   ││
│                          │  └──┬──┘  └───┬───┘ └────┬────┘ └───────────┘   ││
│                          └─────┼─────────┼──────────┼────────────────────── ┘│
└────────────────────────────────┼─────────┼──────────┼─────────────────────── ┘
                                 │         │          │
         ┌───────────────────────▼──┐      │    ┌─────▼────────────────────────┐
         │ Twenty CRM               │      │    │ Google Ads API               │
         │ crm.groundrushlabs.com   │      │    │ (google-ads-googleads SDK)   │
         │ (Lonewolf 117, REST API) │      │    │ google_token.json + env vars │
         └──────────────────────────┘      │    └──────────────────────────────┘
                                    ┌──────▼──────────────────────────────────┐
                                    │ Meta Graph API v21.0                    │
                                    │ (requests lib, OAuth token in .env)     │
                                    └─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         Alfred Claw (Server 101)                              │
│  OpenClaw 2026.2.14 + Telegram bot                                            │
│  SSH accessible from 105 via: ssh -p 2222 brucewayne9@75.43.156.101           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         Lonewolf (Server 117)                                 │
│  LightRAG at port 9621 (currently down, circuit breaker active on 105)        │
│  Twenty CRM at crm.groundrushlabs.com (Dokploy managed)                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|----------------|----------|
| `core/api/main.py` | All HTTP endpoints, auth, WebSocket, chat streaming | FastAPI, ~6900 lines |
| `core/brain/router.py` | LLM routing (Ollama→Cloud→Claude Code), tool call execution loop | Python |
| `core/tools/registry.py` | `@tool` decorator, `TOOL_CATEGORIES`, `CATEGORY_KEYWORDS`, smart filtering | Python dict |
| `core/tools/definitions.py` | All tool implementations (~5573 lines), all `@tool` registrations | Python, imports lazy |
| `integrations/meta_ads/client.py` | Meta Graph API v21.0 wrapper, read + write operations | requests lib |
| `integrations/google_ads/client.py` | Google Ads API wrapper via google-ads SDK, GAQL queries | google-ads SDK |
| `integrations/base_crm/client.py` | Twenty CRM REST API wrapper, fuzzy search, workflow management | requests lib |
| `integrations/google_analytics/client.py` | GA4 Data API, hardcoded property map, service account auth | google-analytics SDK |
| `integrations/lightrag/client.py` | LightRAG knowledge graph, circuit breaker (1hr cooldown) | httpx async |
| `config/settings.py` | Pydantic settings from `config/.env`, all API credentials | Pydantic |

---

## Recommended Project Structure

The existing structure is correct. New work fits within it:

```
integrations/
├── meta_ads/
│   └── client.py          # EXISTS — read + write ops complete
├── google_ads/
│   └── client.py          # EXISTS — missing budget update, ad group control
├── base_crm/
│   └── client.py          # EXISTS — note/task linking workflow has reliability bugs
├── google_analytics/
│   └── client.py          # EXISTS — GA4 property IDs hardcoded, needs sync check

core/tools/
├── registry.py            # EXISTS — TOOL_CATEGORIES needs google_ads budget tools
└── definitions.py         # EXISTS — needs gads_update_budget, gads_pause_ad_group
```

No new top-level directories needed. All new code slots into existing integration clients and `definitions.py`.

---

## Architectural Patterns

### Pattern 1: Integration Client + Tool Wrapper

**What:** Each external service has a standalone client module in `integrations/<service>/client.py`. The client handles auth, HTTP calls, and response normalization. Tool definitions in `core/tools/definitions.py` are thin wrappers that import from the client at call time (lazy import pattern).

**When to use:** Always. This is the established pattern for all 353 tools.

**Trade-offs:** Lazy imports prevent circular imports and keep startup fast. The client handles all API-specific logic. The `@tool` decorator registers name, description, and parameters for the LLM.

**Example (existing pattern to follow for new tools):**
```python
# In integrations/google_ads/client.py
def update_campaign_budget(customer_id: str, campaign_id: str, budget_micros: int) -> dict:
    """Update campaign daily budget. Budget in micros (1 dollar = 1,000,000 micros)."""
    try:
        client = _get_client()
        # ... Google Ads mutation logic
        return {"success": True, "campaign_id": campaign_id, "new_budget": budget_micros / 1_000_000}
    except GoogleAdsException as e:
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        return {"error": str(e)}

# In core/tools/definitions.py
@tool(
    name="gads_update_campaign_budget",
    description="Update a Google Ads campaign's daily budget. Amount in dollars.",
    parameters={
        "campaign_id": {"type": "string", "description": "Campaign ID to update", "required": True},
        "daily_budget": {"type": "number", "description": "New daily budget in dollars (e.g., 50.00)", "required": True},
        "customer_id": "string - Customer ID (optional, uses default)",
    },
)
def gads_update_campaign_budget(campaign_id: str, daily_budget: float, customer_id: str = None) -> dict:
    from integrations.google_ads.client import update_campaign_budget
    return update_campaign_budget(campaign_id=campaign_id, budget_micros=int(daily_budget * 1_000_000), customer_id=customer_id)
```

### Pattern 2: Tool Category Registration

**What:** New tool names must be added to `TOOL_CATEGORIES` in `core/tools/registry.py` and the relevant keywords added to `CATEGORY_KEYWORDS`. The router uses this for smart tool filtering — it only injects tools matching the category detected from the user's query, staying under the 128-tool LLM context limit.

**When to use:** Every new tool requires this. Omitting it means the tool never gets injected into context.

**Trade-offs:** The 128-tool limit is a hard cap enforced in the router. New ad budget tools fit within the existing `google_ads` and `meta_ads` categories.

**Example:**
```python
# In core/tools/registry.py, TOOL_CATEGORIES dict:
"google_ads": [
    "gads_list_accounts", "gads_account_info", "gads_campaigns",
    "gads_campaign_performance", "gads_ad_groups", "gads_keywords",
    "gads_spend", "gads_set_campaign_status",
    # ADD NEW TOOLS HERE:
    "gads_update_campaign_budget", "gads_pause_ad_group", "gads_enable_ad_group",
],
```

### Pattern 3: CRM Response Normalization

**What:** The Twenty CRM REST API returns nested `{"data": {"people": [...]}}` format. The client normalizes every response through `_format_*` functions before returning to the tool. Error handling uses try/except with `resp.raise_for_status()`.

**When to use:** Any new CRM endpoint must follow the same normalization pattern and handle the nested response format.

**Trade-offs:** The nested format is a known quirk of Twenty CRM v1.14.0. Normalization in the client (not the tool definition) keeps tool definitions clean.

**Current reliability bug:** Note/task creation uses a two-step flow: (1) create the note/task, (2) link it via a separate `noteTargets`/`taskTargets` POST. If step 2 fails, the note/task is created but not linked. No rollback exists. This is the primary CRM reliability issue.

---

## Data Flow

### Conversational Ad Campaign Control

```
User: "Pause the Rod Wave campaign on Meta"
    ↓
React Frontend → POST /chat (JWT auth)
    ↓
FastAPI main.py → router.ask()
    ↓
router.py: detect_task_type() → "meta_ads" category detected
    ↓
inject tools from TOOL_CATEGORIES["meta_ads"] (18 tools)
    ↓
LLM (Ollama/Cloud) generates tool call:
    {"tool": "meta_ads_campaigns", "parameters": {}}
    ↓
execute_tool() → meta_ads_campaigns() → integrations/meta_ads/client.list_campaigns()
    ↓
LLM receives campaign list, identifies Rod Wave campaign ID
    ↓
LLM generates: {"tool": "meta_ads_pause_campaign", "parameters": {"campaign_id": "..."}}
    ↓
execute_tool() → meta_ads_pause_campaign() → client.pause_campaign()
    ↓
Meta Graph API POST /{campaign_id}?status=PAUSED
    ↓
Success response streamed back to React frontend
```

### Google Ads Budget Update (Current Gap)

The existing `gads_set_campaign_status` tool handles enable/pause. Budget update is **missing**. The Google Ads API requires a `CampaignBudgetOperation` mutation (not a campaign mutation) because budgets are separate resources in Google Ads. This is architecturally different from Meta Ads where budget is a field on the campaign itself.

```
Google Ads budget flow:
  1. GET campaign to find campaign_budget resource name
  2. CampaignBudgetOperation.update with new amount_micros
  3. mutate_campaign_budgets() call on CampaignBudgetService
```

### CRM Note/Task Linking (Current Reliability Issue)

```
Current (fragile):
  1. POST /rest/notes  → creates note (returns note ID)
  2. POST /rest/noteTargets  → links to person/company/opportunity
     ↑ If this fails, note is orphaned (no rollback)

Recommended fix:
  Wrap steps 1+2 in a single function with error handling:
    - On step 2 failure: attempt to delete the note (cleanup)
    - Return error with context: "Note created but linking failed"
    - Log the orphaned note ID for manual recovery
```

### LightRAG Circuit Breaker Behavior

```
router.py: get_knowledge_context(query)
    ↓
lightrag/client.py: _circuit_is_open()?
    YES → skip LightRAG, return "" (fast path, no 30s timeout)
    NO  → attempt httpx request to http://75.43.156.117:9621
         SUCCESS → _record_success(), return context
         FAILURE → _record_failure(), if >=2 failures → OPEN circuit for 1 hour
```

The circuit breaker is already implemented and working. LightRAG restoration requires fixing the server on Lonewolf (117), not changing Alfred Labs code.

---

## New vs Modified Components

### NEW (does not exist, must be written)

| Component | Location | What It Is |
|-----------|----------|------------|
| `update_campaign_budget()` | `integrations/google_ads/client.py` | Google Ads `CampaignBudgetService.mutate_campaign_budgets()` call |
| `pause_ad_group()` / `enable_ad_group()` | `integrations/google_ads/client.py` | AdGroup status mutation via `AdGroupService` |
| `gads_update_campaign_budget` tool | `core/tools/definitions.py` | Tool wrapper for budget update |
| `gads_pause_ad_group` / `gads_enable_ad_group` tools | `core/tools/definitions.py` | Tool wrappers for ad group control |

### MODIFIED (exists, needs fixes)

| Component | Location | What Changes |
|-----------|----------|-------------|
| `TOOL_CATEGORIES["google_ads"]` | `core/tools/registry.py` | Add new budget/ad group tool names |
| `create_note_for_*` / `create_task_for_*` | `integrations/base_crm/client.py` | Add rollback on linking failure |
| `GA_PROPERTIES` dict | `integrations/google_analytics/client.py` | Verify property IDs match Claw config |

### EXISTING (works, no changes needed)

| Component | Status |
|-----------|--------|
| Meta Ads client (full read + write) | Complete — 18 tools, all write operations present |
| CRM CRUD operations | Structurally correct — reliability bugs in linking only |
| Google Ads read operations (7 tools) | Working |
| LightRAG circuit breaker | Working — just needs server restoration on 117 |

---

## Integration Points

### External Services

| Service | Integration Pattern | Auth Method | Known Issues |
|---------|---------------------|-------------|-------------|
| Meta Graph API v21.0 | REST via `requests`, sync | Long-lived OAuth token in `.env` | Token expiry (not detected at startup) |
| Google Ads API | `google-ads` SDK, GAQL, sync | OAuth `google_token.json` + developer token | Budget update missing (CampaignBudgetService not used) |
| Twenty CRM REST | REST via `requests`, sync | API key Bearer token | Note/task linking is two-step, no rollback |
| Google Analytics Data API | `google-analytics-data` SDK, sync | Service account JSON file | Property IDs hardcoded in client.py — manual sync needed |
| LightRAG | httpx async, circuit breaker | Basic auth (user/pass in `.env`) | Server down on 117, circuit breaker open (1hr cooldown) |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `main.py` ↔ `router.py` | Direct Python import, `ask()` async generator | Streaming via SSE |
| `router.py` ↔ `registry.py` | Direct import, `execute_tool()`, `get_tools()` | Tool cap: 128 max injected |
| `definitions.py` ↔ `integrations/*` | Lazy imports inside function body | Prevents circular import at module load |
| `main.py` ↔ `integrations/lightrag` | Async via `get_knowledge_context()` | Circuit breaker prevents blocking |
| Labs (105) ↔ Claw (101) | SSH via `call_alfred_claw` tool | Port 2222, `brucewayne9` user |
| Labs (105) ↔ Twenty CRM (117) | HTTPS REST to `crm.groundrushlabs.com` | API key in `.env` |

---

## Build Order (Dependency-Aware)

This ordering respects dependencies and delivers operational value early.

### Phase 1: Infrastructure Fixes (No Dependencies)

These are independent fixes that unblock everything else. Do these first.

1. **LightRAG restoration** — Fix server on 117. No code changes on Labs needed (circuit breaker already coded).
2. **GA4 property ID sync** — Verify `GA_PROPERTIES` dict in `google_analytics/client.py` matches Claw's config. Config-only change.
3. **Log rotation on Claw** — SSH fix on 101. No Labs code changes.
4. **Stale gateway cleanup** — SSH fix on 101. No Labs code changes.
5. **Labs git gc** — `git gc --prune=now` on 105. Maintenance.

### Phase 2: Claw Config Fixes (Depends on SSH access, not code)

All done via SSH from 105 to 101.

1. **USER.md trim** — Trim to 3,955 char limit.
2. **HEARTBEAT.md trim** — Trim to 293 char limit.
3. **QUEUE.md grep fix** — Add `-E` flag for alternation.
4. **Tool arg fixes** — `python33→python3`, CRM command args, email args, `HEARTBEAT_OK`.
5. **OpenAI 401** — Unarchive project or switch API key.
6. **Telegram duplicate fix** — Investigate and fix duplicate message bug.

### Phase 3: CRM Reliability (Depends on: running system)

Modify `integrations/base_crm/client.py` only.

1. Add rollback logic to `create_note_for_*` and `create_task_for_*` functions.
2. Add cleanup on linking failure: attempt delete of orphaned record, return descriptive error.
3. No new tools needed — fix is internal to client functions that existing tools already call.

### Phase 4: Google Ads Budget Control (Depends on: Phase 1 complete)

Extends `integrations/google_ads/client.py` and `core/tools/definitions.py`.

1. Add `update_campaign_budget()` to `google_ads/client.py` using `CampaignBudgetService`.
2. Add `pause_ad_group()` and `enable_ad_group()` to `google_ads/client.py`.
3. Register `gads_update_campaign_budget`, `gads_pause_ad_group`, `gads_enable_ad_group` in `definitions.py`.
4. Add new tool names to `TOOL_CATEGORIES["google_ads"]` in `registry.py`.

**Why Google Ads before Meta Ads:** Meta Ads write operations are already complete (18 tools). Google Ads is missing budget control.

### Phase 5: Meta Ads Enhancements (If needed)

Meta Ads is already functionally complete with 18 tools including all write operations. The main gap is the access token expiry — Meta tokens expire every 60 days. If campaign management starts failing, token refresh is the likely cause. No code changes needed unless token refresh automation is required.

---

## Anti-Patterns

### Anti-Pattern 1: Adding Tools Without Category Registration

**What people do:** Add a new `@tool` decorated function to `definitions.py` but forget to add the tool name to `TOOL_CATEGORIES` in `registry.py`.

**Why it's wrong:** The tool gets registered in `_tools` dict but never injected into LLM context because the smart filter uses `TOOL_CATEGORIES` exclusively for category-based injection. The tool is effectively dead.

**Do this instead:** Every new `@tool` registration requires a corresponding entry in `TOOL_CATEGORIES` and verification that relevant trigger words exist in `CATEGORY_KEYWORDS`.

### Anti-Pattern 2: Putting Business Logic in Tool Definitions

**What people do:** Write complex API logic directly inside the `@tool` decorated function in `definitions.py`.

**Why it's wrong:** Makes `definitions.py` unmaintainable (it's already 5573 lines). Breaks the separation between tool schema (what the LLM sees) and implementation (what the API client does). Makes the client untestable in isolation.

**Do this instead:** All API logic lives in `integrations/<service>/client.py`. The tool definition is a thin wrapper with a lazy import. This is the established pattern for all 353 existing tools.

### Anti-Pattern 3: Synchronous Blocking in Async Context

**What people do:** Call integration clients synchronously from `router.py` which runs in async context.

**Why it's wrong:** The existing integration clients (`meta_ads`, `google_ads`, `base_crm`) are synchronous `requests`-based. Calling them directly in the async router blocks the event loop.

**Do this instead:** The existing pattern wraps sync tool execution in `asyncio.get_event_loop().run_in_executor()` or uses `execute_tool()` which handles this. Follow the existing pattern — do not change clients to async unless specifically needed, as the existing sync clients work correctly.

### Anti-Pattern 4: Two-Step CRM Operations Without Rollback

**What people do:** Create a CRM record (note/task) and then link it to an entity in two separate API calls with no error recovery.

**Why it's wrong:** The current CRM client does exactly this and it causes orphaned records when the second call fails. The fix requires explicit rollback: if the link call fails, attempt to delete the created record and return a descriptive error.

**Do this instead:**
```python
def create_note_for_person(title: str, person_id: str, body: str = "") -> dict:
    note = create_note(title, body)
    try:
        _post("/rest/noteTargets", {"noteId": note["id"], "personId": person_id})
        note["linked_to"] = {"person_id": person_id}
        return note
    except Exception as e:
        # Rollback: try to delete the orphaned note
        try:
            _delete(f"/rest/notes/{note['id']}")
        except Exception:
            pass  # Best effort cleanup
        return {"error": f"Note created but linking failed: {e}", "orphaned_note_id": note["id"]}
```

---

## Scaling Considerations

Not applicable at current scale (single user, single server). The architecture is correct for this use case.

| Concern | Current State | If Multi-User Later |
|---------|---------------|---------------------|
| API rate limits | Meta: 200 calls/hour default. Google Ads: per-developer token quota | Add response caching layer in clients |
| Tool count (128 cap) | 353 tools registered, smart filtering keeps injected count under cap | Tune CATEGORY_KEYWORDS if new categories overlap |
| CRM fuzzy search performance | Client-side fuzzy search fetches up to 100 records per search | Add server-side search param when Twenty CRM supports it |

---

## Sources

- Direct inspection of `/home/aialfred/alfred/` codebase (2026-02-20)
- `integrations/meta_ads/client.py` — confirmed 18 write/read operations, all complete
- `integrations/google_ads/client.py` — confirmed 10 functions, budget mutation missing
- `integrations/base_crm/client.py` — confirmed two-step note/task linking pattern
- `core/tools/registry.py` — confirmed 128-tool cap, `TOOL_CATEGORIES` structure
- `config/.env` — confirmed Meta, Google Ads, CRM, LightRAG credentials present
- Google Ads API docs (MEDIUM confidence): CampaignBudgetService is a separate service from CampaignService; budgets are separate resources linked by resource name

---

*Architecture research for: Alfred Platform — Ad Management + CRM Stabilization*
*Researched: 2026-02-20*
