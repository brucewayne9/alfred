# Pitfalls Research

**Domain:** Adding Meta Ads API, Google Ads API, CRM integration, and infrastructure fixes to an existing AI assistant platform
**Researched:** 2026-02-20
**Confidence:** HIGH (codebase read + verified with official docs and web sources)

---

## Critical Pitfalls

### Pitfall 1: Meta Access Token Expiry Silently Kills All Ad Operations

**What goes wrong:**
The Meta access token stored in `config/.env` as `META_ACCESS_TOKEN` is a short-lived user token (expires in 1-2 hours) or an up-to-60-day long-lived token. When it expires, every Meta Ads API call silently fails with an OAuth error (`{"error": {"code": 190, "type": "OAuthException"}}`). Because `meta_ads/client.py` catches all exceptions and returns `{"error": str(e)}`, Alfred reports tool failures without making it clear that a credential rotation is needed. Active Rod Wave / One Music Festival campaigns go dark in the dashboard with no alert.

**Why it happens:**
The token is loaded at module import time (`ACCESS_TOKEN = getattr(settings, 'meta_access_token', '')`). There is no expiry check, no refresh mechanism, and no differentiation between "token expired" and "API is down". The current code stores the token as a static string in `.env` — this works fine for a short integration test but fails for a 24/7 production assistant.

**How to avoid:**
Use a **System User token** from Meta Business Manager instead of a personal user token. System User tokens can be set to non-expiring in Business Manager settings. If a personal user token must be used, generate a long-lived token (60-day) and add an expiry check that alerts Mike via Telegram before it expires. Never use the short-lived token from the Graph API Explorer for production.

To generate a long-lived token from a short-lived one:
```
GET https://graph.facebook.com/oauth/access_token
  ?grant_type=fb_exchange_token
  &client_id={app-id}
  &client_secret={app-secret}
  &fb_exchange_token={short-lived-token}
```

**Warning signs:**
- All Meta Ads tools return `{"error": "...OAuthException..."}` simultaneously
- `is_connected()` returns False
- Error message contains "code 190" or "token expired"

**Phase to address:**
Infrastructure / Claw Config Fixes phase — validate token type and expiry before wiring up write operations. Store token expiry date in `.env` or a separate state file and add a monitor check.

---

### Pitfall 2: Meta Ads Budget API Uses Cents, But Tools Confirm in Dollars — Double Conversion Risk

**What goes wrong:**
The Meta Ads API requires `daily_budget` and `lifetime_budget` in **cents** (integer, USD × 100). The existing `update_ad_set_budget()` and `update_campaign_budget()` functions in `meta_ads/client.py` correctly do `int(daily_budget * 100)` when writing to the API. However, the LLM receives confirmation messages like `"Budget updated: $100.00"` which it computed from the dollar-value parameter it passed in. If the LLM (or a future tool definition change) ever passes a value that is already in cents — for example if it reads back the raw API field value and passes it forward — the budget gets multiplied by 100 again, setting a $10,000 budget when $100 was intended.

**Why it happens:**
The API response after a write does not echo back the new value for confirmation. The tool constructs the confirmation from the input parameter. There is no round-trip read-after-write to verify what was actually set.

**How to avoid:**
After every budget mutation, immediately call `get_campaign_details()` or `get_ad_set_details()` to read back the actual value and include it in the confirmation. The tool should return: `"Budget confirmed via read-back: $X.XX (as reported by Meta API)"`. This prevents double-conversion errors and provides an audit trail.

Add explicit parameter documentation to the tool description: `"daily_budget: float in US dollars (e.g., 100.0 for $100/day — API conversion to cents is handled internally)"`.

**Warning signs:**
- Budget jumps 100x (e.g., $10 becomes $1,000)
- Mike asks "why is my campaign spending 100x more today"
- Budget confirmation message doesn't match what Meta Ads Manager shows

**Phase to address:**
Meta Ads write operations phase — add read-after-write confirmation to all budget mutation tools.

---

### Pitfall 3: Google Ads OAuth Token Is Shared With Google Workspace — Scope Creep Breaks Both

**What goes wrong:**
`google_ads/client.py` loads the refresh token from `config/google_token.json` — the **same file** used by Gmail, Google Calendar, Google Sheets, Google Drive, and other Google integrations. The Google Ads API requires the `https://www.googleapis.com/auth/adwords` scope. If Alfred's Google OAuth flow was originally authorized without this scope, the Google Ads API calls will fail with `PERMISSION_DENIED` or `INSUFFICIENT_AUTHENTICATION_SCOPES`. Attempting to re-authorize to add the scope will **replace** the existing token, potentially breaking the Gmail / Calendar integrations if the re-authorization omits any of their scopes.

**Why it happens:**
OAuth scopes must be declared at authorization time. Adding a new scope (Google Ads) after the token was created requires re-authorization through the full consent screen. The existing code has no scope validation — it just loads whatever token is in `google_token.json` and calls the API.

**How to avoid:**
Before the Google Ads integration phase, check what scopes the current `google_token.json` was authorized with:
```python
from google.oauth2.credentials import Credentials
creds = Credentials.from_authorized_user_file('config/google_token.json')
print(creds.scopes)
```
If `https://www.googleapis.com/auth/adwords` is missing, perform a **combined re-authorization** that includes all required scopes (Gmail, Calendar, Sheets, Drive, Ads) in a single flow. Never re-authorize for just one scope.

**Warning signs:**
- `GoogleAdsException` with `INSUFFICIENT_AUTHENTICATION_SCOPES`
- Gmail or Calendar tools suddenly start failing after re-authorization
- `google_token.json` file was recently regenerated

**Phase to address:**
Google Ads write operations phase — verify scope before adding write operations. Document current authorized scopes in `.env` or a comment in the token file.

---

### Pitfall 4: Google Ads API Field Masks Are Required — Silent No-Op Without Them

**What goes wrong:**
When updating a Google Ads resource (campaign status, budget, ad group), the API requires an explicit `update_mask` (FieldMask proto). Without it, the mutation call succeeds with HTTP 200 but **no fields are actually changed**. The response looks identical to a successful update. This means `set_campaign_status()` can return `{"success": True, "new_status": "PAUSED"}` while the campaign continues running and spending money.

Looking at the current implementation in `google_ads/client.py`, `set_campaign_status()` does include:
```python
client.copy_from(
    campaign_operation.update_mask,
    client.get_type("FieldMask")(paths=["status"])
)
```
This is correct. However, any new mutation operations added (e.g., for budget updates, ad group pausing) must replicate this pattern exactly. It is easy to omit when copying from a similar function.

**Why it happens:**
This is a Google Ads API-specific requirement not present in most REST APIs. Developers coming from REST or Meta's API are accustomed to PATCH requests that update only the sent fields. Google Ads API uses proto-based mutations where you must explicitly state which fields changed.

**How to avoid:**
For every new write operation, immediately verify with a read-after-write. Use the `google.api_core.protobuf_helpers.field_mask()` helper to auto-generate field masks from two proto objects (before/after). Add a comment in every mutation function: `# REQUIRED: update_mask must list all modified fields or API silently ignores them`.

**Warning signs:**
- Mutation returns success but the resource state hasn't changed in Google Ads UI
- No error in logs, but campaign keeps spending after "pause"
- Budget update seems to succeed but spend rate is unchanged

**Phase to address:**
Google Ads write operations phase — add read-after-write to every mutation, add field mask validation unit test.

---

### Pitfall 5: Twenty CRM Search Fetches 100 Records Client-Side — Fails on Large Contact Lists

**What goes wrong:**
`search_people()` in `base_crm/client.py` fetches `limit=100` records and then applies fuzzy matching in Python. When the CRM grows beyond 100 contacts, any contact after the 100-record cutoff is invisible to searches. The `search_companies()` and `search_opportunities()` functions have the same pattern. Mike cannot find recently added contacts if they were created after the first 100 alphabetically (or by creation order).

**Why it happens:**
Twenty CRM's REST API supports server-side filtering via the `filter` query parameter. The original implementation was written assuming a small dataset and used client-side fuzzy matching for flexibility. As the CRM grows, this becomes a correctness bug, not just a performance issue.

**How to avoid:**
Use Twenty CRM's REST API `filter` parameter for name/email searches:
```
GET /rest/people?filter=name.firstName[like]:"%John%"
```
Keep the fuzzy fallback for the results but push the coarse filter to the server. Alternatively, increase the fetch limit to 500 and add explicit handling for the `hasNextPage` / cursor pagination that Twenty CRM supports.

**Warning signs:**
- Mike reports "I can't find [contact name] in CRM" but the contact exists
- CRM contact count grows above 100 and searches become unreliable
- `search_people("John")` returns results but a specific "John" is missing

**Phase to address:**
CRM reliability phase — fix search before adding any new CRM features. This is a correctness bug that will silently return wrong results.

---

### Pitfall 6: OpenClaw `doctor --fix` Wipes All Custom Config

**What goes wrong:**
Running `openclaw doctor --fix` on Alfred Claw (101) resets channel config, model routing, plugin state, and custom settings to defaults. All customizations — Telegram channel config (`@alfredblogbot`), kimi-k2.5:cloud model routing, sub-agent concurrency limits, compaction settings, heartbeat model assignment — are permanently destroyed. Recovery requires manually re-applying every setting from the known-good backup.

**Why it happens:**
`doctor --fix` is designed for first-time setup correction. It is not safe to run on a configured production instance. The command is easy to run accidentally when debugging, especially when `doctor` (without `--fix`) correctly identifies issues that look tempting to auto-fix.

**How to avoid:**
Never run `doctor --fix`. The constraint is already documented in MEMORY.md. Before any SSH session that modifies OpenClaw config, verify the backup exists:
```bash
ls -la ~/.openclaw/openclaw.json.known-good-*
```
Create a dated backup before every change:
```bash
cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.known-good-$(date +%Y%m%d)
```
Use `openclaw doctor` (without `--fix`) to diagnose, then manually apply only the specific fix needed.

**Warning signs:**
- Telegram bot stops responding
- Model routing changes to a default provider
- Sub-agent concurrency limit changes
- Heartbeat starts using an incompatible model

**Phase to address:**
Claw Config Fixes phase — every script that modifies Claw config must: (1) verify backup exists, (2) create new dated backup, (3) apply minimal targeted change, (4) verify config with `openclaw status`.

---

### Pitfall 7: TOOLS.md 20,000-Char Limit — New Tools Silently Truncate

**What goes wrong:**
OpenClaw reads `TOOLS.md` to know what scripts Claw can invoke. The file has a hard 20,000-character limit (currently ~19,766 chars per MEMORY.md). If new tool entries (for Meta Ads, Google Ads, or CRM fixes) are added to TOOLS.md without checking the limit, the file is silently truncated at the limit. Tools defined after the cutoff are invisible to Claw, causing confusing behavior where Claw says a tool doesn't exist even though the script is present.

**Why it happens:**
TOOLS.md is consumed as a context document with a fixed window. There is no error when the limit is exceeded — it just cuts off. The 20K limit is tight enough that adding even a single well-documented tool section can push over.

**How to avoid:**
Before adding any new entries to TOOLS.md: `wc -c ~/.openclaw/workspace/TOOLS.md`. Calculate how many characters the new section requires. If the new total would exceed 19,500 chars, compress or remove an existing section first. Prefer terse documentation in TOOLS.md (function signatures, not prose).

**Warning signs:**
- Claw cannot find a tool that you know is in TOOLS.md
- TOOLS.md `wc -c` output exceeds 19,800
- Last entry in TOOLS.md appears cut off mid-sentence

**Phase to address:**
Claw Config Fixes phase — measure TOOLS.md size before and after every addition. Add a check to the SSH fix scripts.

---

### Pitfall 8: LightRAG Circuit Breaker Stays Open for 1 Hour After Recovery

**What goes wrong:**
The circuit breaker in `integrations/lightrag/client.py` opens after 2 consecutive failures and stays open for exactly 1 hour (hardcoded `CIRCUIT_BREAKER_COOLDOWN = timedelta(hours=1)`). When LightRAG on Lonewolf (117) is restored (e.g., Docker container restarted), Alfred Labs (105) continues returning empty knowledge context for up to 1 hour. Mike notices Alfred has "forgotten" things it should know. There is no admin endpoint to manually reset the circuit breaker.

**Why it happens:**
The circuit breaker state lives in module-level globals (`_circuit_breaker` dict). There is no way to reset it without restarting the FastAPI process or waiting out the 1-hour cooldown. The implementation was intentionally conservative (1 hour) because LightRAG was already down when the circuit breaker was added.

**How to avoid:**
Add a `/admin/lightrag/reset` endpoint that clears the circuit breaker state. This is a 5-line fix. Also add the circuit breaker status to the `/integrations/status` response so it's visible in the UI. When restoring LightRAG, the workflow should be: (1) verify `health_check()` returns healthy, (2) call reset endpoint, (3) confirm knowledge context works in a test query.

**Warning signs:**
- LightRAG Docker container is running on 117 but Alfred still returns empty context
- `/integrations/status` shows LightRAG connected or disconnected (current status response doesn't show circuit breaker state separately)
- LightRAG was recently restored after an outage

**Phase to address:**
Infrastructure Repairs phase — add reset endpoint and circuit breaker status visibility before the LightRAG restoration task. Without this, restoring LightRAG is incomplete.

---

### Pitfall 9: Google Ads Developer Token Is Sandbox-Only Until Approved for Production

**What goes wrong:**
Google Ads API developer tokens have two levels: test (sandbox/test account access only) and standard (production access). A test-level developer token can only query Google Ads test accounts, not real production accounts. If `GOOGLE_ADS_DEVELOPER_TOKEN` in `.env` is a test token, all calls to the real customer accounts fail with `DEVELOPER_TOKEN_NOT_APPROVED` or return empty results from test data. This is especially confusing because the client initializes successfully and some API calls succeed (account info) while others silently return nothing.

**Why it happens:**
Google's token approval process requires submitting an application explaining the use case. Many developers skip this or assume their token is production-approved. The error messages are not always clear about which tier the token is.

**How to avoid:**
Verify token status at `https://developers.google.com/google-ads/api/docs/access-levels`. Check the developer token's approval status in Google Ads → Tools → API Center. If not approved, the approval process takes a few business days. In the meantime, test with a Google Ads test account using the test token.

**Warning signs:**
- `get_campaigns()` returns empty list despite known campaigns existing
- Error: `DEVELOPER_TOKEN_NOT_APPROVED` or `DEVELOPER_TOKEN_PROHIBITED`
- Google Ads Manager shows campaigns but `get_campaigns()` shows none
- No errors but zero results from any query

**Phase to address:**
Google Ads verification phase (before write operations) — verify token approval level as a prerequisite check.

---

### Pitfall 10: Telegram Duplicate Messages From Race Condition in OpenClaw Session

**What goes wrong:**
Alfred Claw sends duplicate Telegram messages to Mike. The active requirement is to fix this. The root cause is typically a session state file that has multiple concurrent message handlers registered, causing the same incoming message to trigger multiple response cycles. On OpenClaw 2026.2.14, session state is at `~/.openclaw/agents/main/sessions/sessions.json` (not `~/.openclaw/sessions.json`). Clearing the wrong file (the non-existent path) has no effect and the bug persists.

**Why it happens:**
The health monitor's auto-fix procedure clears "both sessions files" but must target the correct path (`sessions.json` in the agents subdirectory). If the auto-fix script uses the wrong path, it silently does nothing and the duplicate behavior continues. Additionally, if the gateway is restarted while a session is mid-conversation, the new session can re-process messages the old session already handled.

**How to avoid:**
The correct fix sequence:
1. `rm ~/.openclaw/agents/main/sessions/sessions.json`
2. `openclaw gateway restart`
3. Wait for confirmation the gateway is accepting connections before sending a test message

The health monitor fix script (`alfred_claw_monitor.py`) must use the correct path. Verify the path is right before running the fix: `ls ~/.openclaw/agents/main/sessions/`.

**Warning signs:**
- Mike receives 2-3 identical responses for every message
- `openclaw status` shows multiple active sessions
- Message timestamps show near-simultaneous duplicate sends

**Phase to address:**
Claw Config Fixes phase — fix the sessions path in the monitor script first, then test the fix by inducing a restart.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Load access tokens at module import time | Simple, no lazy init overhead | Token rotation requires full process restart; stale tokens cause cascading failures | Never for production secrets — use lazy init with expiry check |
| Client-side CRM search with 100-record fetch | Works for small datasets, simpler code | Silent data loss when contact count exceeds fetch limit | Only acceptable as MVP with documented limit; fix before CRM grows |
| Hardcode LightRAG circuit breaker to 1-hour cooldown | Protects against hammering a down server | Cannot recover without restart or 1-hour wait after server restored | Acceptable temporarily; add reset endpoint before restoring LightRAG |
| `except Exception: return {"error": str(e)}` everywhere | Prevents crashes; uniform error format | LLM cannot distinguish credential errors from network errors from bugs | Acceptable for MVP; add error categorization before production ads management |
| No read-after-write on budget mutations | Fewer API calls, faster confirmation | Cannot verify mutations succeeded; budget double-conversion risk undetectable | Never for financial mutations — always read back |
| SSH fix scripts that modify OpenClaw config inline | Fast iteration during debugging | Config changes without backup; no audit trail | Never — always backup first |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Meta Ads API | Using short-lived user token from Graph API Explorer for production | Use System User token (non-expiring) from Business Manager |
| Meta Ads API | Passing budget in dollars to API (e.g., 100 instead of 10000 cents) | Always convert: `int(dollars * 100)` before sending; read back after write |
| Meta Ads API | Reading `effective_status` to decide whether to pause (it's read-only) | Write to `status` field; read `effective_status` to understand actual delivery state |
| Meta Ads API | Fetching insights without `level` param returns account-level rollup only | Always specify `"level": "campaign"` or `"level": "adset"` for breakdown |
| Google Ads API | Forgetting `update_mask` on mutations | Every mutate operation must include `update_mask`; silent no-op otherwise |
| Google Ads API | Test developer token against production customer accounts | Verify token approval level at Google Ads API Center before any production work |
| Google Ads API | Re-authorizing OAuth for only Ads scope, breaking other Google integrations | Always re-authorize with all scopes combined in one flow |
| Google Ads API | Passing budget in dollars (API uses micros — dollars × 1,000,000) | `int(dollars * 1_000_000)` for budget micros; current code uses `_format_micros()` correctly for reads |
| Twenty CRM | Assuming search returns all matching contacts | CRM search fetches first 100 records; contacts beyond 100 are invisible |
| Twenty CRM | Passing plain string to name fields | `name` field is a structured object: `{"firstName": "X", "lastName": "Y"}` — already handled in client but easy to break |
| Twenty CRM | Assuming `data.data.people` structure is always present | API returns `{"data":{"people":[...]}}` nested format; missing wrapper causes `AttributeError` silently returning empty list |
| LightRAG | Checking if LightRAG is working after restore without resetting circuit breaker | After restore: check health, reset circuit breaker, then test query |
| LightRAG | Uploading large documents without checking indexing status | Document upload returns immediately; indexing is async. Check `get_document_status()` before querying new documents |
| OpenClaw | Running `openclaw doctor --fix` to fix configuration issues | Use `openclaw doctor` only to diagnose; apply fixes manually |
| OpenClaw | Modifying `openclaw.json` without backup | Always backup to `openclaw.json.known-good-YYYYMMDD` first |
| OpenAI embedding (Claw) | Assuming 401 error is a billing issue | Check if OpenAI project is archived — archived projects return 401 for all API calls |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Meta Ads `get_audience_insights()` makes 3 serial API calls | Dashboard demographic tab takes 10+ seconds | Cache results or run 3 calls concurrently with `asyncio.gather()` | Every time demographics are requested |
| CRM search fetches 100 records and fuzzy-matches in Python | Search takes 2-5 seconds, misses contacts beyond position 100 | Server-side filter + cursor pagination | When CRM exceeds 100 contacts |
| LightRAG query_context_fast has 5s timeout blocking chat hot path | Every chat message waits up to 5s for LightRAG when circuit is just recovering | Circuit breaker reset endpoint; exponential backoff instead of 1hr fixed cooldown | Any LightRAG instability period |
| `/integrations/status` endpoint checks 15+ services serially | Status page loads in 15-20 seconds if any service times out | `asyncio.gather()` all checks concurrently with per-service timeout | Whenever any single integration is slow |
| Tool registry sends all 353 tools in every LLM context | Slow LLM responses; higher cloud API costs; context window pressure | Smart category filtering already in `TOOL_CATEGORIES` but may not be applied to ads tools | As tool count grows beyond current level |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Meta access token stored in `.env` without expiry tracking | Token expires silently; no audit of when last rotated | Store token + expiry date; add pre-expiry Telegram alert |
| Google OAuth token file at `config/google_token.json` world-readable | Refresh token exposes all authorized Google services | `chmod 600 config/google_token.json`; verify permissions after any git operation |
| Budget mutations logged with dollar amounts but no user attribution | Cannot audit who triggered a budget change (it's always "Alfred") | Add tool audit log with timestamp, tool name, args, and requesting conversation ID |
| No confirmation required for destructive ad operations (pause campaign, reduce budget) | LLM misinterpretation could pause active campaigns | Add explicit confirmation step in tool description: "Confirm with user before executing" |
| SSH scripts on 105 targeting 101 store no audit trail | No record of what was changed on Claw remotely | Log all SSH commands with timestamp to a local audit file on 105 |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Campaign pause confirmation only says "PAUSED" without showing current spend | Mike doesn't know if the pause took effect during high-spend period | Show current day spend + confirmation of new status from read-back |
| Insights tools return formatted strings (`"$1,234.56"`) not numbers | LLM cannot do arithmetic on formatted strings for budget recommendations | Return both: `{"spend": "$1,234.56", "spend_raw": 1234.56}` |
| LightRAG circuit open silently returns empty context | Alfred gives generic answers without explaining "I don't have that in my knowledge base right now" | When circuit is open, Alfred should say "My knowledge graph is temporarily unavailable" |
| CRM search returns partial results without warning | Mike thinks he has all contacts when the first 100 were returned | Add `{"warning": "Showing first 100 contacts, search may be incomplete"}` when limit reached |
| Tool errors return raw exception strings | Mike sees Python tracebacks in chat | Wrap all integration errors: `"Meta Ads is not responding — check API token"` not `"HTTPError: 401"` |

---

## "Looks Done But Isn't" Checklist

- [ ] **Meta Ads token:** Token is configured and `is_connected()` returns True — verify it is a System User token or long-lived token, not the short-lived one from Graph API Explorer
- [ ] **Meta Ads budget write:** `update_campaign_budget()` returns success — verify with a read-back of the actual budget from the API
- [ ] **Google Ads scope:** `get_campaigns()` returns campaigns — verify `adwords` scope is in the token's authorized scopes, not just that the call didn't error
- [ ] **Google Ads field mask:** `set_campaign_status()` returns success — verify campaign status actually changed in Google Ads UI within 60 seconds
- [ ] **CRM search:** `search_people("John")` returns results — verify it returns ALL Johns, not just those in the first 100 records
- [ ] **LightRAG restored:** Health check passes on 117 — verify knowledge context actually enriches a chat response (circuit breaker may still be open on 105)
- [ ] **OpenClaw config fixed:** `openclaw status` shows expected config — verify Telegram bot responds, model routing is correct, heartbeat is using llama3.1:8b
- [ ] **Telegram duplicate fix:** Sessions file cleared — verify by sending a test message and confirming exactly one response arrives
- [ ] **TOOLS.md updated:** New tool section added — verify `wc -c` is under 19,500 chars and Claw can find the new tool by name
- [ ] **OpenAI 401 fixed:** API key updated or project unarchived — verify with a direct embedding test call, not just settings check

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Meta token expired, campaigns not visible | LOW | Regenerate long-lived token in Business Manager → update `.env` → restart Labs backend |
| Google OAuth scope missing for Ads | MEDIUM | Re-authorize at `/auth/google` with all scopes combined → verify all Google integrations (Gmail, Calendar) still work |
| OpenClaw config wiped by `doctor --fix` | HIGH | Restore from `openclaw.json.known-good-YYYYMMDD` → `openclaw gateway restart` → verify each integration manually |
| LightRAG circuit breaker stuck open | LOW | Restart Labs FastAPI process (or add reset endpoint first) → verify health → test query |
| CRM search returning incomplete results | LOW | Increase fetch limit to 500 in `search_people()` → add pagination handling → retest with >100 contacts |
| Budget double-conversion (budget set 100x too high) | HIGH | Immediately pause affected campaigns in Meta/Google Ads UI → adjust budget → audit tool call logs to understand how it happened |
| Duplicate Telegram messages | LOW | `rm ~/.openclaw/agents/main/sessions/sessions.json` → `openclaw gateway restart` → test with single message |
| TOOLS.md truncated over 20K limit | MEDIUM | Compress existing entries → remove verbose descriptions → reduce to function signatures → verify Claw finds all tools |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Meta token expiry | Infrastructure / Config Fixes (before ads) | `is_connected()` returns True with token type confirmed as System User or long-lived |
| Meta budget cents vs dollars double-conversion | Meta Ads write operations | Read-after-write shows correct budget amount |
| Google Ads OAuth scope missing | Google Ads pre-work verification | `creds.scopes` includes `adwords`; Gmail still works after any re-auth |
| Google Ads field mask silent no-op | Google Ads write operations | Campaign status change verified in Google Ads UI within 60s |
| Twenty CRM search 100-record limit | CRM Reliability phase | Search returns contact 101+ when CRM has >100 contacts |
| OpenClaw `doctor --fix` destruction | Claw Config Fixes phase | Backup exists before any SSH session; script checks for backup |
| TOOLS.md 20K truncation | Claw Config Fixes phase (after each tool addition) | `wc -c TOOLS.md` < 19,500 after every addition |
| LightRAG circuit breaker stuck | Infrastructure / LightRAG Restoration phase | Admin reset endpoint exists; health check + context test both pass |
| Google Ads developer token sandbox-only | Google Ads verification phase (first task) | `get_campaigns()` returns real campaigns matching Google Ads UI |
| Telegram duplicate messages | Claw Config Fixes phase | Single test message produces exactly one response |
| OpenAI 401 on Claw embeddings | Claw Config Fixes phase | Direct OpenAI API test call succeeds from Claw environment |

---

## Sources

- Meta Ads API official documentation (graph.facebook.com) — token types, budget format in cents, `status` vs `effective_status`
- [Meta Ads API: Complete Guide for Advertisers and Developers (2025)](https://admanage.ai/blog/meta-ads-api) — token expiry patterns
- [How to get a Facebook long-lived access token (GitHub Gist)](https://gist.github.com/msramalho/4fc4bbc2f7ca58e0f6dc4d6de6215dc0) — token exchange endpoint
- [Google Ads API: Updates Using Field Masks](https://developers.google.com/google-ads/api/docs/client-libs/python/field-masks) — field mask requirement for mutations
- [Google Ads API: Mutate Best Practices](https://developers.google.com/google-ads/api/docs/mutating/best-practices) — silent no-op on missing field mask
- [Google Ads Refresh Token: The Nightmare of Data Engineers (Medium)](https://medium.com/@khoadaniel/google-ads-api-refresh-token-the-nightmare-of-data-engineers-df0ed04922e5) — OAuth testing vs production status
- [Twenty CRM: Improve pagination on REST API (GitHub Issue #5798)](https://github.com/twentyhq/twenty/issues/5798) — pagination limitations
- [OpenClaw Configuration Guide](https://eastondev.com/blog/en/posts/ai/20260205-openclaw-config-guide/) — config management best practices
- [Common OpenClaw Pitfalls and How to Fix Them](https://vertu.com/ai-tools/common-openclaw-pitfalls-and-how-to-fix-them/) — session management
- Codebase audit: `integrations/meta_ads/client.py`, `integrations/google_ads/client.py`, `integrations/lightrag/client.py`, `integrations/base_crm/client.py` — direct code review
- `.planning/codebase/CONCERNS.md` — existing known issues and tech debt
- MEMORY.md — system-specific constraints (OpenClaw version, sessions path, TOOLS.md limit, `doctor --fix` prohibition)

---
*Pitfalls research for: Alfred Platform Stabilization & Ad Management*
*Researched: 2026-02-20*
