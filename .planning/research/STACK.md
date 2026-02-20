# Stack Research

**Domain:** AI assistant platform — ad campaign management, CRM integration, infrastructure repair
**Researched:** 2026-02-20
**Confidence:** HIGH (verified against PyPI, official docs, live environment inspection)

---

## Context: What Already Exists (Do Not Re-research)

The Alfred Labs (105) backend uses FastAPI 0.128 + Python 3.11.11 (pyenv), with a 353-tool LLM agent architecture. All Google OAuth infrastructure, httpx, requests, grpcio 1.76, protobuf 6.31.1, and google-auth are already installed. The focus here is only on what is MISSING or BROKEN.

---

## What Is Missing / Broken

### 1. `google-ads` SDK — NOT installed on Labs (105)

The `integrations/google_ads/client.py` imports `from google.ads.googleads.client import GoogleAdsClient` but `google-ads` is NOT installed on Labs. It IS installed on Claw (101) at version 29.1.0.

**Fix:** Install `google-ads==29.1.0` on Labs (105). All dependencies are already present — dry-run confirms clean install with no conflicts.

```bash
pip install google-ads==29.1.0
```

**Verified:** PyPI confirms 29.1.0 is the latest (released 2026-02-11). Protobuf 6.31.1 is within the `<7.0.0,>=4.25.0` requirement. grpcio 1.76 is within `<2.0.0,>=1.59.0`. No conflicts.

### 2. Meta Graph API version — needs upgrade from v21 to v22+

`integrations/meta_ads/client.py` hardcodes `BASE_URL = "https://graph.facebook.com/v21.0"`. Meta deprecated all versions older than v22.0 starting September 9, 2025. The client code uses raw `requests` (no Facebook Business SDK), which is appropriate and already installed.

**Fix:** Update the `BASE_URL` constant from `v21.0` to `v22.0`.

```python
BASE_URL = "https://graph.facebook.com/v22.0"
```

No new packages required. The existing `requests` library handles all Meta API calls correctly.

**Do NOT add `facebook-business` SDK.** The current direct REST approach is simpler, avoids a large dependency (SDK is 18+ MB), and the existing code already implements all needed operations cleanly. The SDK adds complexity for no benefit here.

### 3. Twenty CRM — code is correct, issue is data response mapping

Inspection of `integrations/base_crm/client.py` shows correct API usage. The reliability problems are in the data-access layer: `search_people` fetches only 100 records before fuzzy-matching, and `get_person` has ambiguous response unwrapping (`data.get("person", data.get("data", {}))`). The Twenty REST API at v1.14 returns `{"data": {"people": [...]}}` nesting consistently.

**Fix:** No new packages needed. Code fixes only:
- Add `filter` query param support to Twenty REST calls (Twenty supports `filter=name[like]:%query%` syntax)
- Fix response unwrapping in `get_person`, `get_company`, `get_opportunity`

**Verified:** Twenty v1.14 REST API uses Bearer token auth, standard CRUD at `/rest/{object}`, and supports filter query parameters. No SDK exists for Twenty — raw REST is the correct approach.

### 4. LightRAG — server is running, circuit breakers are stuck

LightRAG is running on Lonewolf (117) via Docker (`ghcr.io/hkuds/lightrag:latest`, up 6 days, port 9621). The Labs circuit breaker has a 1-hour cooldown but may have accumulated failures from when the server was previously down.

**Fix:** No new packages needed. Circuit breaker state in `integrations/lightrag/client.py` needs reset. The `_circuit_breaker` dict is module-level in-memory state — it resets on server restart. If Labs FastAPI is running without a restart, manually reset the breaker or trigger a forced health check.

**The LightRAG client code itself is correct.** The API endpoints (`/login`, `/documents/text`, `/query`, `/graph/label/search`) match the HKUDS/LightRAG server API. Auth is JWT-based (login → Bearer token), which the client handles correctly.

### 5. OpenAI 401 on Claw (101) — project archived or key revoked

The 401 error on Claw's memory embeddings indicates the OpenAI API key is invalid or the project has been archived. This is a configuration issue, not a code issue.

**Fix:** Either unarchive the project in OpenAI dashboard and generate a new API key, or switch to a different embedding provider. Labs already has `openai==2.17.0` (requirements.txt shows `openai==2.15.0` but pip shows 2.17.0 installed — minor version drift, both are current).

**If switching embedding provider on Claw:** ChromaDB on Labs already works. For Claw's OpenClaw memory, Ollama's `nomic-embed-text` model is available locally and free — this is the recommended fallback if the OpenAI key cannot be restored.

### 6. Log rotation on Claw — Python logging misconfiguration

The issue is that OpenClaw (Node.js on Claw) writes to `/tmp/openclaw-1000/openclaw-YYYY-MM-DD.log` but logs from previous days are not rotating correctly. This is a filesystem/systemd log path issue, not a Python package issue.

**Fix on 101:** Verify the log path template in OpenClaw config, ensure the log directory exists, and the process has write permission. No new packages needed.

### 7. QUEUE.md grep — shell command fix, not a package issue

The `-E` flag for extended regex alternation in grep is missing. This is a one-line config change in the escalation bridge script on Labs.

---

## Recommended Stack (New Additions Only)

### Python Packages to Add to Labs (105)

| Package | Version | Purpose | Why |
|---------|---------|---------|-----|
| `google-ads` | 29.1.0 | Google Ads API client | Official SDK, only missing package for Google Ads tools to function. All dependencies already present. |

**That's it.** Only one package is missing. Everything else is code fixes, config changes, or API version updates.

### Python Packages to Add to Claw (101)

None required. `google-ads==29.1.0` is already installed on Claw.

---

## Supporting Libraries (Already Installed, Relevant Context)

| Library | Installed Version | Role | Notes |
|---------|------------------|------|-------|
| `requests` | 2.32.3 | Meta Ads API calls | Correct choice for Meta — sync, simple, no SDK needed |
| `httpx` | 0.28.1 | LightRAG async client | Correct — LightRAG client uses async httpx throughout |
| `google-auth` | 2.48.0 | Google OAuth tokens | Used by both google-ads SDK and google-analytics |
| `grpcio` | 1.76.0 | gRPC transport for Google Ads | Already installed, compatible with google-ads 29.1.0 |
| `protobuf` | 6.31.1 | Protocol buffers for Google Ads | Already installed, within google-ads `<7.0.0,>=4.25.0` range |
| `openai` | 2.17.0 (pip) / 2.15.0 (requirements.txt) | OpenAI embeddings | Minor version drift — both work. Update requirements.txt to 2.17.0 |

---

## Installation

```bash
# On Labs (105) — only new package needed
pip install google-ads==29.1.0
```

No changes needed on Claw (101) for package installation.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Raw `requests` for Meta Ads | `facebook-business` SDK | Only if you need automated creative generation, Pixels API, or advanced audience management. For campaign read/write ops, raw REST is simpler and already works. |
| `google-ads` official SDK | Raw REST for Google Ads | Never — Google Ads API requires OAuth2 + proto-based payloads that the SDK abstracts correctly. Raw REST for Google Ads is extremely complex. |
| Keeping Twenty REST client as-is | GraphQL client for Twenty | Twenty's GraphQL endpoint exists but REST is stable for CRUD. GraphQL only needed for complex joins across objects. |
| OpenClaw `nomic-embed-text` (local) | Restore OpenAI embeddings | If OpenAI project cannot be unarchived, local Ollama embeddings avoid recurring costs and external dependency. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `facebook-business` PyPI SDK | 18MB download, adds complexity, existing raw requests client already handles all campaign operations Alfred needs | `requests` with direct Graph API calls (already implemented) |
| `google-ads` versions < 25.0 | Support for older Google Ads API versions. v19 and earlier API versions were sunset in 2025. | `google-ads==29.1.0` (current) |
| Meta Graph API `v21.0` | Deprecated September 2025, requests will fail | `v22.0` — update the BASE_URL constant |
| `googleads` PyPI package | Completely different, legacy package for older AdWords API. Not the same as `google-ads`. | `google-ads==29.1.0` |

---

## Stack Patterns by Variant

**For Meta Ads write operations (pause/enable/budget):**
- Use existing `_post()` helper in `meta_ads/client.py`
- All write operations use POST to the object ID endpoint with `status` or `daily_budget` fields
- No SDK needed, no new packages needed

**For Google Ads write operations (pause/enable/budget):**
- Use `google-ads` SDK's `CampaignService.mutate_campaigns()` (already implemented in client.py)
- Requires `google-ads` SDK installed — this is the only missing package

**For LightRAG circuit breaker reset:**
- In-process reset: call `_circuit_breaker.update({"failures": 0, "cooldown_until": None})` via a `/admin/reset-lightrag` endpoint, or restart the FastAPI process
- The server is already running on Lonewolf, so the client code just needs its internal state reset

**For Twenty CRM search reliability:**
- Add filter parameter to REST GET calls: `?filter=name[like]:%{query}%` reduces API round-trip from 100-record fetch + fuzzy-match to server-side filter
- This is a code change only, no new packages

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `google-ads==29.1.0` | `protobuf>=4.25.0,<7.0.0` | Installed protobuf 6.31.1 is within range — confirmed via dry-run |
| `google-ads==29.1.0` | `grpcio>=1.59.0,<2.0.0` | Installed grpcio 1.76.0 is within range — confirmed via dry-run |
| `google-ads==29.1.0` | `google-auth>=2.14.1,<3.0.0` | Installed google-auth 2.48.0 is within range |
| `requests==2.32.3` | Meta Graph API v22.0 | Direct REST, no version constraint beyond Python compat |
| `openai==2.17.0` | Python 3.11 | Works on Labs (105). Claw (101) has its own version. |

---

## Sources

- [google-ads PyPI](https://pypi.org/project/google-ads/) — version 29.1.0 current as of 2026-02-11, HIGH confidence
- [Google Ads Python Dependencies](https://developers.google.com/google-ads/api/docs/client-libs/python/dependencies) — dependency ranges verified, MEDIUM confidence (page showed 3.8-3.12 matrix, 3.11 covered)
- [facebook-business PyPI](https://pypi.org/project/facebook-business/) — version 24.0.1 current as of 2025-11-20, HIGH confidence
- [Meta Graph API deprecation](https://web.swipeinsight.app/posts/facebook-launches-graph-api-v22-0-and-marketing-api-v22-0-for-developers-14179) — v21 deprecated September 2025, HIGH confidence (multiple sources confirm)
- [HKUDS/LightRAG README](https://github.com/HKUDS/LightRAG/blob/main/lightrag/api/README.md) — server API endpoints and auth, HIGH confidence
- Live environment inspection (`pip list`, `ssh` to 101, `docker ps` on 117) — HIGH confidence, verified 2026-02-20
- [Twenty CRM API docs](https://twenty.com/developers/section/api-and-webhooks/api) — REST API structure confirmed, MEDIUM confidence

---

*Stack research for: Alfred Platform Stabilization & Ad Management*
*Researched: 2026-02-20*
