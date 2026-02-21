# Phase 1: Infrastructure Repairs - Research

**Researched:** 2026-02-20
**Domain:** Infrastructure restoration — network access, circuit breaker API, GA4 validation, git maintenance
**Confidence:** HIGH (direct system inspection, live connectivity tests, code audit)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**LightRAG restoration**
- Claude decides whether to restore existing data or start fresh (assess what's recoverable)
- LightRAG runs on Labs (105) alongside the FastAPI backend
- LightRAG is best-effort enrichment, NOT a hard dependency — Alfred answers without RAG if LightRAG is unavailable
- When LightRAG has no relevant context for a query, Alfred answers using LLM knowledge alone (no error shown to user)
- LightRAG should be tackled first before other Phase 1 work

**Circuit breaker reset**
- Exposed as an HTTP endpoint (POST) on the Labs API
- Admin-only auth required (bruce/admin role)
- Alfred (the AI) can self-reset stuck circuit breakers — it has permission to call the reset endpoint
- When all breakers are already healthy, the endpoint returns 200 with a "all breakers healthy" message (no-op success, safe to call anytime)

**Work ordering**
- LightRAG restoration first, then circuit breaker, GA4 sync, and git gc can be parallelized

### Claude's Discretion
- LightRAG data recovery vs fresh start (based on what's actually recoverable)
- Circuit breaker endpoint path and response format
- GA4 property ID storage approach
- Git gc strategy
- Exact error handling patterns

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | LightRAG server restored and accessible from Labs (105) | Root cause confirmed: iptables DOCKER-USER DROP rule on 117 blocks 105's public IP. Fix: add RETURN rule for 75.43.156.105 in /etc/iptables/rules.v4. 90 processed docs recoverable. |
| INFRA-02 | LightRAG circuit breaker reset endpoint added (clearable without process restart) | Circuit breaker state lives in `_circuit_breaker` dict in `integrations/lightrag/client.py`. Must expose POST endpoint in `core/api/main.py` that calls a new `reset_circuit_breaker()` function. Admin-only via `require_auth` + role check. |
| INFRA-03 | GA4 property IDs synced to Labs settings | GA4 is ALREADY WORKING. All 8 property IDs match Claw. Live API returning real data. Requirement may be satisfied — needs verification test during execution. |
| INFRA-06 | Labs git repo gc — unreachable loose objects cleaned | Confirmed: 10,933 unreachable objects. Fix: `git gc --prune=now` in `/home/aialfred/alfred`. Simple one-command maintenance. |
</phase_requirements>

---

## Summary

All four requirements have been fully diagnosed with root causes confirmed via live system inspection. No guesswork remains — each fix is well-understood before a single line is written.

**INFRA-01 (LightRAG):** The server is running and healthy on 117 (Up 7 days, 90 processed docs, auth credentials valid). The sole blocker is a DOCKER-USER iptables rule that explicitly DROPs TCP port 9621 from all non-localhost/non-private IPs. Labs (105) connects via its public IP 75.43.156.105, which hits the DROP rule. Fix: insert a RETURN rule for 75.43.156.105 before the DROP in `/etc/iptables/rules.v4` on 117 and apply live. Data is 100% recoverable — no fresh start needed.

**INFRA-02 (Circuit Breaker Reset):** The circuit breaker is an in-process Python dict in `integrations/lightrag/client.py`. Adding a reset endpoint requires: (a) a new `reset_circuit_breaker()` function in the client module, (b) a POST endpoint in `main.py` protected by admin role check, and (c) status reporting so callers know what was reset. The existing `require_auth` dependency handles JWT/cookie auth; admin-only check requires comparing `user["role"] == "admin"`.

**INFRA-03 (GA4 Sync):** Live testing confirms GA4 is fully operational on Labs. The property IDs in `integrations/google_analytics/client.py` match Claw exactly. A live query to RuckTalk returned real traffic data. The success criterion is "Alfred on Labs can answer a GA4 analytics question using the correct property IDs" — this appears already satisfied. Execution task is to run the verification test and mark done, or discover any gap.

**INFRA-06 (Git GC):** `git fsck --unreachable` reports 10,933 unreachable blobs. `git count-objects -v` shows 10,975 loose objects at 62MB. `git gc --prune=now` will pack the reachable objects and delete all unreachable ones. Safe to run — reflog is 181 entries, standard retention.

**Primary recommendation:** Fix iptables on 117 first (INFRA-01), then add circuit breaker endpoint (INFRA-02), then verify GA4 (INFRA-03), then run git gc (INFRA-06). INFRA-03 and INFRA-06 are likely 10-minute tasks.

---

## Standard Stack

### Core (what's already in place — no new dependencies)

| Component | Version/State | Purpose | Notes |
|-----------|--------------|---------|-------|
| LightRAG Docker | `ghcr.io/hkuds/lightrag:latest`, v1.4.9.8 / api 0251 | Knowledge graph server | Running on 117, healthy, auth enabled |
| httpx | Already installed | Async HTTP client for LightRAG calls | Used in `integrations/lightrag/client.py` |
| FastAPI | Already installed | REST API framework for Labs | `core/api/main.py` |
| google-analytics-data-v1beta | Already installed, confirmed working | GA4 Data API client | Returns real data |
| iptables / iptables-save | On 117 | Network access control | `rules.v4` persisted, no iptables-persistent package needed |
| git | Standard | Repository maintenance | `git gc --prune=now` |

### No New Dependencies Required

All four repairs use existing installed software. Phase 1 introduces zero new packages.

---

## Architecture Patterns

### INFRA-01: LightRAG Network Access

**Root cause diagram:**
```
Labs (105) → public IP 75.43.156.105
           → TCP to 75.43.156.117:9621
           → hits DOCKER-USER chain on 117
           → RETURN rule: 127.0.0.0/8 → pass (loopback)
           → RETURN rule: 172.16.0.0/12 → pass (RFC1918 private)
           → DROP rule: 0.0.0.0/0 → *** DROPPED *** ← THIS IS THE PROBLEM
```

**Fix: Edit `/etc/iptables/rules.v4` on 117**

Add one line before the DROP rule:
```
-A DOCKER-USER -s 75.43.156.105/32 -p tcp -m tcp --dport 9621 -j RETURN
```

Then apply live:
```bash
sudo iptables -I DOCKER-USER 3 -p tcp --dport 9621 -s 75.43.156.105 -j RETURN
```

The file already persists (no iptables-persistent package needed — `/etc/iptables/rules.v4` is read by systemd-networkd or a startup script). Editing the file AND applying live ensures survival across reboots.

**Existing circuit breaker state after fix:**

The circuit breaker in `integrations/lightrag/client.py` will be OPEN (if previous failures occurred). After the iptables fix allows connectivity, the circuit breaker must also be reset (INFRA-02). This is the reason work ordering matters: fix network first, then expose reset endpoint, then verify end-to-end.

**LightRAG data assessment:**
- 90 documents, all `processed` status
- Storage: JsonKVStorage, NetworkXStorage, NanoVectorDBStorage — all file-based, intact at `/app/data/rag_storage/`
- Embedding model: `bge-m3:latest` via Ollama on 105:11434 (Labs Ollama serves 117 — this already works)
- LLM: `gpt-4o` via OpenAI API (for document ingestion only, not for context queries) — OpenAI key present and valid
- **Decision: Restore existing data** — no fresh start needed. All 90 docs are processed and the vector/graph stores are intact.

### INFRA-02: Circuit Breaker Reset Endpoint

**Current circuit breaker structure in `integrations/lightrag/client.py`:**
```python
_circuit_breaker = {"failures": 0, "last_failure": None, "cooldown_until": None}
CIRCUIT_BREAKER_THRESHOLD = 2    # Open after 2 failures
CIRCUIT_BREAKER_COOLDOWN = timedelta(hours=1)  # Stay open 1 hour
```

**New function to add in `integrations/lightrag/client.py`:**
```python
def reset_circuit_breaker() -> dict:
    """Reset the LightRAG circuit breaker. Safe to call anytime."""
    cb = _circuit_breaker
    was_open = cb["cooldown_until"] is not None and datetime.now() < cb["cooldown_until"]
    cb["failures"] = 0
    cb["last_failure"] = None
    cb["cooldown_until"] = None
    _token_cache["token"] = None  # Also clear token cache so fresh auth happens
    _token_cache["expires"] = None
    return {
        "reset": True,
        "was_open": was_open,
        "message": "Circuit breaker reset" if was_open else "Circuit breaker was already closed"
    }


def get_circuit_breaker_status() -> dict:
    """Get current circuit breaker state."""
    cb = _circuit_breaker
    is_open = _circuit_is_open()
    return {
        "is_open": is_open,
        "failures": cb["failures"],
        "last_failure": cb["last_failure"].isoformat() if cb["last_failure"] else None,
        "cooldown_until": cb["cooldown_until"].isoformat() if cb["cooldown_until"] else None,
    }
```

**New endpoint to add in `core/api/main.py`:**
```python
@app.post("/api/admin/circuit-breaker/reset")
async def reset_circuit_breaker_endpoint(user: dict = Depends(require_auth)):
    """Reset the LightRAG circuit breaker. Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    from integrations.lightrag.client import reset_circuit_breaker
    result = reset_circuit_breaker()
    return result


@app.get("/api/admin/circuit-breaker/status")
async def circuit_breaker_status_endpoint(user: dict = Depends(require_auth)):
    """Get LightRAG circuit breaker status. Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    from integrations.lightrag.client import get_circuit_breaker_status
    return get_circuit_breaker_status()
```

**Auth pattern used throughout `main.py`:**
- `Depends(require_auth)` handles JWT Bearer + cookie
- `user.get("role") != "admin"` is the admin check pattern (see lines 410-413)
- Returns 403 HTTPException for non-admin (consistent with existing patterns)

**Endpoint path rationale:** `/api/admin/circuit-breaker/reset` groups with future admin operations. Alfred (the AI) will call this via tool when it detects LightRAG is down.

### INFRA-03: GA4 Property ID Verification

**Current state (verified live):**

Labs `integrations/google_analytics/client.py` GA_PROPERTIES dict contains all 8 properties with matching IDs to Claw's `google_analytics.py`:

| Property | Labs ID | Claw ID | Match |
|----------|---------|---------|-------|
| rucktalk | 408395502 | 408395502 | YES |
| nightlife | 442072096 | 442072096 | YES |
| rodwave | 456717749 | 456717749 | YES |
| lenssniper | 472694627 | 472694627 | YES |
| loovacast | 475653248 | 475653248 | YES |
| lumabot | 518920226 | 518920226 | YES |
| myhands | 521064731 | 521064731 | YES |
| agentertainment | 389389502 | 389389502 | YES |

Live test result: `ga_client.get_traffic_summary('rucktalk', 'last_7_days')` returned real data (5 users, 5 sessions, 7 page views). The GA4 SDK is installed, credentials file exists at `/home/aialfred/alfred/config/google_analytics_credentials.json`, and the service account has access.

**Execution task:** Run the verification test (ask Alfred a GA4 question via chat), confirm it returns data, mark INFRA-03 complete. Likely already done — no code changes expected.

**What "synced to Labs settings" meant:** Prior research note indicated property IDs were hardcoded in `client.py` and needed a "manual sync" from Claw config. That sync appears to have already been done at some point. No action needed unless the live chat test reveals a gap.

### INFRA-06: Git GC

**Current state (verified):**
```
git fsck --unreachable: 10,933 unreachable objects
git count-objects: count=10975, size=62032 KB loose
```

**Fix:**
```bash
cd /home/aialfred/alfred
git gc --prune=now
```

`--prune=now` prunes all unreachable objects immediately rather than the default 2-week grace period. This is safe because:
- All important work is on the `main` branch (verified via `git log`)
- The reflog (181 entries) will be retained — `git gc` preserves the reflog
- The unreachable objects are blobs (raw file content), not commits

**After gc, verify:**
```bash
git fsck --unreachable 2>&1 | wc -l   # Should be 0
git count-objects -v                   # loose count should drop dramatically
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LightRAG auth token management | Custom auth flow | Existing `_get_token()` / `_auth_headers()` in `client.py` | Already handles caching, 23h refresh, error recording |
| Network tunnel for LightRAG | SSH tunnel service | Direct iptables RETURN rule on 117 | Simpler, no extra process needed, already has persistence via rules.v4 |
| Admin auth for circuit breaker endpoint | New auth middleware | Existing `Depends(require_auth)` + `user.get("role")` check | Consistent with all other admin patterns in main.py |
| Git object cleanup | Manual file deletion | `git gc --prune=now` | Git handles pack-refs, object compression, cleanup atomically |

---

## Common Pitfalls

### Pitfall 1: iptables Rule Position
**What goes wrong:** Adding the RETURN rule for 105 AFTER the DROP rule in DOCKER-USER means the DROP fires first and the RETURN never executes.
**Why it happens:** iptables rules are evaluated in order; the first match wins.
**How to avoid:** Insert at position 3 in the live chain (`-I DOCKER-USER 3`) so it's before the DROP (currently position 3). Also ensure the rules.v4 file has it before the DROP line.
**Warning signs:** `curl` from 105 still times out after the fix → wrong rule order.

### Pitfall 2: Circuit Breaker Open After LightRAG Restored
**What goes wrong:** After fixing iptables on 117, LightRAG is reachable but the circuit breaker in Labs is still OPEN (1-hour cooldown). Alfred still won't query LightRAG.
**Why it happens:** The circuit breaker persists in memory until either the cooldown expires or it's explicitly reset.
**How to avoid:** After INFRA-01 fix, immediately reset the circuit breaker via the new endpoint (INFRA-02) before running the end-to-end test.
**Warning signs:** Health endpoint shows LightRAG connected, but Alfred still returns empty-context answers.

### Pitfall 3: iptables File Edit Without Live Apply
**What goes wrong:** Only editing `/etc/iptables/rules.v4` without running `iptables -I` means the fix isn't active until reboot.
**Why it happens:** The file is loaded at boot, not watched live.
**How to avoid:** Both edit the file AND run the live `iptables -I` command.
**Warning signs:** After editing file, curl from 105 still times out.

### Pitfall 4: git gc While Process Holds Locks
**What goes wrong:** Running `git gc` while another git operation (commit, fetch) is in progress can fail or leave packs in inconsistent state.
**Why it happens:** gc acquires pack locks.
**How to avoid:** Run when no other git operations are active. `git gc --auto` can be used as a safety check first.
**Warning signs:** `error: Unable to create '$GIT_DIR/gc.pid': File exists`

### Pitfall 5: Token Cache Stale After Circuit Breaker Reset
**What goes wrong:** After reset, `_get_token()` may return a cached but expired or invalid token from `_token_cache`.
**Why it happens:** Token cache is separate from circuit breaker state.
**How to avoid:** The `reset_circuit_breaker()` function should also clear `_token_cache` (included in the pattern above).
**Warning signs:** First query after reset returns 401 from LightRAG, triggering another failure.

---

## Code Examples

### Adding RETURN rule to live iptables (run on 117 via SSH)
```bash
# Apply live (inserts at position 3, before the DROP)
sudo iptables -I DOCKER-USER 3 -p tcp --dport 9621 -s 75.43.156.105 -j RETURN

# Verify rule is in correct position
sudo iptables -L DOCKER-USER -n --line-numbers
# Expected output:
# 1    RETURN     tcp  127.0.0.0/8   ...  dpt:9621
# 2    RETURN     tcp  172.16.0.0/12 ...  dpt:9621
# 3    RETURN     tcp  75.43.156.105 ...  dpt:9621  ← new rule
# 4    DROP       tcp  0.0.0.0/0     ...  dpt:9621

# Save to rules.v4 for persistence across reboots
sudo iptables-save | sudo tee /etc/iptables/rules.v4

# Verify from Labs (run on 105)
curl -s --max-time 5 http://75.43.156.117:9621/health | python3 -m json.tool
```

### Testing LightRAG from Labs after fix
```python
# Run on 105 in /home/aialfred/alfred directory
import asyncio
from integrations.lightrag.client import health_check, get_knowledge_context

async def test():
    health = await health_check()
    print("Health:", health)
    ctx = await get_knowledge_context("Ground Rush Labs operations", top_k=3)
    print("Context length:", len(ctx))
    print("Context preview:", ctx[:200] if ctx else "(empty)")

asyncio.run(test())
```

### Circuit breaker reset endpoint (add to main.py)
```python
@app.post("/api/admin/circuit-breaker/reset")
async def reset_circuit_breaker_endpoint(user: dict = Depends(require_auth)):
    """Reset the LightRAG circuit breaker. Admin only. Safe to call anytime."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    from integrations.lightrag.client import reset_circuit_breaker
    return reset_circuit_breaker()


@app.get("/api/admin/circuit-breaker/status")
async def circuit_breaker_status_endpoint(user: dict = Depends(require_auth)):
    """Get LightRAG circuit breaker status. Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    from integrations.lightrag.client import get_circuit_breaker_status
    return get_circuit_breaker_status()
```

### reset_circuit_breaker() function (add to integrations/lightrag/client.py)
```python
def reset_circuit_breaker() -> dict:
    """Reset the LightRAG circuit breaker to closed state.

    Safe to call anytime — no-op if breaker is already closed.
    Also clears token cache to force fresh authentication.

    Returns:
        dict with reset status and whether breaker was open
    """
    cb = _circuit_breaker
    was_open = bool(cb["cooldown_until"] and datetime.now() < cb["cooldown_until"])
    cb["failures"] = 0
    cb["last_failure"] = None
    cb["cooldown_until"] = None
    # Clear token cache so next request does fresh auth (avoids stale token failure)
    _token_cache["token"] = None
    _token_cache["expires"] = None
    logger.info(f"Circuit breaker reset (was_open={was_open})")
    return {
        "reset": True,
        "was_open": was_open,
        "message": "Circuit breaker reset — LightRAG will be retried" if was_open else "All breakers healthy — no action needed",
    }


def get_circuit_breaker_status() -> dict:
    """Get current circuit breaker state for diagnostics."""
    cb = _circuit_breaker
    return {
        "is_open": _circuit_is_open(),
        "failures": cb["failures"],
        "threshold": CIRCUIT_BREAKER_THRESHOLD,
        "cooldown_hours": CIRCUIT_BREAKER_COOLDOWN.total_seconds() / 3600,
        "last_failure": cb["last_failure"].isoformat() if cb["last_failure"] else None,
        "cooldown_until": cb["cooldown_until"].isoformat() if cb["cooldown_until"] else None,
    }
```

### Git GC
```bash
cd /home/aialfred/alfred

# Run gc
git gc --prune=now

# Verify success
git fsck --unreachable 2>&1 | wc -l   # Target: 0
git count-objects -v                   # loose count should be near 0
```

---

## State of the Art

| Old Assumption | Actual Finding | Impact |
|----------------|----------------|--------|
| LightRAG server is down/broken | Server is UP and healthy (7 days uptime, 90 docs processed) | Only network access is broken, not the server |
| LightRAG data may need fresh start | 90 processed docs intact, all storage files present | Restore existing data — no re-ingestion needed |
| GA4 "needs sync" | GA4 is already fully working on Labs — API returns live data | INFRA-03 is likely a verification task, not a code task |
| Circuit breaker requires process restart | In-process dict can be reset without restart | Add reset function + endpoint, no restart needed |
| Unknown iptables setup on 117 | Explicit DROP rule in DOCKER-USER chain, persisted in /etc/iptables/rules.v4 | Single rule insertion fixes connectivity permanently |

---

## Open Questions

1. **Does iptables-save overwrite Docker's dynamic rules?**
   - What we know: `iptables-save` captures ALL current rules including Docker-generated ones. On 117, `rules.v4` was last saved Feb 13 2026 and does NOT include the many Docker-generated FORWARD rules (only the DOCKER-USER ones we care about). Docker regenerates its rules on startup.
   - What's unclear: Whether running `iptables-save | tee /etc/iptables/rules.v4` after adding the rule will bloat the file with all current Docker rules (which Docker then overwrites on restart, potentially removing our 9621 RETURN rule again).
   - Recommendation: Directly edit `rules.v4` to add the single line for 75.43.156.105 (before the DROP), then apply live with `iptables -I`. Do NOT use `iptables-save` to overwrite the file — Docker's persistent rules are only the DOCKER-USER ones already in the file.

2. **Is INFRA-03 already complete?**
   - What we know: Live API test returned real data. Property IDs match Claw. Tools are registered in registry.
   - What's unclear: Whether there was a specific configuration gap that existed and was already fixed, or if the requirement was written speculatively.
   - Recommendation: During execution, ask Alfred via chat "What's the traffic on RuckTalk this week?" — if it returns data, mark INFRA-03 complete.

3. **Will the circuit breaker be open when INFRA-01 fix is applied?**
   - What we know: The circuit breaker opens after 2 failures and stays open for 1 hour. LightRAG has been unreachable from 105 for an unknown duration.
   - What's unclear: Current state of `_circuit_breaker` dict in the running FastAPI process.
   - Recommendation: After INFRA-01 fix, immediately call the circuit breaker reset endpoint (or wait for the INFRA-02 endpoint to be deployed first, then test end-to-end).

---

## Sources

### Primary (HIGH confidence) — Direct system inspection

- Live SSH to 75.43.156.117: `docker ps`, `iptables -L DOCKER-USER -n`, `/etc/iptables/rules.v4`, `docker exec lightrag env`, `curl localhost:9621/health`
- Live code audit: `/home/aialfred/alfred/integrations/lightrag/client.py` (full read)
- Live code audit: `/home/aialfred/alfred/core/api/main.py` (grep for patterns)
- Live code audit: `/home/aialfred/alfred/core/security/auth.py` (full read)
- Live code audit: `/home/aialfred/alfred/integrations/google_analytics/client.py` (full read)
- Live code audit: `/home/aialfred/alfred/core/tools/registry.py` (grep)
- Live Python test on 105: `ga_client.list_properties()` → 8 properties returned
- Live Python test on 105: `ga_client.get_traffic_summary('rucktalk', 'last_7_days')` → real data returned
- Live git test on 105: `git fsck --unreachable | wc -l` → 10,933
- Live git test on 105: `git count-objects -v` → 10,975 loose objects

### Secondary (MEDIUM confidence) — Prior research

- `/home/aialfred/alfred/.planning/research/FEATURES.md` — Feature landscape, confirmed GA4 "needs sync" note
- `/home/aialfred/alfred/.planning/research/ARCHITECTURE.md` — Architecture patterns, LightRAG circuit breaker description

---

## Metadata

**Confidence breakdown:**
- INFRA-01 (LightRAG): HIGH — Root cause confirmed by live iptables inspection, data confirmed intact
- INFRA-02 (Circuit Breaker): HIGH — Code pattern confirmed, existing auth patterns verified
- INFRA-03 (GA4): HIGH — Live API test passed, property IDs verified against Claw
- INFRA-06 (Git GC): HIGH — Object count confirmed, fix is standard git operation

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable infrastructure, iptables rules don't change without intervention)
