# Phase 5: Ad Workflow Validation & Hardening - Research

**Researched:** 2026-02-21
**Domain:** Meta Graph API (Marketing API), read-after-write verification, token management
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Live testing strategy:**
- Validate against BOTH Rod Wave and One Music Festival campaigns for broader coverage
- Batch by type: validate all read/query tools first, then move to write/mutation tools
- Budget mutation testing: make a small change (e.g. +$1), verify it took via read-back, then revert to original value
- On tool failure: log the failure, continue to the next tool, report all failures at the end (don't halt)

**Read-after-write verification:**
- Applies to ALL write operations, not just budget mutations (includes pause/enable, status changes, etc.)
- Short delay (1-2 seconds) between write and read-back to allow Meta backend propagation
- On mismatch: retry the mutation once, then send Telegram alert to Mike if still mismatched
- Conversational response: normal success response by default, add a warning only if verification failed (no explicit "confirmed" on success)

### Claude's Discretion
- API migration approach (v21→v22 upgrade strategy — all-at-once vs gradual)
- Token verification method and where to document it in config
- Exact retry timing and delay values
- Validation report format

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| META-01 | Meta Ads API updated from v21.0 to v22.0 | Single constant `BASE_URL` in `client.py` line 14; all-at-once swap is safe — no breaking changes affect Alfred's usage pattern |
| META-02 | Meta access token verified as System User (non-expiring) type | `debug_token` endpoint on Graph API returns `type` and `expires_at`; System User tokens have `type: "SYSTEM_USER"` and `expires_at: 0` |
| META-03 | Budget mutations include read-after-write verification | Pattern established in Google Ads client; adapt to Meta's REST API with `time.sleep(1)` then GET read-back |
| META-04 | All 18 existing Meta Ads tools validated against live campaigns | 22 tools actually registered (not 18); all have client functions in `client.py`; validation script needed |
</phase_requirements>

---

## Summary

Phase 5 is a hardening phase with no new tool creation. All work targets the existing `integrations/meta_ads/client.py` and `core/tools/definitions.py`. The three work streams are: (1) upgrade the API version string from v21.0 to v22.0 (a one-line change with no breaking impact on Alfred's usage), (2) add read-after-write verification to all 8 write functions in client.py (following the established Google Ads pattern), and (3) validate all 22 registered Meta Ads tools against live Rod Wave and One Music Festival campaigns.

**Critical discrepancy:** The phase spec says "18 Meta Ads tools" but the registry at `core/tools/registry.py` line 158-169 shows 22 tools. The planner should use 22 as the authoritative count. The discrepancy is likely because the spec was written before write tools were added.

**API version note:** The phase requires upgrading to v22.0 specifically. However, as of February 2026, Meta Graph API is on v24.0 (released October 2025) with v25.0 released February 10, 2026. The phase requirement (META-01) is literally "v21.0 to v22.0" — this should be implemented as written. However, a note in config should document that v22.0+ is the minimum (September 2025 deadline), and v24.0 is current. The planner may wish to flag this version choice as a question.

**Primary recommendation:** Implement read-after-write verification as a wrapper pattern in `client.py` that wraps all `_post()` write calls, with a `time.sleep(1)` then GET read-back, a single retry on mismatch, and Telegram alert via `call_alfred_claw` on persistent mismatch.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `requests` | Already installed | HTTP calls to Meta Graph API | Already used throughout client.py |
| `time` | stdlib | 1-2s sleep before read-back | No dependency needed |
| `logging` | stdlib | Log failures during validation | Already used in client.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python `subprocess` | stdlib | Run validation script or call_alfred_claw | For Telegram alert on persistent mismatch |
| `json` | stdlib | Format validation report | Standard |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `time.sleep(1)` | asyncio.sleep | No async infrastructure in client.py; sync sleep is correct here |
| Telegram via call_alfred_claw tool | Direct HTTP to Telegram Bot API | call_alfred_claw is the established Alfred pattern for Telegram |

**Installation:**
No new dependencies needed. All required libraries are already installed.

---

## Architecture Patterns

### Existing Client Structure

```
integrations/meta_ads/client.py
├── BASE_URL = "https://graph.facebook.com/v21.0"   ← upgrade to v22.0
├── _get(endpoint, params) -> Any                    ← read operations
├── _post(endpoint, data) -> Any                     ← write operations (no verification)
├── Read functions (14 tools): get_ad_account_info, list_campaigns, etc.
└── Write functions (8 tools): pause/enable ad/adset/campaign, update_ad_set_budget, update_campaign_budget
```

### Pattern 1: API Version Upgrade (META-01)

**What:** Change one constant in client.py.
**When to use:** All-at-once is correct — no breaking changes in v22.0 affect Alfred's usage.

```python
# In integrations/meta_ads/client.py line 14
# Before:
BASE_URL = "https://graph.facebook.com/v21.0"
# After:
BASE_URL = "https://graph.facebook.com/v22.0"
```

No other changes required for this upgrade. The breaking changes in v22.0 (STANDARD_ENHANCEMENTS bundle, segment customization, Instagram field renames) do not affect Alfred's budget/status/insights operations.

**Source:** WebSearch via releasebot.io and swipeinsight.app — verified v22.0 released January 21, 2025.

### Pattern 2: Token Verification (META-02)

**What:** Call Meta's `debug_token` endpoint to verify the access token type.
**When to use:** Run once and document the result in `config/.env` or a comment.

```python
import requests

def verify_token_type(access_token: str, app_id: str, app_secret: str) -> dict:
    """Verify that the access token is a non-expiring System User token."""
    app_token = f"{app_id}|{app_secret}"
    resp = requests.get(
        "https://graph.facebook.com/debug_token",
        params={
            "input_token": access_token,
            "access_token": app_token,
        },
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json().get("data", {})
    return {
        "type": data.get("type"),              # "USER", "SYSTEM_USER", "PAGE", etc.
        "expires_at": data.get("expires_at"),  # 0 = never expires
        "is_valid": data.get("is_valid"),
        "app_id": data.get("app_id"),
        "user_id": data.get("user_id"),
    }
```

The `app_id` and `app_secret` are already in `config/.env` as `META_APP_ID` and `META_APP_SECRET`. A System User token returns `type: "SYSTEM_USER"` and `expires_at: 0`.

**Token inspection URL (manual):** `https://developers.facebook.com/tools/debug/accesstoken/` — paste token, press Debug, look for "Expires: Never" and "Type: SYSTEM_USER".

**Where to document:** Add a comment in `config/.env` next to `META_ACCESS_TOKEN` documenting the token type and verification date.

### Pattern 3: Read-After-Write Verification (META-03)

**What:** After each write (_post), sleep briefly then GET the same entity to confirm the mutation applied.
**When to use:** All 8 write functions in client.py.
**Follow the Google Ads pattern from `integrations/google_ads/client.py`** — that codebase already has verified_status / verified_daily_budget return values.

```python
import time

def pause_campaign(campaign_id: str) -> dict:
    """Pause a specific campaign."""
    try:
        result = _post(campaign_id, {"status": "PAUSED"})

        # Read-after-write verification (1s propagation delay)
        time.sleep(1)
        readback = _get(campaign_id, {"fields": "id,status"})
        verified_status = readback.get("status")

        if verified_status != "PAUSED":
            # Retry once
            time.sleep(1)
            _post(campaign_id, {"status": "PAUSED"})
            time.sleep(1)
            readback2 = _get(campaign_id, {"fields": "id,status"})
            verified_status = readback2.get("status")

            if verified_status != "PAUSED":
                # Telegram alert on persistent mismatch
                _send_mismatch_alert("pause_campaign", campaign_id, "PAUSED", verified_status)
                return {
                    "success": True,
                    "campaign_id": campaign_id,
                    "status": "PAUSED",
                    "verified_status": verified_status,
                    "warning": f"Status mismatch after retry: expected PAUSED, got {verified_status}. Mike has been alerted.",
                }

        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": "PAUSED",
            "verified_status": verified_status,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Budget mutation read-back fields:**

```python
# For update_campaign_budget / update_ad_set_budget
readback = _get(campaign_id, {"fields": "id,daily_budget,lifetime_budget"})
verified_budget_cents = int(readback.get("daily_budget", 0))
verified_budget_dollars = verified_budget_cents / 100
```

**The Telegram alert helper** — use `call_alfred_claw` tool pattern from `core/tools/definitions.py`:

```python
import subprocess

def _send_mismatch_alert(operation: str, entity_id: str, expected: str, actual: str) -> None:
    """Send Telegram alert to Mike when read-after-write verification fails."""
    msg = (
        f"Meta Ads mismatch after retry:\\n"
        f"Operation: {operation}\\n"
        f"Entity: {entity_id}\\n"
        f"Expected: {expected}\\n"
        f"Got: {actual}"
    )
    try:
        subprocess.run([
            "openclaw", "message", "send",
            "--channel", "telegram",
            "--target", "7582976864",
            msg
        ], timeout=10, check=False)
    except Exception:
        pass  # Alert failure must never block the primary response
```

Note: The `call_alfred_claw` tool from `core/tools/definitions.py` uses the openclaw CLI directly. The same approach works from client.py, but keep it fire-and-forget wrapped in try/except.

### Pattern 4: Validation Script (META-04)

**What:** A Python script or manual test sequence that calls each of the 22 tools and logs pass/fail.
**When to use:** Run against live campaigns as the final verification gate.
**Structure:**

```python
# validation_script.py approach (run from alfred/ directory)
TOOLS_TO_VALIDATE = [
    # Read tools first
    ("meta_ads_account", {}),
    ("meta_ads_summary", {}),
    ("meta_ads_performance", {"period": "last_7d"}),
    ("meta_ads_campaigns", {}),
    # ... all 22 tools
    # Write tools last (budget change-and-revert)
    ("meta_ads_update_campaign_budget", {"campaign_id": ROD_WAVE_ID, "daily_budget": CURRENT+1}),
    # read-back verify
    ("meta_ads_update_campaign_budget", {"campaign_id": ROD_WAVE_ID, "daily_budget": CURRENT}),  # revert
]
```

Alternatively, run as a series of Alfred chat commands: "Show me all Meta ad campaigns", "Show Meta account info", etc. — this validates the full tool-to-LLM pipeline end-to-end.

### Anti-Patterns to Avoid

- **No time.sleep() before read-back:** Meta's backend is eventually consistent. Writing immediately followed by reading will return the old value. The 1-second sleep is mandatory.
- **Alerting on every mismatch without retry:** First mismatch may be propagation lag. Always retry once before alerting.
- **Blocking the response on alert failure:** The Telegram call must be fire-and-forget (wrapped in try/except). A notification failure must never surface to the user.
- **Upgrading past v22.0 without testing:** The phase spec says v22.0 specifically. Jumping to v24.0 (current) adds untested deprecation risks (Advantage+ campaign restrictions). Use v22.0 as specified.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP retry logic | Custom retry wrapper | Simple single retry inline | One retry is sufficient per decision; retry libraries are overkill |
| Token introspection | Parse JWT | `debug_token` Graph API endpoint | Meta tokens are opaque; only the API knows the type |
| Propagation polling | Poll loop until consistent | Fixed 1-second sleep | Loop adds complexity; 1s is empirically sufficient for status/budget mutations |
| Audit logging | Custom DB | Append to existing `data/ads_audit.jsonl` | Pattern already established in Google Ads client |

**Key insight:** The Google Ads client (`integrations/google_ads/client.py`) already solved the read-after-write verification problem. Port the pattern exactly — `verified_status`/`verified_budget` field in response, same audit log file at `/home/aialfred/alfred/data/ads_audit.jsonl`.

---

## Common Pitfalls

### Pitfall 1: Tool Count Discrepancy
**What goes wrong:** Phase spec says "18 tools" — planner might stop at 18 and miss 4 tools.
**Why it happens:** Spec was written before write operations were added (they came in a later iteration).
**How to avoid:** Use the registry count of 22 as authoritative. The registry at `core/tools/registry.py` line 158-169 is the ground truth.
**Warning signs:** If validation plan shows 18 tools, it's incomplete.

### Pitfall 2: Budget Values in Cents
**What goes wrong:** Meta stores budgets in cents (integer). Reading back `daily_budget` gives cents, not dollars. Comparing to the dollar value passed in will always show mismatch.
**Why it happens:** `update_campaign_budget` takes dollars, converts to cents for `_post`. Read-back returns cents.
**How to avoid:** Read-back comparison: `verified_cents = int(readback.get("daily_budget", 0)); verified_dollars = verified_cents / 100`. Compare `verified_dollars` to the input dollar amount.

### Pitfall 3: API Version on `is_connected()` check
**What goes wrong:** `is_connected()` in client.py calls `/me` endpoint — this works on any version. After upgrading BASE_URL to v22.0, all other endpoints automatically upgrade. No per-endpoint changes needed.
**Why it happens:** Developers assume each endpoint needs separate version pinning.
**How to avoid:** The version is part of `BASE_URL` which is prepended to all endpoints in `_get()` and `_post()`. One change propagates everywhere.

### Pitfall 4: Meta Token Type != Non-Expiring
**What goes wrong:** The current token might be a user token (60-day expiry) not a System User token.
**Why it happens:** Developers use personal user tokens for quick setup; System User tokens require Business Manager configuration.
**How to avoid:** Run the `debug_token` check first. If `expires_at != 0` or `type != "SYSTEM_USER"`, the token needs replacement before any other work proceeds. This is a prerequisite check — if it fails, the phase is blocked.
**Warning signs:** Token in .env is ~200 chars long — System User tokens tend to be shorter (but not always a reliable signal).

### Pitfall 5: Write-to-Paused Entity Errors
**What goes wrong:** Validation script attempts to pause an already-PAUSED campaign/ad/adset — Meta API returns success but the state is already the expected value, making verification trivially pass.
**Why it happens:** Testing with whatever entity is available.
**How to avoid:** When validating pause/enable operations, use `list_campaigns()` first to find one in ACTIVE status, then pause it, verify PAUSED, then re-enable it.

### Pitfall 6: Missing `meta_ads_recommendations` in the Tool Count
**What goes wrong:** `get_campaign_recommendations()` in client.py exists but is NOT registered as a tool in definitions.py or registry.py.
**Why it happens:** Function exists but was intentionally not exposed (comment in client.py: "may require specific permissions").
**How to avoid:** Do not count it in the 22 tools — it is intentionally unregistered.

---

## Code Examples

### Token Verification (verified pattern)

```python
# Source: Meta Graph API debug_token endpoint documentation
# Call from alfred/ directory with venv active

import requests
from config.settings import settings

def verify_meta_token() -> dict:
    app_id = settings.meta_app_id if hasattr(settings, 'meta_app_id') else "2139661656770472"
    app_secret = settings.meta_app_secret if hasattr(settings, 'meta_app_secret') else ""
    access_token = settings.meta_access_token

    app_token = f"{app_id}|{app_secret}"
    resp = requests.get(
        "https://graph.facebook.com/v22.0/debug_token",
        params={"input_token": access_token, "access_token": app_token},
        timeout=10
    )
    data = resp.json().get("data", {})
    is_system_user = data.get("type") == "SYSTEM_USER"
    never_expires = data.get("expires_at") == 0
    return {
        "type": data.get("type"),
        "is_system_user": is_system_user,
        "never_expires": never_expires,
        "is_valid": data.get("is_valid"),
        "expires_at": data.get("expires_at"),
    }
```

### Read-After-Write for Status Changes

```python
# Applies to: pause_ad, enable_ad, pause_ad_set, enable_ad_set,
#             pause_campaign, enable_campaign

import time

def _verify_status(entity_id: str, expected_status: str, operation: str) -> tuple[str, str | None]:
    """Read back status, retry once if mismatched. Returns (verified_status, warning_message)."""
    time.sleep(1)
    readback = _get(entity_id, {"fields": "id,status"})
    verified = readback.get("status")

    if verified == expected_status:
        return verified, None

    # Retry the mutation once
    time.sleep(1)
    _post(entity_id, {"status": expected_status})
    time.sleep(1)
    readback2 = _get(entity_id, {"fields": "id,status"})
    verified = readback2.get("status")

    if verified != expected_status:
        _send_mismatch_alert(operation, entity_id, expected_status, verified)
        return verified, f"Status mismatch after retry: expected {expected_status}, got {verified}. Mike has been alerted."

    return verified, None
```

### Read-After-Write for Budget Changes

```python
# Applies to: update_ad_set_budget, update_campaign_budget

def _verify_budget(entity_id: str, expected_dollars: float, budget_type: str = "daily") -> tuple[float, str | None]:
    """Read back budget value. Returns (verified_dollars, warning_message)."""
    field = "daily_budget" if budget_type == "daily" else "lifetime_budget"

    time.sleep(1)
    readback = _get(entity_id, {"fields": f"id,{field}"})
    verified_cents = int(readback.get(field, 0))
    verified_dollars = verified_cents / 100

    # Allow $0.01 rounding tolerance
    if abs(verified_dollars - expected_dollars) < 0.02:
        return verified_dollars, None

    # Retry
    time.sleep(1)
    _post(entity_id, {field: int(expected_dollars * 100)})
    time.sleep(1)
    readback2 = _get(entity_id, {"fields": f"id,{field}"})
    verified_dollars = int(readback2.get(field, 0)) / 100

    if abs(verified_dollars - expected_dollars) >= 0.02:
        _send_mismatch_alert(f"update_{budget_type}_budget", entity_id, f"${expected_dollars}", f"${verified_dollars}")
        return verified_dollars, f"Budget mismatch after retry. Mike has been alerted."

    return verified_dollars, None
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v21.0 BASE_URL | v22.0 (minimum) or v24.0 (current) | v22.0: Jan 2025; v24.0: Oct 2025 | v21.0 blocked since Sept 9, 2025 |
| No read-after-write | Verified status/budget in response | Phase 5 | Prevents silent failures |
| Personal user token (60d expiry) | System User token (non-expiring) | Business Manager setup | No token rotation required |

**Deprecated/outdated:**
- v21.0: End-of-life September 9, 2025 — any production calls to v21.0 endpoints will fail from that date onward.
- `STANDARD_ENHANCEMENTS` bundle: Removed in v22.0 — but Alfred does not use this, so no action needed.

---

## Open Questions

1. **Should we upgrade to v22.0 or the current v24.0?**
   - What we know: Phase spec says v22.0. Current latest is v24.0 (Oct 2025). v25.0 released Feb 10, 2026.
   - What's unclear: Whether any v23.0/v24.0 features or breaking changes affect Alfred's tools.
   - Recommendation: Implement as v22.0 per spec (it's already past the v21.0 deprecation deadline, so v22.0 is the minimum viable fix). Flag to user that v24.0 is current.

2. **Is the current META_ACCESS_TOKEN in .env a System User token?**
   - What we know: Token is in `.env` as `META_ACCESS_TOKEN=EAAeaAsz...` (long token ~230 chars). Token verification must be the first step of Phase 5.
   - What's unclear: Token type — could be personal user token with 60d expiry or System User.
   - Recommendation: Run `debug_token` check first. If not System User, this is a blocker that requires out-of-band Business Manager access to resolve before any other work.

3. **Are Rod Wave and One Music Festival campaigns currently active (ACTIVE status)?**
   - What we know: The accounts exist (`act_1323671906234016`). Campaign names are known by Mike.
   - What's unclear: Whether they are ACTIVE, PAUSED, or ARCHIVED right now.
   - Recommendation: First validation step should be `meta_ads_campaigns` with no filter to list all campaigns and identify Rod Wave / One Music Festival IDs. Validation plan should derive IDs dynamically.

4. **Where should token type documentation live?**
   - What we know: `config/.env` is the config home for Meta credentials. `config/settings.py` exposes them as `settings.meta_access_token`.
   - Recommendation: Add inline comment in `.env` next to `META_ACCESS_TOKEN` documenting verification date and type. Also add a `META_TOKEN_TYPE=system_user` env var so it is machine-readable.

---

## Sources

### Primary (HIGH confidence)
- Codebase — `integrations/meta_ads/client.py` (full read): current BASE_URL v21.0, all 22 tool functions, _get/_post helpers
- Codebase — `core/tools/registry.py` line 158-169: authoritative 22-tool count
- Codebase — `integrations/google_ads/client.py`: established read-after-write pattern with `verified_status`/`verified_daily_budget`, `_audit_log()`, and retry pattern
- Codebase — `core/tools/definitions.py` line 5555: `call_alfred_claw` Telegram pattern

### Secondary (MEDIUM confidence)
- WebSearch + releasebot.io: v22.0 released January 21, 2025; v24.0 released October 8, 2025; v25.0 released February 10, 2026
- WebSearch + swipeinsight.app: v22.0 breaking changes (STANDARD_ENHANCEMENTS removed, segment customization removed, Instagram impressions → views for media after July 2, 2024)
- WebSearch: September 9, 2025 = mandatory minimum v22.0 for all Meta API calls (v21.0 blocked)
- WebSearch: `debug_token` endpoint returns `type` (USER/SYSTEM_USER/PAGE) and `expires_at` (0 = never)
- WebSearch: Meta budget mutations stored in cents; propagation delay on reads is normal (not a bug)

### Tertiary (LOW confidence)
- Inferred: 1-second sleep is sufficient for status/budget propagation. Meta's own docs say metrics can take 15-20 minutes, but configuration mutations (status, budget) are faster. The 1-2s in user decision is a reasonable default; no official SLA documented.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already present in codebase, no new dependencies
- Architecture: HIGH — Google Ads pattern is directly portable; Meta REST API is simpler than Google Ads SDK
- Pitfalls: HIGH — cents/dollars budget mismatch is documented code fact; tool count discrepancy is read from registry
- API version status: MEDIUM — verified via multiple WebSearch sources but not official Meta developer docs directly (blocked)
- Token verification: MEDIUM — debug_token pattern is well-documented in community sources; official docs blocked

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (Meta API versioning stable; token status depends on actual token in .env)
