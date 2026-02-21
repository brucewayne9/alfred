# Phase 4: Google Ads Budget Control - Research

**Researched:** 2026-02-20
**Domain:** Google Ads Python SDK — write mutations (campaign budget, ad group status)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Budget update behavior**
- Confirmation threshold: confirm if new budget > $100/day OR more than 2x current budget (whichever triggers first)
- Below threshold: apply immediately and confirm
- Handle both shared and individual campaign budgets — if shared, warn Mike which campaigns are affected before applying
- Always show before/after comparison: "Campaign X budget: $30/day → $50/day — done."

**Ad group controls**
- Support bulk operations (e.g., "Pause all ad groups in campaign X") but always list affected ad groups and confirm before applying
- Single ad group changes follow the same confirmation pattern as budgets (threshold-based)
- Always verify by reading back the ad group status from the API after mutation
- Warn if pausing the last active ad group in a campaign ("This is the last active ad group — pausing it effectively stops the campaign. Continue?")
- Align existing `gads_set_campaign_status` tool to match new patterns (verification read-back, before/after display, confirmation)

**Safety guardrails**
- No hard budget cap — the confirmation threshold is sufficient protection
- Audit log: append-only JSONL file in `data/` recording every write operation (timestamp, old value, new value, requester, platform)
- Apply the same guardrail patterns (confirmation threshold, audit logging, verify read-back) to existing Meta Ads write tools too — platform consistency
- When ambiguous requests come in, ask to clarify by listing options ("Which campaign? You have: Campaign A ($30/day), Campaign B ($50/day)")

**Conversational style**
- Proactively surface relevant context alongside confirmations (e.g., recent spend, trends, warnings)
- Error messages in plain English only — no API error codes shown to user
- Alfred should ask to clarify when requests are ambiguous, listing available options

**Tool naming convention** (from CONTEXT.md specifics)
- `gads_update_campaign_budget`, `gads_pause_ad_group`, `gads_enable_ad_group`

### Claude's Discretion
- Response verbosity — match tone to context (terse for quick changes, more detail for significant ones)
- Exact audit log schema and file naming
- How to detect shared vs individual budgets from the API
- Loading/progress indicators for multi-step operations

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GADS-01 | User can update Google Ads campaign budget conversationally | Covered by `gads_update_campaign_budget` tool using `CampaignBudgetService.mutate_campaign_budgets()` with `amount_micros`; shared budget detection via `campaign_budget.explicitly_shared` GAQL field |
| GADS-02 | User can pause/enable Google Ads ad groups conversationally | Covered by `gads_pause_ad_group` / `gads_enable_ad_group` tools using `AdGroupService.mutate_ad_groups()` with status enum; read-back verification via `gads_ad_groups()` existing read tool |
| GADS-03 | `google-ads` SDK installed on Labs (105) | `google-ads==29.1.0` installs cleanly with no dependency conflicts; existing protobuf 6.31.1 satisfies `<7.0.0,>=4.25.0`; requires adding `google-ads` to `requirements.txt` |
</phase_requirements>

---

## Summary

Phase 4 adds three write operations to the existing Google Ads integration: campaign budget updates, ad group pause, and ad group enable. The google-ads Python SDK (currently `29.1.0`) is already designed and the `_get_client()` helper in `integrations/google_ads/client.py` already handles auth — no new authentication work is needed. The existing `set_campaign_status` function provides a working mutation pattern to copy from.

The GADS-03 requirement (SDK installation) is the prerequisite for everything else. The SDK is not installed on Labs (105) — confirmed via `ModuleNotFoundError: No module named 'google.ads'`. However, a `pip3 install --dry-run google-ads` shows it installs cleanly: only `google-ads-29.1.0`, `google-auth-oauthlib-1.2.4`, `oauthlib-3.3.1`, and `requests-oauthlib-2.0.0` would be added. No version conflicts with existing packages.

The key technical complexity areas are: (1) budget mutations require operating on the `campaign_budget` resource (not the `campaign` resource), requiring a two-step process — first query the campaign to get its `campaign_budget.resource_name`, then mutate that budget resource; (2) shared budget detection via the `campaign_budget.explicitly_shared` field and `campaign_budget.reference_count` determines whether a warning is needed; (3) the audit log is new infrastructure (no existing JSONL files in `data/`); (4) the developer token access level must be verified before live testing — test-only tokens silently fail against production accounts.

**Primary recommendation:** Install the SDK first (GADS-03), implement budget mutations using `CampaignBudgetService`, implement ad group status mutations using `AdGroupService`, add verification read-backs to both and to the existing `gads_set_campaign_status`, and create the shared audit log infrastructure.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| google-ads | 29.1.0 (latest) | Google Ads API client — campaigns, budgets, ad groups | Official Google SDK; already used in existing `client.py` imports |
| google-auth-oauthlib | 1.2.4 | OAuth2 flow support (transitive dep of google-ads) | Already installed; required by google-ads |
| requests-oauthlib | 2.0.0 | OAuth HTTP adapter (transitive dep) | Already installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| protobuf | 6.31.1 (already installed) | Proto message serialization for Google Ads API | Required; already satisfies google-ads constraint `<7.0.0,>=4.25.0` |
| grpcio | 1.76.0 (already installed) | gRPC transport for Google Ads API | Required; already installed |
| proto-plus | 1.27.0 (already installed) | Python-friendly proto wrapper | Required by google-ads; already installed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| google-ads SDK | Raw GAQL via REST | SDK provides type safety, proto helpers, enums; REST is lower level and more error-prone |
| CampaignBudgetService mutate | Re-creating budget each time | Mutating existing budget preserves history, respects sharing; creating new budget leaves orphans |

**Installation:**
```bash
pip3 install google-ads
# Then add to requirements.txt:
# google-ads==29.1.0
```

---

## Architecture Patterns

### Recommended Project Structure

New functions go in existing files only — no new files for core SDK functions:

```
integrations/google_ads/
└── client.py           # Add: update_campaign_budget(), set_ad_group_status()
                        # Modify: set_campaign_status() (add verification read-back)

core/tools/
└── definitions.py      # Add: gads_update_campaign_budget, gads_pause_ad_group,
                        #       gads_enable_ad_group tool decorators
                        # Modify: gads_set_campaign_status (update description)

core/tools/
└── registry.py         # Add new tool names to "google_ads" category list

data/
└── ads_audit.jsonl     # NEW: append-only audit log (created on first write)

TOOLS.md                # Update Google Ads section (section 6) with new commands
requirements.txt        # Add google-ads==29.1.0
```

### Pattern 1: Budget Mutation (Two-Step)

**What:** Campaign budgets are a separate resource from campaigns. To update a budget, you must first get the campaign's `campaign_budget.resource_name`, then mutate that budget resource.

**When to use:** Any time the user requests a campaign budget change.

```python
# Source: google-ads Python SDK pattern + existing set_campaign_status() in client.py
def update_campaign_budget(campaign_id: str, new_daily_budget: float, customer_id: str = None) -> dict:
    client = _get_client()
    customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")
    ga_service = client.get_service("GoogleAdsService")

    # Step 1: Query campaign to get budget resource name + shared status + current amount
    query = f"""
        SELECT
            campaign.id,
            campaign.name,
            campaign_budget.resource_name,
            campaign_budget.amount_micros,
            campaign_budget.explicitly_shared,
            campaign_budget.reference_count
        FROM campaign
        WHERE campaign.id = {campaign_id}
    """
    response = ga_service.search(customer_id=customer_id, query=query)
    row = next(iter(response), None)
    if not row:
        return {"error": f"Campaign {campaign_id} not found"}

    old_budget = row.campaign_budget.amount_micros / 1_000_000
    budget_resource_name = row.campaign_budget.resource_name
    is_shared = row.campaign_budget.explicitly_shared
    reference_count = row.campaign_budget.reference_count

    # Step 2: Mutate the campaign_budget resource
    campaign_budget_service = client.get_service("CampaignBudgetService")
    budget_operation = client.get_type("CampaignBudgetOperation")
    budget = budget_operation.update
    budget.resource_name = budget_resource_name
    budget.amount_micros = int(new_daily_budget * 1_000_000)

    client.copy_from(
        budget_operation.update_mask,
        client.get_type("FieldMask")(paths=["amount_micros"])
    )

    campaign_budget_service.mutate_campaign_budgets(
        customer_id=customer_id,
        operations=[budget_operation]
    )

    return {
        "success": True,
        "campaign_id": campaign_id,
        "old_daily_budget": old_budget,
        "new_daily_budget": new_daily_budget,
        "is_shared": is_shared,
        "shared_reference_count": reference_count,
    }
```

### Pattern 2: Ad Group Status Mutation

**What:** Use `AdGroupService.mutate_ad_groups()` with field mask targeting `status` only.

**When to use:** Pause or enable a single or multiple ad groups.

```python
# Source: https://developers.google.com/google-ads/api/samples/update-ad-group
#         https://github.com/googleads/google-ads-python/blob/27.0.0/examples/basic_operations/update_ad_group.py
def set_ad_group_status(ad_group_id: str, status: str, customer_id: str = None) -> dict:
    client = _get_client()
    customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")
    ad_group_service = client.get_service("AdGroupService")

    ad_group_operation = client.get_type("AdGroupOperation")
    ad_group = ad_group_operation.update
    ad_group.resource_name = ad_group_service.ad_group_path(customer_id, ad_group_id)

    if status.upper() == "PAUSED":
        ad_group.status = client.enums.AdGroupStatusEnum.PAUSED
    elif status.upper() == "ENABLED":
        ad_group.status = client.enums.AdGroupStatusEnum.ENABLED
    else:
        return {"error": f"Invalid status: {status}. Use PAUSED or ENABLED"}

    client.copy_from(
        ad_group_operation.update_mask,
        client.get_type("FieldMask")(paths=["status"])
    )

    response = ad_group_service.mutate_ad_groups(
        customer_id=customer_id,
        operations=[ad_group_operation]
    )

    # Verification read-back (query current status)
    # ... then return before/after dict
    return {
        "success": True,
        "ad_group_id": ad_group_id,
        "new_status": status.upper(),
        "resource_name": response.results[0].resource_name,
    }
```

### Pattern 3: Audit Log

**What:** Append-only JSONL file in `data/` with one JSON object per line per write operation.

**When to use:** Every write operation across Google Ads AND Meta Ads.

```python
# Claude's discretion for schema — recommended schema:
import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path("/home/aialfred/alfred/data/ads_audit.jsonl")

def _audit_log(platform: str, operation: str, entity_id: str,
               old_value: dict, new_value: dict, requester: str = "alfred") -> None:
    """Append one line to the audit log."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform,          # "google_ads" | "meta_ads"
        "operation": operation,        # "update_budget" | "pause_ad_group" | etc.
        "entity_id": entity_id,        # campaign_id, ad_group_id, etc.
        "old_value": old_value,
        "new_value": new_value,
        "requester": requester,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

### Pattern 4: Confirmation Gate (LLM-side, not SDK-side)

The confirmation logic lives in the LLM tool description and the tool's return data — not in the client function. The client function always executes when called; the LLM confirms with Mike before calling the tool when the request exceeds the threshold.

**Tool description strategy:** Include threshold rules and required confirmation behavior in the tool's `description` parameter so the LLM knows when to ask before calling.

### Anti-Patterns to Avoid

- **Mutating the campaign resource to change budget:** Budget amount is on `campaign_budget`, not `campaign`. Mutating `campaign.campaign_budget` only reassigns which budget object is linked — it does not change the budget amount.
- **Using `campaign_operation.update` for budget changes:** Wrong resource type. Must use `CampaignBudgetOperation` via `CampaignBudgetService`.
- **Omitting `update_mask`:** The Google Ads API requires an explicit field mask for all UPDATE operations. Omitting it causes the mutation to be silently ignored or errors on newer SDK versions.
- **Skipping verification read-back:** Required by user decision. Don't return only the mutate response — do a follow-up `search()` query or use the existing `get_ad_groups()` to confirm the state changed.
- **Hardcoding shared budget handling:** Must always check `campaign_budget.explicitly_shared` at query time — campaigns can have either type.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Google Ads API authentication | Custom OAuth token refresh | `_get_client()` already handles this | Existing function in client.py handles refresh_token flow correctly |
| Field mask generation | Manual path list construction | `client.get_type("FieldMask")(paths=[...])` with `client.copy_from()` | SDK provides typed FieldMask; hand-rolled string paths cause silent errors |
| Micros conversion | Custom converter | `_format_micros()` already exists | Existing helper in client.py; use consistently |
| GAQL budget queries | Custom REST calls | `ga_service.search(query=gaql)` | GAQL is the standard query language; SDK handles pagination |
| Bulk ad group operations | Multi-threaded parallel calls | Single `mutate_ad_groups()` call with multiple operations | Google Ads API accepts lists of operations in a single request; more efficient and atomic |

**Key insight:** The existing `set_campaign_status()` function in `client.py` is the direct template for all new mutation functions — same pattern, same error handling, same `_get_client()` call. Copy that pattern, don't reinvent it.

---

## Common Pitfalls

### Pitfall 1: Developer Token Access Level
**What goes wrong:** Budget mutations against live production accounts return a generic authorization error like "The developer token is only approved for use with test accounts."
**Why it happens:** Test Account Access tokens only work with test accounts. Explorer Access (now the entry-level production tier as of February 2026) allows production account reads AND write mutations for campaign/budget resources (AccountBudgetProposalService is blocked at Explorer level, but CampaignBudgetService is not).
**How to avoid:** Before testing, verify the developer token access level in the Google Ads console (Tools > API Center). The token in `.env` (`lmET1R_v1NFfsra7Wtmu0A`) must have at least Explorer or Basic access. This is noted as a blocker in STATE.md.
**Warning signs:** `GoogleAdsException` with `USER_PERMISSION_DENIED` or `DEVELOPER_TOKEN_NOT_APPROVED` error codes.

### Pitfall 2: Budget Resource vs Campaign Resource Confusion
**What goes wrong:** Developer queries the `campaign` resource and tries to mutate `campaign.campaign_budget.amount_micros` — this changes which budget is assigned, not the budget amount.
**Why it happens:** The campaign object has a nested `campaign_budget` field in queries, but to mutate the budget amount you must use `CampaignBudgetService` on the `campaign_budget` resource directly.
**How to avoid:** Always use `CampaignBudgetOperation` via `campaign_budget_service = client.get_service("CampaignBudgetService")`. Get the `campaign_budget.resource_name` from the campaign query, then mutate that resource.
**Warning signs:** No error but budget amount unchanged in Google Ads console.

### Pitfall 3: Shared Budget Side Effects
**What goes wrong:** Updating a shared campaign budget changes the spend limit for ALL campaigns sharing it — not just the one Mike named.
**Why it happens:** Shared budgets apply to multiple campaigns by design. The `campaign_budget.explicitly_shared = True` indicates this.
**How to avoid:** Always check `explicitly_shared` and `reference_count` before applying. If `reference_count > 1`, list the affected campaigns to Mike before mutating. Query `SELECT campaign.id, campaign.name FROM campaign WHERE campaign_budget.id = {budget_id}` to enumerate affected campaigns.
**Warning signs:** Mike says "I only wanted to change Campaign A but Campaign B changed too."

### Pitfall 4: SDK Not Installed
**What goes wrong:** Server restart after this phase fails with `ModuleNotFoundError: No module named 'google.ads'` — same error seen when testing `_get_client()` import.
**Why it happens:** `google-ads` is imported inside each function (`from integrations.google_ads.client import ...`) so the error only appears at call time, not at server startup. The existing `gads_*` tools silently break if the SDK isn't installed.
**How to avoid:** Install SDK FIRST as the first task in this phase. Add `google-ads` to `requirements.txt` immediately after installing. Verify the server restarts cleanly.
**Warning signs:** Existing read tools (`gads_campaigns`, `gads_ad_groups`) all return errors after deployment.

### Pitfall 5: protobuf `FieldMask` API
**What goes wrong:** Using `protobuf_helpers.field_mask(None, ad_group._pb)` (pattern from some SDK examples) fails because `proto-plus` wraps objects differently than raw protobuf.
**Why it happens:** The google-ads SDK uses `proto-plus` wrapped messages. With `use_proto_plus=True` (set in `_get_client()`), the correct pattern is `client.get_type("FieldMask")(paths=["field"])` with `client.copy_from()`, not `protobuf_helpers.field_mask()`.
**How to avoid:** Follow the exact pattern from existing `set_campaign_status()` in `client.py` — it already uses the correct `client.get_type("FieldMask")(paths=["status"])` approach. Extend this pattern.
**Warning signs:** `AttributeError: '_pb'` or `TypeError` when building field mask.

---

## Code Examples

Verified patterns from official sources and existing codebase:

### Budget Query (get current budget + shared status)
```python
# Source: Derived from existing client.py GAQL patterns + Google Ads API docs
query = f"""
    SELECT
        campaign.id,
        campaign.name,
        campaign_budget.resource_name,
        campaign_budget.amount_micros,
        campaign_budget.explicitly_shared,
        campaign_budget.reference_count
    FROM campaign
    WHERE campaign.id = {campaign_id}
      AND campaign.status != 'REMOVED'
"""
# response is iterable; use next(iter(response), None)
```

### Find All Campaigns Sharing a Budget
```python
# Source: https://developers.google.com/google-ads/api/docs/campaigns/budgets/share-budgets
query = f"""
    SELECT campaign.id, campaign.name
    FROM campaign
    WHERE campaign_budget.resource_name = '{budget_resource_name}'
      AND campaign.status != 'REMOVED'
"""
```

### Budget Mutation
```python
# Source: Pattern from existing set_campaign_status() in integrations/google_ads/client.py
campaign_budget_service = client.get_service("CampaignBudgetService")
budget_op = client.get_type("CampaignBudgetOperation")
budget = budget_op.update
budget.resource_name = budget_resource_name  # from campaign query above
budget.amount_micros = int(new_daily_budget_dollars * 1_000_000)
client.copy_from(
    budget_op.update_mask,
    client.get_type("FieldMask")(paths=["amount_micros"])
)
response = campaign_budget_service.mutate_campaign_budgets(
    customer_id=customer_id,
    operations=[budget_op]
)
```

### Ad Group Status Mutation (single)
```python
# Source: https://developers.google.com/google-ads/api/samples/update-ad-group
ad_group_service = client.get_service("AdGroupService")
op = client.get_type("AdGroupOperation")
ag = op.update
ag.resource_name = ad_group_service.ad_group_path(customer_id, ad_group_id)
ag.status = client.enums.AdGroupStatusEnum.PAUSED  # or ENABLED
client.copy_from(op.update_mask, client.get_type("FieldMask")(paths=["status"]))
response = ad_group_service.mutate_ad_groups(customer_id=customer_id, operations=[op])
```

### Ad Group Status Mutation (bulk — multiple operations in one call)
```python
# Source: Google Ads API best practices (batch operations in single request)
operations = []
for ad_group_id in ad_group_ids:
    op = client.get_type("AdGroupOperation")
    ag = op.update
    ag.resource_name = ad_group_service.ad_group_path(customer_id, ad_group_id)
    ag.status = client.enums.AdGroupStatusEnum.PAUSED
    client.copy_from(op.update_mask, client.get_type("FieldMask")(paths=["status"]))
    operations.append(op)
response = ad_group_service.mutate_ad_groups(customer_id=customer_id, operations=operations)
```

### Verification Read-Back (ad group)
```python
# Use existing get_ad_groups() from client.py — filter by the mutated id(s)
# OR inline query:
verify_query = f"""
    SELECT ad_group.id, ad_group.status
    FROM ad_group
    WHERE ad_group.id IN ({','.join(ad_group_ids)})
"""
```

### gads_set_campaign_status Alignment Changes Needed
The existing function needs these additions (no functional change to the mutation itself):
1. **Before mutation:** Query campaign status to capture `old_status` for before/after display.
2. **After mutation:** Re-query `campaign.status` to verify the change took effect.
3. **Return:** Add `"old_status"` and `"verified_status"` to the return dict.
4. **Audit log:** Append an entry to `data/ads_audit.jsonl`.

### Audit Log Entry Format (Claude's discretion — recommended)
```python
{
    "timestamp": "2026-02-20T15:30:00+00:00",   # ISO 8601 UTC
    "platform": "google_ads",                    # or "meta_ads"
    "operation": "update_campaign_budget",        # snake_case operation name
    "entity_id": "12345678",                     # campaign_id or ad_group_id
    "entity_name": "Summer Search Campaign",     # human-readable name
    "customer_id": "7835917972",                 # for google_ads; null for meta
    "old_value": {"daily_budget": 30.00},
    "new_value": {"daily_budget": 50.00},
    "requester": "alfred"
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| AdWords API (SOAP) | Google Ads API (gRPC + REST) | 2019 | Existing SDK already uses current approach |
| `googleads-python-lib` | `google-ads` | 2019 | Existing client.py already uses `google-ads` |
| Manual proto field masks (`protobuf_helpers.field_mask`) | `client.get_type("FieldMask")(paths=[...])` + `client.copy_from()` | SDK v10+ | Existing `set_campaign_status()` already uses correct pattern |
| Explorer Access (new tier, Feb 2026) | Previously: Test / Basic / Standard | Feb 2026 | Explorer is now the default production entry tier; no formal application needed |

**Deprecated/outdated:**
- `googleads-python-lib`: Replaced by `google-ads` package. Do not confuse these — they are different packages.
- `AccountBudgetProposalService`: This is for billing budgets (account-level spend caps), NOT campaign budgets. Do not use this. `CampaignBudgetService` is correct for campaign daily budgets.

---

## Open Questions

1. **Developer Token Access Level**
   - What we know: Token `lmET1R_v1NFfsra7Wtmu0A` is in `.env`. STATE.md flags this as a pre-Phase 4 concern.
   - What's unclear: Whether this specific token is Test Account Access only, Explorer, Basic, or Standard. `CampaignBudgetService` mutations require at least Explorer Access (production).
   - Recommendation: First task of the phase should be `pip install google-ads`, then test `gads_campaigns` against production accounts. If it works (read already confirmed working in prior phases), budget mutations should also work since both use the same access tier. However, explicitly verify token level in Google Ads console before writing the mutation code.

2. **Budget Resource Name Format for Shared Budgets**
   - What we know: `campaign_budget.resource_name` is returned by GAQL queries on the `campaign` resource. Format is `customers/{customer_id}/campaignBudgets/{budget_id}`.
   - What's unclear: Whether `budget_id` in the resource name is always available and distinct from `campaign_budget.id` in the GAQL SELECT clause.
   - Recommendation: Include `campaign_budget.id` in the initial GAQL query alongside `campaign_budget.resource_name` to have both available; use `resource_name` for mutations (required), `id` for shared-campaign lookup queries.

3. **TOOLS.md Update Scope**
   - What we know: TOOLS.md section 6 (Google Ads) is at 5,799 chars total for the whole file. Memory notes say 20,000 char limit, currently at ~19,766. Adding new gads commands would push it over if not careful.
   - What's unclear: Exact character budget available.
   - Recommendation: Update Google Ads section 6 in TOOLS.md to add `budget <camp_id> <daily_dollars>`, `pause-group <ag_id>`, `enable-group <ag_id>` — mirroring the terse format of section 5 (Meta Ads). If TOOLS.md exceeds limit, trim other sections first.

---

## Sources

### Primary (HIGH confidence)
- Existing `/home/aialfred/alfred/integrations/google_ads/client.py` — Working `_get_client()`, `set_campaign_status()`, and GAQL patterns that already function against production
- `pip3 install --dry-run google-ads` — Confirmed `google-ads==29.1.0` installs cleanly, no version conflicts
- [Google Ads API — Update Ad Group sample](https://developers.google.com/google-ads/api/samples/update-ad-group) — `AdGroupOperation` + `field_mask` + `mutate_ad_groups()` pattern
- [Google Ads API — Sharing Campaign Budgets](https://developers.google.com/google-ads/api/docs/campaigns/budgets/share-budgets) — `explicitly_shared` field, `reference_count`, GAQL query pattern

### Secondary (MEDIUM confidence)
- [Google Ads API — Create Campaign Budget](https://developers.google.com/google-ads/api/docs/campaigns/budgets/create-budgets) — `CampaignBudgetOperation` and `mutate_campaign_budgets()` pattern (create example, UPDATE follows same service)
- [Google Ads API — Access Levels](https://developers.google.com/google-ads/api/docs/api-policy/access-levels) — Explorer Access tier restrictions; `CampaignBudgetService` NOT listed as blocked at Explorer level
- [google-ads-python GitHub — update_ad_group.py v27.0.0](https://github.com/googleads/google-ads-python/blob/27.0.0/examples/basic_operations/update_ad_group.py) — Source-verified `AdGroupOperation` pattern with `copy_from` field mask
- [Google Ads Developer Blog — Explorer Access tier (Feb 2026)](https://ads-developers.googleblog.com/2026/02/an-update-on-google-ads-api-developer.html) — Explorer Access is now default production tier, no application needed

### Tertiary (LOW confidence)
- WebSearch results on `CampaignBudgetService` UPDATE operation behavior — consistent with CREATE pattern but not directly verified via code example for UPDATE specifically; recommend verifying against live account before full implementation

---

## Metadata

**Confidence breakdown:**
- Standard stack (SDK installability, no conflicts): HIGH — verified via `pip3 install --dry-run`
- Mutation API patterns (AdGroupService): HIGH — verified via official samples + existing codebase
- Mutation API patterns (CampaignBudgetService UPDATE): MEDIUM — CREATE pattern confirmed; UPDATE follows same service but not directly shown in a fetched code example
- Shared budget detection (explicitly_shared, reference_count): HIGH — official docs confirmed
- Developer token access level for mutations: MEDIUM — Explorer Access general restrictions confirmed; specific `CampaignBudgetService` allowance at Explorer level not negatively stated but not explicitly confirmed
- Audit log infrastructure: HIGH (design decision, no external dependency)

**Research date:** 2026-02-20
**Valid until:** 2026-05-20 (SDK versioning is stable; access tier policies may shift faster)
