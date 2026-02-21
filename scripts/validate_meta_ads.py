#!/usr/bin/env python3
"""Validate all 22 Meta Ads tools against live campaigns.

Runs connectivity check, all 14 read tools, then all 8 write tools using
change-and-revert pattern so no live campaign data is permanently modified.

Usage: python3 scripts/validate_meta_ads.py
Output: Prints validation table to stdout, writes JSON to data/meta_ads_validation.json
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.meta_ads.client import (
    is_connected,
    get_ad_account_info,
    get_meta_ads_summary,
    get_account_insights,
    list_campaigns,
    get_campaign_insights,
    list_ad_sets,
    get_ad_set_insights,
    list_ads,
    get_ad_insights,
    get_audience_insights,
    get_placement_insights,
    get_delivery_issues,
    get_spend_by_day,
    # Write operations
    pause_ad,
    enable_ad,
    pause_ad_set,
    enable_ad_set,
    pause_campaign,
    enable_campaign,
    update_ad_set_budget,
    update_campaign_budget,
    # Helpers
    _get,
)
from config.settings import settings

AD_ACCOUNT_ID = getattr(settings, 'meta_ad_account_id', '')

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

results: list[dict] = []


def record(tool_name: str, status: str, notes: str = "", details: dict = None):
    """Record a validation result."""
    entry = {
        "tool": tool_name,
        "status": status,  # PASS / FAIL / SKIP
        "notes": notes,
    }
    if details:
        entry["details"] = details
    results.append(entry)
    icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(status, status)
    print(f"  [{icon}] {tool_name:<40} {notes}")


def call_safely(tool_name: str, fn, *args, **kwargs):
    """Call a function, catching all exceptions. Returns (result, error)."""
    try:
        result = fn(*args, **kwargs)
        return result, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Step 1: Connectivity check
# ---------------------------------------------------------------------------

print()
print("Meta Ads Tool Validation")
print("=" * 60)
print(f"Date: {date.today().isoformat()}")
print(f"API Version: v22.0")
print()

print("Step 1: Connectivity check")
connected = is_connected()
if not connected:
    print("  [FAIL] is_connected() returned False — aborting validation")
    print("  Check META_ACCESS_TOKEN in config/.env")
    sys.exit(1)
print("  [PASS] is_connected() = True")

# ---------------------------------------------------------------------------
# Step 2: Discover campaign IDs dynamically
# ---------------------------------------------------------------------------

print()
print("Step 2: Discover live campaign data")

campaigns = list_campaigns()
if not campaigns or (len(campaigns) == 1 and "error" in campaigns[0]):
    print(f"  [FAIL] list_campaigns() returned error: {campaigns}")
    sys.exit(1)

print(f"  Found {len(campaigns)} total campaigns")
for c in campaigns[:5]:
    print(f"    - {c.get('name', 'Unknown')} [{c.get('status')}] id={c.get('id')}")

# Find any active/paused campaign for write tests
target_campaign = None
for c in campaigns:
    if c.get("status") in ("ACTIVE", "PAUSED") and not c.get("error"):
        target_campaign = c
        break

campaign_id = target_campaign["id"] if target_campaign else None
campaign_status_before = target_campaign["status"] if target_campaign else None
print(f"  Target campaign for write tests: {target_campaign.get('name') if target_campaign else 'None found'}")

# Discover ad sets
ad_sets = []
if campaign_id:
    ad_sets = list_ad_sets(campaign_id)
    valid_ad_sets = [a for a in ad_sets if not a.get("error")]
    print(f"  Found {len(valid_ad_sets)} ad sets in target campaign")

target_ad_set = None
for a in ad_sets:
    if a.get("status") in ("ACTIVE", "PAUSED") and not a.get("error"):
        target_ad_set = a
        break
ad_set_id = target_ad_set["id"] if target_ad_set else None
ad_set_status_before = target_ad_set["status"] if target_ad_set else None

# Discover ads
ads_list = []
if ad_set_id:
    ads_list = list_ads(ad_set_id)
    valid_ads = [a for a in ads_list if not a.get("error")]
    print(f"  Found {len(valid_ads)} ads in target ad set")

target_ad = None
for a in ads_list:
    if a.get("status") in ("ACTIVE", "PAUSED") and not a.get("error"):
        target_ad = a
        break
ad_id = target_ad["id"] if target_ad else None
ad_status_before = target_ad["status"] if target_ad else None

# ---------------------------------------------------------------------------
# Step 3: Read tools (14)
# ---------------------------------------------------------------------------

print()
print("Step 3: Read tools (14 tools)")
print("-" * 60)

# 1. meta_ads_account -> get_ad_account_info
result, err = call_safely("get_ad_account_info", get_ad_account_info)
if err:
    record("meta_ads_account", "FAIL", f"Error: {err}")
elif result and result.get("error"):
    record("meta_ads_account", "FAIL", f"API error: {result['error']}")
else:
    account_name = result.get("name", "Unknown")
    record("meta_ads_account", "PASS", f"Account: {account_name}", {"account": result})

# 2. meta_ads_summary -> get_meta_ads_summary
result, err = call_safely("get_meta_ads_summary", get_meta_ads_summary)
if err:
    record("meta_ads_summary", "FAIL", f"Error: {err}")
elif result and result.get("error"):
    record("meta_ads_summary", "FAIL", f"API error: {result['error']}")
else:
    record("meta_ads_summary", "PASS", f"Active campaigns: {result.get('active_campaigns', 0)}", {"summary": result})

# 3. meta_ads_performance -> get_account_insights
result, err = call_safely("get_account_insights", get_account_insights, "last_7d")
if err:
    record("meta_ads_performance", "FAIL", f"Error: {err}")
elif result and result.get("error"):
    record("meta_ads_performance", "FAIL", f"API error: {result['error']}")
else:
    record("meta_ads_performance", "PASS", f"Spend last 7d: {result.get('spend', 'N/A')}", {"insights": result})

# 4. meta_ads_campaigns -> list_campaigns
result, err = call_safely("list_campaigns", list_campaigns)
if err:
    record("meta_ads_campaigns", "FAIL", f"Error: {err}")
elif result and len(result) == 1 and result[0].get("error"):
    record("meta_ads_campaigns", "FAIL", f"API error: {result[0]['error']}")
else:
    record("meta_ads_campaigns", "PASS", f"Found {len(result)} campaigns")

# 5. meta_ads_campaign_insights -> get_campaign_insights
if campaign_id:
    result, err = call_safely("get_campaign_insights", get_campaign_insights, campaign_id, "last_7d")
    if err:
        record("meta_ads_campaign_insights", "FAIL", f"Error: {err}")
    elif result and len(result) == 1 and result[0].get("error"):
        record("meta_ads_campaign_insights", "FAIL", f"API error: {result[0]['error']}")
    else:
        record("meta_ads_campaign_insights", "PASS", f"Got insights for campaign {campaign_id}")
else:
    record("meta_ads_campaign_insights", "SKIP", "No target campaign ID available")

# 6. meta_ads_ad_sets -> list_ad_sets
if campaign_id:
    result, err = call_safely("list_ad_sets", list_ad_sets, campaign_id)
    if err:
        record("meta_ads_ad_sets", "FAIL", f"Error: {err}")
    elif result and len(result) == 1 and result[0].get("error"):
        record("meta_ads_ad_sets", "FAIL", f"API error: {result[0]['error']}")
    else:
        record("meta_ads_ad_sets", "PASS", f"Found {len(result)} ad sets")
else:
    record("meta_ads_ad_sets", "SKIP", "No target campaign ID available")

# 7. meta_ads_ad_set_insights -> get_ad_set_insights
if campaign_id:
    result, err = call_safely("get_ad_set_insights", get_ad_set_insights, campaign_id, "last_7d")
    if err:
        record("meta_ads_ad_set_insights", "FAIL", f"Error: {err}")
    elif result and len(result) == 1 and result[0].get("error"):
        record("meta_ads_ad_set_insights", "FAIL", f"API error: {result[0]['error']}")
    else:
        record("meta_ads_ad_set_insights", "PASS", f"Got ad set insights for campaign {campaign_id}")
else:
    record("meta_ads_ad_set_insights", "SKIP", "No target campaign ID available")

# 8. meta_ads_ads -> list_ads
if ad_set_id:
    result, err = call_safely("list_ads", list_ads, ad_set_id)
    if err:
        record("meta_ads_ads", "FAIL", f"Error: {err}")
    elif result and len(result) == 1 and result[0].get("error"):
        record("meta_ads_ads", "FAIL", f"API error: {result[0]['error']}")
    else:
        record("meta_ads_ads", "PASS", f"Found {len(result)} ads")
else:
    record("meta_ads_ads", "SKIP", "No target ad set ID available")

# 9. meta_ads_ad_insights -> get_ad_insights (ad-set level)
if ad_set_id:
    result, err = call_safely("get_ad_insights", get_ad_insights, ad_set_id, "last_7d")
    if err:
        record("meta_ads_ad_insights", "FAIL", f"Error: {err}")
    elif result and len(result) == 1 and result[0].get("error"):
        record("meta_ads_ad_insights", "FAIL", f"API error: {result[0]['error']}")
    else:
        record("meta_ads_ad_insights", "PASS", f"Got ad-level insights")
else:
    record("meta_ads_ad_insights", "SKIP", "No target ad set ID available")

# 10. meta_ads_insights -> get_ad_insights (account-level, no ad_set_id)
result, err = call_safely("get_ad_insights (account-level)", get_ad_insights)
if err:
    record("meta_ads_insights", "FAIL", f"Error: {err}")
elif result and len(result) == 1 and result[0].get("error"):
    record("meta_ads_insights", "FAIL", f"API error: {result[0]['error']}")
else:
    record("meta_ads_insights", "PASS", f"Got account-level ad insights ({len(result)} rows)")

# 11. meta_ads_audience -> get_audience_insights
result, err = call_safely("get_audience_insights", get_audience_insights, "last_7d")
if err:
    record("meta_ads_audience", "FAIL", f"Error: {err}")
elif result and result.get("error"):
    record("meta_ads_audience", "FAIL", f"API error: {result['error']}")
else:
    by_platform = result.get("by_platform", [])
    record("meta_ads_audience", "PASS", f"Got audience breakdown ({len(by_platform)} platforms)")

# 12. meta_ads_placements -> get_placement_insights
result, err = call_safely("get_placement_insights", get_placement_insights, "last_7d")
if err:
    record("meta_ads_placements", "FAIL", f"Error: {err}")
elif result and len(result) == 1 and result[0].get("error"):
    record("meta_ads_placements", "FAIL", f"API error: {result[0]['error']}")
else:
    record("meta_ads_placements", "PASS", f"Got placement insights ({len(result)} placements)")

# 13. meta_ads_issues -> get_delivery_issues
result, err = call_safely("get_delivery_issues", get_delivery_issues)
if err:
    record("meta_ads_issues", "FAIL", f"Error: {err}")
elif result and len(result) == 1 and result[0].get("error"):
    record("meta_ads_issues", "FAIL", f"API error: {result[0]['error']}")
else:
    msg = result[0].get("message", "") if result else ""
    notes = "No issues found" if "No delivery issues" in msg else f"{len(result)} issues reported"
    record("meta_ads_issues", "PASS", notes)

# 14. meta_ads_daily_spend -> get_spend_by_day
result, err = call_safely("get_spend_by_day", get_spend_by_day, 7)
if err:
    record("meta_ads_daily_spend", "FAIL", f"Error: {err}")
elif result and len(result) == 1 and result[0].get("error"):
    record("meta_ads_daily_spend", "FAIL", f"API error: {result[0]['error']}")
else:
    record("meta_ads_daily_spend", "PASS", f"Got {len(result)} days of spend data")

# ---------------------------------------------------------------------------
# Step 4: Write tools — status changes (6 tools)
# ---------------------------------------------------------------------------

print()
print("Step 4: Write tools — status changes (6 tools, change-and-revert)")
print("-" * 60)

# Helper to get current status for verification
def get_status(entity_id: str) -> str:
    try:
        data = _get(entity_id, {"fields": "id,status"})
        return data.get("status", "UNKNOWN")
    except Exception as exc:
        return f"ERROR: {exc}"


# 15+16. meta_ads_pause_campaign / meta_ads_enable_campaign
if campaign_id:
    original_status = campaign_status_before
    print(f"  Campaign {campaign_id} original status: {original_status}")

    # Pause it (if ACTIVE) or enable it (if PAUSED) then revert
    if original_status == "ACTIVE":
        result, err = call_safely("pause_campaign", pause_campaign, campaign_id)
        if err:
            record("meta_ads_pause_campaign", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_pause_campaign", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_pause_campaign", "PASS",
                   f"Paused OK, verified_status={verified}",
                   {"verified_status": verified})

        # Re-enable
        result, err = call_safely("enable_campaign", enable_campaign, campaign_id)
        if err:
            record("meta_ads_enable_campaign", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_enable_campaign", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_enable_campaign", "PASS",
                   f"Re-enabled OK, verified_status={verified}",
                   {"verified_status": verified})

    elif original_status == "PAUSED":
        # Enable then re-pause
        result, err = call_safely("enable_campaign", enable_campaign, campaign_id)
        if err:
            record("meta_ads_enable_campaign", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_enable_campaign", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_enable_campaign", "PASS",
                   f"Enabled OK, verified_status={verified}",
                   {"verified_status": verified})

        result, err = call_safely("pause_campaign", pause_campaign, campaign_id)
        if err:
            record("meta_ads_pause_campaign", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_pause_campaign", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_pause_campaign", "PASS",
                   f"Paused OK, verified_status={verified}",
                   {"verified_status": verified})
    else:
        record("meta_ads_pause_campaign", "SKIP", f"Campaign status is {original_status}, not ACTIVE/PAUSED")
        record("meta_ads_enable_campaign", "SKIP", f"Campaign status is {original_status}, not ACTIVE/PAUSED")

    # Verify revert was successful
    final_status = get_status(campaign_id)
    if final_status == original_status:
        print(f"  Campaign status successfully reverted to: {final_status}")
    else:
        print(f"  WARNING: Campaign final status {final_status} != original {original_status}")
else:
    record("meta_ads_pause_campaign", "SKIP", "No campaign ID available")
    record("meta_ads_enable_campaign", "SKIP", "No campaign ID available")

# 17+18. meta_ads_pause_ad_set / meta_ads_enable_ad_set
if ad_set_id:
    original_status = ad_set_status_before
    print(f"  Ad set {ad_set_id} original status: {original_status}")

    if original_status == "ACTIVE":
        result, err = call_safely("pause_ad_set", pause_ad_set, ad_set_id)
        if err:
            record("meta_ads_pause_ad_set", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_pause_ad_set", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_pause_ad_set", "PASS",
                   f"Paused OK, verified_status={verified}",
                   {"verified_status": verified})

        result, err = call_safely("enable_ad_set", enable_ad_set, ad_set_id)
        if err:
            record("meta_ads_enable_ad_set", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_enable_ad_set", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_enable_ad_set", "PASS",
                   f"Re-enabled OK, verified_status={verified}",
                   {"verified_status": verified})

    elif original_status == "PAUSED":
        result, err = call_safely("enable_ad_set", enable_ad_set, ad_set_id)
        if err:
            record("meta_ads_enable_ad_set", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_enable_ad_set", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_enable_ad_set", "PASS",
                   f"Enabled OK, verified_status={verified}",
                   {"verified_status": verified})

        result, err = call_safely("pause_ad_set", pause_ad_set, ad_set_id)
        if err:
            record("meta_ads_pause_ad_set", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_pause_ad_set", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_pause_ad_set", "PASS",
                   f"Paused OK, verified_status={verified}",
                   {"verified_status": verified})
    else:
        record("meta_ads_pause_ad_set", "SKIP", f"Ad set status is {original_status}")
        record("meta_ads_enable_ad_set", "SKIP", f"Ad set status is {original_status}")

    # Verify revert
    final_status = get_status(ad_set_id)
    if final_status == original_status:
        print(f"  Ad set status successfully reverted to: {final_status}")
    else:
        print(f"  WARNING: Ad set final status {final_status} != original {original_status}")
else:
    record("meta_ads_pause_ad_set", "SKIP", "No ad set ID available")
    record("meta_ads_enable_ad_set", "SKIP", "No ad set ID available")

# 19+20. meta_ads_pause_ad / meta_ads_enable_ad
if ad_id:
    original_status = ad_status_before
    print(f"  Ad {ad_id} original status: {original_status}")

    if original_status == "ACTIVE":
        result, err = call_safely("pause_ad", pause_ad, ad_id)
        if err:
            record("meta_ads_pause_ad", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_pause_ad", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_pause_ad", "PASS",
                   f"Paused OK, verified_status={verified}",
                   {"verified_status": verified})

        result, err = call_safely("enable_ad", enable_ad, ad_id)
        if err:
            record("meta_ads_enable_ad", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_enable_ad", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_enable_ad", "PASS",
                   f"Re-enabled OK, verified_status={verified}",
                   {"verified_status": verified})

    elif original_status == "PAUSED":
        result, err = call_safely("enable_ad", enable_ad, ad_id)
        if err:
            record("meta_ads_enable_ad", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_enable_ad", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_enable_ad", "PASS",
                   f"Enabled OK, verified_status={verified}",
                   {"verified_status": verified})

        result, err = call_safely("pause_ad", pause_ad, ad_id)
        if err:
            record("meta_ads_pause_ad", "FAIL", f"Error: {err}")
        elif not result.get("success"):
            record("meta_ads_pause_ad", "FAIL", f"API error: {result.get('error')}")
        else:
            verified = result.get("verified_status")
            record("meta_ads_pause_ad", "PASS",
                   f"Paused OK, verified_status={verified}",
                   {"verified_status": verified})
    else:
        record("meta_ads_pause_ad", "SKIP", f"Ad status is {original_status}")
        record("meta_ads_enable_ad", "SKIP", f"Ad status is {original_status}")

    # Verify revert
    final_status = get_status(ad_id)
    if final_status == original_status:
        print(f"  Ad status successfully reverted to: {final_status}")
    else:
        print(f"  WARNING: Ad final status {final_status} != original {original_status}")
else:
    record("meta_ads_pause_ad", "SKIP", "No active/paused ad found in discovered ad set")
    record("meta_ads_enable_ad", "SKIP", "No active/paused ad found in discovered ad set")

# ---------------------------------------------------------------------------
# Step 5: Write tools — budget changes (2 tools)
# ---------------------------------------------------------------------------

print()
print("Step 5: Write tools — budget changes (2 tools, change-and-revert)")
print("-" * 60)


def get_raw_daily_budget(entity_id: str) -> int | None:
    """Get raw daily_budget in cents from Meta API."""
    try:
        data = _get(entity_id, {"fields": "id,daily_budget,lifetime_budget"})
        raw = data.get("daily_budget")
        return int(raw) if raw else None
    except Exception:
        return None


# 21. meta_ads_update_campaign_budget
if campaign_id:
    raw_cents = get_raw_daily_budget(campaign_id)
    if raw_cents is not None:
        original_dollars = raw_cents / 100
        test_dollars = original_dollars + 1.0
        print(f"  Campaign budget: ${original_dollars:.2f} — will test +$1.00 then revert")

        # Increase by $1
        result, err = call_safely("update_campaign_budget (+$1)", update_campaign_budget,
                                  campaign_id, daily_budget=test_dollars)
        if err:
            record("meta_ads_update_campaign_budget", "FAIL", f"Error on +$1: {err}")
        elif not result.get("success"):
            record("meta_ads_update_campaign_budget", "FAIL", f"API error: {result.get('error')}")
        else:
            verified_after_increase = result.get("verified_budget")
            increase_ok = abs((verified_after_increase or 0) - test_dollars) < 0.02

            # Revert to original
            revert_result, revert_err = call_safely("update_campaign_budget (revert)", update_campaign_budget,
                                                     campaign_id, daily_budget=original_dollars)
            if revert_err:
                record("meta_ads_update_campaign_budget", "FAIL",
                       f"Increased OK but revert failed: {revert_err}")
            elif not revert_result.get("success"):
                record("meta_ads_update_campaign_budget", "FAIL",
                       f"Increased OK but revert API error: {revert_result.get('error')}")
            else:
                verified_after_revert = revert_result.get("verified_budget")
                revert_ok = abs((verified_after_revert or 0) - original_dollars) < 0.02
                if increase_ok and revert_ok:
                    record("meta_ads_update_campaign_budget", "PASS",
                           f"${original_dollars:.2f} -> ${test_dollars:.2f} (verified=${verified_after_increase:.2f}) -> reverted (verified=${verified_after_revert:.2f})",
                           {"verified_budget_after_increase": verified_after_increase,
                            "verified_budget_after_revert": verified_after_revert})
                else:
                    record("meta_ads_update_campaign_budget", "FAIL",
                           f"Budget mismatch: increase_ok={increase_ok}, revert_ok={revert_ok}")
    else:
        # Campaign may use lifetime budget only
        record("meta_ads_update_campaign_budget", "SKIP",
               "Campaign has no daily_budget (may use lifetime budget — out of scope per plan)")
else:
    record("meta_ads_update_campaign_budget", "SKIP", "No campaign ID available")

# 22. meta_ads_update_ad_set_budget
if ad_set_id:
    raw_cents = get_raw_daily_budget(ad_set_id)
    if raw_cents is not None:
        original_dollars = raw_cents / 100
        test_dollars = original_dollars + 1.0
        print(f"  Ad set budget: ${original_dollars:.2f} — will test +$1.00 then revert")

        result, err = call_safely("update_ad_set_budget (+$1)", update_ad_set_budget,
                                  ad_set_id, daily_budget=test_dollars)
        if err:
            record("meta_ads_update_ad_set_budget", "FAIL", f"Error on +$1: {err}")
        elif not result.get("success"):
            record("meta_ads_update_ad_set_budget", "FAIL", f"API error: {result.get('error')}")
        else:
            verified_after_increase = result.get("verified_budget")
            increase_ok = abs((verified_after_increase or 0) - test_dollars) < 0.02

            revert_result, revert_err = call_safely("update_ad_set_budget (revert)", update_ad_set_budget,
                                                     ad_set_id, daily_budget=original_dollars)
            if revert_err:
                record("meta_ads_update_ad_set_budget", "FAIL",
                       f"Increased OK but revert failed: {revert_err}")
            elif not revert_result.get("success"):
                record("meta_ads_update_ad_set_budget", "FAIL",
                       f"Increased OK but revert API error: {revert_result.get('error')}")
            else:
                verified_after_revert = revert_result.get("verified_budget")
                revert_ok = abs((verified_after_revert or 0) - original_dollars) < 0.02
                if increase_ok and revert_ok:
                    record("meta_ads_update_ad_set_budget", "PASS",
                           f"${original_dollars:.2f} -> ${test_dollars:.2f} (verified=${verified_after_increase:.2f}) -> reverted (verified=${verified_after_revert:.2f})",
                           {"verified_budget_after_increase": verified_after_increase,
                            "verified_budget_after_revert": verified_after_revert})
                else:
                    record("meta_ads_update_ad_set_budget", "FAIL",
                           f"Budget mismatch: increase_ok={increase_ok}, revert_ok={revert_ok}")
    else:
        record("meta_ads_update_ad_set_budget", "SKIP",
               "Ad set has no daily_budget (may use campaign-level budget or lifetime budget)")
else:
    record("meta_ads_update_ad_set_budget", "SKIP", "No ad set ID available")

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Meta Ads Tool Validation Report")
print("=" * 60)
print(f"Date: {date.today().isoformat()}")
print(f"API Version: v22.0")
print(f"Account: {AD_ACCOUNT_ID}")
print()
print(f"{'Tool':<45} {'Status':<8} Notes")
print("-" * 90)

pass_count = 0
fail_count = 0
skip_count = 0

for r in results:
    tool = r["tool"]
    status = r["status"]
    notes = r["notes"]
    print(f"  {tool:<43} {status:<8} {notes}")
    if status == "PASS":
        pass_count += 1
    elif status == "FAIL":
        fail_count += 1
    else:
        skip_count += 1

total = pass_count + fail_count + skip_count
print()
print(f"Results: {pass_count}/{total} PASS, {fail_count} FAIL, {skip_count} SKIP")

if fail_count == 0:
    print("All non-skipped tools passed validation.")
else:
    print("FAILURES detected — review results above.")

# Write JSON report
output_path = Path(__file__).parent.parent / "data" / "meta_ads_validation.json"
report = {
    "date": date.today().isoformat(),
    "api_version": "v22.0",
    "account_id": AD_ACCOUNT_ID,
    "summary": {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
    },
    "tools": results,
}
output_path.write_text(json.dumps(report, indent=2))
print()
print(f"JSON report written to: {output_path}")
