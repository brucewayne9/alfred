"""Meta (Facebook/Instagram) Ads API client. Campaign insights, reporting, and analysis."""

import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = "https://graph.facebook.com/v21.0"
ACCESS_TOKEN = getattr(settings, 'meta_access_token', '')
AD_ACCOUNT_ID = getattr(settings, 'meta_ad_account_id', '')


def _get(endpoint: str, params: dict = None) -> Any:
    """Make GET request to Meta API."""
    params = params or {}
    params["access_token"] = ACCESS_TOKEN
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _format_currency(amount_cents: int) -> str:
    """Format amount from cents to dollars."""
    return f"${amount_cents / 100:,.2f}"


def _format_number(num: int | float) -> str:
    """Format large numbers with commas."""
    return f"{num:,.0f}"


# ==================== Account Info ====================

def get_ad_account_info() -> dict:
    """Get ad account details."""
    try:
        data = _get(AD_ACCOUNT_ID, {
            "fields": "id,name,account_status,currency,timezone_name,amount_spent,balance,spend_cap"
        })
        status_map = {1: "Active", 2: "Disabled", 3: "Unsettled", 7: "Pending Review", 9: "In Grace Period", 100: "Pending Closure", 101: "Closed"}
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "status": status_map.get(data.get("account_status"), "Unknown"),
            "currency": data.get("currency"),
            "timezone": data.get("timezone_name"),
            "total_spent": _format_currency(int(data.get("amount_spent", 0))),
            "balance": _format_currency(int(data.get("balance", 0))),
            "spend_cap": _format_currency(int(data.get("spend_cap", 0))) if data.get("spend_cap") else "No limit",
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== Campaigns ====================

def list_campaigns(status_filter: str = None) -> list[dict]:
    """List all campaigns with basic info."""
    try:
        params = {
            "fields": "id,name,status,objective,daily_budget,lifetime_budget,created_time,start_time,stop_time"
        }
        # Only filter if a specific status is provided (not "all" or None)
        if status_filter and status_filter.lower() not in ("all", "any", ""):
            params["filtering"] = f'[{{"field":"effective_status","operator":"IN","value":["{status_filter.upper()}"]}}]'

        data = _get(f"{AD_ACCOUNT_ID}/campaigns", params)
        campaigns = []
        for c in data.get("data", []):
            campaigns.append({
                "id": c.get("id"),
                "name": c.get("name"),
                "status": c.get("status"),
                "objective": c.get("objective"),
                "daily_budget": _format_currency(int(c.get("daily_budget", 0))) if c.get("daily_budget") else None,
                "lifetime_budget": _format_currency(int(c.get("lifetime_budget", 0))) if c.get("lifetime_budget") else None,
                "created": c.get("created_time", "")[:10],
            })
        return campaigns
    except Exception as e:
        return [{"error": str(e)}]


def get_campaign_details(campaign_id: str) -> dict:
    """Get detailed campaign info."""
    try:
        data = _get(campaign_id, {
            "fields": "id,name,status,objective,daily_budget,lifetime_budget,budget_remaining,spend_cap,buying_type,created_time,start_time,stop_time,effective_status"
        })
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "status": data.get("status"),
            "effective_status": data.get("effective_status"),
            "objective": data.get("objective"),
            "buying_type": data.get("buying_type"),
            "daily_budget": _format_currency(int(data.get("daily_budget", 0))) if data.get("daily_budget") else None,
            "lifetime_budget": _format_currency(int(data.get("lifetime_budget", 0))) if data.get("lifetime_budget") else None,
            "budget_remaining": _format_currency(int(data.get("budget_remaining", 0))) if data.get("budget_remaining") else None,
            "created": data.get("created_time"),
            "start_time": data.get("start_time"),
            "stop_time": data.get("stop_time"),
        }
    except Exception as e:
        return {"error": str(e)}


# ==================== Ad Sets ====================

def list_ad_sets(campaign_id: str = None) -> list[dict]:
    """List ad sets, optionally filtered by campaign."""
    try:
        endpoint = f"{campaign_id}/adsets" if campaign_id else f"{AD_ACCOUNT_ID}/adsets"
        data = _get(endpoint, {
            "fields": "id,name,status,daily_budget,lifetime_budget,targeting,optimization_goal,billing_event,bid_amount"
        })
        ad_sets = []
        for a in data.get("data", []):
            ad_sets.append({
                "id": a.get("id"),
                "name": a.get("name"),
                "status": a.get("status"),
                "daily_budget": _format_currency(int(a.get("daily_budget", 0))) if a.get("daily_budget") else None,
                "lifetime_budget": _format_currency(int(a.get("lifetime_budget", 0))) if a.get("lifetime_budget") else None,
                "optimization_goal": a.get("optimization_goal"),
                "billing_event": a.get("billing_event"),
            })
        return ad_sets
    except Exception as e:
        return [{"error": str(e)}]


# ==================== Ads ====================

def list_ads(ad_set_id: str = None) -> list[dict]:
    """List ads, optionally filtered by ad set."""
    try:
        endpoint = f"{ad_set_id}/ads" if ad_set_id else f"{AD_ACCOUNT_ID}/ads"
        data = _get(endpoint, {
            "fields": "id,name,status,effective_status,creative{id,name,thumbnail_url}"
        })
        ads = []
        for a in data.get("data", []):
            creative = a.get("creative", {})
            ads.append({
                "id": a.get("id"),
                "name": a.get("name"),
                "status": a.get("status"),
                "effective_status": a.get("effective_status"),
                "creative_id": creative.get("id"),
                "creative_name": creative.get("name"),
            })
        return ads
    except Exception as e:
        return [{"error": str(e)}]


# ==================== Insights / Reporting ====================

def get_account_insights(date_preset: str = "last_7d") -> dict:
    """Get overall account performance insights.

    date_preset options: today, yesterday, this_week_mon_today, last_7d, last_14d,
                        last_28d, last_30d, this_month, last_month, this_quarter
    """
    try:
        data = _get(f"{AD_ACCOUNT_ID}/insights", {
            "date_preset": date_preset,
            "fields": "impressions,reach,clicks,ctr,cpc,cpm,spend,actions,cost_per_action_type,frequency"
        })

        if not data.get("data"):
            return {"message": "No data for this period"}

        insights = data["data"][0]

        # Extract conversions from actions
        actions = {a["action_type"]: int(a["value"]) for a in insights.get("actions", [])}
        cost_per_action = {a["action_type"]: float(a["value"]) for a in insights.get("cost_per_action_type", [])}

        return {
            "period": date_preset,
            "impressions": _format_number(int(insights.get("impressions", 0))),
            "reach": _format_number(int(insights.get("reach", 0))),
            "clicks": _format_number(int(insights.get("clicks", 0))),
            "ctr": f"{float(insights.get('ctr', 0)):.2f}%",
            "cpc": f"${float(insights.get('cpc', 0)):.2f}",
            "cpm": f"${float(insights.get('cpm', 0)):.2f}",
            "spend": f"${float(insights.get('spend', 0)):,.2f}",
            "frequency": f"{float(insights.get('frequency', 0)):.2f}",
            "conversions": {
                "purchases": actions.get("purchase", actions.get("omni_purchase", 0)),
                "leads": actions.get("lead", 0),
                "page_views": actions.get("landing_page_view", 0),
                "add_to_cart": actions.get("add_to_cart", 0),
                "link_clicks": actions.get("link_click", 0),
            },
            "cost_per_result": {
                "cost_per_purchase": f"${cost_per_action.get('purchase', cost_per_action.get('omni_purchase', 0)):.2f}",
                "cost_per_lead": f"${cost_per_action.get('lead', 0):.2f}",
                "cost_per_link_click": f"${cost_per_action.get('link_click', 0):.2f}",
            }
        }
    except Exception as e:
        return {"error": str(e)}


def get_campaign_insights(campaign_id: str = None, date_preset: str = "last_7d") -> list[dict]:
    """Get insights for campaigns."""
    try:
        endpoint = f"{campaign_id}/insights" if campaign_id else f"{AD_ACCOUNT_ID}/insights"
        params = {
            "date_preset": date_preset,
            "fields": "campaign_id,campaign_name,impressions,reach,clicks,ctr,cpc,spend,actions,cost_per_action_type",
            "level": "campaign"
        }

        data = _get(endpoint, params)

        results = []
        for insights in data.get("data", []):
            actions = {a["action_type"]: int(a["value"]) for a in insights.get("actions", [])}

            results.append({
                "campaign_id": insights.get("campaign_id"),
                "campaign_name": insights.get("campaign_name"),
                "impressions": _format_number(int(insights.get("impressions", 0))),
                "reach": _format_number(int(insights.get("reach", 0))),
                "clicks": _format_number(int(insights.get("clicks", 0))),
                "ctr": f"{float(insights.get('ctr', 0)):.2f}%",
                "cpc": f"${float(insights.get('cpc', 0)):.2f}",
                "spend": f"${float(insights.get('spend', 0)):,.2f}",
                "conversions": actions.get("purchase", actions.get("omni_purchase", actions.get("lead", 0))),
                "link_clicks": actions.get("link_click", 0),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_ad_set_insights(campaign_id: str = None, date_preset: str = "last_7d") -> list[dict]:
    """Get insights at ad set level."""
    try:
        endpoint = f"{campaign_id}/insights" if campaign_id else f"{AD_ACCOUNT_ID}/insights"
        params = {
            "date_preset": date_preset,
            "fields": "adset_id,adset_name,impressions,reach,clicks,ctr,cpc,spend,actions",
            "level": "adset"
        }

        data = _get(endpoint, params)

        results = []
        for insights in data.get("data", []):
            actions = {a["action_type"]: int(a["value"]) for a in insights.get("actions", [])}

            results.append({
                "adset_id": insights.get("adset_id"),
                "adset_name": insights.get("adset_name"),
                "impressions": _format_number(int(insights.get("impressions", 0))),
                "clicks": _format_number(int(insights.get("clicks", 0))),
                "ctr": f"{float(insights.get('ctr', 0)):.2f}%",
                "spend": f"${float(insights.get('spend', 0)):,.2f}",
                "conversions": actions.get("purchase", actions.get("omni_purchase", actions.get("lead", 0))),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_ad_insights(ad_set_id: str = None, date_preset: str = "last_7d") -> list[dict]:
    """Get insights at individual ad level."""
    try:
        endpoint = f"{ad_set_id}/insights" if ad_set_id else f"{AD_ACCOUNT_ID}/insights"
        params = {
            "date_preset": date_preset,
            "fields": "ad_id,ad_name,impressions,clicks,ctr,cpc,spend,actions",
            "level": "ad"
        }

        data = _get(endpoint, params)

        results = []
        for insights in data.get("data", []):
            actions = {a["action_type"]: int(a["value"]) for a in insights.get("actions", [])}

            results.append({
                "ad_id": insights.get("ad_id"),
                "ad_name": insights.get("ad_name"),
                "impressions": _format_number(int(insights.get("impressions", 0))),
                "clicks": _format_number(int(insights.get("clicks", 0))),
                "ctr": f"{float(insights.get('ctr', 0)):.2f}%",
                "spend": f"${float(insights.get('spend', 0)):,.2f}",
                "conversions": actions.get("purchase", actions.get("omni_purchase", actions.get("lead", 0))),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ==================== Demographics & Breakdown ====================

def get_audience_insights(date_preset: str = "last_7d") -> dict:
    """Get audience demographic breakdown."""
    try:
        # Age and gender breakdown
        age_gender = _get(f"{AD_ACCOUNT_ID}/insights", {
            "date_preset": date_preset,
            "fields": "impressions,clicks,spend,actions",
            "breakdowns": "age,gender"
        })

        # Platform breakdown
        platform = _get(f"{AD_ACCOUNT_ID}/insights", {
            "date_preset": date_preset,
            "fields": "impressions,clicks,spend",
            "breakdowns": "publisher_platform"
        })

        # Device breakdown
        device = _get(f"{AD_ACCOUNT_ID}/insights", {
            "date_preset": date_preset,
            "fields": "impressions,clicks,spend",
            "breakdowns": "device_platform"
        })

        return {
            "period": date_preset,
            "by_age_gender": [{
                "age": d.get("age"),
                "gender": d.get("gender"),
                "impressions": _format_number(int(d.get("impressions", 0))),
                "clicks": int(d.get("clicks", 0)),
                "spend": f"${float(d.get('spend', 0)):,.2f}",
            } for d in age_gender.get("data", [])],
            "by_platform": [{
                "platform": d.get("publisher_platform"),
                "impressions": _format_number(int(d.get("impressions", 0))),
                "clicks": int(d.get("clicks", 0)),
                "spend": f"${float(d.get('spend', 0)):,.2f}",
            } for d in platform.get("data", [])],
            "by_device": [{
                "device": d.get("device_platform"),
                "impressions": _format_number(int(d.get("impressions", 0))),
                "clicks": int(d.get("clicks", 0)),
                "spend": f"${float(d.get('spend', 0)):,.2f}",
            } for d in device.get("data", [])],
        }
    except Exception as e:
        return {"error": str(e)}


def get_placement_insights(date_preset: str = "last_7d") -> list[dict]:
    """Get performance by placement (feed, stories, reels, etc.)."""
    try:
        data = _get(f"{AD_ACCOUNT_ID}/insights", {
            "date_preset": date_preset,
            "fields": "impressions,clicks,ctr,spend,actions",
            "breakdowns": "publisher_platform,platform_position"
        })

        results = []
        for d in data.get("data", []):
            actions = {a["action_type"]: int(a["value"]) for a in d.get("actions", [])}
            results.append({
                "platform": d.get("publisher_platform"),
                "position": d.get("platform_position"),
                "impressions": _format_number(int(d.get("impressions", 0))),
                "clicks": int(d.get("clicks", 0)),
                "ctr": f"{float(d.get('ctr', 0)):.2f}%",
                "spend": f"${float(d.get('spend', 0)):,.2f}",
                "conversions": actions.get("purchase", actions.get("omni_purchase", 0)),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ==================== Issues & Recommendations ====================

def get_delivery_issues() -> list[dict]:
    """Check for ads with delivery issues."""
    try:
        # Get ads with issues
        data = _get(f"{AD_ACCOUNT_ID}/ads", {
            "fields": "id,name,effective_status,issues_info",
            "filtering": '[{"field":"effective_status","operator":"IN","value":["DISAPPROVED","WITH_ISSUES","PENDING_REVIEW"]}]'
        })

        issues = []
        for ad in data.get("data", []):
            ad_issues = ad.get("issues_info", [])
            for issue in ad_issues:
                issues.append({
                    "ad_id": ad.get("id"),
                    "ad_name": ad.get("name"),
                    "status": ad.get("effective_status"),
                    "issue_level": issue.get("level"),
                    "issue_message": issue.get("error_summary"),
                })

        if not issues:
            return [{"message": "No delivery issues found"}]
        return issues
    except Exception as e:
        return [{"error": str(e)}]


def get_campaign_recommendations() -> list[dict]:
    """Get optimization recommendations from Meta."""
    try:
        data = _get(f"{AD_ACCOUNT_ID}/adrules_library", {
            "fields": "id,name,status,evaluation_spec,execution_spec"
        })
        return data.get("data", [{"message": "No automated rules configured"}])
    except Exception as e:
        # Recommendations endpoint may require specific permissions
        return [{"message": "Recommendations require additional permissions or check Ads Manager for suggestions"}]


# ==================== Spend & Budget ====================

def get_spend_by_day(days: int = 7) -> list[dict]:
    """Get daily spend breakdown."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = _get(f"{AD_ACCOUNT_ID}/insights", {
            "time_range": f'{{"since":"{start_date.strftime("%Y-%m-%d")}","until":"{end_date.strftime("%Y-%m-%d")}"}}',
            "fields": "spend,impressions,clicks,actions",
            "time_increment": 1
        })

        results = []
        for d in data.get("data", []):
            actions = {a["action_type"]: int(a["value"]) for a in d.get("actions", [])}
            results.append({
                "date": d.get("date_start"),
                "spend": f"${float(d.get('spend', 0)):,.2f}",
                "impressions": _format_number(int(d.get("impressions", 0))),
                "clicks": int(d.get("clicks", 0)),
                "conversions": actions.get("purchase", actions.get("omni_purchase", actions.get("lead", 0))),
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


# ==================== Connection Check ====================

def is_connected() -> bool:
    """Check if Meta Ads API is accessible."""
    if not ACCESS_TOKEN:
        return False
    try:
        resp = requests.get(f"{BASE_URL}/me", params={"access_token": ACCESS_TOKEN}, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_meta_ads_summary() -> dict:
    """Get a quick summary of Meta Ads status."""
    try:
        account = get_ad_account_info()
        insights = get_account_insights("last_7d")
        campaigns = list_campaigns("ACTIVE")

        return {
            "connected": True,
            "account": account.get("name"),
            "status": account.get("status"),
            "active_campaigns": len([c for c in campaigns if not c.get("error")]),
            "last_7_days": {
                "spend": insights.get("spend"),
                "impressions": insights.get("impressions"),
                "clicks": insights.get("clicks"),
                "ctr": insights.get("ctr"),
            }
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
