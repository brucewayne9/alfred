"""Google Ads API client for Alfred."""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

logger = logging.getLogger(__name__)

# Load credentials from environment
DEVELOPER_TOKEN = os.environ.get("GOOGLE_ADS_DEVELOPER_TOKEN", "")
LOGIN_CUSTOMER_ID = os.environ.get("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "").replace("-", "")

# Support multiple customer IDs (comma-separated)
_customer_ids_raw = os.environ.get("GOOGLE_ADS_CUSTOMER_ID", "")
CUSTOMER_IDS = [cid.replace("-", "").strip() for cid in _customer_ids_raw.split(",") if cid.strip()]
DEFAULT_CUSTOMER_ID = CUSTOMER_IDS[0] if CUSTOMER_IDS else ""

# Path to Google OAuth token
TOKEN_FILE = Path("/home/aialfred/alfred/config/google_token.json")


def list_accounts() -> dict:
    """List all configured Google Ads customer IDs."""
    accounts = []
    for cid in CUSTOMER_IDS:
        # Format as XXX-XXX-XXXX
        formatted = f"{cid[:3]}-{cid[3:6]}-{cid[6:]}" if len(cid) == 10 else cid
        accounts.append({"customer_id": formatted, "customer_id_raw": cid})
    return {"accounts": accounts, "count": len(accounts), "default": DEFAULT_CUSTOMER_ID}


def _get_client() -> GoogleAdsClient:
    """Get authenticated Google Ads client using existing OAuth credentials."""
    if not DEVELOPER_TOKEN:
        raise RuntimeError("GOOGLE_ADS_DEVELOPER_TOKEN not configured")
    if not CUSTOMER_IDS:
        raise RuntimeError("GOOGLE_ADS_CUSTOMER_ID not configured")

    # Load OAuth credentials from existing token file
    if not TOKEN_FILE.exists():
        raise RuntimeError("Google not connected. Visit /auth/google to authorize.")

    with open(TOKEN_FILE) as f:
        token_data = json.load(f)

    # Get client ID and secret from settings
    from config.settings import settings

    # Build credentials dict for google-ads library
    credentials = {
        "developer_token": DEVELOPER_TOKEN,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "refresh_token": token_data.get("refresh_token"),
        "use_proto_plus": True,
    }

    if LOGIN_CUSTOMER_ID:
        credentials["login_customer_id"] = LOGIN_CUSTOMER_ID

    return GoogleAdsClient.load_from_dict(credentials)


def _format_micros(micros: int) -> float:
    """Convert micros to standard currency value."""
    return micros / 1_000_000 if micros else 0.0


def get_account_info(customer_id: str = None) -> dict:
    """Get Google Ads account information."""
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        ga_service = client.get_service("GoogleAdsService")

        query = """
            SELECT
                customer.id,
                customer.descriptive_name,
                customer.currency_code,
                customer.time_zone,
                customer.status
            FROM customer
            LIMIT 1
        """

        response = ga_service.search(customer_id=customer_id, query=query)

        for row in response:
            return {
                "id": str(row.customer.id),
                "name": row.customer.descriptive_name,
                "currency": row.customer.currency_code,
                "timezone": row.customer.time_zone,
                "status": row.customer.status.name,
            }

        return {"error": "No account found"}
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}


def get_campaigns(customer_id: str = None, status: str = None) -> dict:
    """Get all campaigns for the account.

    Args:
        customer_id: Optional customer ID (uses default if not provided)
        status: Filter by status (ENABLED, PAUSED, REMOVED)
    """
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        ga_service = client.get_service("GoogleAdsService")

        query = """
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                campaign.advertising_channel_type,
                campaign.bidding_strategy_type,
                campaign_budget.amount_micros
            FROM campaign
            WHERE campaign.status != 'REMOVED'
            ORDER BY campaign.name
        """

        if status:
            query = query.replace(
                "WHERE campaign.status != 'REMOVED'",
                f"WHERE campaign.status = '{status.upper()}'"
            )

        response = ga_service.search(customer_id=customer_id, query=query)

        campaigns = []
        for row in response:
            campaigns.append({
                "id": str(row.campaign.id),
                "name": row.campaign.name,
                "status": row.campaign.status.name,
                "channel": row.campaign.advertising_channel_type.name,
                "bidding_strategy": row.campaign.bidding_strategy_type.name,
                "daily_budget": _format_micros(row.campaign_budget.amount_micros),
            })

        return {"campaigns": campaigns, "count": len(campaigns)}
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}


def get_campaign_performance(
    customer_id: str = None,
    campaign_id: str = None,
    days: int = 30
) -> dict:
    """Get campaign performance metrics.

    Args:
        customer_id: Optional customer ID
        campaign_id: Optional specific campaign ID (all campaigns if not provided)
        days: Number of days to look back (default 30)
    """
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        ga_service = client.get_service("GoogleAdsService")

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        query = f"""
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                metrics.impressions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.conversions,
                metrics.conversions_value,
                metrics.ctr,
                metrics.average_cpc
            FROM campaign
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                AND campaign.status != 'REMOVED'
        """

        if campaign_id:
            query += f" AND campaign.id = {campaign_id}"

        query += " ORDER BY metrics.cost_micros DESC"

        response = ga_service.search(customer_id=customer_id, query=query)

        campaigns = []
        total_cost = 0
        total_clicks = 0
        total_impressions = 0
        total_conversions = 0

        for row in response:
            cost = _format_micros(row.metrics.cost_micros)
            total_cost += cost
            total_clicks += row.metrics.clicks
            total_impressions += row.metrics.impressions
            total_conversions += row.metrics.conversions

            campaigns.append({
                "id": str(row.campaign.id),
                "name": row.campaign.name,
                "status": row.campaign.status.name,
                "impressions": row.metrics.impressions,
                "clicks": row.metrics.clicks,
                "cost": round(cost, 2),
                "conversions": round(row.metrics.conversions, 2),
                "conversion_value": round(_format_micros(int(row.metrics.conversions_value * 1_000_000)), 2),
                "ctr": round(row.metrics.ctr * 100, 2),
                "avg_cpc": round(_format_micros(row.metrics.average_cpc), 2),
            })

        return {
            "campaigns": campaigns,
            "period": f"{start_date} to {end_date}",
            "summary": {
                "total_cost": round(total_cost, 2),
                "total_clicks": total_clicks,
                "total_impressions": total_impressions,
                "total_conversions": round(total_conversions, 2),
                "avg_ctr": round((total_clicks / total_impressions * 100) if total_impressions else 0, 2),
            }
        }
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}


def get_ad_groups(customer_id: str = None, campaign_id: str = None) -> dict:
    """Get ad groups, optionally filtered by campaign."""
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        ga_service = client.get_service("GoogleAdsService")

        query = """
            SELECT
                ad_group.id,
                ad_group.name,
                ad_group.status,
                ad_group.type,
                campaign.id,
                campaign.name
            FROM ad_group
            WHERE ad_group.status != 'REMOVED'
        """

        if campaign_id:
            query += f" AND campaign.id = {campaign_id}"

        query += " ORDER BY campaign.name, ad_group.name"

        response = ga_service.search(customer_id=customer_id, query=query)

        ad_groups = []
        for row in response:
            ad_groups.append({
                "id": str(row.ad_group.id),
                "name": row.ad_group.name,
                "status": row.ad_group.status.name,
                "type": row.ad_group.type_.name,
                "campaign_id": str(row.campaign.id),
                "campaign_name": row.campaign.name,
            })

        return {"ad_groups": ad_groups, "count": len(ad_groups)}
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}


def get_keywords(
    customer_id: str = None,
    campaign_id: str = None,
    ad_group_id: str = None,
    days: int = 30
) -> dict:
    """Get keyword performance metrics."""
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        ga_service = client.get_service("GoogleAdsService")

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        query = f"""
            SELECT
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                ad_group_criterion.status,
                ad_group.name,
                campaign.name,
                metrics.impressions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.conversions,
                metrics.ctr,
                metrics.average_cpc
            FROM keyword_view
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
        """

        if campaign_id:
            query += f" AND campaign.id = {campaign_id}"
        if ad_group_id:
            query += f" AND ad_group.id = {ad_group_id}"

        query += " ORDER BY metrics.cost_micros DESC LIMIT 100"

        response = ga_service.search(customer_id=customer_id, query=query)

        keywords = []
        for row in response:
            keywords.append({
                "keyword": row.ad_group_criterion.keyword.text,
                "match_type": row.ad_group_criterion.keyword.match_type.name,
                "status": row.ad_group_criterion.status.name,
                "ad_group": row.ad_group.name,
                "campaign": row.campaign.name,
                "impressions": row.metrics.impressions,
                "clicks": row.metrics.clicks,
                "cost": round(_format_micros(row.metrics.cost_micros), 2),
                "conversions": round(row.metrics.conversions, 2),
                "ctr": round(row.metrics.ctr * 100, 2),
                "avg_cpc": round(_format_micros(row.metrics.average_cpc), 2),
            })

        return {
            "keywords": keywords,
            "count": len(keywords),
            "period": f"{start_date} to {end_date}"
        }
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}


def get_account_spend(
    customer_id: str = None,
    days: int = 30,
    by_day: bool = False
) -> dict:
    """Get account spend summary.

    Args:
        customer_id: Optional customer ID
        days: Number of days to look back
        by_day: If True, return daily breakdown
    """
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        ga_service = client.get_service("GoogleAdsService")

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        if by_day:
            query = f"""
                SELECT
                    segments.date,
                    metrics.cost_micros,
                    metrics.clicks,
                    metrics.impressions,
                    metrics.conversions
                FROM customer
                WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY segments.date DESC
            """
        else:
            query = f"""
                SELECT
                    metrics.cost_micros,
                    metrics.clicks,
                    metrics.impressions,
                    metrics.conversions,
                    metrics.conversions_value
                FROM customer
                WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
            """

        response = ga_service.search(customer_id=customer_id, query=query)

        if by_day:
            daily = []
            for row in response:
                daily.append({
                    "date": row.segments.date,
                    "cost": round(_format_micros(row.metrics.cost_micros), 2),
                    "clicks": row.metrics.clicks,
                    "impressions": row.metrics.impressions,
                    "conversions": round(row.metrics.conversions, 2),
                })
            return {
                "daily_spend": daily,
                "period": f"{start_date} to {end_date}",
                "total_cost": round(sum(d["cost"] for d in daily), 2),
            }
        else:
            total_cost = 0
            total_clicks = 0
            total_impressions = 0
            total_conversions = 0
            total_value = 0

            for row in response:
                total_cost += _format_micros(row.metrics.cost_micros)
                total_clicks += row.metrics.clicks
                total_impressions += row.metrics.impressions
                total_conversions += row.metrics.conversions
                total_value += row.metrics.conversions_value

            return {
                "period": f"{start_date} to {end_date}",
                "days": days,
                "total_cost": round(total_cost, 2),
                "total_clicks": total_clicks,
                "total_impressions": total_impressions,
                "total_conversions": round(total_conversions, 2),
                "total_conversion_value": round(total_value, 2),
                "avg_daily_spend": round(total_cost / days, 2) if days else 0,
                "cost_per_conversion": round(total_cost / total_conversions, 2) if total_conversions else 0,
            }
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}


def set_campaign_status(campaign_id: str, status: str, customer_id: str = None) -> dict:
    """Enable or pause a campaign.

    Args:
        campaign_id: The campaign ID to update
        status: New status (ENABLED or PAUSED)
        customer_id: Optional customer ID
    """
    try:
        client = _get_client()
        customer_id = (customer_id or DEFAULT_CUSTOMER_ID).replace("-", "")

        campaign_service = client.get_service("CampaignService")

        # Build the campaign resource name
        resource_name = client.get_service("GoogleAdsService").campaign_path(
            customer_id, campaign_id
        )

        # Create the operation
        campaign_operation = client.get_type("CampaignOperation")
        campaign = campaign_operation.update
        campaign.resource_name = resource_name

        # Set status
        if status.upper() == "ENABLED":
            campaign.status = client.enums.CampaignStatusEnum.ENABLED
        elif status.upper() == "PAUSED":
            campaign.status = client.enums.CampaignStatusEnum.PAUSED
        else:
            return {"error": f"Invalid status: {status}. Use ENABLED or PAUSED"}

        # Set the field mask
        client.copy_from(
            campaign_operation.update_mask,
            client.get_type("FieldMask")(paths=["status"])
        )

        # Execute
        response = campaign_service.mutate_campaigns(
            customer_id=customer_id,
            operations=[campaign_operation]
        )

        return {
            "success": True,
            "campaign_id": campaign_id,
            "new_status": status.upper(),
            "resource_name": response.results[0].resource_name,
        }
    except GoogleAdsException as e:
        logger.error(f"Google Ads API error: {e}")
        return {"error": str(e.failure.errors[0].message if e.failure.errors else e)}
    except Exception as e:
        logger.error(f"Google Ads error: {e}")
        return {"error": str(e)}
