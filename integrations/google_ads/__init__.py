"""Google Ads integration."""

from integrations.google_ads.client import (
    get_account_info,
    get_campaigns,
    get_campaign_performance,
    get_ad_groups,
    get_keywords,
    get_account_spend,
    set_campaign_status,
)

__all__ = [
    "get_account_info",
    "get_campaigns",
    "get_campaign_performance",
    "get_ad_groups",
    "get_keywords",
    "get_account_spend",
    "set_campaign_status",
]
