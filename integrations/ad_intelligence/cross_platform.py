"""Cross-platform ad performance aggregation.

Combines Meta (Facebook/Instagram) and Google Ads performance data into a single
unified response. If one platform fails, the other is still returned with an error note.
"""

import logging

logger = logging.getLogger(__name__)


def _parse_meta_float(value) -> float:
    """Parse Meta formatted currency/number strings to float.

    Meta client returns strings like '$1,234.56' or '1,234' for numbers.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    # Strip $, commas, % signs
    cleaned = str(value).replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def _get_meta_summary(days: int) -> dict:
    """Pull Meta Ads account totals and per-campaign breakdown.

    Returns normalized dict or raises exception.
    """
    from integrations.meta_ads.client import get_campaign_insights, get_account_insights

    period = f"last_{days}d"

    account_data = get_account_insights(date_preset=period)
    campaign_data = get_campaign_insights(date_preset=period)

    # Account-level totals
    total_spend = _parse_meta_float(account_data.get("spend", 0))
    total_impressions = int(_parse_meta_float(account_data.get("impressions", 0)))
    total_clicks = int(_parse_meta_float(account_data.get("clicks", 0)))
    avg_ctr = _parse_meta_float(account_data.get("ctr", 0))
    avg_cpc = _parse_meta_float(account_data.get("cpc", 0))
    avg_cpm = _parse_meta_float(account_data.get("cpm", 0))

    # Conversions: sum purchases + leads
    conversions_block = account_data.get("conversions", {})
    if isinstance(conversions_block, dict):
        total_conversions = (
            conversions_block.get("purchases", 0) +
            conversions_block.get("leads", 0)
        )
    else:
        total_conversions = int(_parse_meta_float(conversions_block))

    # Per-campaign breakdown
    campaigns = []
    for c in campaign_data:
        if c.get("error"):
            continue
        c_spend = _parse_meta_float(c.get("spend", 0))
        c_impressions = int(_parse_meta_float(c.get("impressions", 0)))
        c_clicks = int(_parse_meta_float(c.get("clicks", 0)))
        c_ctr = _parse_meta_float(c.get("ctr", 0))
        c_cpc = _parse_meta_float(c.get("cpc", 0))
        c_conversions = int(c.get("conversions", 0))
        c_cpa = round(c_spend / c_conversions, 2) if c_conversions else 0.0

        campaigns.append({
            "name": c.get("campaign_name", "Unknown"),
            "status": c.get("status", "UNKNOWN"),
            "spend": round(c_spend, 2),
            "impressions": c_impressions,
            "clicks": c_clicks,
            "ctr": round(c_ctr, 4),
            "cpc": round(c_cpc, 2),
            "conversions": c_conversions,
            "cpa": c_cpa,
        })

    return {
        "total_spend": round(total_spend, 2),
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "avg_ctr": round(avg_ctr, 4),
        "avg_cpc": round(avg_cpc, 2),
        "avg_cpm": round(avg_cpm, 2),
        "campaigns": campaigns,
        "error": None,
    }


def _get_google_summary(days: int) -> dict:
    """Pull Google Ads account totals and per-campaign breakdown.

    Returns normalized dict or raises exception.
    """
    from integrations.google_ads.client import get_campaign_performance, get_account_spend

    perf_data = get_campaign_performance(days=days)
    spend_data = get_account_spend(days=days)

    if perf_data.get("error"):
        raise RuntimeError(perf_data["error"])
    if spend_data.get("error"):
        raise RuntimeError(spend_data["error"])

    summary = perf_data.get("summary", {})
    total_spend = float(spend_data.get("total_cost", summary.get("total_cost", 0)))
    total_impressions = int(summary.get("total_impressions", 0))
    total_clicks = int(summary.get("total_clicks", 0))
    total_conversions = int(summary.get("total_conversions", 0))
    avg_ctr = float(summary.get("avg_ctr", 0))
    avg_cpc = (total_spend / total_clicks) if total_clicks else 0.0

    campaigns = []
    for c in perf_data.get("campaigns", []):
        campaigns.append({
            "name": c.get("name", "Unknown"),
            "status": c.get("status", "UNKNOWN"),
            "spend": round(float(c.get("cost", 0)), 2),
            "impressions": int(c.get("impressions", 0)),
            "clicks": int(c.get("clicks", 0)),
            "ctr": round(float(c.get("ctr", 0)), 4),
            "cpc": round(float(c.get("avg_cpc", 0)), 2),
            "conversions": int(c.get("conversions", 0)),
        })

    return {
        "total_spend": round(total_spend, 2),
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "total_conversions": total_conversions,
        "avg_ctr": round(avg_ctr, 4),
        "avg_cpc": round(avg_cpc, 2),
        "campaigns": campaigns,
        "error": None,
    }


def get_cross_platform_summary(days: int = 7) -> dict:
    """Get combined Meta + Google Ads performance summary.

    Calls both platforms and aggregates metrics. If one platform fails,
    the other is still returned with an error note on the failed platform.

    Args:
        days: Number of days to analyze. Common values: 7, 14, 30.
              Maps to Meta's last_Nd preset and Google's date range.

    Returns:
        dict with keys: period_days, meta, google, combined
    """
    meta_result = None
    google_result = None
    meta_error = None
    google_error = None

    # Attempt Meta
    try:
        meta_result = _get_meta_summary(days)
    except Exception as exc:
        meta_error = str(exc)
        logger.error(f"Meta Ads fetch failed in cross_platform summary: {exc}")
        meta_result = {
            "total_spend": 0.0,
            "total_impressions": 0,
            "total_clicks": 0,
            "total_conversions": 0,
            "avg_ctr": 0.0,
            "avg_cpc": 0.0,
            "avg_cpm": 0.0,
            "campaigns": [],
            "error": meta_error,
        }

    # Attempt Google
    try:
        google_result = _get_google_summary(days)
    except Exception as exc:
        google_error = str(exc)
        logger.error(f"Google Ads fetch failed in cross_platform summary: {exc}")
        google_result = {
            "total_spend": 0.0,
            "total_impressions": 0,
            "total_clicks": 0,
            "total_conversions": 0,
            "avg_ctr": 0.0,
            "avg_cpc": 0.0,
            "campaigns": [],
            "error": google_error,
        }

    # Build combined totals — only include platforms that succeeded
    combined_spend = 0.0
    combined_impressions = 0
    combined_clicks = 0
    combined_conversions = 0

    if not meta_error:
        combined_spend += meta_result["total_spend"]
        combined_impressions += meta_result["total_impressions"]
        combined_clicks += meta_result["total_clicks"]
        combined_conversions += meta_result["total_conversions"]

    if not google_error:
        combined_spend += google_result["total_spend"]
        combined_impressions += google_result["total_impressions"]
        combined_clicks += google_result["total_clicks"]
        combined_conversions += google_result["total_conversions"]

    combined_ctr = round((combined_clicks / combined_impressions * 100) if combined_impressions else 0.0, 4)
    combined_cpc = round((combined_spend / combined_clicks) if combined_clicks else 0.0, 2)

    return {
        "period_days": days,
        "meta": meta_result,
        "google": google_result,
        "combined": {
            "total_spend": round(combined_spend, 2),
            "total_impressions": combined_impressions,
            "total_clicks": combined_clicks,
            "total_conversions": combined_conversions,
            "avg_ctr": combined_ctr,
            "avg_cpc": combined_cpc,
        },
    }
