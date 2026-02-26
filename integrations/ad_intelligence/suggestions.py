"""AI-powered ad optimization suggestions engine.

Analyzes live Meta + Google Ads campaign data using rule-based heuristics
and returns prioritized, actionable recommendations.
"""

import logging
from integrations.ad_intelligence.cross_platform import get_cross_platform_summary

logger = logging.getLogger(__name__)

# Priority sort order
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _safe_div(numerator: float, denominator: float) -> float:
    """Safe division — returns 0.0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _platform_avg_cpc(platform_data: dict) -> float:
    """Get average CPC for a platform from summary data."""
    return float(platform_data.get("avg_cpc", 0.0))


def _platform_avg_ctr(platform_data: dict) -> float:
    """Get average CTR for a platform from summary data."""
    return float(platform_data.get("avg_ctr", 0.0))


def _check_high_cpc_outlier(campaigns: list, platform_avg_cpc: float, platform: str) -> list:
    """Rule a: High CPC outlier — flag campaigns where CPC > 2x platform average.

    Args:
        campaigns: List of campaign dicts from platform summary.
        platform_avg_cpc: Average CPC for the platform.
        platform: 'meta' or 'google'

    Returns:
        List of suggestion dicts.
    """
    suggestions = []
    if platform_avg_cpc <= 0:
        logger.debug(f"Skipping high CPC outlier check for {platform}: platform avg CPC is 0")
        return suggestions

    for c in campaigns:
        if c.get("impressions", 0) == 0:
            continue
        cpc = float(c.get("cpc", 0.0))
        if cpc <= 0:
            continue
        ratio = _safe_div(cpc, platform_avg_cpc)
        if ratio > 2.0:
            logger.debug(f"High CPC outlier: {c.get('name')} CPC={cpc} vs avg={platform_avg_cpc}")
            suggestions.append({
                "priority": "high",
                "type": "investigate",
                "platform": platform,
                "campaign_name": c.get("name", "Unknown"),
                "campaign_id": str(c.get("id", "")),
                "reason": f"CPC of ${cpc:.2f} is {ratio:.1f}x the platform average of ${platform_avg_cpc:.2f}.",
                "metric_detail": f"Campaign CPC: ${cpc:.2f} | Platform avg: ${platform_avg_cpc:.2f} | Ratio: {ratio:.1f}x",
                "suggested_action": "Review targeting and ad creative for this campaign.",
            })
    return suggestions


def _check_zero_conversions_with_spend(campaigns: list, platform: str, days: int) -> list:
    """Rule b: Zero conversions with significant spend — campaigns that spent >$10 but got 0 conversions.

    Args:
        campaigns: List of campaign dicts.
        platform: 'meta' or 'google'
        days: Analysis period.

    Returns:
        List of suggestion dicts.
    """
    suggestions = []
    for c in campaigns:
        spend = float(c.get("spend", 0.0))
        conversions = int(c.get("conversions", 0))
        if spend > 10.0 and conversions == 0:
            logger.debug(f"Zero conversions with spend: {c.get('name')} spend={spend}")
            suggestions.append({
                "priority": "high",
                "type": "pause",
                "platform": platform,
                "campaign_name": c.get("name", "Unknown"),
                "campaign_id": str(c.get("id", "")),
                "reason": f"Spent ${spend:.2f} with zero conversions over {days} days.",
                "metric_detail": f"Total spend: ${spend:.2f} | Conversions: 0 | Period: {days}d",
                "suggested_action": "Consider pausing this campaign or revising the conversion setup.",
            })
    return suggestions


def _check_low_ctr(campaigns: list, platform: str) -> list:
    """Rule c: Low CTR — flag campaigns below platform-specific CTR thresholds.

    Meta threshold: 0.5%, Google threshold: 1.0%.

    Args:
        campaigns: List of campaign dicts.
        platform: 'meta' or 'google'

    Returns:
        List of suggestion dicts.
    """
    suggestions = []
    # CTR in these campaigns is stored as a decimal (e.g., 0.01 = 1%) for Meta
    # but we need to check the actual values from the platform data
    threshold_pct = 0.5 if platform == "meta" else 1.0

    for c in campaigns:
        impressions = int(c.get("impressions", 0))
        if impressions == 0:
            continue
        # CTR may be stored as decimal (0.01 = 1%) or percentage (1.0 = 1%)
        # cross_platform.py stores as ctr rounded to 4 decimals from the raw API value
        # Meta returns CTR as a percentage string (e.g. "1.2345"), we parse to float
        # Google returns as a decimal (e.g. 0.012345)
        # cross_platform.py normalizes: Meta avg_ctr from account API (already %)
        # For campaigns: Meta ctr = float percentage (e.g. 1.5 = 1.5%)
        # Google: ctr = decimal (0.015 = 1.5%)
        raw_ctr = float(c.get("ctr", 0.0))
        # Normalize to percentage
        if platform == "meta":
            ctr_pct = raw_ctr  # Meta already returns as % value
        else:
            # Google returns as decimal fraction (0.015 = 1.5%)
            ctr_pct = raw_ctr * 100.0

        if ctr_pct <= 0:
            continue

        if ctr_pct < threshold_pct:
            logger.debug(f"Low CTR: {c.get('name')} CTR={ctr_pct:.2f}% threshold={threshold_pct}%")
            suggestions.append({
                "priority": "medium",
                "type": "investigate",
                "platform": platform,
                "campaign_name": c.get("name", "Unknown"),
                "campaign_id": str(c.get("id", "")),
                "reason": f"CTR of {ctr_pct:.2f}% is below the {threshold_pct}% threshold.",
                "metric_detail": f"CTR: {ctr_pct:.2f}% | Threshold: {threshold_pct}% | Impressions: {impressions:,}",
                "suggested_action": "Test new ad copy or adjust audience targeting.",
            })
    return suggestions


def _check_strong_performer_underfunded(
    campaigns: list,
    platform_avg_cpc: float,
    platform_avg_ctr: float,
    platform: str,
) -> list:
    """Rule d: Strong performer underfunded — good CPC, good CTR, has conversions.

    Args:
        campaigns: List of campaign dicts.
        platform_avg_cpc: Platform average CPC.
        platform_avg_ctr: Platform average CTR (same unit as campaign ctr field).
        platform: 'meta' or 'google'

    Returns:
        List of suggestion dicts.
    """
    suggestions = []
    if platform_avg_cpc <= 0 or platform_avg_ctr <= 0:
        return suggestions

    for c in campaigns:
        if int(c.get("impressions", 0)) == 0:
            continue
        cpc = float(c.get("cpc", 0.0))
        raw_ctr = float(c.get("ctr", 0.0))
        conversions = int(c.get("conversions", 0))

        if cpc <= 0 or raw_ctr <= 0 or conversions == 0:
            continue

        # Both CPC below average AND CTR above average
        if cpc < platform_avg_cpc and raw_ctr > platform_avg_ctr:
            if platform == "meta":
                ctr_pct = raw_ctr
                avg_ctr_pct = platform_avg_ctr
            else:
                ctr_pct = raw_ctr * 100.0
                avg_ctr_pct = platform_avg_ctr * 100.0

            logger.debug(f"Strong performer: {c.get('name')} CPC={cpc} avg={platform_avg_cpc} CTR={ctr_pct:.2f}%")
            suggestions.append({
                "priority": "medium",
                "type": "budget_increase",
                "platform": platform,
                "campaign_name": c.get("name", "Unknown"),
                "campaign_id": str(c.get("id", "")),
                "reason": (
                    f"This campaign is outperforming on CPC (${cpc:.2f} vs avg ${platform_avg_cpc:.2f}) "
                    f"and CTR ({ctr_pct:.2f}% vs avg {avg_ctr_pct:.2f}%)."
                ),
                "metric_detail": (
                    f"CPC: ${cpc:.2f} (avg: ${platform_avg_cpc:.2f}) | "
                    f"CTR: {ctr_pct:.2f}% (avg: {avg_ctr_pct:.2f}%) | "
                    f"Conversions: {conversions}"
                ),
                "suggested_action": "Consider increasing the daily budget to capture more of this high-performing traffic.",
            })
    return suggestions


def _check_paused_with_good_metrics(
    campaigns: list,
    platform_avg_cpc: float,
    platform_avg_ctr: float,
    platform: str,
) -> list:
    """Rule e: Paused campaign with good historical metrics.

    Args:
        campaigns: List of campaign dicts.
        platform_avg_cpc: Platform average CPC.
        platform_avg_ctr: Platform average CTR (same unit as campaign ctr field).
        platform: 'meta' or 'google'

    Returns:
        List of suggestion dicts.
    """
    suggestions = []
    if platform_avg_cpc <= 0 or platform_avg_ctr <= 0:
        return suggestions

    for c in campaigns:
        status = str(c.get("status", "")).upper()
        if "PAUSE" not in status:
            continue

        cpc = float(c.get("cpc", 0.0))
        raw_ctr = float(c.get("ctr", 0.0))

        if cpc <= 0 or raw_ctr <= 0:
            continue

        if raw_ctr > platform_avg_ctr and cpc < platform_avg_cpc:
            if platform == "meta":
                ctr_pct = raw_ctr
                avg_ctr_pct = platform_avg_ctr
            else:
                ctr_pct = raw_ctr * 100.0
                avg_ctr_pct = platform_avg_ctr * 100.0

            logger.debug(f"Paused strong performer: {c.get('name')}")
            suggestions.append({
                "priority": "low",
                "type": "reactivate",
                "platform": platform,
                "campaign_name": c.get("name", "Unknown"),
                "campaign_id": str(c.get("id", "")),
                "reason": "This paused campaign had strong metrics when active.",
                "metric_detail": (
                    f"CPC: ${cpc:.2f} (avg: ${platform_avg_cpc:.2f}) | "
                    f"CTR: {ctr_pct:.2f}% (avg: {avg_ctr_pct:.2f}%) | Status: PAUSED"
                ),
                "suggested_action": "Consider reactivating this campaign.",
            })
    return suggestions


def _check_budget_imbalance(
    meta_data: dict,
    google_data: dict,
) -> list:
    """Rule f: Budget imbalance across platforms.

    If one platform accounts for >80% of total spend but <50% of total conversions, flag it.

    Args:
        meta_data: Meta platform summary dict.
        google_data: Google platform summary dict.

    Returns:
        List of suggestion dicts (0 or 1).
    """
    suggestions = []
    meta_spend = float(meta_data.get("total_spend", 0.0))
    google_spend = float(google_data.get("total_spend", 0.0))
    meta_conversions = int(meta_data.get("total_conversions", 0))
    google_conversions = int(google_data.get("total_conversions", 0))

    total_spend = meta_spend + google_spend
    total_conversions = meta_conversions + google_conversions

    if total_spend <= 0:
        return suggestions

    meta_spend_pct = _safe_div(meta_spend, total_spend) * 100
    google_spend_pct = _safe_div(google_spend, total_spend) * 100

    if total_conversions > 0:
        meta_conv_pct = _safe_div(meta_conversions, total_conversions) * 100
        google_conv_pct = _safe_div(google_conversions, total_conversions) * 100
    else:
        meta_conv_pct = 0.0
        google_conv_pct = 0.0

    # Check if Meta dominates spend but underperforms on conversions
    if meta_spend_pct > 80 and meta_conv_pct < 50:
        logger.debug(f"Budget imbalance: Meta has {meta_spend_pct:.0f}% of spend but {meta_conv_pct:.0f}% of conversions")
        suggestions.append({
            "priority": "medium",
            "type": "cross_platform",
            "platform": "cross_platform",
            "campaign_name": "Cross-platform",
            "campaign_id": "",
            "reason": f"Meta accounts for {meta_spend_pct:.0f}% of spend but only {meta_conv_pct:.0f}% of conversions.",
            "metric_detail": (
                f"Meta: ${meta_spend:.2f} spend ({meta_spend_pct:.0f}%), {meta_conversions} conversions ({meta_conv_pct:.0f}%) | "
                f"Google: ${google_spend:.2f} spend ({google_spend_pct:.0f}%), {google_conversions} conversions ({google_conv_pct:.0f}%)"
            ),
            "suggested_action": "Consider shifting budget from Meta to Google.",
        })
    # Check if Google dominates spend but underperforms on conversions
    elif google_spend_pct > 80 and google_conv_pct < 50:
        logger.debug(f"Budget imbalance: Google has {google_spend_pct:.0f}% of spend but {google_conv_pct:.0f}% of conversions")
        suggestions.append({
            "priority": "medium",
            "type": "cross_platform",
            "platform": "cross_platform",
            "campaign_name": "Cross-platform",
            "campaign_id": "",
            "reason": f"Google accounts for {google_spend_pct:.0f}% of spend but only {google_conv_pct:.0f}% of conversions.",
            "metric_detail": (
                f"Google: ${google_spend:.2f} spend ({google_spend_pct:.0f}%), {google_conversions} conversions ({google_conv_pct:.0f}%) | "
                f"Meta: ${meta_spend:.2f} spend ({meta_spend_pct:.0f}%), {meta_conversions} conversions ({meta_conv_pct:.0f}%)"
            ),
            "suggested_action": "Consider shifting budget from Google to Meta.",
        })

    return suggestions


def _check_declining_performance(
    summary_7: dict,
    summary_14: dict,
    platform: str,
) -> list:
    """Rule g: Declining performance — compare recent 7d vs prior 7d (within 14d window).

    Only triggered when days >= 14.

    Args:
        summary_7: Cross-platform summary for last 7 days.
        summary_14: Cross-platform summary for last 14 days.
        platform: 'meta' or 'google'

    Returns:
        List of suggestion dicts.
    """
    suggestions = []

    platform_7 = summary_7.get(platform, {})
    platform_14 = summary_14.get(platform, {})

    # If either platform errored out, skip
    if platform_7.get("error") or platform_14.get("error"):
        return suggestions

    campaigns_7 = {c["name"]: c for c in platform_7.get("campaigns", [])}
    campaigns_14 = {c["name"]: c for c in platform_14.get("campaigns", [])}

    for name, c7 in campaigns_7.items():
        if name not in campaigns_14:
            continue

        c14 = campaigns_14[name]
        cpc_7 = float(c7.get("cpc", 0.0))
        cpc_14 = float(c14.get("cpc", 0.0))
        ctr_7 = float(c7.get("ctr", 0.0))
        ctr_14 = float(c14.get("ctr", 0.0))

        if c7.get("impressions", 0) == 0:
            continue

        # CPC increased by >30%
        if cpc_14 > 0 and cpc_7 > cpc_14 * 1.3:
            pct_increase = ((cpc_7 - cpc_14) / cpc_14) * 100
            logger.debug(f"Declining performance (CPC): {name} CPC 7d={cpc_7} 14d={cpc_14}")
            suggestions.append({
                "priority": "high",
                "type": "investigate",
                "platform": platform,
                "campaign_name": name,
                "campaign_id": str(c7.get("id", "")),
                "reason": f"CPC has increased {pct_increase:.0f}% in the last 7 days vs the prior 7-day period.",
                "metric_detail": f"Recent 7d CPC: ${cpc_7:.2f} | Prior 7d CPC: ${cpc_14:.2f} | Change: +{pct_increase:.0f}%",
                "suggested_action": "Review recent targeting changes, competition, and ad creative performance.",
            })

        # CTR decreased by >30%
        elif ctr_14 > 0 and ctr_7 < ctr_14 * 0.7:
            pct_decrease = ((ctr_14 - ctr_7) / ctr_14) * 100
            logger.debug(f"Declining performance (CTR): {name} CTR 7d={ctr_7} 14d={ctr_14}")
            suggestions.append({
                "priority": "high",
                "type": "investigate",
                "platform": platform,
                "campaign_name": name,
                "campaign_id": str(c7.get("id", "")),
                "reason": f"CTR has dropped {pct_decrease:.0f}% in the last 7 days vs the prior 7-day period.",
                "metric_detail": f"Recent 7d CTR: {ctr_7:.4f} | Prior 7d CTR: {ctr_14:.4f} | Change: -{pct_decrease:.0f}%",
                "suggested_action": "Review recent targeting changes, competition, and ad creative performance.",
            })

    return suggestions


def generate_suggestions(days: int = 7) -> dict:
    """Generate prioritized ad optimization suggestions from live campaign data.

    Analyzes Meta and Google Ads campaigns using rule-based heuristics to identify
    underperformers, budget reallocation opportunities, and optimization actions.

    Args:
        days: Number of days of data to analyze (default: 7).
              Use 14 or 30 for trend detection (enables Rule g: declining performance).

    Returns:
        dict with keys:
            period_days: int — analysis window
            suggestions: list of suggestion dicts, sorted by priority (high first)
            summary: str — plain English overview for Alfred to relay
    """
    logger.debug(f"Generating ad optimization suggestions for last {days} days")

    # Fetch cross-platform summary
    summary = get_cross_platform_summary(days)

    meta_data = summary.get("meta", {})
    google_data = summary.get("google", {})

    meta_error = meta_data.get("error")
    google_error = google_data.get("error")

    meta_campaigns = meta_data.get("campaigns", []) if not meta_error else []
    google_campaigns = google_data.get("campaigns", []) if not google_error else []

    meta_avg_cpc = _platform_avg_cpc(meta_data)
    meta_avg_ctr = _platform_avg_ctr(meta_data)
    google_avg_cpc = _platform_avg_cpc(google_data)
    google_avg_ctr = _platform_avg_ctr(google_data)

    all_suggestions = []

    # --- Rule a: High CPC outlier ---
    if meta_campaigns:
        logger.debug("Running Rule a (high CPC outlier) for Meta")
        all_suggestions.extend(_check_high_cpc_outlier(meta_campaigns, meta_avg_cpc, "meta"))
    if google_campaigns:
        logger.debug("Running Rule a (high CPC outlier) for Google")
        all_suggestions.extend(_check_high_cpc_outlier(google_campaigns, google_avg_cpc, "google"))

    # --- Rule b: Zero conversions with significant spend ---
    if meta_campaigns:
        logger.debug("Running Rule b (zero conversions) for Meta")
        all_suggestions.extend(_check_zero_conversions_with_spend(meta_campaigns, "meta", days))
    if google_campaigns:
        logger.debug("Running Rule b (zero conversions) for Google")
        all_suggestions.extend(_check_zero_conversions_with_spend(google_campaigns, "google", days))

    # --- Rule c: Low CTR ---
    if meta_campaigns:
        logger.debug("Running Rule c (low CTR) for Meta")
        all_suggestions.extend(_check_low_ctr(meta_campaigns, "meta"))
    if google_campaigns:
        logger.debug("Running Rule c (low CTR) for Google")
        all_suggestions.extend(_check_low_ctr(google_campaigns, "google"))

    # --- Rule d: Strong performer underfunded ---
    if meta_campaigns:
        logger.debug("Running Rule d (strong performer) for Meta")
        all_suggestions.extend(
            _check_strong_performer_underfunded(meta_campaigns, meta_avg_cpc, meta_avg_ctr, "meta")
        )
    if google_campaigns:
        logger.debug("Running Rule d (strong performer) for Google")
        all_suggestions.extend(
            _check_strong_performer_underfunded(google_campaigns, google_avg_cpc, google_avg_ctr, "google")
        )

    # --- Rule e: Paused campaign with good historical metrics ---
    if meta_campaigns:
        logger.debug("Running Rule e (paused good metrics) for Meta")
        all_suggestions.extend(
            _check_paused_with_good_metrics(meta_campaigns, meta_avg_cpc, meta_avg_ctr, "meta")
        )
    if google_campaigns:
        logger.debug("Running Rule e (paused good metrics) for Google")
        all_suggestions.extend(
            _check_paused_with_good_metrics(google_campaigns, google_avg_cpc, google_avg_ctr, "google")
        )

    # --- Rule f: Budget imbalance across platforms ---
    if not meta_error and not google_error:
        logger.debug("Running Rule f (budget imbalance)")
        all_suggestions.extend(_check_budget_imbalance(meta_data, google_data))

    # --- Rule g: Declining performance (only if days >= 14) ---
    if days >= 14:
        logger.debug(f"Running Rule g (declining performance) — fetching 7-day summary for comparison")
        try:
            summary_7 = get_cross_platform_summary(7)
            for platform in ("meta", "google"):
                all_suggestions.extend(
                    _check_declining_performance(summary_7, summary, platform)
                )
        except Exception as exc:
            logger.warning(f"Rule g (declining performance) skipped due to error: {exc}")
    else:
        logger.debug(f"Skipping Rule g (declining performance) — days={days} < 14")

    # Sort by priority: high first, then medium, then low
    all_suggestions.sort(key=lambda s: _PRIORITY_ORDER.get(s.get("priority", "low"), 2))

    # Generate summary string
    high_count = sum(1 for s in all_suggestions if s.get("priority") == "high")
    medium_count = sum(1 for s in all_suggestions if s.get("priority") == "medium")
    low_count = sum(1 for s in all_suggestions if s.get("priority") == "low")

    if not all_suggestions:
        summary_text = "All campaigns are performing within normal parameters. No immediate action needed."
    else:
        parts = []
        if high_count:
            parts.append(f"{high_count} high-priority")
        if medium_count:
            parts.append(f"{medium_count} medium-priority")
        if low_count:
            parts.append(f"{low_count} low-priority")
        summary_text = f"Found {', '.join(parts)} optimization {'opportunity' if len(all_suggestions) == 1 else 'opportunities'} across Meta and Google campaigns."

    logger.debug(
        f"Suggestions generated: {high_count} high, {medium_count} medium, {low_count} low "
        f"(total: {len(all_suggestions)})"
    )

    return {
        "period_days": days,
        "suggestions": all_suggestions,
        "summary": summary_text,
    }
