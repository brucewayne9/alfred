"""Rank tracking — pulls SERPs for the active keyword punch list and stores
positions over time. Reads seo_keywords for the target keyword set, writes
captures to seo_rankings_daily and refreshes the cached rank fields on
seo_keywords.

Multi-tenant from day 1: track_site(slug) is the public entry point.
"""
from core.seo.rankings.tracker import (
    RankResult,
    SiteRankReport,
    track_site,
)

__all__ = ["RankResult", "SiteRankReport", "track_site"]
