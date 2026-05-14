# core/seo/api_clients/gsc_client.py
"""Thin wrapper around Search Console API."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from googleapiclient.discovery import build

from integrations.google_seo import get_credentials

logger = logging.getLogger(__name__)


def get_client():
    """Return an authenticated Search Console API client."""
    creds = get_credentials()
    return build("searchconsole", "v1", credentials=creds, cache_discovery=False)


def query_analytics(client, property_uri: str, start: dt.date, end: dt.date, row_limit: int = 1000) -> dict:
    """Run a search analytics query for date range, dimensioned by query."""
    body = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dimensions": ["query"],
        "rowLimit": row_limit,
    }
    return client.searchanalytics().query(siteUrl=property_uri, body=body).execute()


def list_top_linking_sites(client, property_uri: str, days_back: int = 90) -> list[dict]:
    """GSC links report — top linking external sites."""
    # GSC links report uses sites().listAllExternalLinks (legacy) or the data
    # is browseable via property metadata only. Practical alternative: scrape
    # via the searchanalytics with a different report type isn't supported.
    # For Phase 1 we use a simpler pull: backlinks come from a CSV export
    # endpoint or are populated by a manual export. Returns [] for now;
    # task 30 wires up the data path properly.
    return []
