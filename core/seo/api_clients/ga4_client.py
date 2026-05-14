# core/seo/api_clients/ga4_client.py
"""GA4 Data API wrapper."""
from __future__ import annotations

import datetime as dt
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange, Dimension, Metric, RunReportRequest,
)

from integrations.google_seo import get_credentials


def get_client() -> BetaAnalyticsDataClient:
    creds = get_credentials()
    return BetaAnalyticsDataClient(credentials=creds)


def run_page_organic_report(client: BetaAnalyticsDataClient, property_id: str, date: dt.date) -> dict:
    """Pull organic sessions + conversions by page path for one date."""
    req = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="pagePath")],
        metrics=[Metric(name="sessions"), Metric(name="conversions")],
        date_ranges=[DateRange(start_date=date.isoformat(), end_date=date.isoformat())],
        dimension_filter=None,  # GA4 reports default to all traffic; refine via channel grouping if needed
        limit=1000,
    )
    response = client.run_report(req)
    # Convert to plain dict for downstream code + test fixtures.
    rows = []
    for r in response.rows:
        rows.append({
            "dimensionValues": [{"value": dv.value} for dv in r.dimension_values],
            "metricValues":    [{"value": mv.value} for mv in r.metric_values],
        })
    return {"rows": rows}
