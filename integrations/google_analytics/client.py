"""Google Analytics Data API client for Alfred."""

import logging
from datetime import datetime, timedelta
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    RunRealtimeReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy,
)
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# GA4 Property configurations
GA_PROPERTIES = {
    "rucktalk": {"id": "408395502", "name": "RuckTalk"},
    "nightlife": {"id": "442072096", "name": "NightLife Functions"},
    "rodwave": {"id": "456717749", "name": "Rod Wave"},
    "rod": {"id": "456717749", "name": "Rod Wave"},
    "lenssniper": {"id": "472694627", "name": "LensSniper"},
    "lens": {"id": "472694627", "name": "LensSniper"},
    "loovacast": {"id": "475653248", "name": "LoovaCast"},
    "loova": {"id": "475653248", "name": "LoovaCast"},
    "lumabot": {"id": "518920226", "name": "Luma Bot"},
    "luma": {"id": "518920226", "name": "Luma Bot"},
    "myhands": {"id": "521064731", "name": "My Hands Car Wash"},
    "carwash": {"id": "521064731", "name": "My Hands Car Wash"},
    "agentertainment": {"id": "389389502", "name": "AG Entertainment"},
    "ag": {"id": "389389502", "name": "AG Entertainment"},
}

CREDENTIALS_PATH = "/home/aialfred/alfred/config/google_analytics_credentials.json"


class GoogleAnalyticsClient:
    """Client for Google Analytics Data API."""

    def __init__(self):
        self._client = None
        self._credentials = None

    def _get_client(self) -> BetaAnalyticsDataClient:
        """Get or create authenticated client."""
        if self._client is None:
            self._credentials = service_account.Credentials.from_service_account_file(
                CREDENTIALS_PATH,
                scopes=["https://www.googleapis.com/auth/analytics.readonly"],
            )
            self._client = BetaAnalyticsDataClient(credentials=self._credentials)
        return self._client

    def _resolve_property(self, property_name: str) -> tuple[str, str]:
        """Resolve property name/alias to ID and display name."""
        key = property_name.lower().replace(" ", "").replace("-", "").replace("_", "")

        # Direct ID lookup
        if key.isdigit():
            for prop in GA_PROPERTIES.values():
                if prop["id"] == key:
                    return prop["id"], prop["name"]
            return key, f"Property {key}"

        # Name/alias lookup
        if key in GA_PROPERTIES:
            return GA_PROPERTIES[key]["id"], GA_PROPERTIES[key]["name"]

        # Partial match
        for alias, prop in GA_PROPERTIES.items():
            if key in alias or key in prop["name"].lower().replace(" ", ""):
                return prop["id"], prop["name"]

        raise ValueError(f"Unknown property: {property_name}. Available: {', '.join(p['name'] for p in GA_PROPERTIES.values())}")

    def _parse_period(self, period: str) -> tuple[str, str]:
        """Convert period string to start/end dates."""
        today = datetime.now()
        period_lower = period.lower().replace("_", "").replace("-", "").replace(" ", "")

        if "today" in period_lower:
            return today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "yesterday" in period_lower:
            yesterday = today - timedelta(days=1)
            return yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y-%m-%d")
        elif "7" in period_lower or "week" in period_lower:
            start = today - timedelta(days=7)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "14" in period_lower:
            start = today - timedelta(days=14)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "30" in period_lower or "month" in period_lower:
            start = today - timedelta(days=30)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "90" in period_lower or "quarter" in period_lower:
            start = today - timedelta(days=90)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif "365" in period_lower or "year" in period_lower:
            start = today - timedelta(days=365)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        else:
            # Default to last 7 days
            start = today - timedelta(days=7)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    def list_properties(self) -> dict:
        """List all configured GA4 properties."""
        properties = []
        seen = set()
        for prop in GA_PROPERTIES.values():
            if prop["id"] not in seen:
                properties.append({"id": prop["id"], "name": prop["name"]})
                seen.add(prop["id"])
        return {"properties": properties, "count": len(properties)}

    def get_traffic_summary(self, property_name: str, period: str = "last_7_days") -> dict:
        """Get traffic summary for a property."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)
            start_date, end_date = self._parse_period(period)

            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                metrics=[
                    Metric(name="activeUsers"),
                    Metric(name="sessions"),
                    Metric(name="screenPageViews"),
                    Metric(name="bounceRate"),
                    Metric(name="averageSessionDuration"),
                    Metric(name="newUsers"),
                ],
            )
            response = client.run_report(request)

            if response.rows:
                row = response.rows[0]
                return {
                    "property": display_name,
                    "property_id": property_id,
                    "period": f"{start_date} to {end_date}",
                    "active_users": int(row.metric_values[0].value),
                    "sessions": int(row.metric_values[1].value),
                    "page_views": int(row.metric_values[2].value),
                    "bounce_rate": f"{float(row.metric_values[3].value) * 100:.1f}%",
                    "avg_session_duration": f"{float(row.metric_values[4].value):.0f}s",
                    "new_users": int(row.metric_values[5].value),
                }
            return {"property": display_name, "error": "No data available"}
        except Exception as e:
            logger.error(f"GA traffic summary error: {e}")
            return {"error": str(e)}

    def get_realtime(self, property_name: str) -> dict:
        """Get real-time active users for a property."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)

            request = RunRealtimeReportRequest(
                property=f"properties/{property_id}",
                metrics=[Metric(name="activeUsers")],
            )
            response = client.run_realtime_report(request)

            active_users = 0
            if response.rows:
                active_users = int(response.rows[0].metric_values[0].value)

            return {
                "property": display_name,
                "property_id": property_id,
                "active_users_now": active_users,
            }
        except Exception as e:
            logger.error(f"GA realtime error: {e}")
            return {"error": str(e)}

    def get_traffic_sources(self, property_name: str, period: str = "last_7_days", limit: int = 10) -> dict:
        """Get traffic sources breakdown."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)
            start_date, end_date = self._parse_period(period)

            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=[Dimension(name="sessionSource")],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="activeUsers"),
                ],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"), desc=True)],
                limit=limit,
            )
            response = client.run_report(request)

            sources = []
            for row in response.rows:
                sources.append({
                    "source": row.dimension_values[0].value or "(direct)",
                    "sessions": int(row.metric_values[0].value),
                    "users": int(row.metric_values[1].value),
                })

            return {
                "property": display_name,
                "period": f"{start_date} to {end_date}",
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"GA sources error: {e}")
            return {"error": str(e)}

    def get_top_pages(self, property_name: str, period: str = "last_7_days", limit: int = 10) -> dict:
        """Get top pages by views."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)
            start_date, end_date = self._parse_period(period)

            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=[Dimension(name="pagePath")],
                metrics=[
                    Metric(name="screenPageViews"),
                    Metric(name="activeUsers"),
                    Metric(name="averageSessionDuration"),
                ],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)],
                limit=limit,
            )
            response = client.run_report(request)

            pages = []
            for row in response.rows:
                pages.append({
                    "page": row.dimension_values[0].value,
                    "views": int(row.metric_values[0].value),
                    "users": int(row.metric_values[1].value),
                    "avg_time": f"{float(row.metric_values[2].value):.0f}s",
                })

            return {
                "property": display_name,
                "period": f"{start_date} to {end_date}",
                "pages": pages,
            }
        except Exception as e:
            logger.error(f"GA pages error: {e}")
            return {"error": str(e)}

    def get_devices(self, property_name: str, period: str = "last_7_days") -> dict:
        """Get device category breakdown."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)
            start_date, end_date = self._parse_period(period)

            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=[Dimension(name="deviceCategory")],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="activeUsers"),
                ],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"), desc=True)],
            )
            response = client.run_report(request)

            devices = []
            total_sessions = 0
            for row in response.rows:
                sessions = int(row.metric_values[0].value)
                total_sessions += sessions
                devices.append({
                    "device": row.dimension_values[0].value,
                    "sessions": sessions,
                    "users": int(row.metric_values[1].value),
                })

            # Add percentages
            for d in devices:
                d["percentage"] = f"{(d['sessions'] / total_sessions * 100):.1f}%" if total_sessions > 0 else "0%"

            return {
                "property": display_name,
                "period": f"{start_date} to {end_date}",
                "devices": devices,
            }
        except Exception as e:
            logger.error(f"GA devices error: {e}")
            return {"error": str(e)}

    def get_countries(self, property_name: str, period: str = "last_7_days", limit: int = 10) -> dict:
        """Get geographic breakdown by country."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)
            start_date, end_date = self._parse_period(period)

            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=[Dimension(name="country")],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="activeUsers"),
                ],
                order_bys=[OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"), desc=True)],
                limit=limit,
            )
            response = client.run_report(request)

            countries = []
            for row in response.rows:
                countries.append({
                    "country": row.dimension_values[0].value or "(not set)",
                    "sessions": int(row.metric_values[0].value),
                    "users": int(row.metric_values[1].value),
                })

            return {
                "property": display_name,
                "period": f"{start_date} to {end_date}",
                "countries": countries,
            }
        except Exception as e:
            logger.error(f"GA countries error: {e}")
            return {"error": str(e)}

    def get_daily_traffic(self, property_name: str, period: str = "last_30_days") -> dict:
        """Get daily traffic breakdown."""
        try:
            client = self._get_client()
            property_id, display_name = self._resolve_property(property_name)
            start_date, end_date = self._parse_period(period)

            request = RunReportRequest(
                property=f"properties/{property_id}",
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimensions=[Dimension(name="date")],
                metrics=[
                    Metric(name="activeUsers"),
                    Metric(name="sessions"),
                    Metric(name="screenPageViews"),
                ],
                order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name="date"), desc=False)],
            )
            response = client.run_report(request)

            daily = []
            for row in response.rows:
                date_str = row.dimension_values[0].value
                # Format date from YYYYMMDD to YYYY-MM-DD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                daily.append({
                    "date": formatted_date,
                    "users": int(row.metric_values[0].value),
                    "sessions": int(row.metric_values[1].value),
                    "page_views": int(row.metric_values[2].value),
                })

            return {
                "property": display_name,
                "period": f"{start_date} to {end_date}",
                "daily": daily,
            }
        except Exception as e:
            logger.error(f"GA daily error: {e}")
            return {"error": str(e)}

    def get_all_properties_summary(self, period: str = "last_7_days") -> dict:
        """Get summary for all configured properties."""
        summaries = []
        seen = set()

        for prop in GA_PROPERTIES.values():
            if prop["id"] not in seen:
                seen.add(prop["id"])
                summary = self.get_traffic_summary(prop["id"], period)
                if "error" not in summary:
                    summaries.append(summary)

        # Sort by active users descending
        summaries.sort(key=lambda x: x.get("active_users", 0), reverse=True)

        return {
            "period": period,
            "properties": summaries,
            "total_properties": len(summaries),
        }


# Singleton instance
ga_client = GoogleAnalyticsClient()
