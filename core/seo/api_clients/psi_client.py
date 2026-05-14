# core/seo/api_clients/psi_client.py
"""PageSpeed Insights API wrapper. Uses simple API key, not OAuth."""
from __future__ import annotations

import logging
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"


def get_cwv(url: str, strategy: str = "mobile") -> dict:
    """Returns the raw PSI response. Caller parses metrics."""
    api_key = settings.seo_psi_api_key
    if not api_key:
        raise RuntimeError("SEO_PSI_API_KEY not set in config/.env")
    params = {
        "url": url,
        "key": api_key,
        "strategy": strategy,
        "category": "PERFORMANCE",
    }
    r = requests.get(ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    return r.json()
