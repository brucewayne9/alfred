# tests/core/seo/test_ga4_ingest.py
import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoPage, SeoSite
from core.seo.ingest.ga4 import sync_site_for_date
from core.seo.sites.registry import register_site, deactivate_site


FIXTURE = json.loads(Path("tests/fixtures/ga4_response.json").read_text())


def _purge_site(slug: str) -> None:
    """Hard-delete any prior fixture remnants (deactivate_site is soft-only)."""
    with SessionLocal() as s:
        existing = s.query(SeoSite).filter_by(slug=slug).all()
        for site in existing:
            s.query(SeoPage).filter_by(site_id=site.id).delete()
            s.delete(site)
        s.commit()


@pytest.fixture
def roen_site():
    _purge_site("roen-ga4-test")
    site = register_site(
        slug="roen-ga4-test", domain="roen.invalid", display_name="Roen GA4",
        wp_rest_url="https://x/wp-json", ga4_property_id="123456789",
    )
    yield site
    deactivate_site("roen-ga4-test")
    _purge_site("roen-ga4-test")


def test_sync_writes_page_rows(roen_site):
    fake_client = MagicMock()
    # NOTE: plan's test patched fake_client.run_report.return_value = FIXTURE, but
    # run_page_organic_report iterates response.rows (a proto, not a dict). We patch
    # the report function directly to inject the already-parsed dict shape.
    with patch("core.seo.ingest.ga4._build_client", return_value=fake_client), \
         patch("core.seo.ingest.ga4.run_page_organic_report", return_value=FIXTURE):
        n = sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))
    assert n == 3
    with SessionLocal() as s:
        rows = s.query(SeoPage).filter_by(site_id=roen_site.id).all()
        urls = {r.url for r in rows}
        assert "https://roen.invalid/product/red-bead-toggle-necklace/" in urls
