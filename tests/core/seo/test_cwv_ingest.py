# tests/core/seo/test_cwv_ingest.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoPage, SeoSite
from core.seo.ingest.cwv import sync_url
from core.seo.sites.registry import register_site, deactivate_site


FIXTURE = json.loads(Path("tests/fixtures/psi_response.json").read_text())


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
    _purge_site("roen-cwv-test")
    site = register_site(
        slug="roen-cwv-test", domain="roen.invalid", display_name="Roen CWV",
        wp_rest_url="https://x/wp-json",
    )
    yield site
    deactivate_site("roen-cwv-test")
    _purge_site("roen-cwv-test")


def test_sync_url_writes_cwv_metrics(roen_site, monkeypatch):
    from config.settings import settings as _settings
    monkeypatch.setattr(_settings, "seo_psi_api_key", "test-api-key")
    fake_resp = MagicMock()
    fake_resp.json.return_value = FIXTURE
    fake_resp.status_code = 200
    with patch("requests.get", return_value=fake_resp):
        sync_url(roen_site.id, "https://roen.invalid/")
    with SessionLocal() as s:
        page = s.query(SeoPage).filter_by(site_id=roen_site.id).first()
        assert page is not None
        assert page.cwv_lcp_ms == 2200
        assert float(page.cwv_cls) == 0.08
        assert page.cwv_inp_ms == 180
