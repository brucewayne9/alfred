# tests/core/seo/test_gsc_ingest.py
import datetime as dt
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoQuery, SeoSite
from core.seo.ingest.gsc import sync_site_for_date
from core.seo.sites.registry import register_site, deactivate_site


FIXTURE = json.loads(Path("tests/fixtures/gsc_response.json").read_text())


def _purge_site(slug: str) -> None:
    """Hard-delete any prior fixture remnants (deactivate_site is soft-only)."""
    with SessionLocal() as s:
        existing = s.query(SeoSite).filter_by(slug=slug).all()
        for site in existing:
            s.query(SeoQuery).filter_by(site_id=site.id).delete()
            s.delete(site)
        s.commit()


@pytest.fixture
def roen_site():
    _purge_site("roen-gsc-test")
    site = register_site(
        slug="roen-gsc-test", domain="roen.invalid", display_name="Roen GSC",
        wp_rest_url="https://x/wp-json", gsc_property="sc-domain:roen.invalid",
    )
    yield site
    deactivate_site("roen-gsc-test")
    _purge_site("roen-gsc-test")


def test_sync_site_writes_query_rows(roen_site):
    fake_client = MagicMock()
    fake_client.searchanalytics().query().execute.return_value = FIXTURE
    with patch("core.seo.ingest.gsc._build_client", return_value=fake_client):
        n = sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))
    assert n == 3
    with SessionLocal() as s:
        rows = s.query(SeoQuery).filter_by(site_id=roen_site.id).all()
        assert len(rows) == 3
        assert any(r.query == "evil eye bracelet meaning" for r in rows)


def test_sync_is_idempotent(roen_site):
    fake_client = MagicMock()
    fake_client.searchanalytics().query().execute.return_value = FIXTURE
    with patch("core.seo.ingest.gsc._build_client", return_value=fake_client):
        sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))
        sync_site_for_date(roen_site.id, dt.date(2026, 5, 14))  # re-run
    with SessionLocal() as s:
        rows = s.query(SeoQuery).filter_by(site_id=roen_site.id).all()
        assert len(rows) == 3  # not 6 — UNIQUE (site, query, date) enforced
