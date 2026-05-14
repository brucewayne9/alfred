# tests/core/seo/test_backlinks_ingest.py
from unittest.mock import patch

import pytest

from core.seo.db import SessionLocal
from core.seo.models import SeoBacklink, SeoSite
from core.seo.ingest.backlinks import record_backlinks_for_site
from core.seo.sites.registry import register_site, deactivate_site


def _purge_site(slug: str) -> None:
    """Hard-delete any prior fixture remnants (deactivate_site is soft-only)."""
    with SessionLocal() as s:
        existing = s.query(SeoSite).filter_by(slug=slug).all()
        for site in existing:
            s.query(SeoBacklink).filter_by(site_id=site.id).delete()
            s.delete(site)
        s.commit()


@pytest.fixture
def roen_site():
    _purge_site("roen-bl-test")
    site = register_site(
        slug="roen-bl-test", domain="roen.invalid", display_name="Roen BL",
        wp_rest_url="https://x/wp-json", gsc_property="sc-domain:roen.invalid",
    )
    yield site
    deactivate_site("roen-bl-test")
    _purge_site("roen-bl-test")


def test_records_new_and_existing(roen_site):
    snapshot = [
        ("https://atlanta-mag.example.com/spring", "https://roen.invalid/", "Roen Atlanta studio"),
        ("https://crafts-blog.example.com/finds", "https://roen.invalid/products/", "handmade jewelry"),
    ]
    record_backlinks_for_site(roen_site.id, snapshot)
    record_backlinks_for_site(roen_site.id, snapshot)  # re-run, no duplicates
    with SessionLocal() as s:
        rows = s.query(SeoBacklink).filter_by(site_id=roen_site.id).all()
        assert len(rows) == 2
        assert all(r.lost_at is None for r in rows)


def test_marks_lost_when_missing_in_new_snapshot(roen_site):
    record_backlinks_for_site(roen_site.id, [
        ("https://a.example.com/", "https://roen.invalid/", "a"),
        ("https://b.example.com/", "https://roen.invalid/", "b"),
    ])
    record_backlinks_for_site(roen_site.id, [
        ("https://a.example.com/", "https://roen.invalid/", "a"),
    ])
    with SessionLocal() as s:
        b = s.query(SeoBacklink).filter_by(site_id=roen_site.id, source_url="https://b.example.com/").one()
        assert b.lost_at is not None
