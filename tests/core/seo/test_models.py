# tests/core/seo/test_models.py
import datetime as dt
import pytest
from core.seo.db import SessionLocal, Base, engine
from core.seo.models import SeoSite, SeoQuery


@pytest.fixture(autouse=True)
def _tables():
    Base.metadata.create_all(engine, tables=[SeoSite.__table__, SeoQuery.__table__])
    yield
    # Best-effort cleanup
    with SessionLocal() as s:
        s.execute(SeoQuery.__table__.delete())
        s.execute(SeoSite.__table__.delete())
        s.commit()


def test_create_site_and_query():
    with SessionLocal() as s:
        site = SeoSite(
            slug="roen-test", domain="roenhandmade-test.invalid",
            display_name="Roen Test", wp_rest_url="https://x/wp-json",
        )
        s.add(site)
        s.commit()
        q = SeoQuery(
            site_id=site.id, query="evil eye bracelet",
            position=14.2, impressions=1247, clicks=47, ctr=0.0377,
            captured_at=dt.date(2026, 5, 14),
        )
        s.add(q)
        s.commit()
        assert q.id is not None
        assert q.site.slug == "roen-test"
