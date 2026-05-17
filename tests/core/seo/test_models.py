# tests/core/seo/test_models.py
import datetime as dt
import pytest
from core.seo.db import SessionLocal, Base, engine
from core.seo.models import SeoSite, SeoQuery

# SCOPED test slug — cleanup MUST filter to this slug only. An unfiltered
# delete in this fixture wiped production seo_* tables on 2026-05-16 because
# core.seo.db.SessionLocal is bound to the live alfred_main database.
TEST_SLUG = "roen-models-test"


@pytest.fixture(autouse=True)
def _tables():
    Base.metadata.create_all(engine, tables=[SeoSite.__table__, SeoQuery.__table__])
    # Pre-clean any leftover row from a previous aborted run, scoped to TEST_SLUG.
    with SessionLocal() as s:
        site = s.query(SeoSite).filter(SeoSite.slug == TEST_SLUG).one_or_none()
        if site:
            s.execute(SeoQuery.__table__.delete().where(SeoQuery.site_id == site.id))
            s.delete(site)
            s.commit()
    yield
    with SessionLocal() as s:
        site = s.query(SeoSite).filter(SeoSite.slug == TEST_SLUG).one_or_none()
        if site:
            s.execute(SeoQuery.__table__.delete().where(SeoQuery.site_id == site.id))
            s.delete(site)
            s.commit()


def test_create_site_and_query():
    with SessionLocal() as s:
        site = SeoSite(
            slug=TEST_SLUG, domain="roenhandmade-test.invalid",
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
        assert q.site.slug == TEST_SLUG
