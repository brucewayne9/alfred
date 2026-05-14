# tests/core/seo/test_registry.py
import pytest
from core.seo.db import SessionLocal
from core.seo.sites.registry import (
    register_site, get_site_by_slug, list_sites, update_site, deactivate_site
)


def test_register_and_fetch_site():
    site = register_site(
        slug="roen-test-22",
        domain="roen22.invalid",
        display_name="Roen 22",
        wp_rest_url="https://roen22.invalid/wp-json",
    )
    assert site.id is not None
    fetched = get_site_by_slug("roen-test-22")
    assert fetched.display_name == "Roen 22"
    deactivate_site("roen-test-22")  # cleanup-ish
    fetched = get_site_by_slug("roen-test-22")
    assert fetched.status == "inactive"


def test_register_rejects_duplicate_slug():
    register_site(slug="dup-test", domain="x.invalid", display_name="X", wp_rest_url="https://x")
    with pytest.raises(ValueError):
        register_site(slug="dup-test", domain="y.invalid", display_name="Y", wp_rest_url="https://y")
    deactivate_site("dup-test")


def test_list_sites_only_active():
    register_site(slug="active-22", domain="a.invalid", display_name="A", wp_rest_url="https://a")
    register_site(slug="inactive-22", domain="b.invalid", display_name="B", wp_rest_url="https://b")
    deactivate_site("inactive-22")
    slugs = {s.slug for s in list_sites()}
    assert "active-22" in slugs
    assert "inactive-22" not in slugs
    deactivate_site("active-22")
