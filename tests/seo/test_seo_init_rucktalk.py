"""Verify the rucktalk site registers (and idempotently re-registers) cleanly."""
import importlib
import os
import sys

sys.path.insert(0, "/home/aialfred/alfred")

from core.seo.sites.registry import get_site_by_slug


def test_rucktalk_registration_creates_or_updates_row(monkeypatch):
    monkeypatch.setenv("RUCKTALK_WP_APP_PASSWORD", "test-password-xxxx-xxxx-xxxx")
    monkeypatch.setenv("RUCKTALK_GA4_PROPERTY_ID", "123456789")

    mod = importlib.import_module("scripts.seo_init_rucktalk")
    rc = mod.main()
    assert rc == 0

    site = get_site_by_slug("rucktalk")
    assert site is not None
    assert site.domain == "rucktalk.com"
    assert site.gsc_property == "sc-domain:rucktalk.com"
    assert site.brand_profile_path.endswith("data/seo/sites/rucktalk/brand.yaml")
    assert site.business_type == "Organization"  # NOT LocalBusiness — RuckTalk is a media brand
