#!/usr/bin/env python3
"""Register RuckTalk as Site #2 in the SEO orchestrator.

Idempotent — safe to re-run. Reads WP app password from env or skips it.
Mirrors scripts/seo_init_roen.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/aialfred/alfred")

from core.seo.sites.registry import get_site_by_slug, register_site, update_site

SLUG = "rucktalk"
DOMAIN = "rucktalk.com"
WP_REST = "https://rucktalk.com/wp-json"
GSC_PROPERTY = "sc-domain:rucktalk.com"
BUSINESS_TYPE = "Organization"  # podcast brand, not local business
BRAND_PROFILE = "data/seo/sites/rucktalk/brand.yaml"


def main() -> int:
    existing = get_site_by_slug(SLUG)
    wp_password = os.environ.get("RUCKTALK_WP_APP_PASSWORD", "")
    if not wp_password:
        print("WARN: RUCKTALK_WP_APP_PASSWORD env not set — site registered without WP credentials.")
        print("      Set it later via update_site() once Mike generates the application password.")
    fields = dict(
        domain=DOMAIN,
        display_name="RuckTalk",
        wp_rest_url=WP_REST,
        wp_username="alfred-seo",
        wp_app_password=wp_password or None,
        gsc_property=GSC_PROPERTY,
        ga4_property_id=os.environ.get("RUCKTALK_GA4_PROPERTY_ID", "") or None,
        brand_profile_path=BRAND_PROFILE,
        business_type=BUSINESS_TYPE,
    )
    if existing:
        update_site(SLUG, **{k: v for k, v in fields.items() if v is not None})
        print(f"UPDATED site {SLUG} (id={existing.id})")
    else:
        site = register_site(slug=SLUG, **fields)
        print(f"REGISTERED site {SLUG} (id={site.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
