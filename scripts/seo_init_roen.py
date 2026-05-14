#!/usr/bin/env python3
"""Register Roen as Site #1 in the SEO orchestrator.

Idempotent — safe to re-run. Reads WP app password from env or prompts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/aialfred/alfred")

from core.seo.sites.registry import get_site_by_slug, register_site, update_site

ROEN_SLUG = "roen"
ROEN_DOMAIN = "roenhandmade.com"
ROEN_WP_REST = "https://www.roenhandmade.com/wp-json"
ROEN_GSC_PROPERTY = "sc-domain:roenhandmade.com"  # adjust if URL-prefix-only verification
ROEN_BUSINESS_TYPE = "LocalBusiness"
ROEN_BRAND_PROFILE = "data/seo/sites/roen/brand.yaml"


def main() -> int:
    existing = get_site_by_slug(ROEN_SLUG)
    wp_password = os.environ.get("ROEN_WP_APP_PASSWORD", "")
    if not wp_password:
        print("WARN: ROEN_WP_APP_PASSWORD env not set — site registered without WP credentials.")
        print("      Set it later via update_site() once Mike generates the application password.")
    fields = dict(
        domain=ROEN_DOMAIN,
        display_name="Roen",
        wp_rest_url=ROEN_WP_REST,
        wp_username="alfred-seo",
        wp_app_password=wp_password or None,
        gsc_property=ROEN_GSC_PROPERTY,
        ga4_property_id=os.environ.get("ROEN_GA4_PROPERTY_ID", "") or None,
        brand_profile_path=ROEN_BRAND_PROFILE,
        business_type=ROEN_BUSINESS_TYPE,
    )
    if existing:
        update_site(ROEN_SLUG, **{k: v for k, v in fields.items() if v is not None})
        print(f"UPDATED site {ROEN_SLUG} (id={existing.id})")
    else:
        site = register_site(slug=ROEN_SLUG, **fields)
        print(f"REGISTERED site {ROEN_SLUG} (id={site.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
