"""Site URL inventory — fetch the list of pages/products on a site so the
Keyword Engine can pick a target URL for each candidate keyword.

For WordPress sites, we hit the public WP REST API (no auth needed for read-
only on public posts/pages/products). We DO NOT crawl arbitrary HTML — the
DataForSEO On-Page audit handles that. This is purely "what URLs does the
WordPress side know about".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# Per-page cap on how many of each post-type we pull. WP REST max is 100.
WP_REST_PER_PAGE = 100
# How many pages of each type we'll walk (so total cap = PER_PAGE * MAX_PAGES).
WP_REST_MAX_PAGES = 5


@dataclass
class SiteUrl:
    url: str
    title: str
    post_type: str  # post | page | product | category | tag
    excerpt: str = ""
    slug: str = ""


def _fetch_post_type(
    wp_base: str,
    rest_namespace: str,
    post_type: str,
    *,
    timeout: float = 30.0,
) -> list[SiteUrl]:
    """Pull a single post-type from WP REST API, paginated."""
    out: list[SiteUrl] = []
    url = f"{wp_base.rstrip('/')}/{rest_namespace}/{post_type}"
    for page in range(1, WP_REST_MAX_PAGES + 1):
        try:
            resp = httpx.get(
                url,
                params={"per_page": WP_REST_PER_PAGE, "page": page, "_fields": "id,link,title,excerpt,slug,type"},
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            log.warning("url_inventory %s page=%d HTTP error: %s", post_type, page, e)
            return out

        if resp.status_code == 400 and page > 1:
            # WP returns 400 "rest_post_invalid_page_number" past the last page; that's normal.
            break
        if resp.status_code != 200:
            log.warning("url_inventory %s page=%d status=%d body=%s",
                        post_type, page, resp.status_code, resp.text[:200])
            break

        items = resp.json() or []
        if not items:
            break
        for item in items:
            title_obj = item.get("title") or {}
            excerpt_obj = item.get("excerpt") or {}
            out.append(SiteUrl(
                url=item.get("link", ""),
                title=(title_obj.get("rendered") or "").strip(),
                post_type=item.get("type", post_type),
                excerpt=(excerpt_obj.get("rendered") or "").strip(),
                slug=item.get("slug", ""),
            ))
        if len(items) < WP_REST_PER_PAGE:
            break  # last page

    return out


def fetch_wp_url_inventory(wp_rest_url: str) -> list[SiteUrl]:
    """Return all public posts, pages, and (where present) products for a WP site.

    `wp_rest_url` is the full root, e.g. https://www.roenhandmade.com/wp-json
    """
    base = wp_rest_url.rstrip("/")
    if not base.endswith("/wp-json"):
        # Tolerate either "https://x.com" or "https://x.com/wp-json"
        if "/wp-json" not in base:
            base = base + "/wp-json"
    inventory: list[SiteUrl] = []

    # Core post types + WooCommerce product CPT (when exposed via wp/v2/product,
    # which most stores do — wp/v2 has consistent field shape vs wc/store/v1).
    for ptype in ("posts", "pages", "product"):
        rows = _fetch_post_type(base, "wp/v2", ptype)
        if rows:
            log.info("url_inventory %s: %d urls", ptype, len(rows))
        else:
            log.info("url_inventory %s: 0 urls (endpoint may be hidden)", ptype)
        inventory.extend(rows)

    # Dedupe by URL
    seen: set[str] = set()
    unique: list[SiteUrl] = []
    for u in inventory:
        if u.url and u.url not in seen:
            seen.add(u.url)
            unique.append(u)
    log.info("url_inventory total unique urls: %d", len(unique))
    return unique
