"""Product-image index for the SEO content engine.

Pulls every WooCommerce product from a Roen-style site via the public Store
API and builds a lookup of {attachment_id → ProductImage}. Each blog post
draft picks unused images from this pool and renders them with a soft-CTA
caption that links back to the product page.

Index is cached on disk (6h TTL) so we don't hammer WC every generation.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

from core.seo.models import SeoSite

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 6 * 3600
CACHE_DIR = Path("/home/aialfred/alfred/data/seo/image_index")
STORE_API_PATH = "/wp-json/wc/store/v1/products"
DEFAULT_PER_PAGE = 100


@dataclass
class ProductImage:
    attachment_id: int       # WP media attachment ID
    src: str                 # full image URL
    thumbnail: str           # thumbnail URL
    alt: str                 # alt text (often empty on Roen)
    product_id: int          # owning WC product ID
    product_name: str        # e.g. "Red Bead Toggle Necklace"
    product_slug: str        # e.g. "red-bead-toggle-necklace"
    product_url: str         # full permalink

    @property
    def caption_markdown(self) -> str:
        return f"*Featured: [{self.product_name}]({self.product_url})*"


def _cache_path(site_slug: str) -> Path:
    return CACHE_DIR / f"{site_slug}_product_images.json"


def _read_cache(site_slug: str) -> Optional[list[ProductImage]]:
    p = _cache_path(site_slug)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if time.time() - data.get("cached_at", 0) > CACHE_TTL_SECONDS:
        return None
    return [ProductImage(**row) for row in data.get("images", [])]


def _write_cache(site_slug: str, images: list[ProductImage]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cached_at": int(time.time()),
        "site_slug": site_slug,
        "images": [asdict(img) for img in images],
    }
    _cache_path(site_slug).write_text(json.dumps(payload, indent=2))


def fetch_product_images(site: SeoSite, *, force_refresh: bool = False) -> list[ProductImage]:
    """Return every product image on the site. Cached 6h on disk."""
    if not force_refresh:
        cached = _read_cache(site.slug)
        if cached is not None:
            log.info("product image index: cache hit for %s (%d images)", site.slug, len(cached))
            return cached

    base = site.wp_rest_url.rstrip("/").removesuffix("/wp-json")
    if not base.endswith("/"):
        base = base + "/"
    url = f"{base.rstrip('/')}{STORE_API_PATH}"

    images: list[ProductImage] = []
    page = 1
    while True:
        r = requests.get(url, params={"per_page": DEFAULT_PER_PAGE, "page": page}, timeout=30)
        if r.status_code == 400:
            # WC returns 400 when paging past the end on some setups
            break
        r.raise_for_status()
        batch = r.json() or []
        if not batch:
            break
        for product in batch:
            for img in product.get("images") or []:
                if not img.get("id") or not img.get("src"):
                    continue
                images.append(ProductImage(
                    attachment_id=int(img["id"]),
                    src=img["src"],
                    thumbnail=img.get("thumbnail") or img["src"],
                    alt=img.get("alt") or product.get("name", ""),
                    product_id=int(product["id"]),
                    product_name=product.get("name", ""),
                    product_slug=product.get("slug", ""),
                    product_url=product.get("permalink", ""),
                ))
        total_pages = int(r.headers.get("X-WP-TotalPages", "1"))
        if page >= total_pages:
            break
        page += 1

    log.info("product image index: refreshed for %s (%d images across %d pages)",
             site.slug, len(images), page)
    _write_cache(site.slug, images)
    return images


def get_used_attachment_ids(site_id: int) -> set[int]:
    """Union of attachment_ids already attached to drafts for this site.

    Pulls from seo_pending + seo_decided so a freshly-enqueued draft's images
    don't get re-picked by the next brief while it's awaiting approval.
    """
    from sqlalchemy import select
    from core.seo.db import SessionLocal
    from core.seo.models import SeoDecided, SeoPending

    used: set[int] = set()
    with SessionLocal() as s:
        for model in (SeoPending, SeoDecided):
            q = select(model.body_payload).where(model.site_id == site_id)
            for row in s.scalars(q).all():
                if not row:
                    continue
                ids = (row.get("image_ids") or [])
                for v in ids:
                    try:
                        used.add(int(v))
                    except (TypeError, ValueError):
                        pass
                fid = row.get("featured_image_id")
                if fid:
                    try:
                        used.add(int(fid))
                    except (TypeError, ValueError):
                        pass
    return used
