"""Pick + splice product images into a generated draft.

Mike's spec (2026-05-15): blog posts get a featured image at the top + 2
inline images at logical paragraph breaks. Each inline gets a soft-CTA
caption like *Featured: [Bracelet Name](product_url)* so the photo doubles
as a product discovery surface.

Image relevance is intentionally LOOSE — the image just needs to be on-brand
(every product photo on Roen is). We rotate through unused images so the
same shot doesn't appear twice across drafts.

Selection diversifies across products: we pick from N distinct products,
not 3 photos of the same bracelet.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from core.seo.images.wp_media import (
    ProductImage,
    fetch_product_images,
    get_used_attachment_ids,
)
from core.seo.models import SeoSite

log = logging.getLogger(__name__)


@dataclass
class ImagedDraft:
    """Result of splicing — body has images embedded as markdown,
    plus the WP attachment IDs we used so the publisher + queue can reference them."""
    body: str
    featured_image_id: int
    featured_image_url: str
    inline_image_ids: list[int]
    all_image_ids: list[int]


def select_images(
    site: SeoSite,
    *,
    count: int = 3,
    rng: Optional[random.Random] = None,
) -> list[ProductImage]:
    """Pick N images from the site's product pool, diversified by product.

    Falls back to allowing repeats from the pool if the unused pool is too
    small (e.g. very young site, or pool exhausted).
    """
    rng = rng or random.Random()
    pool = fetch_product_images(site)
    if not pool:
        log.warning("select_images: empty product pool for site=%s", site.slug)
        return []

    used_ids = get_used_attachment_ids(site.id)
    unused = [img for img in pool if img.attachment_id not in used_ids]
    log.info("select_images: site=%s pool=%d used=%d unused=%d",
             site.slug, len(pool), len(used_ids), len(unused))

    candidates = unused if len(unused) >= count else pool
    rng.shuffle(candidates)

    # Diversify by product_id — don't pick 3 photos of the same product
    chosen: list[ProductImage] = []
    seen_products: set[int] = set()
    for img in candidates:
        if img.product_id in seen_products:
            continue
        chosen.append(img)
        seen_products.add(img.product_id)
        if len(chosen) >= count:
            break
    # Backfill if we couldn't diversify enough (small site, few products)
    if len(chosen) < count:
        for img in candidates:
            if img not in chosen:
                chosen.append(img)
                if len(chosen) >= count:
                    break
    return chosen


def _split_paragraphs(body: str) -> list[str]:
    """Split a markdown body into paragraph chunks (separated by blank lines)."""
    return [p for p in body.split("\n\n") if p.strip()]


def _image_block(img: ProductImage) -> str:
    """Render a WordPress-native <figure> block with image + product-link caption.

    Uses HTML (not markdown) so WP renders the image even when no markdown
    plugin is active. Plain WP wpautop honors raw HTML inside post content.
    """
    import html as _html
    alt = _html.escape(img.alt or img.product_name or "Roen handmade jewelry")
    name = _html.escape(img.product_name or "")
    href = _html.escape(img.product_url or "", quote=True)
    src = _html.escape(img.src, quote=True)
    return (
        f'<figure style="margin: 28px 0;">\n'
        f'  <img src="{src}" alt="{alt}" loading="lazy" style="width:100%;height:auto;border-radius:6px;" />\n'
        f'  <figcaption style="font-size:13px;color:#666;text-align:center;margin-top:8px;font-style:italic;">'
        f'Featured: <a href="{href}" style="color:#B85C3D;text-decoration:none;border-bottom:1px dotted;">{name}</a>'
        f'</figcaption>\n'
        f'</figure>'
    )


def splice_images_into_body(
    body: str,
    featured: ProductImage,
    inline: list[ProductImage],
) -> str:
    """Render featured at the very top + inline images at ~33% and ~67% paragraph
    boundaries. Body must be markdown without an existing H1 image.
    """
    paragraphs = _split_paragraphs(body)
    if not paragraphs:
        return _image_block(featured) + "\n\n" + body

    # Featured image goes after the first paragraph (which is usually a hook),
    # not literally first — that way the post reads "headline → opening → image"
    # instead of "headline → image → opening" which feels brochure-y.
    n = len(paragraphs)
    if n <= 2 or not inline:
        slots: list[int] = [1]
    elif len(inline) == 1:
        slots = [1, max(2, int(n * 0.6))]
    else:
        slots = [1, max(2, int(n * 0.4)), max(3, int(n * 0.75))]

    # Cap slots at len(paragraphs)
    slots = sorted(set(min(s, n) for s in slots))
    images_to_place = [featured] + inline

    out: list[str] = []
    img_idx = 0
    for i, para in enumerate(paragraphs):
        out.append(para)
        if img_idx < len(images_to_place) and (i + 1) in slots:
            out.append(_image_block(images_to_place[img_idx]))
            img_idx += 1
    # Spill any remaining images at the end (if slots ran out due to short body)
    while img_idx < len(images_to_place):
        out.append(_image_block(images_to_place[img_idx]))
        img_idx += 1

    return "\n\n".join(out)


def add_images_to_draft(
    body: str,
    site: SeoSite,
    *,
    count: int = 3,
    rng: Optional[random.Random] = None,
) -> ImagedDraft:
    """Pick `count` images for `site` and splice them into `body`.

    If the site has no product pool, returns the body unchanged with no images.
    """
    chosen = select_images(site, count=count, rng=rng)
    if not chosen:
        return ImagedDraft(
            body=body,
            featured_image_id=0,
            featured_image_url="",
            inline_image_ids=[],
            all_image_ids=[],
        )
    featured = chosen[0]
    inline = chosen[1:]
    new_body = splice_images_into_body(body, featured, inline)
    return ImagedDraft(
        body=new_body,
        featured_image_id=featured.attachment_id,
        featured_image_url=featured.src,
        inline_image_ids=[img.attachment_id for img in inline],
        all_image_ids=[img.attachment_id for img in chosen],
    )
