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
    """Render a native Gutenberg image block.

    Theme-respecting: uses .wp-block-image / .wp-element-caption classes that
    the roen-minimal stylesheet styles globally, with no inline pixel hacks.
    Block delimiters let WP treat this as a real image block in the editor.

    Two render modes:
      - PRODUCT image (img.product_url set): wraps img in <a> to product page,
        figcaption shows "Shop: [Product Name]" linking to the product.
      - EDITORIAL hero (img.product_url empty): no <a> wrap, no caption — the
        image stands alone as a magazine-style hero.
    """
    import html as _html
    alt = _html.escape(img.alt or img.product_name or "Roen handmade jewelry")
    src = _html.escape(img.src, quote=True)
    aid = int(img.attachment_id)

    if img.product_url and img.product_name:
        name = _html.escape(img.product_name)
        href = _html.escape(img.product_url, quote=True)
        return (
            f'<!-- wp:image {{"id":{aid},"sizeSlug":"large","linkDestination":"custom"}} -->\n'
            f'<figure class="wp-block-image size-large">'
            f'<a href="{href}"><img src="{src}" alt="{alt}" class="wp-image-{aid}"/></a>'
            f'<figcaption class="wp-element-caption">'
            f'Shop: <a href="{href}">{name}</a>'
            f'</figcaption>'
            f'</figure>\n'
            f'<!-- /wp:image -->'
        )

    return (
        f'<!-- wp:image {{"id":{aid},"sizeSlug":"large"}} -->\n'
        f'<figure class="wp-block-image size-large">'
        f'<img src="{src}" alt="{alt}" class="wp-image-{aid}"/>'
        f'</figure>\n'
        f'<!-- /wp:image -->'
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
    """All-product version. Pick `count` images and splice them in.

    Used as fallback when ComfyUI hero generation is unavailable. For the
    primary blog flow use compose_blog_images() instead.
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


def compose_blog_images(
    body: str,
    site: SeoSite,
    *,
    topic: str,
    target_keyword: Optional[str] = None,
    inline_count: int = 2,
    use_comfyui_hero: bool = True,
    rng: Optional[random.Random] = None,
) -> ImagedDraft:
    """Editorial composition for blog/cluster posts.

    Hero: ComfyUI-generated editorial shot (or first product photo as fallback).
    Inline: N product photo Gutenberg blocks with "Shop:" caption links.

    Hero is used as the WP featured_image AND inserted at the top of the body.
    Inline images splice in at logical paragraph boundaries.
    """
    from core.seo.images.comfyui_hero import generate_hero_for_topic

    rng = rng or random.Random()
    hero: Optional[ProductImage] = None
    if use_comfyui_hero:
        log.info("compose_blog_images: generating ComfyUI hero for %r", topic[:60])
        hero = generate_hero_for_topic(site, topic=topic, target_keyword=target_keyword, rng=rng)
        if hero:
            log.info("compose_blog_images: hero attachment_id=%d", hero.attachment_id)

    # Pick inline product photos (don't include hero in dedup since it's editorial)
    inline = select_images(site, count=inline_count, rng=rng)

    if hero is None and not inline:
        # No imagery at all — return body unchanged
        return ImagedDraft(
            body=body, featured_image_id=0, featured_image_url="",
            inline_image_ids=[], all_image_ids=[],
        )

    # Fallback: if no ComfyUI hero, promote the first product image to hero
    if hero is None:
        hero = inline[0]
        inline = inline[1:]

    new_body = splice_images_into_body(body, hero, inline)
    all_ids = [hero.attachment_id] + [img.attachment_id for img in inline]
    return ImagedDraft(
        body=new_body,
        featured_image_id=hero.attachment_id,
        featured_image_url=hero.src,
        inline_image_ids=[img.attachment_id for img in inline],
        all_image_ids=all_ids,
    )
