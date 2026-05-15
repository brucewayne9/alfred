"""ComfyUI editorial hero generator for Roen blog posts.

Generates magazine-quality hero shots styled to Roen's locked aesthetic:
marble tabletop or soft studio light, terracotta accent, no on-model
imagery, no text in image. Mejuri/Aritzia editorial DNA.

Variation pool prevents siblings — every generation rotates through
lighting / composition / prop combinations so heroes feel curated, not
batch-produced.

Pipeline:
  1. Build a topical prompt (brief topic + brand baseline + random variation)
  2. Call ComfyUI on 105 to generate
  3. Upload resulting JPG to the site's WP media library
  4. Return ProductImage-shaped record so the splicer can use it identically
     to product photos
"""
from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Optional

import requests

from core.seo.images.wp_media import ProductImage
from core.seo.models import SeoSite

log = logging.getLogger(__name__)

# Brand-locked baseline applied to every Roen hero generation.
ROEN_BASELINE = (
    "editorial product photography, handmade beaded bracelets and necklaces "
    "arranged on a flat surface, marble tabletop backdrop, soft natural studio "
    "light, terracotta and warm neutral accents, Mejuri Aritzia magazine "
    "aesthetic, minimal composition, no people, no faces, no text, no logos, "
    "no watermarks, photorealistic, sharp focus, shallow depth of field, "
    "muted color palette with selective terracotta warmth"
)

# Hard negatives — what the image must NOT contain.
ROEN_NEGATIVE = (
    "people, person, model, hands, fingers, face, body, text, words, letters, "
    "logo, watermark, signature, busy background, plastic, cartoon, illustration, "
    "low quality, blurry, oversaturated, neon, dark moody, gothic"
)

# Variation pools — rotate for non-sibling images.
LIGHTING_POOL = [
    "soft window light from the side",
    "overhead diffused studio softbox",
    "warm late-afternoon natural light",
    "bright overhead noon light with soft shadows",
    "gentle backlight with rim highlight",
]

COMPOSITION_POOL = [
    "overhead flat-lay composition",
    "three-quarter angle with shallow depth",
    "close detail shot with negative space",
    "side angle with one piece in foreground sharp",
    "centered single-piece minimal layout",
]

PROP_POOL = [
    "with a small terracotta ceramic dish",
    "alongside a sprig of dried eucalyptus",
    "with a folded linen napkin in cream",
    "near a small ceramic vessel",
    "with a single dried wildflower stem",
    "beside a smooth river stone",
    "with crumpled raw silk fabric backdrop",
]


def build_hero_prompt(topic: str, *, rng: Optional[random.Random] = None) -> str:
    """Compose a ComfyUI prompt for a Roen blog hero shot.

    Topic informs subject specifics (e.g. "evil eye bracelet meaning" → "evil
    eye bracelet"). Variation pools prevent visual sibling-ness.
    """
    rng = rng or random.Random()
    topic_specific = ""
    topic_lower = topic.lower()
    # Lightweight topic→subject mapping. Falls through to generic "jewelry pieces".
    if "evil eye" in topic_lower:
        topic_specific = "evil eye bracelet with small blue glass eye charm, "
    elif "necklace" in topic_lower:
        topic_specific = "delicate beaded necklace with toggle clasp, "
    elif "layer" in topic_lower or "stack" in topic_lower:
        topic_specific = "two or three stacked beaded bracelets in complementary tones, "
    elif "bracelet" in topic_lower:
        topic_specific = "beaded bracelet with small charm, "

    parts = [
        topic_specific + ROEN_BASELINE,
        rng.choice(LIGHTING_POOL),
        rng.choice(COMPOSITION_POOL),
        rng.choice(PROP_POOL),
    ]
    return ", ".join(p for p in parts if p)


def _generate_image_sync(prompt: str, *, width: int = 1280, height: int = 960) -> dict:
    """Sync wrapper around the async ComfyUI client. Returns the same dict
    shape: {success, image_path, error, ...}.
    """
    from integrations.comfyui.client import generate_image
    return asyncio.run(generate_image(
        prompt=prompt,
        width=width,
        height=height,
        mode="quality",
        upscale=False,
    ))


def _upload_to_wp_media(
    site: SeoSite,
    file_path: Path,
    *,
    alt: str = "",
    title: str = "",
    timeout: int = 60,
) -> dict:
    """Upload an image file to WP media library. Returns the WP media item dict."""
    url = f"{site.wp_rest_url.rstrip('/')}/wp/v2/media"
    auth = (site.wp_username, site.wp_app_password)
    with file_path.open("rb") as fh:
        headers = {
            "Content-Disposition": f'attachment; filename="{file_path.name}"',
            "Content-Type": "image/jpeg" if file_path.suffix.lower() in (".jpg", ".jpeg") else "image/png",
        }
        r = requests.post(url, data=fh.read(), headers=headers, auth=auth, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"WP media upload failed: {r.status_code} {r.text[:300]}")
    item = r.json()
    # Patch alt + title if provided (uploads default both to filename)
    if alt or title:
        patch = {}
        if alt:
            patch["alt_text"] = alt
        if title:
            patch["title"] = title
        try:
            requests.post(
                f"{url}/{item['id']}", json=patch, auth=auth, timeout=timeout,
            )
        except Exception:
            pass  # alt/title are nice-to-have, don't fail the upload
    return item


def generate_hero_for_topic(
    site: SeoSite,
    *,
    topic: str,
    target_keyword: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Optional[ProductImage]:
    """Generate + upload a hero image for a blog topic. Returns a ProductImage
    record (with product_url empty — heros aren't product-tied) or None on failure.
    """
    prompt = build_hero_prompt(topic, rng=rng)
    log.info("comfyui hero: topic=%r prompt=%r", topic[:60], prompt[:160])

    try:
        result = _generate_image_sync(prompt)
    except Exception:
        log.exception("comfyui hero: generation failed")
        return None
    if not result.get("success"):
        log.warning("comfyui hero: generation unsuccessful: %s", result.get("error"))
        return None

    img_path = Path(result["image_path"])
    if not img_path.exists():
        log.warning("comfyui hero: returned image_path missing: %s", img_path)
        return None

    alt_text = topic[:120]
    title = f"Editorial — {(target_keyword or topic)[:60]}"

    try:
        media = _upload_to_wp_media(site, img_path, alt=alt_text, title=title)
    except Exception:
        log.exception("comfyui hero: WP upload failed")
        return None

    return ProductImage(
        attachment_id=int(media["id"]),
        src=media.get("source_url", ""),
        thumbnail=(media.get("media_details", {}).get("sizes", {}).get("thumbnail", {}).get("source_url")
                   or media.get("source_url", "")),
        alt=alt_text,
        product_id=0,                 # not tied to a product
        product_name="",              # no caption needed for editorial heros
        product_slug="",
        product_url="",
    )
