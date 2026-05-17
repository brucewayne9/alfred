"""Batch-generate Roen blog briefs through the same path /admin/seo/new uses.

Picks a curated set of high-intent jewelry topics (meaning / styling / care /
gifting mix), runs each through writer → image pipeline → queue, and prints
a one-line summary per topic. Failures don't abort the batch.

Run: source venv/bin/activate && python scripts/seo_batch_roen.py
"""
from __future__ import annotations

import logging
import sys
import time
import traceback

from core.seo.content.writer import Brief, generate_with_retry
from core.seo.images.selector import compose_blog_images
from core.seo.queue.pending import enqueue_draft
from core.seo.sites.profile import load_profile
from core.seo.sites.registry import get_site_by_slug

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("seo_batch_roen")

SITE_SLUG = "roen"

# Curated topic set — high-intent, on-brand, varied across discovery modes.
# (topic, target_keyword, title_hint, extra_keywords[csv], audience)
TOPICS: list[tuple[str, str, str, str, str]] = [
    (
        "How to layer beaded bracelets without overdoing it",
        "layering beaded bracelets",
        "",
        "stacking bracelets, bracelet layering, beaded jewelry styling",
        "women 25-45 building a daily bracelet stack",
    ),
    (
        "What a hamsa bracelet actually means",
        "hamsa bracelet meaning",
        "",
        "hamsa hand, evil eye protection, hamsa symbolism",
        "first-time hamsa shoppers researching meaning before buying",
    ),
    (
        "Stacking necklaces — three rules that keep it tasteful",
        "how to stack necklaces",
        "",
        "necklace layering, layered necklaces, stacking dainty necklaces",
        "women 25-45 looking to layer everyday necklaces",
    ),
    (
        "Why handmade jewelry feels different on the wrist",
        "handmade jewelry quality",
        "",
        "handmade vs mass produced, small batch jewelry, artisan jewelry",
        "shoppers weighing handmade vs chain-store brands",
    ),
    (
        "Birthstone bracelets — a quick reference for every month",
        "birthstone bracelet meaning",
        "",
        "birthstones by month, gemstone meaning, birthstone gift",
        "gift shoppers researching meaningful stones",
    ),
    (
        "How to care for natural-stone bracelets at home",
        "how to care for beaded bracelets",
        "",
        "cleaning beaded jewelry, jewelry care, stone bracelet care",
        "new owners of handmade beaded jewelry",
    ),
    (
        "Picking a bracelet she will actually wear — a gift guide",
        "meaningful bracelet gift",
        "",
        "bracelet gift ideas, handmade bracelet gift, gift bracelet women",
        "gift-shoppers buying for partners, friends, mothers",
    ),
    (
        "Toggle clasp vs lobster clasp — which one wears better",
        "toggle clasp bracelet",
        "",
        "bracelet clasps, toggle clasp, lobster clasp comparison",
        "design-curious shoppers picking between clasp styles",
    ),
]


def run_one(idx: int, total: int, topic_tuple: tuple) -> dict:
    topic, kw, title_hint, extra_kw, audience = topic_tuple
    extras = [k.strip() for k in extra_kw.split(",") if k.strip()]
    brief = Brief(
        topic=topic,
        content_type="blog",
        target_keyword=kw,
        audience=audience or None,
        title_hint=title_hint or None,
        extra_keywords=extras,
        source_signal="batch_seo_roen_v1",
    )

    site = get_site_by_slug(SITE_SLUG)
    if not site:
        raise RuntimeError(f"site not found: {SITE_SLUG}")
    profile = load_profile(SITE_SLUG)

    log.info("[%d/%d] WRITER topic=%r kw=%r", idx, total, topic[:70], kw)
    t0 = time.time()
    draft = generate_with_retry(brief, profile)
    log.info("[%d/%d] writer done in %.1fs (title=%r words=%d)",
             idx, total, time.time() - t0, draft.title[:80], len(draft.body.split()))

    log.info("[%d/%d] IMAGES (ComfyUI hero + 2 product inlines)…", idx, total)
    t1 = time.time()
    imaged = compose_blog_images(
        draft.body, site,
        topic=brief.topic, target_keyword=brief.target_keyword,
        inline_count=2, use_comfyui_hero=True,
    )
    draft.body = imaged.body
    log.info("[%d/%d] images done in %.1fs (hero=%d inline=%d)",
             idx, total, time.time() - t1, imaged.featured_image_id, len(imaged.inline_image_ids))

    image_meta = {
        "featured_image_id": imaged.featured_image_id,
        "featured_image_url": imaged.featured_image_url,
        "image_ids": imaged.all_image_ids,
        "inline_image_ids": imaged.inline_image_ids,
    }
    result = enqueue_draft(
        site.id,
        draft=draft,
        source_signal={
            "manual_by": "alfred-batch",
            "topic": brief.topic[:120],
            "target_keyword": brief.target_keyword,
            "featured_image_id": imaged.featured_image_id,
        },
        meta_description=None,
    )

    # Stash image metadata on the persisted body_payload (matches /admin/seo/new path).
    if image_meta:
        from core.seo.db import SessionLocal
        from core.seo.models import SeoPending
        with SessionLocal() as s:
            row = s.get(SeoPending, result.pending_id)
            if row and row.body_payload is not None:
                payload = dict(row.body_payload)
                payload.update(image_meta)
                payload["body"] = draft.body
                row.body_payload = payload
                s.commit()

    return {
        "pending_id": result.pending_id,
        "title": draft.title,
        "topic": topic,
        "target_keyword": kw,
        "hero_id": imaged.featured_image_id,
        "inline_count": len(imaged.inline_image_ids),
    }


def main() -> int:
    successes: list[dict] = []
    failures: list[dict] = []
    total = len(TOPICS)
    log.info("=== Roen batch v1 — %d topics ===", total)
    overall_start = time.time()

    for i, t in enumerate(TOPICS, start=1):
        try:
            out = run_one(i, total, t)
            successes.append(out)
            log.info("[%d/%d] ✓ pending_id=%d", i, total, out["pending_id"])
        except Exception as e:
            log.error("[%d/%d] ✗ FAILED topic=%r: %s", i, total, t[0][:70], e)
            log.error(traceback.format_exc())
            failures.append({"topic": t[0], "error": str(e)})

    elapsed = time.time() - overall_start
    log.info("=== batch done in %.1fs (%.1f min) — %d ok, %d failed ===",
             elapsed, elapsed / 60, len(successes), len(failures))
    print("\n=== SUMMARY ===")
    for s in successes:
        print(f"  ✓ #{s['pending_id']:>4}  hero={s['hero_id']:>4}  {s['title'][:80]}")
    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print(f"  ✗ {f['topic'][:70]}\n    {f['error']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
