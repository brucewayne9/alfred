"""Batch-generate product_enrichment drafts for Roen's keyword-mapped products.

Picks the 6 products that the Keyword Engine v1 mapped highest-opportunity
keywords to, fetches each product's current WP description as context,
generates a ~120-160 word enrichment via Kimi K2.6 (cloud), validates against
the Roen brand profile, and enqueues approved drafts to seo_pending so Mike
can review at /admin/seo/pending.

Runs in parallel (4 workers) — each generation is 60-170s on Kimi cloud;
serial would take ~12 min, parallel ~3-4 min.

Usage:
    PYTHONPATH=. venv/bin/python scripts/seo_roen_product_enrichment_batch.py
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import httpx

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("roen_product_enrichment")

from core.seo.content.writer import Brief, generate_with_retry
from core.seo.content.validator import validate_draft
from core.seo.queue.pending import enqueue_draft
from core.seo.sites.profile import load_profile

ROEN_SITE_ID = 59

# Products to enrich, ordered by cumulative keyword opportunity (per
# Keyword Engine v1 run 2026-05-17). Each entry maps to its top primary
# keyword + 1-2 secondaries the discovery engine surfaced.
TARGETS = [
    {
        "slug": "willow-beaded-bracelet",
        "primary_kw": "beaded bracelet",
        "extra_kw": ["handmade bracelet", "stretch bracelet"],
        "note": "Hero product — 16 keywords mapped, ~228K monthly vol",
    },
    {
        "slug": "turquoise-flower-smiley-bracelet",
        "primary_kw": "gold beaded bracelet",
        "extra_kw": ["beaded bracelet", "handmade bracelet"],
        "note": "3 keywords mapped, ~29K monthly vol",
    },
    {
        "slug": "evil-eye-glass-bracelet",
        "primary_kw": "glass beaded bracelet",
        "extra_kw": ["evil eye bracelet", "beaded glass bracelet"],
        "note": "2 keywords mapped, ~16K monthly vol",
    },
    {
        "slug": "gold-ribbed-stretch-bracelet",
        "primary_kw": "handmade bracelet",
        "extra_kw": ["stretch bracelet", "bracelet handmade"],
        "note": "2 keywords mapped, ~13K monthly vol",
    },
    {
        "slug": "turquoise-gold-bracelet",
        "primary_kw": "beaded bracelet",
        "extra_kw": ["gold beaded bracelet", "stretch bracelet"],
        "note": "1 keyword mapped, 33K monthly vol",
    },
]


@dataclass
class ProductContext:
    slug: str
    product_id: int
    name: str
    description_html: str


def fetch_product(slug: str) -> Optional[ProductContext]:
    """Pull a Roen product's current WP data via public REST."""
    r = httpx.get(
        f"https://www.roenhandmade.com/wp-json/wp/v2/product?slug={slug}",
        timeout=15.0,
    )
    items = r.json()
    if not items:
        return None
    p = items[0]
    return ProductContext(
        slug=slug,
        product_id=int(p["id"]),
        name=p.get("title", {}).get("rendered", slug),
        description_html=(p.get("content") or {}).get("rendered", "").strip(),
    )


def _topic_for(ctx: ProductContext, target: dict) -> str:
    """Construct the writer brief topic from the product context."""
    # Strip HTML tags lightly for the writer's context.
    desc = ctx.description_html.replace("<p>", "").replace("</p>", " ").strip()
    # Cap to keep prompt lean
    desc = " ".join(desc.split())[:600]
    return (
        f"{ctx.name} — Roen product. Existing description: \"{desc}\" "
        f"Write enrichment to APPEND below this description; do not redescribe "
        f"the piece. Focus sections on materials/care + outfit pairings."
    )


def process_one(target: dict, profile) -> dict:
    """Generate + validate + enqueue a single product enrichment."""
    slug = target["slug"]
    started = time.monotonic()
    try:
        ctx = fetch_product(slug)
        if ctx is None:
            return {"slug": slug, "status": "skip", "reason": "wp lookup miss"}

        brief = Brief(
            topic=_topic_for(ctx, target),
            content_type="product_enrichment",
            target_keyword=target["primary_kw"],
            extra_keywords=target["extra_kw"],
            source_signal=f"keyword_engine_v1: {target['note']}",
        )
        draft = generate_with_retry(brief, profile)
        v = validate_draft(
            draft.body,
            profile,
            content_type="product_enrichment",
            target_keyword=target["primary_kw"],
        )
        if not v.ok:
            log.warning("validation soft-fail for %s: %s", slug, v.issues)

        # Title shipped to queue: product name (not auto-generated).
        draft.title = ctx.name
        result = enqueue_draft(
            site_id=ROEN_SITE_ID,
            draft=draft,
            source_signal={
                "kind": "product_enrichment",
                "product_slug": slug,
                "product_id": ctx.product_id,
                "wp_path": f"/product/{slug}/",
                "primary_kw": target["primary_kw"],
                "extra_kw": target["extra_kw"],
                "note": target["note"],
                "validation": {
                    "ok": v.ok,
                    "flesch": v.flesch,
                    "word_count": v.word_count,
                    "issues": v.issues,
                },
            },
        )
        return {
            "slug": slug,
            "status": "ok",
            "pending_id": result.pending_id,
            "words": v.word_count,
            "flesch": v.flesch,
            "validator_ok": v.ok,
            "elapsed_s": round(time.monotonic() - started, 1),
            "title": ctx.name,
        }
    except Exception as e:
        log.exception("enrichment failed for %s", slug)
        return {
            "slug": slug,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": round(time.monotonic() - started, 1),
        }


def main() -> int:
    profile = load_profile("roen")
    log.info("starting batch — %d products, profile=%s", len(TARGETS), profile.slug)
    started = time.monotonic()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_one, t, profile): t["slug"] for t in TARGETS}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            log.info("done %s: %s", r["slug"], r.get("status"))

    elapsed = time.monotonic() - started
    print("\n" + "=" * 90)
    print(f"BATCH DONE in {elapsed:.1f}s")
    print("=" * 90)
    print(f"  {'STATUS':<6}  {'WORDS':>5}  {'FLESCH':>6}  {'VALID':<5}  {'PEND_ID':>7}  PRODUCT")
    print("-" * 90)
    ok_count = 0
    for r in sorted(results, key=lambda x: x["slug"]):
        status = r["status"]
        words = r.get("words", "—")
        flesch = r.get("flesch", "—")
        val = "✓" if r.get("validator_ok") else "—"
        pid = r.get("pending_id", "—")
        title = r.get("title", r["slug"])
        print(f"  {status:<6}  {str(words):>5}  {str(flesch):>6}  {val:<5}  {str(pid):>7}  {title}")
        if status == "ok":
            ok_count += 1
    print("-" * 90)
    print(f"  {ok_count}/{len(TARGETS)} drafts queued. Review at /admin/seo/pending")
    return 0 if ok_count == len(TARGETS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
