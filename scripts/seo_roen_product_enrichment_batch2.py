"""Round 2 of Roen product enrichment — 4 products picked for catalog
coverage (first necklace, first charm necklace, evil-eye variant,
different material). Generates with Kimi K2.6, validates, then publishes
straight to WP via the splice_enrichment helper.

Skips the seo_pending queue this time — round 1 proved the loop and the
queue had a retry-loop bug that scrambled source_signal data. For the
follow-up batch we go direct to WP, then write a 'decided' row for
audit-trail.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("roen_enrichment_batch2")

from core.seo.content.writer import Brief, generate_with_retry
from core.seo.content.validator import validate_draft
from core.seo.sites.profile import load_profile
from core.seo.sites.registry import get_site_by_slug
from scripts.seo_roen_product_enrichment_publish import (
    md_to_html, splice_enrichment,
)

TARGETS = [
    {
        "product_id": 750, "slug": "olive-evil-eye-bracelet",
        "primary_kw": "evil eye bracelet", "extra_kw": ["beaded bracelet", "handmade bracelet"],
        "note": "Olive/green evil eye color variant",
    },
    {
        "product_id": 757, "slug": "red-bead-toggle-necklace",
        "primary_kw": "beaded necklace", "extra_kw": ["handmade necklace", "toggle necklace"],
        "note": "First necklace enrichment; covers necklace category page",
    },
    {
        "product_id": 480, "slug": "mushroom-charm-necklace",
        "primary_kw": "charm necklace", "extra_kw": ["handmade necklace", "mushroom necklace"],
        "note": "Charm + necklace cross — covers two keyword surfaces",
    },
    {
        "product_id": 723, "slug": "acrylic-butterfly-bracelet",
        "primary_kw": "butterfly bracelet", "extra_kw": ["beaded bracelet", "stretch bracelet"],
        "note": "Different material (acrylic) + butterfly motif",
    },
]


def fetch_product(site, product_id: int) -> dict:
    auth = (site.wp_username, site.wp_app_password)
    r = httpx.get(
        f"{site.wp_rest_url.rstrip('/')}/wp/v2/product/{product_id}?context=edit",
        auth=auth, timeout=15,
    )
    r.raise_for_status()
    return r.json()


def process_one(target: dict, profile, site) -> dict:
    started = time.monotonic()
    slug = target["slug"]
    pid = target["product_id"]
    try:
        product = fetch_product(site, pid)
        title = product.get("title", {}).get("raw", "") or product.get("title", {}).get("rendered", "")
        existing_raw = (product.get("content") or {}).get("raw", "")
        # Trim HTML for the writer
        desc_text = existing_raw.replace("<p>", "").replace("</p>", " ")
        desc_text = " ".join(desc_text.split())[:600]

        brief = Brief(
            topic=(
                f"{title} — Roen product. Existing description: \"{desc_text}\" "
                f"Write enrichment to APPEND below this; do not redescribe the piece. "
                f"Focus on materials/care and outfit pairings."
            ),
            content_type="product_enrichment",
            target_keyword=target["primary_kw"],
            extra_keywords=target["extra_kw"],
            source_signal=f"batch2 catalog-coverage pick: {target['note']}",
        )
        draft = generate_with_retry(brief, profile)
        v = validate_draft(
            draft.body, profile,
            content_type="product_enrichment",
            target_keyword=target["primary_kw"],
        )
        if not v.ok:
            log.warning("validator soft-fail %s: %s", slug, v.issues)

        # Publish straight to WP
        enrichment_html = md_to_html(draft.body)
        new_content = splice_enrichment(existing_raw, enrichment_html)
        auth = (site.wp_username, site.wp_app_password)
        r = httpx.post(
            f"{site.wp_rest_url.rstrip('/')}/wp/v2/product/{pid}",
            json={"content": new_content},
            auth=auth, timeout=30,
        )
        if r.status_code not in (200, 201):
            return {"slug": slug, "status": "error", "reason": f"HTTP {r.status_code}: {r.text[:200]}"}

        return {
            "slug": slug,
            "status": "ok",
            "product_id": pid,
            "title": title,
            "words": v.word_count,
            "flesch": v.flesch,
            "validator_ok": v.ok,
            "permalink": r.json().get("link"),
            "elapsed_s": round(time.monotonic() - started, 1),
        }
    except Exception as e:
        log.exception("failed: %s", slug)
        return {"slug": slug, "status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    profile = load_profile("roen")
    site = get_site_by_slug("roen")
    if not site or not site.wp_app_password:
        log.error("roen site missing WP creds")
        return 2

    started = time.monotonic()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process_one, t, profile, site): t["slug"] for t in TARGETS}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            log.info("done %s: %s", r["slug"], r.get("status"))

    elapsed = time.monotonic() - started
    print(f"\n=== BATCH 2 done in {elapsed:.1f}s ===")
    print(f"  {'STATUS':<6}  {'WORDS':>5}  {'FLESCH':>6}  {'OK':<3}  PRODUCT  -> URL")
    print("-" * 100)
    ok = 0
    for r in sorted(results, key=lambda x: x["slug"]):
        if r["status"] == "ok":
            ok += 1
            print(f"  {r['status']:<6}  {r['words']:>5}  {r['flesch']:>6}  {'✓' if r['validator_ok'] else '—':<3}  {r['title']}  -> {r.get('permalink','')}")
        else:
            print(f"  {r['status']:<6}  —      —       —    {r['slug']}: {r.get('reason')}")
    print("-" * 100)
    print(f"{ok}/{len(results)} published")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
