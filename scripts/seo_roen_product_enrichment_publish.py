"""Publish queued product_enrichment drafts by APPENDING them to the
existing WooCommerce product description.

The standard core.seo.content.adapter_wp.publish_to_wp was built for blog
posts — it CREATES new posts. Product enrichment is different: we keep
Sarah's existing product copy intact and append our materials/care +
styling sections below it.

Flow per draft:
  1. Fetch the SeoPending row's source_signal → product_id + slug
  2. GET the product via WP REST (authenticated)
  3. Convert the draft markdown body to safe HTML
  4. Build new content = existing description + sentinel marker + new HTML
  5. POST to /wp/v2/product/{id} with the combined content
  6. Mark the SeoPending row decided (approved) with the WP post id

Idempotency: looks for a sentinel comment `<!-- alfred-seo:enrichment v1 -->`
in the existing content. If present, replaces only the enrichment block
(so re-runs swap the enrichment instead of stacking duplicates).
"""
from __future__ import annotations

import logging
import re
import sys
import datetime as dt
from typing import Optional

import httpx
from sqlalchemy import select

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("roen_enrichment_publish")

from core.seo.db import SessionLocal
from core.seo.models import SeoPending
from core.seo.sites.registry import get_site_by_slug

ROEN_SLUG = "roen"
SENTINEL_OPEN = "<!-- alfred-seo:enrichment v1 -->"
SENTINEL_CLOSE = "<!-- /alfred-seo:enrichment v1 -->"


def md_to_html(md: str) -> str:
    """Convert the writer's narrow markdown subset to HTML.

    Handles:
      ## heading        -> <h2>heading</h2>
      - list item       -> <ul><li>item</li></ul>
      blank line breaks -> <p>...</p>
    """
    lines = md.strip().split("\n")
    out: list[str] = []
    para_buf: list[str] = []
    list_buf: list[str] = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            text = " ".join(s.strip() for s in para_buf if s.strip())
            if text:
                out.append(f"<p>{text}</p>")
            para_buf = []

    def flush_list():
        nonlocal list_buf
        if list_buf:
            items = "".join(f"<li>{item}</li>" for item in list_buf)
            out.append(f"<ul>{items}</ul>")
            list_buf = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            flush_para()
            flush_list()
            out.append(f"<h2>{stripped[3:].strip()}</h2>")
        elif stripped.startswith("- "):
            flush_para()
            list_buf.append(stripped[2:].strip())
        elif not stripped:
            flush_para()
            flush_list()
        else:
            flush_list()
            para_buf.append(stripped)
    flush_para()
    flush_list()
    return "\n".join(out)


def splice_enrichment(existing_html: str, enrichment_html: str) -> str:
    """Append or replace the enrichment block inside an existing description."""
    block = (
        f"\n\n{SENTINEL_OPEN}\n{enrichment_html}\n{SENTINEL_CLOSE}\n"
    )
    if SENTINEL_OPEN in existing_html and SENTINEL_CLOSE in existing_html:
        # Replace prior enrichment in-place.
        pattern = re.escape(SENTINEL_OPEN) + r".*?" + re.escape(SENTINEL_CLOSE)
        return re.sub(pattern, block.strip(), existing_html, flags=re.DOTALL)
    return existing_html.rstrip() + block


# Fallback title→product map for rows whose source_signal got overwritten
# by retry loops before we wired auth. Sourced from the original batch
# script's TARGETS list. Used only when the row's own source_signal is
# missing product_id (which happens after publish_failed retries).
_TITLE_TO_PRODUCT = {
    "Willow Beaded Bracelet":              (637, "willow-beaded-bracelet"),
    "Evil Eye Glass Bracelet":             (679, "evil-eye-glass-bracelet"),
    "Turquoise Flower Smiley Bracelet":    (397, "turquoise-flower-smiley-bracelet"),
    "Gold Ribbed Stretch Bracelet":        (541, "gold-ribbed-stretch-bracelet"),
    "Turquoise Gold Bracelet":             (564, "turquoise-gold-bracelet"),
}


def publish_one(site, row: SeoPending) -> dict:
    """Append the draft's body to the WP product's description."""
    sig = row.source_signal or {}
    slug = sig.get("product_slug")
    product_id = sig.get("product_id")
    if not (slug and product_id):
        # Recovery path for rows whose source_signal got overwritten by
        # retry loops. Match by title against our hard-coded map.
        fallback = _TITLE_TO_PRODUCT.get((row.title or "").strip())
        if fallback:
            product_id, slug = fallback
        else:
            return {"pending_id": row.id, "status": "skip", "reason": "missing product_id/slug + no title match"}

    auth = (site.wp_username, site.wp_app_password)
    base = site.wp_rest_url.rstrip("/")

    # 1. Fetch current product (authenticated to read raw markdown content)
    rget = httpx.get(
        f"{base}/wp/v2/product/{product_id}?context=edit",
        auth=auth,
        timeout=20,
    )
    rget.raise_for_status()
    product = rget.json()
    existing = (product.get("content") or {}).get("raw", "")

    # 2. Build new content — body is stored under body_payload['body']
    body_md = (row.body_payload or {}).get("body", "") if isinstance(row.body_payload, dict) else ""
    if not body_md:
        return {"pending_id": row.id, "status": "skip", "reason": "empty body_payload.body"}
    enrichment_html = md_to_html(body_md)
    new_content = splice_enrichment(existing, enrichment_html)

    # 3. Push update via WP REST
    rpost = httpx.post(
        f"{base}/wp/v2/product/{product_id}",
        json={"content": new_content},
        auth=auth,
        timeout=30,
    )
    if rpost.status_code not in (200, 201):
        return {
            "pending_id": row.id,
            "status": "error",
            "reason": f"HTTP {rpost.status_code}: {rpost.text[:200]}",
        }

    return {
        "pending_id": row.id,
        "status": "ok",
        "product_id": product_id,
        "product_slug": slug,
        "title": row.title,
        "wp_status": rpost.json().get("status"),
        "permalink": rpost.json().get("link"),
    }


def main() -> int:
    site = get_site_by_slug(ROEN_SLUG)
    if not site or not site.wp_app_password:
        print("ERROR: roen site missing or wp_app_password not set", file=sys.stderr)
        return 2

    with SessionLocal() as s:
        rows = s.execute(
            select(SeoPending).where(
                SeoPending.site_id == site.id,
                SeoPending.status == "pending",
                SeoPending.content_type == "product_enrichment",
            )
        ).scalars().all()

    log.info("publishing %d pending product enrichments", len(rows))

    results: list[dict] = []
    for row in rows:
        try:
            r = publish_one(site, row)
        except Exception as e:
            r = {"pending_id": row.id, "status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        log.info("publish %s: %s", row.id, r.get("status"))
        results.append(r)

        # Mark seo_pending decided on success
        if r["status"] == "ok":
            with SessionLocal() as s2:
                pending = s2.get(SeoPending, row.id)
                if pending:
                    pending.status = "decided"
                    s2.commit()

    print("\n=== Publish results ===")
    for r in results:
        line = f"  pend={r['pending_id']:>3}  {r['status']:<5}  "
        if r["status"] == "ok":
            line += f"product_id={r['product_id']:>4}  {r['title']}  -> {r.get('permalink', '?')}"
        else:
            line += r.get("reason", "?")
        print(line)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{len(results)} published")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
