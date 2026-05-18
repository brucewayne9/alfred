"""Weekly blog engine — generates one blog draft per configured site,
enqueues it to seo_pending for Mike's approval. NEVER auto-publishes.

Flow per site:
  1. pick_next_blog_topic     → highest-volume not-yet-blogged cluster
  2. generate_angle           → LLM proposes a specific angle (1 sentence)
  3. writer.generate_with_retry → 800-1200 word blog draft
  4. validator.validate_draft   → soft check (Flesch, keyword density, voice)
  5. compose_blog_images      → ComfyUI hero + 2 inline product photos
  6. enqueue_draft            → seo_pending (status=pending)
  7. Telegram + email summary

Skips a site when picker returns None (all clusters exhausted) — logs and
moves on. Mike will get an "all topics exhausted" telegram so he knows to
expand keyword discovery or pivot to editorial topics.

Cron (host TZ = America/New_York):
  0 6 * * 0 /home/aialfred/alfred/venv/bin/python -m scripts.seo_weekly_blog_engine \
    >> /home/aialfred/alfred/data/seo/blog_engine.log 2>&1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

import httpx

sys.path.insert(0, "/home/aialfred/alfred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("blog_engine")

from config.settings import settings
from core.seo.content.blog_planner import (
    generate_angle,
    pick_next_blog_topic,
)
from core.seo.content.validator import validate_draft
from core.seo.content.writer import Brief, generate_with_retry
from core.seo.db import SessionLocal
from core.seo.images.selector import compose_blog_images
from core.seo.models import SeoPending
from core.seo.queue.pending import enqueue_draft
from core.seo.sites.profile import load_profile
from core.seo.sites.registry import get_site_by_slug

# Sites currently in the auto-blog rotation. Manual list rather than a flag
# on seo_sites so we change cadence without DB migrations.
ACTIVE_SITES = ["roen"]

MIKE_CHAT_ID = "7582976864"


@dataclass
class EngineResult:
    site_slug: str
    status: str                      # ok | skipped | error
    pending_id: Optional[int] = None
    primary_kw: Optional[str] = None
    angle: Optional[str] = None
    title: Optional[str] = None
    word_count: Optional[int] = None
    flesch: Optional[float] = None
    validator_ok: Optional[bool] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None


def _send_telegram(text: str) -> None:
    token = getattr(settings, "telegram_bot_token", "") or ""
    if not token:
        log.warning("telegram skipped — TELEGRAM_BOT_TOKEN missing")
        return
    try:
        r = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": MIKE_CHAT_ID,
                "text": text,
                "disable_web_page_preview": True,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        r.raise_for_status()
    except Exception:
        log.exception("telegram send failed")


def process_site(slug: str) -> EngineResult:
    started = time.monotonic()
    site = get_site_by_slug(slug)
    if not site:
        return EngineResult(site_slug=slug, status="error", error=f"site not found")

    try:
        profile = load_profile(slug)
    except Exception as e:
        return EngineResult(site_slug=slug, status="error", error=f"profile load: {e}")

    plan = pick_next_blog_topic(site.id)
    if not plan:
        log.info("site=%s — no unblogged clusters remain", slug)
        return EngineResult(site_slug=slug, status="skipped",
                            error="all keyword clusters already blogged — expand keyword discovery or add editorial topics")

    log.info("site=%s pick: primary=%r vol=%d product=%s",
             slug, plan.primary_kw, plan.total_volume, plan.target_product_name)

    angle = generate_angle(plan, profile)
    log.info("site=%s angle: %s", slug, angle[:140])

    brief = Brief(
        topic=angle,
        content_type="blog",
        target_keyword=plan.primary_kw,
        extra_keywords=plan.extra_kws,
        source_signal=f"weekly_blog_engine: cluster vol={plan.total_volume} product={plan.target_product_name}",
    )

    try:
        draft = generate_with_retry(brief, profile)
    except Exception as e:
        log.exception("draft generation failed for site=%s", slug)
        return EngineResult(site_slug=slug, status="error", error=f"writer: {e}")

    v = validate_draft(draft.body, profile, content_type="blog",
                       target_keyword=plan.primary_kw)
    if not v.ok:
        log.warning("validator soft-fail site=%s: %s", slug, v.issues)

    # Images — non-fatal if ComfyUI hiccups
    image_meta: dict = {}
    try:
        imaged = compose_blog_images(
            draft.body, site,
            topic=brief.topic, target_keyword=brief.target_keyword,
            inline_count=2, use_comfyui_hero=True,
        )
        draft.body = imaged.body
        image_meta = {
            "featured_image_id": imaged.featured_image_id,
            "featured_image_url": imaged.featured_image_url,
            "image_ids": imaged.all_image_ids,
            "inline_image_ids": imaged.inline_image_ids,
        }
        log.info("site=%s images composed hero=%d inlines=%d",
                 slug, imaged.featured_image_id, len(imaged.inline_image_ids))
    except Exception:
        log.exception("image composition failed for site=%s — continuing text-only", slug)

    result = enqueue_draft(
        site.id,
        draft=draft,
        source_signal={
            "kind": "weekly_blog_engine",
            "primary_kw": plan.primary_kw,
            "target_keyword": plan.primary_kw,
            "extra_kw": plan.extra_kws,
            "cluster_volume": plan.total_volume,
            "target_product_url": plan.target_product_url,
            "angle": angle[:200],
            "validation": {
                "ok": v.ok, "flesch": v.flesch,
                "word_count": v.word_count, "issues": v.issues,
            },
            **{k: v_ for k, v_ in image_meta.items() if k in {"featured_image_id"}},
        },
        meta_description=None,
    )

    # Stash image metadata onto body_payload so the publisher sets featured_media
    if image_meta:
        with SessionLocal() as s:
            row = s.get(SeoPending, result.pending_id)
            if row and row.body_payload is not None:
                payload = dict(row.body_payload)
                payload.update(image_meta)
                payload["body"] = draft.body  # body now includes spliced inlines
                row.body_payload = payload
                s.commit()

    return EngineResult(
        site_slug=slug,
        status="ok",
        pending_id=result.pending_id,
        primary_kw=plan.primary_kw,
        angle=angle,
        title=draft.title,
        word_count=v.word_count,
        flesch=v.flesch,
        validator_ok=v.ok,
        elapsed_s=round(time.monotonic() - started, 1),
    )


def _summary_message(results: list[EngineResult]) -> str:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    errors = [r for r in results if r.status == "error"]

    lines = [f"📝 *Weekly blog engine* — {len(ok)}/{len(results)} drafted"]
    for r in ok:
        v_badge = "✓" if r.validator_ok else "⚠"
        lines.append(
            f"\n• *{r.site_slug}* {v_badge} _{r.title or r.primary_kw}_"
            f"\n  kw={r.primary_kw} · {r.word_count}w · Flesch {r.flesch} · {r.elapsed_s}s"
            f"\n  → https://aialfred.groundrushcloud.com/admin/seo/pending"
        )
    for r in skipped:
        lines.append(f"\n• *{r.site_slug}* skipped — {r.error}")
    for r in errors:
        lines.append(f"\n• *{r.site_slug}* 🚨 {r.error}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=",".join(ACTIVE_SITES),
                    help=f"comma-sep site slugs (default: {','.join(ACTIVE_SITES)})")
    ap.add_argument("--dry-run", action="store_true", help="run but don't telegram")
    args = ap.parse_args()

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    log.info("blog engine starting — sites=%s", sites)

    results = [process_site(s) for s in sites]

    print("\n=== weekly blog engine results ===")
    for r in results:
        if r.status == "ok":
            print(f"  ✓ {r.site_slug}  pending_id={r.pending_id}  kw={r.primary_kw}  {r.word_count}w  Flesch {r.flesch}")
        elif r.status == "skipped":
            print(f"  — {r.site_slug}  skipped: {r.error}")
        else:
            print(f"  ✗ {r.site_slug}  ERROR: {r.error}")

    if not args.dry_run:
        _send_telegram(_summary_message(results))

    ok_count = sum(1 for r in results if r.status == "ok")
    # Skipped sites aren't failures — only real errors count
    has_errors = any(r.status == "error" for r in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
