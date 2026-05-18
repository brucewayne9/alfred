"""Pick the next blog topic for a site.

Strategy (v1):
  1. Group active keywords by target_url (each product → its own cluster)
  2. Drop clusters whose primary kw already appears in seo_briefs or
     seo_pending (content_type='blog') — we don't want duplicates
  3. From what's left, pick the cluster with highest summed search volume
  4. LLM generates a single-sentence angle for that cluster using the
     brand profile + product context

When all clusters are exhausted, returns None — caller decides whether
to skip the week or hand-pick an editorial topic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx
from sqlalchemy import text

from config.settings import settings
from core.seo.db import SessionLocal
from core.seo.sites.profile import BrandProfile

log = logging.getLogger("seo.blog_planner")

# Use a fast non-reasoning cloud model for angle gen — Kimi's reasoning mode
# hits num_predict before emitting content, so we don't use it here. Verified
# 2026-05-18: ministral-3:14b-cloud returns clean output in ~1.2s, vs Kimi
# burning 200 tokens of thinking with content='' on the same prompt.
ANGLE_MODEL = "ministral-3:14b-cloud"
OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass
class BlogPlan:
    primary_kw: str
    extra_kws: list[str]
    total_volume: int
    target_product_url: str
    target_product_name: str           # best-effort, from URL slug
    angle: str = ""                    # filled by generate_angle
    cluster_keywords: list[str] = field(default_factory=list)


def _product_name_from_url(url: str) -> str:
    """Pull a human title from a /product/foo-bar-baz/ URL."""
    if not url:
        return ""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").title()


def _load_already_blogged(site_id: int) -> set[str]:
    """Return set of primary keywords already covered by a blog brief
    (queued, pending, or decided). Lower-cased for matching."""
    covered: set[str] = set()
    with SessionLocal() as s:
        # From seo_briefs — every blog brief we've created
        briefs = s.execute(text("""
            SELECT target_keywords FROM seo_briefs
            WHERE site_id = :sid AND content_type IN ('blog', 'cluster')
        """), {"sid": site_id}).all()
        for row in briefs:
            kws = row.target_keywords or []
            if isinstance(kws, list):
                covered.update(k.lower() for k in kws if k)
            elif isinstance(kws, dict):
                # legacy shape: {primary: x, extras: [...]}
                if kws.get("primary"):
                    covered.add(str(kws["primary"]).lower())

        # From seo_pending — blog drafts awaiting approval
        pend = s.execute(text("""
            SELECT source_signal FROM seo_pending
            WHERE site_id = :sid AND content_type IN ('blog', 'cluster')
              AND status = 'pending'
        """), {"sid": site_id}).all()
        for row in pend:
            sig = row.source_signal or {}
            if sig.get("target_keyword"):
                covered.add(str(sig["target_keyword"]).lower())
            if sig.get("primary_kw"):
                covered.add(str(sig["primary_kw"]).lower())
    return covered


def _load_clusters(site_id: int) -> list[BlogPlan]:
    """Group active keywords by target_url. Each cluster gets one BlogPlan."""
    with SessionLocal() as s:
        # Tiebreaker on keyword pick: among same-volume keywords, the shortest
        # one is usually the cleanest head term ("beaded bracelet" beats
        # "beaded bracelet beads"). Sort vol DESC, then char length ASC.
        rows = s.execute(text("""
            SELECT
                target_url,
                SUM(search_volume) AS total_vol,
                ARRAY_AGG(keyword ORDER BY search_volume DESC NULLS LAST, LENGTH(keyword) ASC) AS kws,
                ARRAY_AGG(search_volume ORDER BY search_volume DESC NULLS LAST, LENGTH(keyword) ASC) AS vols
            FROM seo_keywords
            WHERE site_id = :sid AND status = 'active' AND target_url IS NOT NULL
            GROUP BY target_url
            ORDER BY total_vol DESC NULLS LAST
        """), {"sid": site_id}).all()

    plans: list[BlogPlan] = []
    for r in rows:
        kws = list(r.kws or [])
        if not kws:
            continue
        primary = kws[0]
        extras = kws[1:6]  # cap extras for prompt size
        plans.append(BlogPlan(
            primary_kw=primary,
            extra_kws=extras,
            total_volume=int(r.total_vol or 0),
            target_product_url=r.target_url,
            target_product_name=_product_name_from_url(r.target_url),
            cluster_keywords=kws,
        ))
    return plans


def pick_next_blog_topic(site_id: int) -> Optional[BlogPlan]:
    """Return the highest-opportunity not-yet-blogged cluster, or None."""
    covered = _load_already_blogged(site_id)
    clusters = _load_clusters(site_id)
    for plan in clusters:  # already sorted by total_vol DESC
        if plan.primary_kw.lower() in covered:
            continue
        # Also skip if any cluster keyword overlaps a covered one — prevents
        # near-duplicate "beaded bracelet" vs "beaded bracelet beads" double-up.
        if any(k.lower() in covered for k in plan.cluster_keywords):
            continue
        return plan
    return None


def generate_angle(plan: BlogPlan, profile: BrandProfile) -> str:
    """Ask Kimi to propose ONE specific blog angle for this cluster.

    Returns a single sentence/short paragraph the writer can use as brief.topic.
    Falls back to a templated angle if the LLM call fails so we never block
    the engine.
    """
    voice_tone = (profile.voice.get("tone") if isinstance(profile.voice, dict) else "") or ""
    system = (
        f"You are an SEO editorial planner for {profile.display_name}. "
        f"Brand: {profile.brand_one_liner or profile.display_name}. "
        f"Voice tone: {voice_tone or 'on-brand'}. "
        f"Audience: {profile.target_audience or 'general'}. "
        "Your job is to propose ONE specific, useful, non-generic blog angle "
        "that targets the given keyword. Avoid 'ultimate guide' phrasing. "
        "The angle should give the writer enough hook to write a 800-1200 word "
        "post that helps a real reader, not just chases SEO. "
        "Reply with ONE sentence — no numbering, no preamble, no quotes."
    )
    user = (
        f"Target keyword: {plan.primary_kw}\n"
        f"Cluster variants: {', '.join(plan.extra_kws)}\n"
        f"Related product: {plan.target_product_name} ({plan.target_product_url})\n\n"
        "Propose the blog angle."
    )
    try:
        r = httpx.post(
            OLLAMA_URL,
            json={
                "model": ANGLE_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.6, "num_predict": 120},
            },
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {}) or {}
        # Reasoning models on local Ollama (Kimi/Qwen3.5) emit nothing in
        # `content`, output lands in `thinking`/`reasoning`. Non-reasoning
        # models put it in `content`. Read both so we tolerate either shape.
        text_out = (
            msg.get("content")
            or msg.get("thinking")
            or msg.get("reasoning")
            or ""
        ).strip()
        # Strip surrounding quotes/asterisks if model wrapped the response
        text_out = text_out.strip('"\'* ')
        # Take only first paragraph — if the model rambled despite instructions
        text_out = text_out.split("\n\n")[0].strip()
        if text_out:
            log.info("angle generated for %r: %r", plan.primary_kw, text_out[:100])
            return text_out
    except Exception:
        log.exception("angle generation failed; falling back to template")

    # Templated fallback — safe, generic, but better than no angle
    return (
        f"Write a useful guide for shoppers searching '{plan.primary_kw}' — "
        f"cover what to look for, how to style with everyday outfits, and why "
        f"handmade pieces from a small studio differ from mass-produced ones. "
        f"Naturally reference {plan.target_product_name} as an example without "
        f"making the post feel like a product ad."
    )
