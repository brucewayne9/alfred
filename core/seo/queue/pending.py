"""Approval queue state machine for SEO drafts.

Mirrors the Roen social_queue.py pattern Mike already trusts:
  - Drafts land in seo_pending with status='pending'
  - Mike approves → publisher pushes to WP → row moves to seo_decided (outcome=approved)
  - Mike rejects → row moves to seo_decided (outcome=rejected, no WP write)
  - On publish failure → row moves to seo_decided (outcome=publish_failed) with error text;
    a NEW pending row is enqueued for retry (don't strand failed drafts).

State diagram:
  pending → publishing → decided(approved, wp_post_id)
  pending → decided(rejected)
  pending → decided(publish_failed, error)  + new pending row for retry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select

from core.seo.content.adapter_wp import (
    PublishError,
    PublishedPost,
    publish_to_wp,
    slugify,
)
from core.seo.content.writer import GeneratedDraft
from core.seo.db import SessionLocal
from core.seo.models import SeoDecided, SeoPending, SeoSite

log = logging.getLogger(__name__)


@dataclass
class EnqueueResult:
    pending_id: int
    site_slug: str
    title: str


@dataclass
class DecisionResult:
    decided_id: int
    outcome: str                         # approved | rejected | publish_failed
    wp_post_id: Optional[int] = None
    wp_url: Optional[str] = None
    error: Optional[str] = None


def enqueue_draft(
    site_id: int,
    *,
    draft: GeneratedDraft,
    brief_id: Optional[int] = None,
    source_signal: Optional[dict[str, Any]] = None,
    meta_description: Optional[str] = None,
    initial_status: str = "pending",
) -> EnqueueResult:
    """Insert a generated draft into seo_pending for human review."""
    body_payload = {
        "title": draft.title,
        "body": draft.body,
        "model": draft.model,
        "latency_s": draft.latency_s,
        "target_keyword": draft.target_keyword,
        "meta_description": meta_description,
        "slug": slugify(draft.title),
        "validation": _validation_to_dict(draft.validation),
    }
    with SessionLocal() as s:
        site = s.get(SeoSite, site_id)
        if not site:
            raise ValueError(f"unknown site_id: {site_id}")
        row = SeoPending(
            site_id=site_id,
            brief_id=brief_id,
            content_type=draft.content_type,
            title=draft.title,
            body_payload=body_payload,
            source_signal=source_signal or {},
            status=initial_status,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        log.info(
            "enqueue_draft: site=%s pending_id=%d title=%r",
            site.slug, row.id, draft.title,
        )
        return EnqueueResult(pending_id=row.id, site_slug=site.slug, title=draft.title)


def list_pending(site_id: Optional[int] = None) -> list[SeoPending]:
    """Return open pending rows, newest first. Detached from session."""
    with SessionLocal() as s:
        q = select(SeoPending).where(SeoPending.status == "pending").order_by(
            SeoPending.created_at.desc()
        )
        if site_id is not None:
            q = q.where(SeoPending.site_id == site_id)
        rows = list(s.scalars(q).all())
        for r in rows:
            s.expunge(r)
        return rows


def get_pending(pending_id: int) -> Optional[SeoPending]:
    with SessionLocal() as s:
        row = s.get(SeoPending, pending_id)
        if row:
            s.expunge(row)
        return row


def reject(pending_id: int, *, decided_by: str, reason: Optional[str] = None) -> DecisionResult:
    """Mark a pending draft as rejected. No WP write."""
    with SessionLocal() as s:
        row = s.get(SeoPending, pending_id)
        if not row:
            raise ValueError(f"unknown pending_id: {pending_id}")
        if row.status != "pending":
            raise ValueError(f"pending {pending_id} not in pending state: {row.status}")

        decided = SeoDecided(
            site_id=row.site_id,
            brief_id=row.brief_id,
            content_type=row.content_type,
            title=row.title,
            body_payload=_with_reject_reason(row.body_payload, reason),
            decided_by=decided_by,
            outcome="rejected",
        )
        s.add(decided)
        s.delete(row)
        s.commit()
        s.refresh(decided)
        log.info("reject: pending_id=%d → decided_id=%d (by %s)", pending_id, decided.id, decided_by)
        return DecisionResult(decided_id=decided.id, outcome="rejected")


def approve_and_publish(
    pending_id: int,
    *,
    decided_by: str,
    publish_status: str = "draft",
) -> DecisionResult:
    """Approve a draft and POST it to WP via the alfred-seo plugin.

    publish_status='draft' (safe default) leaves the post as a WP draft so
    Mike can preview before going live. Pass publish_status='publish' to
    push live in one step.

    On publish failure: mark the seo_decided row outcome='publish_failed'
    with the error text, AND re-enqueue a fresh seo_pending row so the work
    isn't lost (per spec line 431).
    """
    with SessionLocal() as s:
        row = s.get(SeoPending, pending_id)
        if not row:
            raise ValueError(f"unknown pending_id: {pending_id}")
        if row.status != "pending":
            raise ValueError(f"pending {pending_id} not in pending state: {row.status}")

        site = s.get(SeoSite, row.site_id)
        if not site:
            raise ValueError(f"site {row.site_id} for pending {pending_id} missing")

        body = row.body_payload or {}
        title = body.get("title") or row.title
        body_md = body.get("body") or ""
        slug = body.get("slug") or slugify(title)
        meta_desc = body.get("meta_description")

        # Map content_type → WP post_type. Cluster pages and ad landings live
        # as WP "pages"; blog and product enrichment land as posts.
        wp_post_type = "page" if row.content_type in {"cluster", "cluster_pages", "ad_landing"} else "post"

        try:
            published = publish_to_wp(
                site,
                title=title,
                body=body_md,
                meta_description=meta_desc,
                slug=slug,
                post_type=wp_post_type,
                status=publish_status,
            )
        except PublishError as e:
            log.error("approve_and_publish: WP publish failed for pending %d: %s", pending_id, e)
            decided = SeoDecided(
                site_id=row.site_id,
                brief_id=row.brief_id,
                content_type=row.content_type,
                title=row.title,
                body_payload=row.body_payload,
                decided_by=decided_by,
                outcome="publish_failed",
                error=str(e)[:1000],
            )
            s.add(decided)
            # Re-enqueue: fresh pending row referencing the same brief
            retry_row = SeoPending(
                site_id=row.site_id,
                brief_id=row.brief_id,
                content_type=row.content_type,
                title=row.title,
                body_payload=row.body_payload,
                source_signal={"retry_after": "publish_failed", "previous_pending_id": row.id},
                status="pending",
            )
            s.add(retry_row)
            s.delete(row)
            s.commit()
            s.refresh(decided)
            return DecisionResult(
                decided_id=decided.id,
                outcome="publish_failed",
                error=str(e),
            )

        decided = SeoDecided(
            site_id=row.site_id,
            brief_id=row.brief_id,
            content_type=row.content_type,
            title=row.title,
            body_payload=row.body_payload,
            decided_by=decided_by,
            outcome="approved",
            wp_post_id=published.post_id,
        )
        s.add(decided)
        s.delete(row)
        s.commit()
        s.refresh(decided)
        log.info(
            "approve_and_publish: pending_id=%d → decided_id=%d wp_post=%d (deduped=%s)",
            pending_id, decided.id, published.post_id, published.deduped,
        )
        return DecisionResult(
            decided_id=decided.id,
            outcome="approved",
            wp_post_id=published.post_id,
            wp_url=published.url,
        )


def _validation_to_dict(v: Any) -> dict[str, Any]:
    if v is None:
        return {}
    return {
        "ok": v.ok,
        "flesch": v.flesch,
        "word_count": v.word_count,
        "sentence_count": v.sentence_count,
        "issues": list(v.issues),
    }


def _with_reject_reason(payload: Any, reason: Optional[str]) -> dict[str, Any]:
    base = dict(payload or {})
    if reason:
        base["reject_reason"] = reason
        base["rejected_at"] = datetime.now(timezone.utc).isoformat()
    return base
