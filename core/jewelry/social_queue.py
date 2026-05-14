"""Pending social-post queue for Roen Handmade FB Page (and later IG).

Drafts created by the publish flow land here as JSON files. Mike approves
or rejects each via /admin/roen/social-pending. Approval posts to the
roenhandmade FB Page via the Meta integration.

Storage layout:
    data/roen/social_pending/<draft_id>.json   — awaiting decision
    data/roen/social_decided/<draft_id>.json   — approved/rejected/posted/failed
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path("/home/aialfred/alfred/data/roen")
PENDING_DIR = ROOT / "social_pending"
DECIDED_DIR = ROOT / "social_decided"


def _ensure_dirs() -> None:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    DECIDED_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enqueue_draft(
    wc_product_id: int,
    product_name: str,
    product_price: str,
    product_url: str,
    image_url: str,
    caption: str,
    source: str = "publish",
) -> dict:
    """Save a new FB Page draft awaiting Mike's approval. Returns the saved record."""
    _ensure_dirs()
    draft_id = secrets.token_urlsafe(8)
    record = {
        "draft_id": draft_id,
        "wc_product_id": wc_product_id,
        "product_name": product_name,
        "product_price": product_price,
        "product_url": product_url,
        "image_url": image_url,
        "caption": caption,
        "source": source,
        "channel": "fb_page",
        "created_at": _now(),
        "status": "pending",
        "post_id": None,
        "error": None,
    }
    path = PENDING_DIR / f"{draft_id}.json"
    path.write_text(json.dumps(record, indent=2))
    logger.info("enqueued roen draft %s for wc#%s", draft_id, wc_product_id)
    return record


def list_pending() -> list[dict]:
    _ensure_dirs()
    out = []
    for p in sorted(PENDING_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            logger.exception("failed to read %s", p)
    return out


def list_recent_decisions(limit: int = 20) -> list[dict]:
    _ensure_dirs()
    files = sorted(DECIDED_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    out = []
    for p in files[:limit]:
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            logger.exception("failed to read %s", p)
    return out


def _load(draft_id: str) -> tuple[Path, dict] | None:
    p = PENDING_DIR / f"{draft_id}.json"
    if not p.exists():
        return None
    return p, json.loads(p.read_text())


def _move_to_decided(path: Path, record: dict) -> None:
    target = DECIDED_DIR / path.name
    target.write_text(json.dumps(record, indent=2))
    path.unlink()


def reject(draft_id: str, reason: str = "") -> dict | None:
    found = _load(draft_id)
    if not found:
        return None
    path, record = found
    record["status"] = "rejected"
    record["decided_at"] = _now()
    record["reason"] = reason
    _move_to_decided(path, record)
    return record


def approve_and_post(draft_id: str) -> dict | None:
    """Approve a draft → publish to FB Page AND Instagram.

    Failures on one channel don't block the other. Each channel records its
    own post_id + any error. Overall status reflects whether at least one
    landed.
    """
    from integrations.meta_roen import client as meta
    found = _load(draft_id)
    if not found:
        return None
    path, record = found

    image_url = record["image_url"]
    caption = record["caption"]

    # ---- FB Page ----
    fb_ok = False
    try:
        resp = meta.page_post_photo(image_url=image_url, message=caption, published=True)
        record["fb_post_id"] = resp.get("post_id") or resp.get("id")
        record["fb_response"] = resp
        fb_ok = True
    except Exception as e:
        logger.exception("page_post_photo failed for draft %s", draft_id)
        record["fb_error"] = str(e)

    # ---- Instagram ----
    ig_ok = False
    try:
        container_id = meta.ig_create_media_container(image_url, caption, media_type="IMAGE")
        ig_resp = meta.ig_publish_media(container_id)
        record["ig_media_id"] = ig_resp.get("id")
        record["ig_response"] = ig_resp
        ig_ok = True
    except Exception as e:
        logger.exception("ig publish failed for draft %s", draft_id)
        record["ig_error"] = str(e)

    if fb_ok and ig_ok:
        record["status"] = "posted"
    elif fb_ok or ig_ok:
        record["status"] = "partial"
    else:
        record["status"] = "failed"
    # Legacy field for the existing admin UI ("Recent decisions" Link/note column)
    record["post_id"] = record.get("fb_post_id") or record.get("ig_media_id")
    record["error"] = record.get("fb_error") or record.get("ig_error")

    record["decided_at"] = _now()
    _move_to_decided(path, record)
    return record


def build_default_caption(name: str, price: str, url: str) -> str:
    """Naïve fallback caption. Bot replaces this with an LLM-written one in production."""
    return f"{name} — {price}\n\nHandmade in Atlanta. Shop: {url}"
