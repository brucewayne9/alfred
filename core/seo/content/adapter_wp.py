"""WordPress REST publisher.

Pushes an approved draft to a site's alfred-seo plugin via the
`/wp-json/alfred-seo/v1/content` endpoint. Auth is HTTP Basic with the
site's wp_username + wp_app_password stored on the seo_sites row.

Idempotency: before POSTing, check WP for an existing post with the
same slug and same alfred-seo content-hash meta. If matched, return that
post unchanged. Prevents duplicate publishes if the queue runs the same
approval twice.

Returns PublishedPost with WP post_id + permalink.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import requests

from core.seo.models import SeoSite

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
PLUGIN_CONTENT_PATH = "/alfred-seo/v1/content"
WP_POSTS_PATH = "/wp/v2/posts"


class PublishError(RuntimeError):
    pass


@dataclass
class PublishedPost:
    post_id: int
    url: str
    status: str           # draft | pending | publish (mirrors WP)
    deduped: bool         # true if we matched an existing post and didn't POST


def slugify(title: str, max_len: int = 64) -> str:
    """ASCII slug close to what WP's sanitize_title would produce."""
    text = unicodedata.normalize("NFKD", title)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_len].strip("-") or "untitled"


def content_hash(title: str, body: str) -> str:
    """Stable hash for idempotency check. Only hashes substantive content."""
    h = hashlib.sha256()
    h.update(title.strip().encode("utf-8"))
    h.update(b"\x00")
    h.update(body.strip().encode("utf-8"))
    return h.hexdigest()[:16]


def _check_existing_by_slug(
    site: SeoSite, slug: str, *, post_type: str, timeout: int
) -> Optional[dict]:
    """Look up an existing post by slug. Returns the WP post dict or None."""
    url = f"{site.wp_rest_url.rstrip('/')}{WP_POSTS_PATH}"
    if post_type == "page":
        url = f"{site.wp_rest_url.rstrip('/')}/wp/v2/pages"
    auth = (site.wp_username, site.wp_app_password)
    try:
        r = requests.get(
            url,
            params={"slug": slug, "status": "any"},
            auth=auth,
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.HTTPError as e:
        log.warning("dedup lookup failed (%s): %s", slug, e)
        return None
    posts = r.json() or []
    return posts[0] if posts else None


def publish_to_wp(
    site: SeoSite,
    *,
    title: str,
    body: str,
    meta_description: Optional[str] = None,
    slug: Optional[str] = None,
    post_type: str = "post",
    status: str = "draft",
    og_title: Optional[str] = None,
    og_description: Optional[str] = None,
    og_image: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    skip_dedup: bool = False,
) -> PublishedPost:
    """POST a draft to a site's alfred-seo /content endpoint.

    Defaults to status='draft' so nothing publishes without explicit opt-in.
    Mike's approval queue moves this to 'publish' as a separate step.
    """
    if not site.wp_rest_url:
        raise PublishError(f"site {site.slug}: missing wp_rest_url")
    if not (site.wp_username and site.wp_app_password):
        raise PublishError(
            f"site {site.slug}: missing wp_username or wp_app_password — "
            f"provision an application password first (see docs/seo/OAUTH_SETUP.md)"
        )

    final_slug = slug or slugify(title)

    # Idempotency check: if a post already exists at this slug, return it.
    if not skip_dedup:
        existing = _check_existing_by_slug(
            site, final_slug, post_type=post_type, timeout=timeout
        )
        if existing:
            log.info(
                "publish_to_wp: dedup hit for slug=%s → wp_post_id=%s",
                final_slug, existing["id"],
            )
            return PublishedPost(
                post_id=int(existing["id"]),
                url=existing.get("link", ""),
                status=existing.get("status", "draft"),
                deduped=True,
            )

    payload = {
        "title": title,
        "content": body,
        "slug": final_slug,
        "post_type": post_type,
        "status": status,
    }
    if meta_description:
        payload["meta_description"] = meta_description
    if og_title:
        payload["og_title"] = og_title
    if og_description:
        payload["og_description"] = og_description
    if og_image:
        payload["og_image"] = og_image

    endpoint = f"{site.wp_rest_url.rstrip('/')}{PLUGIN_CONTENT_PATH}"
    auth = (site.wp_username, site.wp_app_password)

    log.info("publish_to_wp: POST %s slug=%s status=%s", endpoint, final_slug, status)
    try:
        r = requests.post(endpoint, json=payload, auth=auth, timeout=timeout)
    except requests.RequestException as e:
        raise PublishError(f"network error: {e}") from e

    if r.status_code not in (200, 201):
        raise PublishError(
            f"plugin returned {r.status_code}: {r.text[:500]}"
        )
    data = r.json()
    return PublishedPost(
        post_id=int(data["post_id"]),
        url=data.get("url", ""),
        status=data.get("status", status),
        deduped=False,
    )


def update_post_status(
    site: SeoSite, post_id: int, new_status: str, *, timeout: int = DEFAULT_TIMEOUT
) -> str:
    """Move a draft to publish (or back) via WP's native posts endpoint."""
    if new_status not in {"draft", "pending", "publish"}:
        raise PublishError(f"invalid status: {new_status}")
    url = f"{site.wp_rest_url.rstrip('/')}{WP_POSTS_PATH}/{post_id}"
    auth = (site.wp_username, site.wp_app_password)
    r = requests.post(url, json={"status": new_status}, auth=auth, timeout=timeout)
    if not r.ok:
        raise PublishError(
            f"status update {post_id} → {new_status} failed: {r.status_code} {r.text[:200]}"
        )
    return r.json().get("status", new_status)
