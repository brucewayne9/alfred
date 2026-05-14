"""Meta Graph API client for Roen Handmade.

Three surfaces:
- Catalog: batch upsert / delete product items in the Roen Commerce catalog
- FB Page: publish + delete posts on the roenhandmade Page
- IG: media publish (gated on App Review approval — code present, raises until live)

Auth: System User token (long-lived, never expires) — see config/.env.
All required IDs (app, page, IG user, catalog) live in config.settings.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import requests

logger = logging.getLogger(__name__)

GRAPH_VERSION = "v23.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_VERSION}"


def _config() -> dict:
    """Pull Roen Meta config via the settings module (per never-read-env-at-import rule)."""
    from config.settings import settings
    token = settings.roen_meta_system_user_token
    if not token:
        raise RuntimeError(
            "ROEN_META_SYSTEM_USER_TOKEN is not set. Check /home/aialfred/alfred/config/.env."
        )
    return {
        "token": token,
        "page_id": settings.roen_meta_page_id,
        "ig_user_id": settings.roen_meta_ig_user_id,
        "catalog_id": settings.roen_meta_catalog_id,
        "site_base": settings.roen_site_base_url,
    }


# Page Access Token (derived from system user token via /me/accounts).
# Process-local cache: Page Access Tokens derived from a never-expiring System
# User token are themselves long-lived. Re-fetch on first need or on auth error.
_PAGE_TOKEN_CACHE: dict[str, str] = {}


def _page_access_token(force_refresh: bool = False) -> str:
    """Fetch the Page Access Token for the configured Roen Page.

    The system user token can manage catalogs but Page write endpoints
    (/photos, /feed) require a Page Access Token derived via /me/accounts.
    """
    cfg = _config()
    pid = cfg["page_id"]
    if not force_refresh and pid in _PAGE_TOKEN_CACHE:
        return _PAGE_TOKEN_CACHE[pid]
    r = requests.get(
        f"{BASE_URL}/me/accounts",
        params={"fields": "id,access_token", "access_token": cfg["token"]},
        timeout=15,
    )
    r.raise_for_status()
    for page in r.json().get("data", []):
        if page.get("id") == pid:
            tok = page.get("access_token")
            if not tok:
                raise RuntimeError(f"no access_token returned for page {pid}")
            _PAGE_TOKEN_CACHE[pid] = tok
            return tok
    raise RuntimeError(f"page {pid} not in /me/accounts response")


# ---------- Catalog ----------

def list_catalog_products(after: str | None = None, limit: int = 100) -> dict:
    """Page through products currently in the Meta catalog. Used to detect stale items."""
    cfg = _config()
    params = {
        "fields": "id,retailer_id,name,availability,price",
        "limit": limit,
        "access_token": cfg["token"],
    }
    if after:
        params["after"] = after
    r = requests.get(f"{BASE_URL}/{cfg['catalog_id']}/products", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def list_all_catalog_retailer_ids() -> list[str]:
    """Walk full pagination, return every retailer_id in the catalog."""
    out: list[str] = []
    after: str | None = None
    while True:
        page = list_catalog_products(after=after, limit=100)
        for item in page.get("data", []):
            rid = item.get("retailer_id")
            if rid:
                out.append(rid)
        cursors = page.get("paging", {}).get("cursors", {})
        next_after = cursors.get("after")
        has_next = bool(page.get("paging", {}).get("next"))
        if not (has_next and next_after) or next_after == after:
            break
        after = next_after
    return out


def items_batch(requests_body: list[dict], allow_upsert: bool = True) -> dict:
    """Submit a batch of catalog mutations.

    Each request is {"method": "UPDATE"|"DELETE", "retailer_id": "...", "data": {...}}.
    Meta caps at ~5000 requests per batch — caller should chunk.
    """
    cfg = _config()
    body = {
        "access_token": cfg["token"],
        "item_type": "PRODUCT_ITEM",
        "allow_upsert": allow_upsert,
        "requests": requests_body,
    }
    r = requests.post(
        f"{BASE_URL}/{cfg['catalog_id']}/items_batch",
        json=body,
        timeout=60,
    )
    if r.status_code >= 400:
        logger.error("catalog items_batch failed: %s %s", r.status_code, r.text)
        r.raise_for_status()
    return r.json()


def upsert_products(items: list[dict]) -> dict:
    """Upsert a list of products. Each item must have retailer_id + the data fields.

    Required data fields per Meta: name, description, url, image_url, price, availability,
    condition, brand. We always send these.
    """
    if not items:
        return {"handles": [], "status": "noop"}
    batch = []
    for it in items:
        rid = it["retailer_id"]
        # Meta items_batch wants the retailer_id INSIDE data as "id"
        data = {k: v for k, v in it.items() if k != "retailer_id"}
        data["id"] = rid
        batch.append({"method": "UPDATE", "data": data})
    return items_batch(batch, allow_upsert=True)


def delete_products(retailer_ids: Iterable[str]) -> dict:
    rids = list(retailer_ids)
    if not rids:
        return {"handles": [], "status": "noop"}
    batch = [{"method": "DELETE", "data": {"id": rid}} for rid in rids]
    return items_batch(batch, allow_upsert=False)


# ---------- FB Page ----------

def page_post_photo(image_url: str, message: str, published: bool = True) -> dict:
    """Post a photo to the roenhandmade FB Page.

    For draft-and-approve flow, we store the intent locally and call this with
    published=True at approval time. We don't use Meta's own draft system because
    it's clunky and Mike wants the approval surface in his admin UI.
    """
    cfg = _config()
    if not cfg["page_id"]:
        raise RuntimeError("ROEN_META_PAGE_ID is empty.")
    body = {
        "url": image_url,
        "message": message,
        "published": "true" if published else "false",
        "access_token": _page_access_token(),
    }
    r = requests.post(f"{BASE_URL}/{cfg['page_id']}/photos", data=body, timeout=30)
    # Auth issues sometimes mean a stale cached token — retry once after refresh.
    if r.status_code in (401, 403):
        logger.warning("page photo 401/403 — refreshing page access token and retrying")
        body["access_token"] = _page_access_token(force_refresh=True)
        r = requests.post(f"{BASE_URL}/{cfg['page_id']}/photos", data=body, timeout=30)
    if r.status_code >= 400:
        logger.error("page photo post failed: %s %s", r.status_code, r.text)
        r.raise_for_status()
    return r.json()


def page_post_link(link: str, message: str, published: bool = True) -> dict:
    cfg = _config()
    body = {
        "link": link,
        "message": message,
        "published": "true" if published else "false",
        "access_token": _page_access_token(),
    }
    r = requests.post(f"{BASE_URL}/{cfg['page_id']}/feed", data=body, timeout=30)
    if r.status_code in (401, 403):
        body["access_token"] = _page_access_token(force_refresh=True)
        r = requests.post(f"{BASE_URL}/{cfg['page_id']}/feed", data=body, timeout=30)
    if r.status_code >= 400:
        logger.error("page feed post failed: %s %s", r.status_code, r.text)
        r.raise_for_status()
    return r.json()


def page_delete_post(post_id: str) -> bool:
    r = requests.delete(
        f"{BASE_URL}/{post_id}",
        params={"access_token": _page_access_token()},
        timeout=15,
    )
    return r.status_code in (200, 204)


# ---------- Instagram (gated on App Review for instagram_content_publish) ----------

def ig_create_media_container(image_url: str, caption: str, media_type: str = "IMAGE") -> str:
    """Step 1 of IG publish. Returns container_id to feed into ig_publish_media.

    Will raise (200) without instagram_content_publish Advanced Access until App
    Review clears. Keep the call path wired so it just-works on approval.
    """
    cfg = _config()
    if not cfg["ig_user_id"]:
        raise RuntimeError("ROEN_META_IG_USER_ID is empty.")
    body: dict[str, Any] = {
        "caption": caption,
        "access_token": cfg["token"],
        "media_type": media_type,
    }
    if media_type in ("IMAGE", "CAROUSEL_ITEM"):
        body["image_url"] = image_url
    elif media_type in ("REELS", "VIDEO"):
        body["video_url"] = image_url
    r = requests.post(f"{BASE_URL}/{cfg['ig_user_id']}/media", data=body, timeout=30)
    if r.status_code >= 400:
        logger.error("ig media container create failed: %s %s", r.status_code, r.text)
        r.raise_for_status()
    return r.json()["id"]


def ig_publish_media(container_id: str, poll_interval: float = 3.0, max_wait: float = 90.0) -> dict:
    """Step 2 of IG publish. Polls the container until FINISHED, then publishes."""
    cfg = _config()
    # Poll container readiness
    deadline = time.time() + max_wait
    while time.time() < deadline:
        r = requests.get(
            f"{BASE_URL}/{container_id}",
            params={"fields": "status_code,status", "access_token": cfg["token"]},
            timeout=15,
        )
        r.raise_for_status()
        status = r.json().get("status_code", "")
        if status == "FINISHED":
            break
        if status in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"IG container {container_id} {status}: {r.json()}")
        time.sleep(poll_interval)
    # Publish
    pub = requests.post(
        f"{BASE_URL}/{cfg['ig_user_id']}/media_publish",
        data={"creation_id": container_id, "access_token": cfg["token"]},
        timeout=30,
    )
    if pub.status_code >= 400:
        logger.error("ig media publish failed: %s %s", pub.status_code, pub.text)
        pub.raise_for_status()
    return pub.json()


# ---------- Diagnostics ----------

def verify_access() -> dict:
    """End-to-end credential check. Returns a dict of what's reachable."""
    cfg = _config()
    out: dict[str, Any] = {"ok": True, "errors": []}
    try:
        r = requests.get(
            f"{BASE_URL}/{cfg['page_id']}",
            params={"fields": "id,name,fan_count", "access_token": cfg["token"]},
            timeout=15,
        )
        r.raise_for_status()
        out["page"] = r.json()
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"page: {e}")
    try:
        r = requests.get(
            f"{BASE_URL}/{cfg['ig_user_id']}",
            params={"fields": "id,username,followers_count", "access_token": cfg["token"]},
            timeout=15,
        )
        r.raise_for_status()
        out["ig"] = r.json()
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"ig: {e}")
    try:
        r = requests.get(
            f"{BASE_URL}/{cfg['catalog_id']}",
            params={
                "fields": "id,name,product_count,vertical",
                "access_token": cfg["token"],
            },
            timeout=15,
        )
        r.raise_for_status()
        out["catalog"] = r.json()
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"catalog: {e}")
    return out
