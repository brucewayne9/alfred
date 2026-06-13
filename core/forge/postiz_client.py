"""Forge ⇄ Postiz (Mainstay org) — push finished clips as TikTok/IG **drafts**.

Forge's burner accounts live in their own Postiz *organization* (Mainstay Music
Group), separate from the Ground Rush org. That org is reached with its own API
key, so Forge does NOT reuse Oracle's ``postiz.py`` (which is hard-wired to the
Ground Rush key and never sends TikTok's required ``settings`` block).

Two auth surfaces, mirroring the proven Postiz contract:
  * ``/public/v1/posts``        — org-scoped, uses the Mainstay **API key**.
  * ``/media/upload-simple``    — session-scoped, uses a **JWT** (instance secret).

Everything here creates **drafts** only (``type: "draft"``) with
``content_posting_method: "UPLOAD"`` — TikTok drops the video into the creator's
app inbox/drafts and the human taps Post. Nothing this module does auto-publishes.

Secrets come from ``config/.env`` at call-time (never at import) because
``forge-web.service`` runs with a stripped environment and pydantic Settings
ignores undeclared fields — so we read the file directly. See
``feedback_never_read_env_at_import``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_ENV_FILE = Path(__file__).resolve().parents[2] / "config" / ".env"
_DEFAULT_BASE = "https://social.groundrushlabs.com/api"


def _env(name: str, default: str | None = None) -> str | None:
    """Read one key from config/.env at call-time (no import-time env reads)."""
    try:
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass
    import os
    return os.environ.get(name, default)


# Per-org Postiz API keys. Each org's burners live in its own Postiz org,
# reached with its own key. mainstay = Rod Wave burners; rucktalk = the
# RuckTalk / Ground Rush org.
_ORG_KEY_ENV = {
    "mainstay": "POSTIZ_MAINSTAY_API_KEY",
    "rucktalk": "POSTIZ_RUCKTALK_API_KEY",
}


def key_for_org(org: str) -> str | None:
    return _env(_ORG_KEY_ENV.get(org, ""))


def _base() -> str:
    return (_env("POSTIZ_BASE_URL") or _DEFAULT_BASE).rstrip("/")


def is_configured() -> bool:
    """True when the Mainstay org API key is present."""
    return bool(_env("POSTIZ_MAINSTAY_API_KEY"))


def _jwt() -> str | None:
    """Session JWT for media upload (instance-wide secret + Postiz user id)."""
    secret = _env("POSTIZ_JWT_SECRET")
    uid = _env("POSTIZ_USER_ID")
    if not (secret and uid):
        return None
    import jwt  # PyJWT
    return jwt.encode(
        {"id": uid, "email": "mjohnson@groundrushlabs.com",
         "providerName": "LOCAL", "activated": True},
        secret, algorithm="HS256",
    )


def list_integrations(org: str = "mainstay") -> list[dict]:
    """All channels connected in ``org``'s Postiz org (id, name, provider)."""
    key = key_for_org(org)
    if not key:
        logger.error("Postiz API key for org '%s' missing — cannot list integrations", org)
        return []
    try:
        r = requests.get(f"{_base()}/public/v1/integrations",
                         headers={"Authorization": key}, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else (data.get("integrations") or [])
    except Exception as exc:  # noqa: BLE001
        logger.error("Postiz list_integrations failed: %s", str(exc)[:200])
        return []


def upload_media(local_path: str | Path) -> dict | None:
    """Upload a local asset to the Postiz media library; return {id, path} or None."""
    tok = _jwt()
    if not tok:
        logger.error("Postiz JWT not configured — cannot upload media")
        return None
    p = Path(local_path)
    ext = p.suffix.lower().lstrip(".") or "mp4"
    ct = {"mp4": "video/mp4", "mov": "video/quicktime", "png": "image/png",
          "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
          "gif": "image/gif"}.get(ext, "application/octet-stream")
    try:
        with open(p, "rb") as fh:
            r = requests.post(f"{_base()}/media/upload-simple",
                              headers={"auth": tok},
                              files={"file": (p.name, fh, ct)}, timeout=120)
        if r.status_code >= 400:
            logger.error("Postiz media upload HTTP %s: %s", r.status_code, r.text[:200])
            return None
        j = r.json()
        return {"id": j.get("id"), "path": j.get("path")}
    except Exception as exc:  # noqa: BLE001
        logger.error("Postiz media upload error: %s", str(exc)[:200])
        return None


def _settings_for(platform: str, *, title: str = "") -> dict:
    """Platform-specific ``settings`` block Postiz requires for a draft."""
    p = (platform or "").lower()
    if p == "tiktok":
        # UPLOAD = drop the video into the creator's TikTok app drafts/inbox; they tap
        # Post (and may choose public). This is the ONLY method an unaudited/sandbox app
        # can use — DIRECT_POST is rejected by TikTok with "App not approved for public
        # posting" (even for private) until the production audit clears. Flip back to
        # DIRECT_POST for hands-off auto-posting once the app is approved.
        return {
            "content_posting_method": "UPLOAD",
            "privacy_level": "SELF_ONLY",
            "duet": False, "stitch": False, "comment": True,
            "autoAddMusic": "no",
            "brand_content_toggle": False, "brand_organic_toggle": False,
            "title": title or "",
        }
    if p in ("instagram", "instagram reels", "instagram-reels"):
        return {"post_type": "post"}
    return {}


def create_draft(content: str, integration_id: str, *,
                 media_path: str | Path | None = None,
                 platform: str = "tiktok",
                 when: str | None = None,
                 title: str = "",
                 scheduled: bool = False,
                 org: str = "mainstay") -> dict:
    """Create ONE Postiz post on ``integration_id`` (in ``org``'s Postiz org).

    ``scheduled=False`` → a **draft** (sits in Postiz until a human hits send).
    ``scheduled=True``  → a **scheduled** post that auto-fires at ``when`` (UTC ISO).

    Returns ``{"ok": bool, "postId": str|None, "error": str|None}``.
    """
    key = key_for_org(org)
    if not key:
        return {"ok": False, "postId": None,
                "error": f"Postiz API key for org '{org}' missing"}

    image_list: list[dict] = []
    if media_path:
        media = upload_media(media_path)
        if not media or not media.get("id"):
            return {"ok": False, "postId": None, "error": "media upload failed"}
        image_list = [{"id": media["id"], "path": media.get("path")}]

    date = when or (datetime.now(timezone.utc) + timedelta(hours=1)) \
        .replace(microsecond=0).isoformat().replace("+00:00", "")
    body = {
        "type": "schedule" if scheduled else "draft", "date": date,
        "shortLink": False, "tags": [],
        "posts": [{
            "integration": {"id": integration_id},
            "value": [{"content": content, "image": image_list}],
            "settings": _settings_for(platform, title=title),
        }],
    }
    try:
        r = requests.post(f"{_base()}/public/v1/posts",
                          headers={"Authorization": key}, json=body, timeout=60)
        if r.status_code >= 400:
            return {"ok": False, "postId": None, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        data = r.json()
        post_id = None
        if isinstance(data, list) and data:
            post_id = data[0].get("postId") or data[0].get("id")
        return {"ok": True, "postId": post_id, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "postId": None, "error": str(exc)[:200]}
