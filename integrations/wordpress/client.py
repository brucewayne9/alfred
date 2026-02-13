"""WordPress REST API client with multi-site support.

Supports:
- Posts, Pages, Media management
- RankMath SEO integration
- Plugin and Theme management
- Site health monitoring
- Elementor compatibility
"""

import base64
import logging
import os
from typing import Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")

logger = logging.getLogger(__name__)

# Site registry - loaded from environment
WORDPRESS_SITES: dict[str, dict] = {}


def _load_sites():
    """Load WordPress sites from environment variables."""
    global WORDPRESS_SITES

    # Format: WP_SITE_<name>_URL, WP_SITE_<name>_USER, WP_SITE_<name>_PASS
    site_names = os.getenv("WP_SITES", "").split(",")

    for name in site_names:
        name = name.strip()
        if not name:
            continue

        url = os.getenv(f"WP_SITE_{name.upper()}_URL", "")
        user = os.getenv(f"WP_SITE_{name.upper()}_USER", "")
        password = os.getenv(f"WP_SITE_{name.upper()}_PASS", "")

        if url and user and password:
            WORDPRESS_SITES[name.lower()] = {
                "name": name,
                "url": url.rstrip("/"),
                "user": user,
                "password": password.replace(" ", ""),  # Remove spaces from app passwords
            }
            logger.info(f"Loaded WordPress site: {name} ({url})")


# Load sites on module import
_load_sites()


def _get_site(site_name: str) -> dict:
    """Get site configuration by name."""
    site = WORDPRESS_SITES.get(site_name.lower())
    if not site:
        available = ", ".join(WORDPRESS_SITES.keys()) or "none configured"
        raise ValueError(f"Unknown WordPress site: {site_name}. Available: {available}")
    return site


def _get_auth_header(site: dict) -> dict:
    """Generate Basic Auth header for WordPress REST API."""
    credentials = f"{site['user']}:{site['password']}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def _api_request(
    site_name: str,
    endpoint: str,
    method: str = "GET",
    data: dict = None,
    files: dict = None,
) -> dict | list:
    """Make authenticated request to WordPress REST API."""
    site = _get_site(site_name)
    url = urljoin(site["url"] + "/", f"wp-json/wp/v2/{endpoint}")
    headers = _get_auth_header(site)

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data, timeout=30)
        elif method == "PUT":
            headers["Content-Type"] = "application/json"
            response = requests.put(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json() if response.content else {}

    except requests.exceptions.HTTPError as e:
        logger.error(f"WordPress API error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"WordPress request failed: {e}")
        raise


def _rankmath_request(
    site_name: str,
    endpoint: str,
    method: str = "GET",
    data: dict = None,
) -> dict:
    """Make request to RankMath REST API."""
    site = _get_site(site_name)
    url = urljoin(site["url"] + "/", f"wp-json/rankmath/v1/{endpoint}")
    headers = _get_auth_header(site)

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data, timeout=30)
        elif method == "POST":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, headers=headers, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json() if response.content else {}

    except Exception as e:
        logger.error(f"RankMath API error: {e}")
        return {"error": str(e)}


# ============ Site Management ============

def list_sites() -> list[dict]:
    """List all configured WordPress sites."""
    return [
        {"name": name, "url": site["url"]}
        for name, site in WORDPRESS_SITES.items()
    ]


def get_site(site_name: str) -> dict:
    """Get site details."""
    site = _get_site(site_name)
    return {"name": site["name"], "url": site["url"]}


def test_connection(site_name: str) -> dict:
    """Test connection to a WordPress site."""
    try:
        site = _get_site(site_name)
        url = urljoin(site["url"] + "/", "wp-json/wp/v2/users/me")
        headers = _get_auth_header(site)

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        user_data = response.json()

        return {
            "success": True,
            "site": site_name,
            "url": site["url"],
            "connected_as": user_data.get("name", "Unknown"),
            "roles": user_data.get("roles", []),
        }
    except Exception as e:
        return {"success": False, "site": site_name, "error": str(e)}


# ============ Posts ============

def get_posts(
    site_name: str,
    per_page: int = 10,
    page: int = 1,
    status: str = "any",
    search: str = None,
) -> list[dict]:
    """Get posts from a WordPress site."""
    params = {"per_page": per_page, "page": page, "status": status}
    if search:
        params["search"] = search

    posts = _api_request(site_name, "posts", data=params)

    return [
        {
            "id": p["id"],
            "title": p["title"]["rendered"],
            "slug": p["slug"],
            "status": p["status"],
            "date": p["date"],
            "link": p["link"],
            "excerpt": p["excerpt"]["rendered"][:200] if p.get("excerpt") else "",
        }
        for p in posts
    ]


def get_post(site_name: str, post_id: int) -> dict:
    """Get a single post by ID."""
    post = _api_request(site_name, f"posts/{post_id}")
    return {
        "id": post["id"],
        "title": post["title"]["rendered"],
        "content": post["content"]["rendered"],
        "slug": post["slug"],
        "status": post["status"],
        "date": post["date"],
        "link": post["link"],
        "categories": post.get("categories", []),
        "tags": post.get("tags", []),
        "featured_media": post.get("featured_media"),
    }


def create_post(
    site_name: str,
    title: str,
    content: str,
    status: str = "draft",
    categories: list[int] = None,
    tags: list[int] = None,
    featured_media: int = None,
) -> dict:
    """Create a new post."""
    data = {
        "title": title,
        "content": content,
        "status": status,
    }
    if categories:
        data["categories"] = categories
    if tags:
        data["tags"] = tags
    if featured_media:
        data["featured_media"] = featured_media

    post = _api_request(site_name, "posts", method="POST", data=data)
    return {
        "success": True,
        "id": post["id"],
        "title": post["title"]["rendered"],
        "link": post["link"],
        "status": post["status"],
    }


def update_post(
    site_name: str,
    post_id: int,
    title: str = None,
    content: str = None,
    status: str = None,
    categories: list[int] = None,
    tags: list[int] = None,
) -> dict:
    """Update an existing post."""
    data = {}
    if title:
        data["title"] = title
    if content:
        data["content"] = content
    if status:
        data["status"] = status
    if categories is not None:
        data["categories"] = categories
    if tags is not None:
        data["tags"] = tags

    post = _api_request(site_name, f"posts/{post_id}", method="POST", data=data)
    return {
        "success": True,
        "id": post["id"],
        "title": post["title"]["rendered"],
        "link": post["link"],
    }


def delete_post(site_name: str, post_id: int, force: bool = False) -> dict:
    """Delete a post (moves to trash unless force=True)."""
    params = {"force": force}
    _api_request(site_name, f"posts/{post_id}", method="DELETE", data=params)
    return {"success": True, "id": post_id, "deleted": True}


# ============ Pages ============

def get_pages(
    site_name: str,
    per_page: int = 10,
    page: int = 1,
    status: str = "any",
    search: str = None,
) -> list[dict]:
    """Get pages from a WordPress site."""
    params = {"per_page": per_page, "page": page, "status": status}
    if search:
        params["search"] = search

    pages = _api_request(site_name, "pages", data=params)

    return [
        {
            "id": p["id"],
            "title": p["title"]["rendered"],
            "slug": p["slug"],
            "status": p["status"],
            "link": p["link"],
            "parent": p.get("parent", 0),
            "template": p.get("template", ""),
        }
        for p in pages
    ]


def get_page(site_name: str, page_id: int) -> dict:
    """Get a single page by ID."""
    page = _api_request(site_name, f"pages/{page_id}")
    return {
        "id": page["id"],
        "title": page["title"]["rendered"],
        "content": page["content"]["rendered"],
        "slug": page["slug"],
        "status": page["status"],
        "link": page["link"],
        "parent": page.get("parent", 0),
        "template": page.get("template", ""),
        "featured_media": page.get("featured_media"),
    }


def create_page(
    site_name: str,
    title: str,
    content: str,
    status: str = "draft",
    parent: int = None,
    template: str = None,
) -> dict:
    """Create a new page."""
    data = {
        "title": title,
        "content": content,
        "status": status,
    }
    if parent:
        data["parent"] = parent
    if template:
        data["template"] = template

    page = _api_request(site_name, "pages", method="POST", data=data)
    return {
        "success": True,
        "id": page["id"],
        "title": page["title"]["rendered"],
        "link": page["link"],
        "status": page["status"],
    }


def update_page(
    site_name: str,
    page_id: int,
    title: str = None,
    content: str = None,
    status: str = None,
) -> dict:
    """Update an existing page."""
    data = {}
    if title:
        data["title"] = title
    if content:
        data["content"] = content
    if status:
        data["status"] = status

    page = _api_request(site_name, f"pages/{page_id}", method="POST", data=data)
    return {
        "success": True,
        "id": page["id"],
        "title": page["title"]["rendered"],
        "link": page["link"],
    }


# ============ Media ============

def get_media(
    site_name: str,
    per_page: int = 10,
    page: int = 1,
    media_type: str = None,
    search: str = None,
) -> list[dict]:
    """Get media items from the library."""
    params = {"per_page": per_page, "page": page}
    if media_type:
        params["media_type"] = media_type
    if search:
        params["search"] = search

    media = _api_request(site_name, "media", data=params)

    return [
        {
            "id": m["id"],
            "title": m["title"]["rendered"],
            "url": m["source_url"],
            "type": m["media_type"],
            "mime_type": m["mime_type"],
            "alt_text": m.get("alt_text", ""),
        }
        for m in media
    ]


def upload_media(
    site_name: str,
    file_path: str,
    title: str = None,
    alt_text: str = None,
) -> dict:
    """Upload media to WordPress."""
    import mimetypes
    from pathlib import Path

    file_path = Path(file_path)
    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    mime_type, _ = mimetypes.guess_type(str(file_path))

    site = _get_site(site_name)
    url = urljoin(site["url"] + "/", "wp-json/wp/v2/media")
    headers = _get_auth_header(site)
    headers["Content-Disposition"] = f'attachment; filename="{file_path.name}"'
    if mime_type:
        headers["Content-Type"] = mime_type

    with open(file_path, "rb") as f:
        response = requests.post(url, headers=headers, data=f, timeout=120)

    response.raise_for_status()
    media = response.json()

    # Update title/alt if provided
    if title or alt_text:
        update_data = {}
        if title:
            update_data["title"] = title
        if alt_text:
            update_data["alt_text"] = alt_text
        _api_request(site_name, f"media/{media['id']}", method="POST", data=update_data)

    return {
        "success": True,
        "id": media["id"],
        "url": media["source_url"],
        "title": media["title"]["rendered"],
    }


def upload_media_base64(
    site_name: str,
    base64_data: str,
    filename: str,
    title: str = None,
    alt_text: str = None,
) -> dict:
    """Upload media to WordPress from base64-encoded data.

    Args:
        site_name: Site identifier
        base64_data: Base64-encoded file content (with or without data URI prefix)
        filename: Desired filename (e.g., 'banner.jpg')
        title: Media title (optional)
        alt_text: Alt text for images (optional)

    Returns:
        dict with success, id, url, title
    """
    import tempfile
    from pathlib import Path

    try:
        # Strip data URI prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in base64_data and base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]

        # Decode base64
        file_bytes = base64.b64decode(base64_data)

        # Save to temp file
        temp_dir = Path(tempfile.gettempdir()) / "alfred_uploads"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / filename

        with open(temp_file, "wb") as f:
            f.write(file_bytes)

        # Upload using existing function
        result = upload_media(site_name, str(temp_file), title, alt_text)

        # Clean up temp file
        try:
            temp_file.unlink()
        except Exception:
            pass

        return result

    except Exception as e:
        logger.error(f"Failed to upload base64 media: {e}")
        return {"success": False, "error": str(e)}


def get_media_item(site_name: str, media_id: int) -> dict:
    """Get details of a specific media item.

    Args:
        site_name: Site identifier
        media_id: Media ID

    Returns:
        Media details including URL, title, alt text
    """
    try:
        media = _api_request(site_name, f"media/{media_id}")
        return {
            "success": True,
            "id": media["id"],
            "title": media["title"]["rendered"],
            "url": media["source_url"],
            "alt_text": media.get("alt_text", ""),
            "mime_type": media["mime_type"],
            "media_type": media["media_type"],
            "width": media.get("media_details", {}).get("width"),
            "height": media.get("media_details", {}).get("height"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_media(site_name: str, media_id: int, force: bool = True) -> dict:
    """Delete a media item from WordPress.

    Args:
        site_name: Site identifier
        media_id: Media ID to delete
        force: Permanently delete (True) or trash (False)

    Returns:
        Success status
    """
    try:
        _api_request(site_name, f"media/{media_id}", method="DELETE", data={"force": force})
        return {"success": True, "id": media_id, "deleted": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ SEO (RankMath) ============

def get_seo_meta(site_name: str, post_id: int, post_type: str = "post") -> dict:
    """Get RankMath SEO metadata for a post/page."""
    try:
        # RankMath stores meta in post meta
        endpoint = f"posts/{post_id}" if post_type == "post" else f"pages/{post_id}"
        post = _api_request(site_name, endpoint, data={"context": "edit"})

        # RankMath meta fields
        meta = post.get("meta", {})
        return {
            "success": True,
            "post_id": post_id,
            "seo_title": meta.get("rank_math_title", ""),
            "seo_description": meta.get("rank_math_description", ""),
            "focus_keyword": meta.get("rank_math_focus_keyword", ""),
            "seo_score": meta.get("rank_math_seo_score", 0),
            "canonical_url": meta.get("rank_math_canonical_url", ""),
            "robots": meta.get("rank_math_robots", []),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_seo_meta(
    site_name: str,
    post_id: int,
    post_type: str = "post",
    seo_title: str = None,
    seo_description: str = None,
    focus_keyword: str = None,
    canonical_url: str = None,
) -> dict:
    """Update RankMath SEO metadata for a post/page."""
    try:
        endpoint = f"posts/{post_id}" if post_type == "post" else f"pages/{post_id}"

        meta = {}
        if seo_title:
            meta["rank_math_title"] = seo_title
        if seo_description:
            meta["rank_math_description"] = seo_description
        if focus_keyword:
            meta["rank_math_focus_keyword"] = focus_keyword
        if canonical_url:
            meta["rank_math_canonical_url"] = canonical_url

        _api_request(site_name, endpoint, method="POST", data={"meta": meta})

        return {
            "success": True,
            "post_id": post_id,
            "updated_fields": list(meta.keys()),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_seo_score(site_name: str, post_id: int, post_type: str = "post") -> dict:
    """Get RankMath SEO score for a post/page."""
    meta = get_seo_meta(site_name, post_id, post_type)
    if not meta.get("success"):
        return meta

    return {
        "success": True,
        "post_id": post_id,
        "score": meta.get("seo_score", 0),
        "focus_keyword": meta.get("focus_keyword", ""),
        "has_title": bool(meta.get("seo_title")),
        "has_description": bool(meta.get("seo_description")),
    }


# ============ Plugins ============

def get_plugins(site_name: str) -> list[dict]:
    """Get list of installed plugins."""
    plugins = _api_request(site_name, "plugins")

    return [
        {
            "plugin": p["plugin"],
            "name": p["name"],
            "status": p["status"],
            "version": p.get("version", ""),
            "author": p.get("author", ""),
            "requires_wp": p.get("requires", ""),
        }
        for p in plugins
    ]


def activate_plugin(site_name: str, plugin_slug: str) -> dict:
    """Activate a plugin."""
    try:
        _api_request(site_name, f"plugins/{plugin_slug}", method="POST", data={"status": "active"})
        return {"success": True, "plugin": plugin_slug, "status": "active"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def deactivate_plugin(site_name: str, plugin_slug: str) -> dict:
    """Deactivate a plugin."""
    try:
        _api_request(site_name, f"plugins/{plugin_slug}", method="POST", data={"status": "inactive"})
        return {"success": True, "plugin": plugin_slug, "status": "inactive"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_plugin(site_name: str, plugin_slug: str) -> dict:
    """Update a plugin to latest version."""
    # WordPress REST API doesn't have native plugin update
    # This would require WP-CLI or a custom endpoint
    return {"success": False, "error": "Plugin updates require WP-CLI or custom endpoint"}


# ============ Themes ============

def get_themes(site_name: str) -> list[dict]:
    """Get list of installed themes."""
    themes = _api_request(site_name, "themes")

    return [
        {
            "stylesheet": t["stylesheet"],
            "name": t["name"]["rendered"] if isinstance(t["name"], dict) else t["name"],
            "status": t["status"],
            "version": t.get("version", ""),
            "author": t.get("author", {}).get("raw", "") if isinstance(t.get("author"), dict) else "",
        }
        for t in themes
    ]


def activate_theme(site_name: str, theme_slug: str) -> dict:
    """Activate a theme."""
    try:
        _api_request(site_name, f"themes/{theme_slug}", method="POST", data={"status": "active"})
        return {"success": True, "theme": theme_slug, "status": "active"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============ Users ============

def get_users(site_name: str, per_page: int = 10) -> list[dict]:
    """Get list of users."""
    users = _api_request(site_name, "users", data={"per_page": per_page})

    return [
        {
            "id": u["id"],
            "name": u["name"],
            "slug": u["slug"],
            "email": u.get("email", ""),
            "roles": u.get("roles", []),
        }
        for u in users
    ]


# ============ Site Health ============

def get_site_health(site_name: str) -> dict:
    """Get basic site health information."""
    try:
        site = _get_site(site_name)

        # Get some basic stats
        posts = _api_request(site_name, "posts", data={"per_page": 1, "status": "publish"})
        pages = _api_request(site_name, "pages", data={"per_page": 1, "status": "publish"})

        # Get plugins
        try:
            plugins = get_plugins(site_name)
            active_plugins = [p for p in plugins if p["status"] == "active"]
            inactive_plugins = [p for p in plugins if p["status"] == "inactive"]
        except Exception:
            active_plugins = []
            inactive_plugins = []

        return {
            "success": True,
            "site": site_name,
            "url": site["url"],
            "total_posts": len(posts) if posts else 0,
            "total_pages": len(pages) if pages else 0,
            "active_plugins": len(active_plugins),
            "inactive_plugins": len(inactive_plugins),
            "plugins_need_update": 0,  # Would need WP-CLI to check
        }
    except Exception as e:
        return {"success": False, "site": site_name, "error": str(e)}


# ============ Bulk Operations ============

def test_all_connections() -> list[dict]:
    """Test connections to all configured WordPress sites."""
    results = []
    for site_name in WORDPRESS_SITES:
        results.append(test_connection(site_name))
    return results


# ============ Elementor Integration ============

def get_elementor_data(site_name: str, post_id: int) -> dict:
    """Get Elementor data for a page/post."""
    try:
        site = _get_site(site_name)
        # Elementor data is stored in post meta
        endpoint = f"pages/{post_id}" if True else f"posts/{post_id}"  # Try pages first
        try:
            data = _api_request(site_name, endpoint, data={"context": "edit"})
        except:
            data = _api_request(site_name, f"posts/{post_id}", data={"context": "edit"})

        meta = data.get("meta", {})
        elementor_data = meta.get("_elementor_data", "[]")

        # Parse if string
        if isinstance(elementor_data, str):
            import json
            elementor_data = json.loads(elementor_data) if elementor_data else []

        return {
            "success": True,
            "post_id": post_id,
            "elementor_data": elementor_data,
            "edit_mode": meta.get("_elementor_edit_mode", ""),
            "template_type": meta.get("_elementor_template_type", ""),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_elementor_data(
    site_name: str,
    post_id: int,
    elementor_data: list | str,
    post_type: str = "page",
) -> dict:
    """Save Elementor data to a page/post.

    Args:
        site_name: Site identifier
        post_id: Page or post ID
        elementor_data: Elementor JSON data (list of sections or JSON string)
        post_type: 'page' or 'post'
    """
    import json

    try:
        # Convert to string if needed
        if isinstance(elementor_data, list):
            elementor_json = json.dumps(elementor_data)
        else:
            elementor_json = elementor_data

        endpoint = f"pages/{post_id}" if post_type == "page" else f"posts/{post_id}"

        # Update post meta with Elementor data
        data = {
            "meta": {
                "_elementor_data": elementor_json,
                "_elementor_edit_mode": "builder",
                "_elementor_template_type": "wp-page" if post_type == "page" else "wp-post",
            }
        }

        result = _api_request(site_name, endpoint, method="POST", data=data)

        return {
            "success": True,
            "post_id": post_id,
            "message": f"Elementor design saved to {post_type} {post_id}",
            "link": result.get("link", ""),
        }
    except Exception as e:
        logger.error(f"Failed to save Elementor data: {e}")
        return {"success": False, "error": str(e)}


def create_elementor_page(
    site_name: str,
    title: str,
    elementor_data: list | str,
    status: str = "draft",
    template: str = "elementor_header_footer",
) -> dict:
    """Create a new page with Elementor design.

    Args:
        site_name: Site identifier
        title: Page title
        elementor_data: Elementor JSON data
        status: 'draft' or 'publish'
        template: Page template (elementor_header_footer for full-width)
    """
    import json

    try:
        # Convert to string if needed
        if isinstance(elementor_data, list):
            elementor_json = json.dumps(elementor_data)
        else:
            elementor_json = elementor_data

        # Create page with Elementor meta
        data = {
            "title": title,
            "content": "",  # Elementor handles content
            "status": status,
            "template": template,
            "meta": {
                "_elementor_data": elementor_json,
                "_elementor_edit_mode": "builder",
                "_elementor_template_type": "wp-page",
                "_elementor_page_settings": {
                    "hide_title": "yes",
                },
            }
        }

        page = _api_request(site_name, "pages", method="POST", data=data)

        return {
            "success": True,
            "id": page["id"],
            "title": page["title"]["rendered"],
            "link": page["link"],
            "edit_link": f"{_get_site(site_name)['url']}/wp-admin/post.php?post={page['id']}&action=elementor",
            "status": page["status"],
        }
    except Exception as e:
        logger.error(f"Failed to create Elementor page: {e}")
        return {"success": False, "error": str(e)}


def create_elementor_popup(
    site_name: str,
    title: str,
    elementor_data: list | str,
    trigger: str = "page_load",
    trigger_delay: int = 5,
    status: str = "publish",
) -> dict:
    """Create an Elementor Pro popup.

    Creates the popup as an elementor_library post type with popup template type.

    Args:
        site_name: Site identifier
        title: Popup title
        elementor_data: Elementor JSON data (list of sections)
        trigger: Popup trigger type (page_load, scroll, click, exit_intent)
        trigger_delay: Delay in seconds before showing (for page_load trigger)
        status: 'draft' or 'publish'

    Returns:
        Dict with popup details including edit link
    """
    import json

    try:
        if isinstance(elementor_data, list):
            elementor_json = json.dumps(elementor_data)
        else:
            elementor_json = elementor_data

        site = _get_site(site_name)

        # Build display conditions and triggers
        popup_settings = {
            "a11y_navigation": "yes",
            "timing": {},
        }

        if trigger == "page_load":
            popup_settings["timing"]["page_load"] = "yes"
            popup_settings["timing"]["page_load_delay"] = trigger_delay
        elif trigger == "scroll":
            popup_settings["timing"]["scrolling"] = "yes"
            popup_settings["timing"]["scrolling_direction"] = "down"
            popup_settings["timing"]["scrolling_offset"] = 30
        elif trigger == "exit_intent":
            popup_settings["timing"]["on_exit_intent"] = "yes"
        elif trigger == "click":
            popup_settings["open_selector"] = ".newsletter-popup-trigger"

        # Create as elementor_library custom post type
        data = {
            "title": title,
            "content": "",
            "status": status,
            "meta": {
                "_elementor_data": elementor_json,
                "_elementor_edit_mode": "builder",
                "_elementor_template_type": "popup",
                "_elementor_page_settings": popup_settings,
            }
        }

        # Try elementor_library endpoint first (registered by Elementor Pro)
        try:
            popup = _api_request(site_name, "elementor_library", method="POST", data=data)
        except Exception:
            # Fallback: create as a page with popup meta (less ideal but works)
            data["template"] = "elementor_canvas"
            popup = _api_request(site_name, "pages", method="POST", data=data)

        popup_id = popup["id"]
        return {
            "success": True,
            "id": popup_id,
            "title": popup.get("title", {}).get("rendered", title),
            "status": popup.get("status", status),
            "trigger": trigger,
            "edit_link": f"{site['url']}/wp-admin/post.php?post={popup_id}&action=elementor",
            "message": f"Popup '{title}' created. Open the edit link in Elementor to set display conditions and preview.",
        }
    except Exception as e:
        logger.error(f"Failed to create Elementor popup: {e}")
        return {"success": False, "error": str(e)}


def get_elementor_templates(site_name: str) -> list[dict]:
    """Get saved Elementor templates from a site."""
    try:
        # Elementor templates are stored as custom post type 'elementor_library'
        site = _get_site(site_name)
        url = urljoin(site["url"] + "/", "wp-json/elementor/v1/templates")
        headers = _get_auth_header(site)

        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 404:
            # Elementor API might not be available, try standard endpoint
            templates = _api_request(site_name, "elementor_library", data={"per_page": 50})
            return [
                {
                    "id": t["id"],
                    "title": t["title"]["rendered"],
                    "type": t.get("template_type", ""),
                }
                for t in templates
            ]

        response.raise_for_status()
        templates = response.json()

        template_list = templates.get("data", templates) if isinstance(templates, dict) else templates
        return [
            {
                "id": t.get("id"),
                "title": t.get("title", ""),
                "type": t.get("type", ""),
            }
            for t in template_list
        ]
    except Exception as e:
        logger.warning(f"Could not fetch Elementor templates: {e}")
        return []


# ============================================================================
# CODE SNIPPETS & TRACKING (WPCode integration)
# ============================================================================

def get_wpcode_snippets(site_name: str) -> list[dict]:
    """Get all code snippets from WPCode plugin.

    Args:
        site_name: Site identifier

    Returns:
        List of snippets with id, title, code_type, status
    """
    try:
        # WPCode stores snippets as a custom post type
        snippets = _api_request(site_name, "wpcode-snippet", method="GET", params={"per_page": 100})
        return [
            {
                "id": s["id"],
                "title": s["title"]["rendered"] if isinstance(s["title"], dict) else s["title"],
                "status": s["status"],
                "code_type": s.get("code_type", "html"),
            }
            for s in snippets
        ]
    except Exception as e:
        logger.warning(f"WPCode not available or no snippets: {e}")
        return []


def create_wpcode_snippet(
    site_name: str,
    title: str,
    code: str,
    code_type: str = "js",
    location: str = "site_wide_header",
    status: str = "publish",
) -> dict:
    """Create a new code snippet via WPCode.

    Args:
        site_name: Site identifier
        title: Snippet name
        code: The actual code (JS, HTML, PHP, CSS)
        code_type: 'js', 'html', 'php', 'css'
        location: 'site_wide_header', 'site_wide_footer', 'frontend_only', etc.
        status: 'publish' or 'draft'

    Returns:
        Created snippet info
    """
    try:
        data = {
            "title": title,
            "status": status,
            "wpcode_snippet_code": code,
            "wpcode_code_type": code_type,
            "wpcode_auto_insert_location": location,
        }

        result = _api_request(site_name, "wpcode-snippet", method="POST", data=data)
        return {
            "success": True,
            "id": result["id"],
            "title": title,
            "message": f"Snippet '{title}' created and active"
        }
    except Exception as e:
        logger.error(f"Failed to create WPCode snippet: {e}")
        return {"success": False, "error": str(e)}


def add_tracking_script(
    site_name: str,
    script_name: str,
    script_code: str,
    location: str = "header"
) -> dict:
    """Add a tracking script to a WordPress site.

    This tries Elementor snippets first (most reliable), then WPCode.

    Args:
        site_name: Site identifier
        script_name: Name for the script (e.g., "Meta Pixel Conversion Events")
        script_code: JavaScript code to add
        location: 'header' or 'footer'

    Returns:
        Result with success status
    """
    # Clean the code (remove script tags if present, we'll handle placement)
    clean_code = script_code.strip()
    if clean_code.startswith('<script'):
        import re
        clean_code = re.sub(r'^<script[^>]*>', '', clean_code)
        clean_code = re.sub(r'</script>$', '', clean_code.strip())

    # Try Elementor snippets first (works on all Elementor sites)
    try:
        elementor_loc = "elementor_head" if location == "header" else "elementor_body_end"

        # Create the snippet
        data = {
            "title": script_name,
            "status": "publish",
            "content": "",
        }
        result = _api_request(site_name, "elementor_snippet", method="POST", data=data)
        snippet_id = result["id"]

        # Update with the actual code
        meta_data = {
            "meta": {
                "_elementor_location": elementor_loc,
                "_elementor_priority": 10,
                "_elementor_code": clean_code,
            }
        }
        _api_request(site_name, f"elementor_snippet/{snippet_id}", method="POST", data=meta_data)

        return {
            "success": True,
            "method": "elementor_snippet",
            "id": snippet_id,
            "title": script_name,
            "location": elementor_loc,
            "message": f"Tracking code '{script_name}' added via Elementor (ID: {snippet_id})"
        }
    except Exception as e:
        logger.warning(f"Elementor snippet failed: {e}")

    # Try WPCode as fallback
    wrapped_code = f"<script>\n{clean_code}\n</script>"
    loc = "site_wide_header" if location == "header" else "site_wide_footer"

    result = create_wpcode_snippet(
        site_name,
        title=script_name,
        code=wrapped_code,
        code_type="html",
        location=loc,
        status="publish"
    )

    if result.get("success"):
        return result

    # If both fail, return instructions for manual addition
    return {
        "success": False,
        "error": "Could not inject code automatically",
        "manual_instructions": f"""
To add this tracking code manually:
1. Go to WordPress Admin > Code Snippets > Add Snippet
2. Or use Elementor > Custom Code
3. Add this JavaScript:

{clean_code}
        """
    }


def get_meta_pixel_events(site_name: str) -> dict:
    """Analyze what Meta Pixel events are configured on a site.

    Args:
        site_name: Site identifier

    Returns:
        Analysis of pixel setup
    """
    import re

    site = _get_site(site_name)

    try:
        # Fetch homepage HTML
        response = requests.get(site["url"], timeout=15)
        html = response.text

        result = {
            "site": site_name,
            "url": site["url"],
            "pixel_found": False,
            "pixel_id": None,
            "events_found": [],
            "issues": [],
        }

        # Check for pixel initialization
        pixel_init = re.search(r"fbq\(['\"]init['\"],\s*['\"](\d+)['\"]", html)
        if pixel_init:
            result["pixel_found"] = True
            result["pixel_id"] = pixel_init.group(1)
        else:
            result["issues"].append("No Meta Pixel initialization found")
            return result

        # Find all fbq track calls
        events = re.findall(r"fbq\(['\"]track['\"],\s*['\"](\w+)['\"]", html)
        result["events_found"] = list(set(events))

        # Check for standard conversion events
        standard_events = ["Lead", "Contact", "Schedule", "Purchase", "CompleteRegistration", "AddToCart"]
        missing = [e for e in standard_events if e not in result["events_found"]]

        if "PageView" not in result["events_found"]:
            result["issues"].append("PageView event not found - pixel may not be firing")

        if missing:
            result["issues"].append(f"Missing conversion events: {', '.join(missing)}")

        # Check for consent blocking
        if "consent" in html.lower() or "cookie" in html.lower():
            result["issues"].append("Consent/cookie notice detected - may block pixel until consent")

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


def add_meta_pixel_events(
    site_name: str,
    pixel_id: str = None,
    track_forms: bool = True,
    track_phone_clicks: bool = True,
    track_buttons: bool = False,
) -> dict:
    """Add Meta Pixel conversion event tracking to a WordPress site.

    Args:
        site_name: Site identifier
        pixel_id: Meta Pixel ID (will auto-detect if not provided)
        track_forms: Track form submissions as Lead events
        track_phone_clicks: Track tel: link clicks as Contact events
        track_buttons: Track CTA button clicks

    Returns:
        Result with success status
    """
    # First, analyze current setup
    analysis = get_meta_pixel_events(site_name)

    if not analysis.get("pixel_found"):
        return {
            "success": False,
            "error": "No Meta Pixel found on site. Install Meta Pixel plugin first.",
        }

    pixel_id = pixel_id or analysis.get("pixel_id")

    # Build event tracking code
    code_parts = [
        "// Meta Pixel Conversion Events - Added by Alfred",
        "document.addEventListener('DOMContentLoaded', function() {",
    ]

    if track_forms:
        code_parts.append("""
    // Track all form submissions as Lead events
    document.querySelectorAll('form').forEach(function(form) {
        form.addEventListener('submit', function() {
            if (typeof fbq !== 'undefined') {
                fbq('track', 'Lead', {content_name: 'Form Submission'});
            }
        });
    });
""")

    if track_phone_clicks:
        code_parts.append("""
    // Track phone number clicks as Contact events
    document.querySelectorAll('a[href^="tel:"]').forEach(function(el) {
        el.addEventListener('click', function() {
            if (typeof fbq !== 'undefined') {
                fbq('track', 'Contact', {content_name: 'Phone Click'});
            }
        });
    });
""")

    if track_buttons:
        code_parts.append("""
    // Track CTA button clicks
    document.querySelectorAll('.elementor-button, .wp-block-button, button[type="submit"]').forEach(function(el) {
        el.addEventListener('click', function() {
            if (typeof fbq !== 'undefined') {
                fbq('track', 'Lead', {content_name: el.textContent || 'Button Click'});
            }
        });
    });
""")

    code_parts.append("});")

    full_code = "\n".join(code_parts)

    # Add the tracking script
    result = add_tracking_script(
        site_name,
        script_name="Meta Pixel Conversion Events",
        script_code=full_code,
        location="footer"
    )

    if result.get("success"):
        result["events_added"] = []
        if track_forms:
            result["events_added"].append("Lead (form submissions)")
        if track_phone_clicks:
            result["events_added"].append("Contact (phone clicks)")
        if track_buttons:
            result["events_added"].append("Lead (button clicks)")
        result["message"] = f"Added conversion tracking: {', '.join(result['events_added'])}"

    return result