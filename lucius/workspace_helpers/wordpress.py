#!/usr/bin/env python3
"""WordPress Multi-Site Manager — Full Control CLI for OpenClaw

Capabilities: content, pages, themes, custom CSS/JS, menus, widgets,
media upload, settings, search-replace, comments, WP-CLI on any site,
and full Gutenberg block design (always as draft for Mike's review).

All design commands create/update pages as DRAFT — Mike publishes.
"""
import os, sys, json, urllib.request, urllib.parse, urllib.error, base64, subprocess, re
from openclaw_env import env as _env

WP_SITES_STR = _env("WP_SITES", "groundrush,loovacast,rucktalk,nightlife,lumabot")
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
_SSH_USER = "brucewayne9"
_SSH_KEY_DIR = os.path.expanduser("~/.ssh")

# Map host IPs to SSH identity files (matches ~/.ssh/config)
_SSH_KEYS = {
    "75.43.156.98":  os.path.join(_SSH_KEY_DIR, "alfred_98"),
    "75.43.156.100": os.path.join(_SSH_KEY_DIR, "alfred_100"),
    "75.43.156.101": os.path.join(_SSH_KEY_DIR, "alfred_101"),
    "75.43.156.104": os.path.join(_SSH_KEY_DIR, "alfred_104"),
    "75.43.156.111": os.path.join(_SSH_KEY_DIR, "alfred_111"),
    "75.43.156.117": os.path.join(_SSH_KEY_DIR, "alfred_117"),
    "75.43.156.121": os.path.join(_SSH_KEY_DIR, "alfred_121"),
}

def _ssh_base(host):
    """Build base SSH command args with the correct identity file for a host."""
    args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    key = _SSH_KEYS.get(host)
    if key and os.path.exists(key):
        args.extend(["-i", key])
    args.append(f"{_SSH_USER}@{host}")
    return args

# ── Container registry ──────────────────────────────────────
# Loaded from config/wp_sites.json — Claw can add/remove sites himself.
# Format: {"shortname": {"host": "ip", "container": "docker-name", "url": "https://..."}}

_WP_SITES_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "wp_sites.json")

def _load_registry():
    """Load container registry from wp_sites.json."""
    try:
        with open(_WP_SITES_CONFIG) as f:
            data = json.load(f)
        return {k: (v["host"], v["container"], v.get("url", "")) for k, v in data.items()}
    except Exception:
        return {}

def _save_registry(registry):
    """Save container registry back to wp_sites.json."""
    data = {}
    for name, (host, container, url) in registry.items():
        data[name] = {"host": host, "container": container, "url": url}
    with open(_WP_SITES_CONFIG, "w") as f:
        json.dump(data, f, indent=2)

def add_site(name, host, container, url=""):
    """Register a new WordPress site.
    Example: add-site newclient 75.43.156.104 newclient-wp https://newclient.com
    """
    registry = _load_registry()
    if name in registry:
        return {"error": f"Site '{name}' already exists. Use remove-site first to replace."}
    registry[name] = (host, container, url)
    _save_registry(registry)
    # Verify it works
    _ensure_wpcli(host, container)
    check = wpcli(name, "eval 'echo \"OK\";'", timeout=15)
    status = "healthy" if "OK" in check.get("output", "") else "unreachable"
    return {"status": "added", "name": name, "host": host, "container": container, "url": url, "health": status}

def remove_site(name):
    """Unregister a WordPress site."""
    registry = _load_registry()
    if name not in registry:
        return {"error": f"Site '{name}' not found."}
    removed = registry.pop(name)
    _save_registry(registry)
    return {"status": "removed", "name": name}

# Property-like accessor for backward compat
CONTAINER_REGISTRY = _load_registry()

# ── Site config (REST API auth) ─────────────────────────────

def _get_sites():
    sites = {}
    for name in WP_SITES_STR.split(","):
        name = name.strip()
        url = _env(f"WP_SITE_{name.upper()}_URL", "")
        user = _env(f"WP_SITE_{name.upper()}_USER", "")
        pwd = _env(f"WP_SITE_{name.upper()}_PASS", "")
        if url:
            sites[name] = {"url": url.rstrip("/"), "user": user, "pass": pwd}
    return sites

def _auth_header(cfg):
    creds = base64.b64encode(f"{cfg['user']}:{cfg['pass']}".encode()).decode()
    return f"Basic {creds}"

def _request(site, endpoint, method="GET", data=None, api="wp/v2"):
    sites = _get_sites()
    if site not in sites:
        return {"error": f"Unknown site: {site}. Available: {list(sites.keys())}"}
    cfg = sites[site]
    url = f"{cfg['url']}/wp-json/{api}/{endpoint}"
    headers = {"Authorization": _auth_header(cfg), "Content-Type": "application/json", "User-Agent": _UA}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"error": str(e)}

def _upload(site, filepath):
    """Upload a file (image/media) to the media library."""
    sites = _get_sites()
    if site not in sites:
        return {"error": f"Unknown site: {site}"}
    cfg = sites[site]
    filename = os.path.basename(filepath)
    import mimetypes
    mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    with open(filepath, "rb") as f:
        body = f.read()
    url = f"{cfg['url']}/wp-json/wp/v2/media"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", _auth_header(cfg))
    req.add_header("Content-Type", mime)
    req.add_header("Content-Disposition", f'attachment; filename="{filename}"')
    req.add_header("User-Agent", _UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:500]}"}
    except Exception as e:
        return {"error": str(e)}

# ── WP-CLI (multi-site, any container) ──────────────────────

def _resolve_container(site):
    """Resolve a site name to (host, container). Returns None if not found."""
    registry = _load_registry()
    if site in registry:
        host, container, _ = registry[site]
        return host, container
    # Try fuzzy match
    for key in registry:
        if key in site or site in key:
            host, container, _ = registry[key]
            return host, container
    return None, None

def _ensure_wpcli(host, container):
    """Install WP-CLI in a container if it's missing. Idempotent."""
    check_cmd = _ssh_base(host) + [
        f"docker exec {container} which wp 2>/dev/null || echo MISSING"
    ]
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=15)
        if "MISSING" in result.stdout:
            install_cmd = _ssh_base(host) + [
                f"docker exec {container} bash -c "
                f"'curl -sO https://raw.githubusercontent.com/wp-cli/builds/gh-pages/phar/wp-cli.phar "
                f"&& chmod +x wp-cli.phar && mv wp-cli.phar /usr/local/bin/wp'"
            ]
            subprocess.run(install_cmd, capture_output=True, text=True, timeout=60)
            return True
    except Exception:
        pass
    return False

def wpcli(site, command, timeout=60):
    """Run a WP-CLI command inside any WordPress container.

    Usage: wpcli <site> <command>
    Example: wpcli nightlife 'theme list --format=json'
             wpcli rucktalk 'plugin update --all'
             wpcli groundrush 'post list --post_type=page --format=json'
    """
    host, container = _resolve_container(site)
    if not host:
        return {"error": f"Unknown site: {site}. Available: {list(CONTAINER_REGISTRY.keys())}"}

    _ensure_wpcli(host, container)

    ssh_cmd = _ssh_base(host) + [
        f"docker exec {container} wp {command} --allow-root 2>/dev/null"
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
        # Filter out PHP warnings/notices
        lines = [l for l in output.splitlines()
                 if not l.startswith(("[", "Warning:", "PHP")) or l.startswith("[{")]
        output = "\n".join(lines).strip() if lines else output
        if result.returncode != 0 and not output:
            return {"error": result.stderr.strip() or "Command failed"}
        # Try to parse as JSON
        if output.startswith(("[", "{")):
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                pass
        return {"output": output}
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}

# ── Content ──────────────────────────────────────────────────

def list_sites():
    """List all sites with REST API config and container info."""
    registry = _load_registry()
    api_sites = {k: v["url"] for k, v in _get_sites().items()}
    result = []
    for name, (host, container, url) in registry.items():
        result.append({
            "name": name,
            "url": url or api_sites.get(name, ""),
            "server": host.split(".")[-1],
            "container": container,
            "has_api": name in api_sites,
        })
    return result

def posts(site, count=10, status="publish"):
    data = _request(site, f"posts?per_page={count}&status={status}")
    if isinstance(data, list):
        return [{"id": p["id"], "title": p["title"]["rendered"], "status": p["status"],
                 "date": p["date"], "link": p["link"]} for p in data]
    return data


def elementor_cache_flush(site):
    """Clear Elementor + WP object cache for a site. Auto-called after every update."""
    # 1. Elementor REST API flush
    try:
        _request(site, "cache", "DELETE", api="elementor/v1")
    except:
        pass
    # 2. WP-CLI guaranteed fallback — fires even if REST API fails
    try:
        wpcli(site, "elementor flush-css")
    except:
        pass
    try:
        wpcli(site, "cache flush")
    except:
        pass

def get_post(site, post_id):
    return _request(site, f"posts/{post_id}")

def create_post(site, data):
    return _request(site, "posts", "POST", data)

def update_post(site, post_id, data):
    result = _request(site, f"posts/{post_id}", "POST", data)
    elementor_cache_flush(site)
    return result

def delete_post(site, post_id):
    return _request(site, f"posts/{post_id}?force=true", "DELETE")

# ── Pages ────────────────────────────────────────────────────

def pages(site, count=10):
    data = _request(site, f"pages?per_page={count}&status=any")
    if isinstance(data, list):
        return [{"id": p["id"], "title": p["title"]["rendered"], "status": p["status"],
                 "link": p["link"]} for p in data]
    return data

def get_page(site, page_id):
    return _request(site, f"pages/{page_id}")

def create_page(site, data):
    result = _request(site, "pages", "POST", data)
    elementor_cache_flush(site)
    return result

def update_page(site, page_id, data):
    result = _request(site, f"pages/{page_id}", "POST", data)
    elementor_cache_flush(site)
    return result

def delete_page(site, page_id):
    return _request(site, f"pages/{page_id}?force=true", "DELETE")

# ── Themes ───────────────────────────────────────────────────

def themes(site):
    data = _request(site, "themes")
    if isinstance(data, list):
        return [{"stylesheet": t.get("stylesheet"), "name": t.get("name",{}).get("raw",""),
                 "status": t.get("status"), "version": t.get("version")} for t in data]
    return data

def activate_theme(site, stylesheet):
    return _request(site, f"themes/{stylesheet}", "POST", {"status": "active"})

def get_theme_mods(site):
    """Get current theme customizer settings."""
    return _request(site, "settings", api="wp/v2")

# ── Custom CSS ───────────────────────────────────────────────

def get_custom_css(site):
    """Get the current custom CSS via WP-CLI."""
    result = wpcli(site, 'eval "echo wp_get_custom_css();"')
    output = result.get("output", "")
    return {"css": output} if output else {"css": "", "note": "No custom CSS set"}

def set_custom_css(site, css):
    """Set custom CSS via WP-CLI. Replaces existing Additional CSS."""
    safe = css.replace("'", "'\\''")
    return wpcli(site, f"eval \"wp_update_custom_css_post('{safe}');echo 'OK';\"")

def append_custom_css(site, css):
    """Append CSS to existing Additional CSS via WP-CLI."""
    safe = css.replace("'", "'\\''")
    return wpcli(site, f"eval \"\\$existing=wp_get_custom_css();wp_update_custom_css_post(\\$existing.'\\n{safe}');echo 'OK';\"")

# ── Menus ────────────────────────────────────────────────────

def menus(site):
    return _request(site, "menus", api="wp/v2")

def get_menu(site, menu_id):
    return _request(site, f"menus/{menu_id}", api="wp/v2")

def menu_items(site, menu_id):
    return _request(site, f"menu-items?menus={menu_id}&per_page=100", api="wp/v2")

def create_menu_item(site, data):
    """Create a menu item. data: {menus: id, title, url, type, status: publish}"""
    return _request(site, "menu-items", "POST", data, api="wp/v2")

def update_menu_item(site, item_id, data):
    return _request(site, f"menu-items/{item_id}", "POST", data, api="wp/v2")

def delete_menu_item(site, item_id):
    return _request(site, f"menu-items/{item_id}?force=true", "DELETE", api="wp/v2")

# ── Widgets ──────────────────────────────────────────────────

def widget_areas(site):
    return _request(site, "sidebars", api="wp/v2")

def widgets(site):
    return _request(site, "widgets", api="wp/v2")

def create_widget(site, data):
    """data: {sidebar: 'sidebar-1', id_base: 'text', instance: {title, text}}"""
    return _request(site, "widgets", "POST", data, api="wp/v2")

def update_widget(site, widget_id, data):
    return _request(site, f"widgets/{widget_id}", "POST", data, api="wp/v2")

def delete_widget(site, widget_id):
    return _request(site, f"widgets/{widget_id}?force=true", "DELETE", api="wp/v2")

# ── Media ────────────────────────────────────────────────────

def media(site, count=10):
    data = _request(site, f"media?per_page={count}")
    if isinstance(data, list):
        return [{"id": m["id"], "title": m["title"]["rendered"],
                 "url": m["source_url"], "type": m["mime_type"]} for m in data]
    return data

def upload_media(site, filepath):
    return _upload(site, filepath)

def delete_media(site, media_id):
    return _request(site, f"media/{media_id}?force=true", "DELETE")

# ── Plugins ──────────────────────────────────────────────────

def plugins(site):
    return _request(site, "plugins")

def activate_plugin(site, plugin):
    return _request(site, f"plugins/{plugin}", "POST", {"status": "active"})

def deactivate_plugin(site, plugin):
    return _request(site, f"plugins/{plugin}", "POST", {"status": "inactive"})

def update_all_plugins(site):
    """Update all plugins on a site via WP-CLI."""
    return wpcli(site, "plugin update --all", timeout=120)

# ── Settings ─────────────────────────────────────────────────

def get_settings(site):
    return _request(site, "settings")

def update_settings(site, data):
    """data: {title, description, timezone_string, date_format, etc.}"""
    return _request(site, "settings", "POST", data)

# ── Users / Taxonomy ─────────────────────────────────────────

def users(site):
    return _request(site, "users")

def categories(site):
    return _request(site, "categories?per_page=100")

def create_category(site, data):
    return _request(site, "categories", "POST", data)

def tags(site):
    return _request(site, "tags?per_page=100")

def create_tag(site, data):
    return _request(site, "tags", "POST", data)

# ── Comments ─────────────────────────────────────────────────

def comments(site, count=10):
    data = _request(site, f"comments?per_page={count}")
    if isinstance(data, list):
        return [{"id": c["id"], "author_name": c["author_name"],
                 "content": c["content"]["rendered"][:200], "status": c["status"],
                 "post": c["post"], "date": c["date"]} for c in data]
    return data

def delete_comment(site, comment_id):
    return _request(site, f"comments/{comment_id}?force=true", "DELETE")

# ── Search / Replace ─────────────────────────────────────────

def search(site, query, count=10):
    data = _request(site, f"search?search={urllib.parse.quote(query)}&per_page={count}")
    if isinstance(data, list):
        return [{"id": r["id"], "title": r["title"], "type": r["type"],
                 "url": r["url"]} for r in data]
    return data

# ── SEO / Health ─────────────────────────────────────────────

def seo(site, post_id):
    sites = _get_sites()
    if site not in sites:
        return {"error": f"Unknown site: {site}"}
    cfg = sites[site]
    url = f"{cfg['url']}/wp-json/rankmath/v1/getHead?url={cfg['url']}/?p={post_id}"
    headers = {"Authorization": _auth_header(cfg)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def site_health(site):
    sites = _get_sites()
    if site not in sites:
        return {"error": f"Unknown site: {site}"}
    cfg = sites[site]
    url = f"{cfg['url']}/wp-json/wp-site-health/v1/tests/background-updates"
    headers = {"Authorization": _auth_header(cfg)}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return {"note": "Site health endpoint may not be available", "status": "reachable"}

# ══════════════════════════════════════════════════════════════
# DESIGN ENGINE — Gutenberg block design via WP-CLI
# All design commands create as DRAFT. Mike publishes.
# ══════════════════════════════════════════════════════════════

def get_block_content(site, page_id):
    """Get raw Gutenberg block markup for any post/page.
    Returns the raw post_content with block comments intact.
    """
    result = wpcli(site, f"post get {page_id} --field=post_content")
    return result

def design_page(site, title, block_content):
    """Create a new page with Gutenberg block markup as DRAFT.

    Args:
        site: Site short name (e.g., 'nightlife')
        title: Page title
        block_content: Raw Gutenberg block HTML markup

    Returns: dict with page_id and preview_url
    """
    # Escape single quotes in content for shell
    safe_title = title.replace("'", "'\\''")
    safe_content = block_content.replace("'", "'\\''")

    result = wpcli(
        site,
        f"post create --post_type=page --post_title='{safe_title}' "
        f"--post_status=draft --post_content='{safe_content}' --porcelain",
        timeout=30
    )

    output = result.get("output", "").strip()
    if output and output.isdigit():
        page_id = output
        # Get the preview URL
        preview = wpcli(site, f"post get {page_id} --field=guid")
        preview_url = preview.get("output", "").strip()
        return {
            "status": "draft_created",
            "page_id": int(page_id),
            "title": title,
            "preview_url": f"{preview_url}&preview=true" if preview_url else "",
            "note": "Page created as DRAFT. Mike must review and publish.",
        }
    return {"error": "Failed to create page", "details": result}

def update_block_content(site, page_id, block_content):
    """Update an existing page's block content. Sets status to DRAFT.

    Args:
        site: Site short name
        page_id: Post/page ID to update
        block_content: New Gutenberg block HTML markup
    """
    # Write content to a temp file on the remote server to avoid shell escaping issues
    import tempfile, hashlib
    content_hash = hashlib.md5(block_content.encode()).hexdigest()[:8]
    remote_tmp = f"/tmp/wp-design-{content_hash}.html"

    host, container = _resolve_container(site)
    if not host:
        return {"error": f"Unknown site: {site}"}

    # Write content to temp file on the server
    safe_content = block_content.replace("'", "'\\''")
    write_cmd = _ssh_base(host) + [
        f"cat > {remote_tmp} << 'WPBLOCKEOF'\n{block_content}\nWPBLOCKEOF"
    ]
    try:
        subprocess.run(write_cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        return {"error": f"Failed to write temp file: {e}"}

    # Copy into container and update
    copy_cmd = _ssh_base(host) + [
        f"docker cp {remote_tmp} {container}:{remote_tmp} && "
        f"docker exec {container} wp post update {page_id} {remote_tmp} "
        f"--post_status=draft --allow-root 2>/dev/null && "
        f"rm -f {remote_tmp}"
    ]
    try:
        result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {
                "status": "draft_updated",
                "page_id": int(page_id),
                "note": "Page updated as DRAFT. Mike must review and publish.",
            }
        return {"error": result.stderr.strip() or "Update failed"}
    except Exception as e:
        return {"error": str(e)}

def get_theme_json(site):
    """Read the active theme's theme.json (design system: colors, fonts, spacing)."""
    result = wpcli(site, "eval 'echo file_get_contents(get_template_directory().\"/theme.json\");'")
    output = result.get("output", "")
    if output and output.startswith("{"):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            pass
    return result

def update_theme_json(site, theme_json_data):
    """Update the active theme's theme.json. Creates a backup first.

    Args:
        site: Site short name
        theme_json_data: dict with theme.json content
    """
    host, container = _resolve_container(site)
    if not host:
        return {"error": f"Unknown site: {site}"}

    # Backup existing theme.json
    wpcli(site, 'eval \'$d=get_template_directory();copy("$d/theme.json","$d/theme.json.bak");\'')

    # Write new theme.json
    remote_tmp = "/tmp/wp-theme-json-update.json"
    content = json.dumps(theme_json_data, indent=2)

    write_cmd = _ssh_base(host) + [
        f"cat > {remote_tmp} << 'THEMEJSONEOF'\n{content}\nTHEMEJSONEOF"
    ]
    try:
        subprocess.run(write_cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        return {"error": f"Failed to write temp file: {e}"}

    # Get theme directory and copy
    theme_dir_result = wpcli(site, "eval 'echo get_template_directory();'")
    theme_dir = theme_dir_result.get("output", "").strip()
    if not theme_dir:
        return {"error": "Could not determine theme directory"}

    copy_cmd = _ssh_base(host) + [
        f"docker cp {remote_tmp} {container}:{theme_dir}/theme.json && rm -f {remote_tmp}"
    ]
    try:
        result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            # Clear theme caches
            wpcli(site, "cache flush")
            return {"status": "theme_json_updated", "backup": f"{theme_dir}/theme.json.bak"}
        return {"error": result.stderr.strip() or "Copy failed"}
    except Exception as e:
        return {"error": str(e)}

def site_audit(site):
    """Full audit of a site: theme, plugins, pages, posts count, PHP version."""
    results = {}
    results["theme"] = wpcli(site, "theme list --status=active --format=json")
    results["plugins"] = wpcli(site, "plugin list --format=json")
    results["pages"] = wpcli(site, "post list --post_type=page --post_status=any --format=json --fields=ID,post_title,post_status")
    results["post_count"] = wpcli(site, "post list --post_type=post --format=count")
    results["php_version"] = wpcli(site, "eval 'echo phpversion();'")
    results["wp_version"] = wpcli(site, "core version")
    results["home_url"] = wpcli(site, "option get home")
    return results

def bulk_update_plugins():
    """Update all plugins on ALL sites. Returns per-site results."""
    results = {}
    for site in _load_registry():
        results[site] = wpcli(site, "plugin update --all", timeout=120)
    return results

def bulk_health_check():
    """Quick health check across all sites: is WordPress responding?"""
    results = {}
    for site in _load_registry():
        r = wpcli(site, "eval 'echo \"OK\";'", timeout=15)
        output = r.get("output", "")
        results[site] = "healthy" if "OK" in output else r.get("error", "unknown")
    return results

# ── CLI dispatcher ───────────────────────────────────────────

COMMANDS = {
    # Content
    "sites": (lambda: list_sites(), "List all configured sites with container info"),
    "posts": (lambda s, n="10": posts(s, int(n)), "posts <site> [count]"),
    "get-post": (lambda s, i: get_post(s, i), "get-post <site> <id>"),
    "create-post": (lambda s, d: create_post(s, json.loads(d)), "create-post <site> <json>"),
    "update-post": (lambda s, i, d: update_post(s, i, json.loads(d)), "update-post <site> <id> <json>"),
    "delete-post": (lambda s, i: delete_post(s, i), "delete-post <site> <id>"),
    # Pages
    "pages": (lambda s, n="10": pages(s, int(n)), "pages <site> [count]"),
    "get-page": (lambda s, i: get_page(s, i), "get-page <site> <id>"),
    "create-page": (lambda s, d: create_page(s, json.loads(d)), "create-page <site> <json>"),
    "update-page": (lambda s, i, d: update_page(s, i, json.loads(d)), "update-page <site> <id> <json>"),
    "delete-page": (lambda s, i: delete_page(s, i), "delete-page <site> <id>"),
    # Themes
    "themes": (lambda s: themes(s), "themes <site>"),
    "activate-theme": (lambda s, t: activate_theme(s, t), "activate-theme <site> <stylesheet>"),
    "theme-json": (lambda s: get_theme_json(s), "theme-json <site> — read theme.json"),
    "update-theme-json": (lambda s, d: update_theme_json(s, json.loads(d)), "update-theme-json <site> <json>"),
    # Custom CSS
    "get-css": (lambda s: get_custom_css(s), "get-css <site>"),
    "set-css": (lambda s, css: set_custom_css(s, css), "set-css <site> <css_string>"),
    "append-css": (lambda s, css: append_custom_css(s, css), "append-css <site> <css_string>"),
    # Menus
    "menus": (lambda s: menus(s), "menus <site>"),
    "menu-items": (lambda s, i: menu_items(s, i), "menu-items <site> <menu_id>"),
    "create-menu-item": (lambda s, d: create_menu_item(s, json.loads(d)), "create-menu-item <site> <json>"),
    "update-menu-item": (lambda s, i, d: update_menu_item(s, i, json.loads(d)), "update-menu-item <site> <id> <json>"),
    "delete-menu-item": (lambda s, i: delete_menu_item(s, i), "delete-menu-item <site> <id>"),
    # Widgets
    "widget-areas": (lambda s: widget_areas(s), "widget-areas <site>"),
    "widgets": (lambda s: widgets(s), "widgets <site>"),
    "create-widget": (lambda s, d: create_widget(s, json.loads(d)), "create-widget <site> <json>"),
    "update-widget": (lambda s, i, d: update_widget(s, i, json.loads(d)), "update-widget <site> <id> <json>"),
    "delete-widget": (lambda s, i: delete_widget(s, i), "delete-widget <site> <id>"),
    # Media
    "media": (lambda s, n="10": media(s, int(n)), "media <site> [count]"),
    "upload-media": (lambda s, f: upload_media(s, f), "upload-media <site> <filepath>"),
    "delete-media": (lambda s, i: delete_media(s, i), "delete-media <site> <id>"),
    # Plugins
    "plugins": (lambda s: plugins(s), "plugins <site>"),
    "activate-plugin": (lambda s, p: activate_plugin(s, p), "activate-plugin <site> <slug>"),
    "deactivate-plugin": (lambda s, p: deactivate_plugin(s, p), "deactivate-plugin <site> <slug>"),
    "update-plugins": (lambda s: update_all_plugins(s), "update-plugins <site>"),
    "bulk-update-plugins": (lambda: bulk_update_plugins(), "Update all plugins on ALL sites"),
    # Settings
    "settings": (lambda s: get_settings(s), "settings <site>"),
    "update-settings": (lambda s, d: update_settings(s, json.loads(d)), 'update-settings <site> <json>'),
    # Users / Taxonomy
    "users": (lambda s: users(s), "users <site>"),
    "categories": (lambda s: categories(s), "categories <site>"),
    "create-category": (lambda s, d: create_category(s, json.loads(d)), "create-category <site> <json>"),
    "tags": (lambda s: tags(s), "tags <site>"),
    "create-tag": (lambda s, d: create_tag(s, json.loads(d)), "create-tag <site> <json>"),
    # Comments
    "comments": (lambda s, n="10": comments(s, int(n)), "comments <site> [count]"),
    "delete-comment": (lambda s, i: delete_comment(s, i), "delete-comment <site> <id>"),
    # Search
    "search": (lambda s, q, n="10": search(s, q, int(n)), "search <site> <query> [count]"),
    # SEO / Health
    "seo": (lambda s, i: seo(s, i), "seo <site> <post_id>"),
    "health": (lambda s: site_health(s), "health <site>"),
    "bulk-health": (lambda: bulk_health_check(), "Quick health check on ALL sites"),
    # WP-CLI (any site)
    "wpcli": (lambda s, *a: wpcli(s, " ".join(a)), "wpcli <site> <wp-cli command...>"),
    # Design Engine
    "get-blocks": (lambda s, i: get_block_content(s, i), "get-blocks <site> <page_id> — raw block markup"),
    "design-page": (lambda s, t, c: design_page(s, t, c), "design-page <site> <title> <block_html> — creates DRAFT"),
    "update-blocks": (lambda s, i, c: update_block_content(s, i, c), "update-blocks <site> <page_id> <block_html> — updates as DRAFT"),
    "audit": (lambda s: site_audit(s), "audit <site> — full site audit"),
    # Site Management
    "add-site": (lambda n, h, c, u="": add_site(n, h, c, u), "add-site <name> <host_ip> <container> [url]"),
    "remove-site": (lambda n: remove_site(n), "remove-site <name>"),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print("WordPress Full Control + Design Engine — CLI for OpenClaw")
        print("Usage: wordpress.py <command> [args]\n")
        print("Content:    sites | posts | get-post | create-post | update-post | delete-post")
        print("Pages:      pages | get-page | create-page | update-page | delete-page")
        print("Themes:     themes | activate-theme | theme-json | update-theme-json")
        print("CSS:        get-css | set-css | append-css")
        print("Menus:      menus | menu-items | create-menu-item | update-menu-item | delete-menu-item")
        print("Widgets:    widget-areas | widgets | create-widget | update-widget | delete-widget")
        print("Media:      media | upload-media | delete-media")
        print("Plugins:    plugins | activate-plugin | deactivate-plugin | update-plugins | bulk-update-plugins")
        print("Settings:   settings | update-settings")
        print("Users:      users | categories | create-category | tags | create-tag")
        print("Comments:   comments | delete-comment")
        print("Search:     search <site> <query>")
        print("SEO:        seo <site> <post_id> | health <site> | bulk-health")
        print("WP-CLI:     wpcli <site> <any wp-cli command>  (runs on any site)")
        print("Design:     get-blocks <site> <id> | design-page <site> <title> <html>")
        print("            update-blocks <site> <id> <html> | audit <site>")
        print("Manage:     add-site <name> <host> <container> [url] | remove-site <name>")
        print(f"\nSites: {', '.join(_load_registry().keys())}")
        print("\n** All design commands create as DRAFT — Mike publishes. **")
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

    fn, _ = COMMANDS[cmd]
    try:
        result = fn(*args)
        print(json.dumps(result, indent=2, default=str))
    except TypeError as e:
        print(json.dumps({"error": f"Wrong number of arguments: {e}"}, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    main()
