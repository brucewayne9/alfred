"""
WooCommerce step: create a draft product on roenhandmade.com via WP-CLI.

We shell out to `ssh server-104 'docker exec roenhandmade-wp wp ...'`
because (a) we already have key-based SSH set up, (b) WP-CLI inside the
container runs as root with full access, no API auth dance.

For a v1 we create the product as `status=draft` so Sarah reviews before
publishing. After 2 weeks of calibration we flip to publish.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

SSH_HOST = "server-104"
CONTAINER = "roenhandmade-wp"
WP_PATH = "/var/www/html"
SITE_URL = "https://www.roenhandmade.com"


def _ssh_docker_wp(args: List[str], stdin_bytes: Optional[bytes] = None, timeout: int = 60) -> Tuple[int, str, str]:
    """Run `wp <args>` inside the roenhandmade-wp container. Returns (rc, stdout, stderr)."""
    inner = "wp " + " ".join(shlex.quote(a) for a in args) + f" --allow-root --path={WP_PATH}"
    if stdin_bytes is not None:
        cmd = ["ssh", SSH_HOST, f"docker exec -i {CONTAINER} {inner}"]
    else:
        cmd = ["ssh", SSH_HOST, f"docker exec {CONTAINER} {inner}"]
    logger.info("wp-cli: %s", inner)
    res = subprocess.run(
        cmd,
        input=stdin_bytes,
        capture_output=True,
        timeout=timeout,
    )
    return res.returncode, res.stdout.decode("utf-8", "replace"), res.stderr.decode("utf-8", "replace")


def _scp_into_container(local_path: Path, container_path: str) -> None:
    """Copy a local file into the WP container by piping through ssh+docker exec."""
    inner = f"docker exec -i {CONTAINER} bash -c 'cat > {shlex.quote(container_path)}'"
    cmd = ["ssh", SSH_HOST, inner]
    with local_path.open("rb") as f:
        res = subprocess.run(cmd, stdin=f, capture_output=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"upload {local_path} failed: {res.stderr.decode()!r}")


def _ensure_category(slug: str, name: str) -> int:
    """Return the term_id for a product_cat, creating it if missing."""
    rc, out, err = _ssh_docker_wp(
        ["term", "list", "product_cat", "--slug=" + slug, "--format=ids"],
        timeout=30,
    )
    if rc == 0 and out.strip():
        return int(out.strip().split()[0])
    rc, out, err = _ssh_docker_wp(
        ["term", "create", "product_cat", name, "--slug=" + slug, "--porcelain"],
        timeout=30,
    )
    if rc != 0:
        raise RuntimeError(f"could not create category {slug}: {err}")
    return int(out.strip())


def upload_image(local_path: Path) -> int:
    """Upload an image to WP media library. Returns attachment ID."""
    container_path = f"/tmp/roen_upload_{local_path.name}"
    _scp_into_container(local_path, container_path)
    rc, out, err = _ssh_docker_wp(
        ["media", "import", container_path, "--porcelain"],
        timeout=60,
    )
    # Best-effort cleanup of the staged file. Failure is non-fatal.
    _ssh_docker_wp(["eval", f"unlink('{container_path}');"], timeout=15)
    if rc != 0:
        raise RuntimeError(f"media import failed for {local_path}: {err}")
    return int(out.strip())


def create_draft_product(
    name: str,
    sku: str,
    price_cents: int,
    short_description: str,
    long_description: str,
    category_slug: str,
    tags: List[str],
    image_attachment_ids: List[int],
) -> int:
    """Create a draft WooCommerce product. Returns the post_id."""
    if not image_attachment_ids:
        raise ValueError("at least one image is required for a product")

    price_dollars = f"{price_cents / 100:.2f}"

    # Map category slug to nice display name.
    cat_display = category_slug.capitalize()
    cat_term_id = _ensure_category(category_slug, cat_display)

    # Build a tiny PHP snippet that creates the product. WP-CLI's `wp post create`
    # only does posts; for WooCommerce products we want WC's API to set price
    # and SKU through the proper hooks, so we run a `wp eval` script.
    php = f"""
$p = new WC_Product_Simple();
$p->set_name({_php_str(name)});
$p->set_sku({_php_str(sku)});
$p->set_status('draft');
$p->set_regular_price({_php_str(price_dollars)});
$p->set_short_description({_php_str(short_description)});
$p->set_description({_php_str(long_description)});
$p->set_image_id({image_attachment_ids[0]});
$p->set_gallery_image_ids({_php_array_int(image_attachment_ids[1:])});
$p->set_category_ids([{cat_term_id}]);
$p->set_manage_stock(false);
$p->set_stock_status('instock');
$pid = $p->save();
$tags = {_php_array_str(tags)};
if (!empty($tags)) {{ wp_set_object_terms($pid, $tags, 'product_tag'); }}
echo $pid;
"""
    rc, out, err = _ssh_docker_wp(
        ["eval", php],
        timeout=60,
    )
    if rc != 0 or not out.strip().isdigit():
        raise RuntimeError(f"create_draft_product failed (rc={rc}): out={out!r} err={err!r}")
    return int(out.strip())


def admin_edit_url(post_id: int) -> str:
    return f"{SITE_URL}/wp-admin/post.php?post={post_id}&action=edit"


def preview_url(post_id: int) -> str:
    return f"{SITE_URL}/?p={post_id}&preview=true"


def update_product_field(post_id: int, field: str, value: str) -> None:
    """Update one field on an existing WooCommerce product via wp eval."""
    setters = {
        "name": "set_name",
        "regular_price": "set_regular_price",
        "short_description": "set_short_description",
        "description": "set_description",
        "sku": "set_sku",
    }
    if field not in setters:
        raise ValueError(f"unsupported field: {field}")
    php = (
        f"$p = wc_get_product({int(post_id)});"
        f"if(!$p){{ echo 'NOTFOUND'; return; }}"
        f"$p->{setters[field]}({_php_str(str(value))});"
        f"$p->save();"
        f"echo 'OK';"
    )
    rc, out, err = _ssh_docker_wp(["eval", php], timeout=30)
    if rc != 0 or "OK" not in out:
        raise RuntimeError(f"update {field} on post {post_id} failed: rc={rc} out={out!r} err={err!r}")


def trash_product(post_id: int) -> None:
    rc, out, err = _ssh_docker_wp(
        ["post", "delete", str(int(post_id))],
        timeout=30,
    )
    if rc != 0:
        raise RuntimeError(f"trash post {post_id} failed: {err}")


def publish_product(post_id: int) -> None:
    rc, out, err = _ssh_docker_wp(
        ["post", "update", str(int(post_id)), "--post_status=publish"],
        timeout=30,
    )
    if rc != 0:
        raise RuntimeError(f"publish post {post_id} failed: {err}")


# --- PHP literal builders (so we don't have to escape user content into a php string at the shell level) ---

def _php_str(s: str) -> str:
    """PHP single-quoted string literal — escape \\ and '."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _php_array_int(items: List[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in items) + "]"


def _php_array_str(items: List[str]) -> str:
    return "[" + ",".join(_php_str(x) for x in items) + "]"
