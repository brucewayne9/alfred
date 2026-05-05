#!/usr/bin/env python3
"""
One-time backfill: re-run vision on every published bracelet that lacks
the four _roen_* meta keys, and write the tags. Idempotent — skips products
that already have all four keys set.

Usage:
    python3 scripts/roen_retag_bracelets.py [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/aialfred/alfred")

import requests

from core.jewelry import vision, woocommerce
from core.jewelry.woocommerce import _ssh_docker_wp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retag")

REQUIRED_KEYS = ('_roen_color_family', '_roen_material_class',
                 '_roen_style_class', '_roen_dominant_hex')


def list_bracelets() -> list[dict]:
    """Return [{id, name, image_url, has_all_tags}] for published bracelets.

    Uses wp eval to dump the data in one round-trip.
    """
    php = """
$args = [
    'post_type' => 'product',
    'post_status' => 'publish',
    'posts_per_page' => -1,
    'tax_query' => [[
        'taxonomy' => 'product_cat',
        'field' => 'slug',
        'terms' => 'bracelets',
    ]],
];
$q = new WP_Query($args);
$out = [];
while ($q->have_posts()) {
    $q->the_post();
    $pid = get_the_ID();
    $img_id = get_post_thumbnail_id($pid);
    $img_url = $img_id ? wp_get_attachment_url($img_id) : '';
    $meta = get_post_meta($pid);
    $has = [
        '_roen_color_family'  => !empty($meta['_roen_color_family'][0]),
        '_roen_material_class'=> !empty($meta['_roen_material_class'][0]),
        '_roen_style_class'   => !empty($meta['_roen_style_class'][0]),
        '_roen_dominant_hex'  => !empty($meta['_roen_dominant_hex'][0]),
    ];
    $has_all = $has['_roen_color_family'] && $has['_roen_material_class']
            && $has['_roen_style_class'] && $has['_roen_dominant_hex'];
    $out[] = [
        'id' => $pid,
        'name' => get_the_title(),
        'image_url' => $img_url,
        'has_all_tags' => $has_all,
    ];
}
wp_reset_postdata();
echo json_encode($out);
"""
    rc, out, err = _ssh_docker_wp(["eval", php], timeout=60)
    if rc != 0:
        raise RuntimeError(f"list_bracelets failed: rc={rc} err={err!r}")
    try:
        return json.loads(out.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"list_bracelets JSON parse failed: {e}; out={out!r}")


def retag_one(product: dict, dry_run: bool) -> bool:
    """Returns True if tags were written (or would be in dry-run), False if skipped."""
    if product['has_all_tags']:
        return False
    pid = product['id']
    name = product['name']
    img_url = product['image_url']
    if not img_url:
        log.warning("skip %d (%s): no featured image", pid, name)
        return False

    log.info("tagging %d (%s) ...", pid, name)
    if dry_run:
        return True

    # Download image to tempfile
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            try:
                r = requests.get(img_url, timeout=30)
                r.raise_for_status()
                tmp.write(r.content)
                tmp.flush()
                tmp_path = Path(tmp.name)
            except Exception:
                log.exception("download failed for %d (%s)", pid, name)
                return False

        result = vision.describe_piece([tmp_path])
        for key, val in [
            ('_roen_color_family',   result['color_family']),
            ('_roen_material_class', result['material_class']),
            ('_roen_style_class',    result['style_class']),
            ('_roen_dominant_hex',   result['dominant_hex']),
        ]:
            try:
                woocommerce.update_product_meta(pid, key, val)
            except Exception:
                log.exception("write %s=%s on %d failed", key, val, pid)
        return True
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(
        description="Backfill _roen_* vision tags on published bracelets.")
    p.add_argument('--dry-run', action='store_true',
                   help='list what would be retagged without doing it')
    args = p.parse_args()

    if args.dry_run:
        log.info("DRY RUN — no writes will be made")

    log.info("listing published bracelets...")
    bracelets = list_bracelets()
    log.info("%d bracelets total", len(bracelets))

    total = len(bracelets)
    tagged = 0
    skipped = 0
    for prod in bracelets:
        if prod['has_all_tags']:
            skipped += 1
            continue
        if retag_one(prod, args.dry_run):
            tagged += 1

    log.info("done: %d total, %d tagged, %d skipped (already had tags)",
             total, tagged, skipped)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log.exception("backfill failed")
        sys.exit(1)
