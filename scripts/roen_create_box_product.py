#!/usr/bin/env python3
"""
One-time, idempotent: create the hidden 'bracelet-box' WooCommerce
product on roenhandmade.com via wp-cli. Does nothing if it already exists.

Usage:
    python3 scripts/roen_create_box_product.py
"""
from __future__ import annotations

import logging
import sys

sys.path.insert(0, "/home/aialfred/alfred")

from core.jewelry.woocommerce import _ssh_docker_wp, _php_str

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("create-box")


def find_existing_id() -> int:
    rc, out, err = _ssh_docker_wp(
        ["eval", "echo wc_get_product_id_by_sku('bracelet-box');"],
        timeout=30,
    )
    if rc != 0:
        raise RuntimeError(f"sku lookup failed: rc={rc} err={err!r}")
    return int(out.strip() or 0)


def create() -> int:
    php = f"""
$p = new WC_Product_Simple();
$p->set_name({_php_str("Roen's Bracelet Box")});
$p->set_slug('bracelet-box');
$p->set_sku('bracelet-box');
$p->set_status('publish');
$p->set_regular_price('25.00');
$p->set_manage_stock(true);
$p->set_stock_quantity(0);
$p->set_stock_status('outofstock');
$p->set_catalog_visibility('hidden');
$p->set_description({_php_str('Five hand-picked bracelets, curated by Roen. $25. Shipped within five business days with a personal card.')});
$p->set_short_description({_php_str('Five hand-picked bracelets. One curated note. $25.')});
echo $p->save();
"""
    rc, out, err = _ssh_docker_wp(["eval", php], timeout=60)
    if rc != 0 or not out.strip().isdigit():
        raise RuntimeError(f"create failed (rc={rc}): out={out!r} err={err!r}")
    return int(out.strip())


def main() -> int:
    existing = find_existing_id()
    if existing:
        print(f"OK: bracelet-box already exists (id={existing})")
        return 0
    new_id = create()
    print(f"CREATED: bracelet-box (id={new_id})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log.exception("bootstrap failed")
        sys.exit(1)
