#!/usr/bin/env python3
"""
Backfill Printful-synced WC products with their mockup images.

Two modes:
- (default) Set featured image on each PARENT product.
- --variations  Set per-variation images so the PDP color picker swaps the
  main product image when the customer switches colors.

Idempotent: skips items that already have an image attached.

Usage:
    python3 scripts/printful_image_backfill.py                 # parents
    python3 scripts/printful_image_backfill.py --variations    # per-color
    python3 scripts/printful_image_backfill.py --variations --dry-run
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.request


PF_API = "https://api.printful.com"
SSH_HOST = "brucewayne9@75.43.156.104"
SSH_KEY = "/home/aialfred/.ssh/alfred_104"
WP_CONTAINER = "ag-wordpress"


def pf_get(token: str, path: str) -> dict:
    req = urllib.request.Request(
        f"{PF_API}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


_SSH_CTRL = "/tmp/ssh-pf-backfill.sock"


def ssh_exec(cmd: str, timeout: int = 120) -> tuple[int, str]:
    full = [
        "ssh",
        "-o", "IdentitiesOnly=yes",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={_SSH_CTRL}",
        "-o", "ControlPersist=10m",
        "-i", SSH_KEY, SSH_HOST,
        f"docker exec {WP_CONTAINER} {cmd}",
    ]
    proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def has_featured_image(post_id: int) -> bool:
    rc, out = ssh_exec(
        f"wp post meta get {post_id} _thumbnail_id --allow-root 2>/dev/null"
    )
    return rc == 0 and out.strip().isdigit() and int(out.strip()) > 0


def import_featured(post_id: int, url: str, title: str) -> tuple[bool, str]:
    safe_title = title.replace("'", "")[:80]
    cmd = (
        f"wp media import '{url}' --post_id={post_id} --featured_image "
        f"--title='{safe_title}' --allow-root"
    )
    rc, out = ssh_exec(cmd)
    return rc == 0, out


def import_attachment(url: str, title: str) -> tuple[int | None, str]:
    """Create a media attachment from URL. Returns (attachment_id, output)."""
    safe_title = title.replace("'", "")[:80]
    cmd = (
        f"wp media import '{url}' --title='{safe_title}' --porcelain --allow-root"
    )
    rc, out = ssh_exec(cmd)
    if rc != 0:
        return None, out
    for line in reversed(out.splitlines()):
        s = line.strip()
        if s.isdigit():
            return int(s), out
    return None, out


def set_thumbnail(post_id: int, attachment_id: int) -> bool:
    rc, _ = ssh_exec(
        f"wp post meta update {post_id} _thumbnail_id {attachment_id} --allow-root"
    )
    return rc == 0


def first_preview_url(variant: dict) -> str | None:
    """Designed mockup if available; fall back to bare product photo."""
    for wanted in ("preview", "mockup"):
        for f in variant.get("files", []):
            if f.get("type") == wanted and f.get("preview_url"):
                return f["preview_url"]
    return variant.get("product", {}).get("image")


def fetch_token() -> str:
    """Pull Printful OAuth key from WP option (no local secret file)."""
    cmd = (
        "wp option get woocommerce_printful_settings --format=json --allow-root"
    )
    rc, out = ssh_exec(cmd)
    if rc != 0:
        raise RuntimeError(f"Could not read Printful settings: {out}")
    data = json.loads(out)
    token = data.get("printful_oauth_key")
    if not token:
        raise RuntimeError("printful_oauth_key not set in WC options")
    return token


def run_parents(token: str, dry_run: bool) -> int:
    listing = pf_get(token, "/sync/products?limit=100")
    products = listing.get("result", [])
    print(f"Found {len(products)} Printful sync products\n")

    done, skipped, failed = 0, 0, 0
    for p in products:
        external_id = int(p["external_id"])
        name = p["name"]
        print(f"[{external_id}] {name}")

        if has_featured_image(external_id):
            print("  -> already has featured image, skip")
            skipped += 1
            continue

        detail = pf_get(token, f"/sync/products/{p['id']}")
        variants = detail.get("result", {}).get("sync_variants", [])
        url = None
        for v in variants:
            url = first_preview_url(v)
            if url:
                break
        if not url:
            print("  -> NO preview mockup found in any variant")
            failed += 1
            continue

        print(f"  mockup: {url}")
        if dry_run:
            print("  -> DRY RUN, skip import")
            continue

        ok, msg = import_featured(external_id, url, name)
        last = msg.splitlines()[-1] if msg else "ok"
        if ok:
            print(f"  -> imported & set featured ({last})")
            done += 1
        else:
            print(f"  -> FAILED: {last}")
            failed += 1

    print(f"\nDone: {done} imported, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def run_variations(token: str, dry_run: bool) -> int:
    listing = pf_get(token, "/sync/products?limit=100")
    products = listing.get("result", [])
    print(f"Found {len(products)} Printful sync products\n")

    url_to_attachment: dict[str, int] = {}
    done, skipped, failed = 0, 0, 0
    total_variants = 0

    for p in products:
        print(f"[{p['external_id']}] {p['name']}")
        detail = pf_get(token, f"/sync/products/{p['id']}")
        variants = detail.get("result", {}).get("sync_variants", [])
        total_variants += len(variants)

        for v in variants:
            try:
                var_id = int(v["external_id"])
            except (KeyError, ValueError):
                continue
            url = first_preview_url(v)
            if not url:
                print(f"  {var_id} ({v.get('name','?')}): no mockup")
                failed += 1
                continue

            if has_featured_image(var_id):
                skipped += 1
                continue

            if dry_run:
                print(f"  {var_id} ({v['name']}): would set from {url[:70]}")
                continue

            if url in url_to_attachment:
                att_id = url_to_attachment[url]
            else:
                att_id, msg = import_attachment(url, v["name"])
                if not att_id:
                    print(f"  {var_id}: IMPORT FAILED: {msg.splitlines()[-1] if msg else 'no output'}")
                    failed += 1
                    continue
                url_to_attachment[url] = att_id

            if set_thumbnail(var_id, att_id):
                done += 1
                # Compact one-line summary per variant
                print(f"  {var_id} ({v['name']}) -> att {att_id}")
            else:
                print(f"  {var_id}: meta update FAILED")
                failed += 1

    print(
        f"\nDone: {done} variants imaged, {skipped} skipped, {failed} failed "
        f"(of {total_variants} total variants; {len(url_to_attachment)} unique mockups downloaded)"
    )
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--variations",
        action="store_true",
        help="Set per-variation images instead of parent featured image",
    )
    args = ap.parse_args()

    token = fetch_token()
    if args.variations:
        return run_variations(token, args.dry_run)
    return run_parents(token, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
