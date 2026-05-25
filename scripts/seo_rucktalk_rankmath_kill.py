#!/usr/bin/env python3
"""Deactivate Rank Math on rt-wordpress and scrub its stored data.

Idempotent — safe to re-run. Does NOT delete plugin files (Mike can reinstate
manually if needed) — only deactivates and clears state so Rank Math stops
emitting head/sitemap/post_meta output and the alfred-seo plugin owns SEO.

Before running this, capture a backup of rank_math_* postmeta via:
  ssh server-100 "docker exec rt-wordpress wp eval '...wpdb query...' --allow-root --path=/var/www/html" > backup.tsv

Backup for this rt-wordpress instance lives under
data/seo/sites/rucktalk/backups/rankmath-postmeta-<timestamp>.tsv (401 rows
captured 2026-05-24).
"""
from __future__ import annotations

import subprocess
import sys

WP = "docker exec rt-wordpress wp"
WP_FLAGS = "--allow-root --path=/var/www/html"

# Each is run via `ssh server-100 "timeout 60 <wp_cmd>"`.
CMDS = [
    # 1. Deactivate both Rank Math plugins.
    f"{WP} plugin deactivate seo-by-rank-math seo-by-rank-math-pro {WP_FLAGS} || true",
    # 2. Delete the autoloaded options Rank Math owns (rank_math_*, rank-math-*).
    #    Use wp eval — wp option list with --search returns globs that
    #    occasionally collide with shell expansion in the ssh quoting layer.
    (
        f"{WP} eval 'global $wpdb; "
        '$opts = $wpdb->get_col("SELECT option_name FROM $wpdb->options WHERE option_name LIKE \\"rank_math_%\\" OR option_name LIKE \\"rank-math-%\\""); '
        "foreach($opts as $o){ delete_option($o); } "
        'echo count($opts).\" options deleted\\n\";'
        f"' {WP_FLAGS}"
    ),
    # 3. Drop all rank_math_* post_meta (backup already captured).
    (
        f"{WP} eval 'global $wpdb; "
        '$n = $wpdb->query("DELETE FROM $wpdb->postmeta WHERE meta_key LIKE \\"rank_math_%\\""); '
        'echo $n.\" postmeta rows deleted\\n\";'
        f"' {WP_FLAGS}"
    ),
    # 4. Flush rewrite rules so Rank Math sitemap URLs stop resolving.
    f"{WP} rewrite flush {WP_FLAGS}",
    # 5. Clear caches so the next request rebuilds head/sitemap without RM.
    f"{WP} w3-total-cache flush all {WP_FLAGS} || true",
    f"{WP} cache flush {WP_FLAGS} || true",
    f"{WP} eval 'if(function_exists(\"opcache_reset\")){{opcache_reset();echo \"opcache reset\\n\";}}' {WP_FLAGS}",
]


def main() -> int:
    for cmd in CMDS:
        full = f'ssh server-100 "timeout 60 {cmd}"'
        print(f">> {cmd[:140]}")
        rc = subprocess.call(full, shell=True)
        if rc != 0:
            print(f"   non-zero rc={rc} — continuing (idempotent)", file=sys.stderr)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
