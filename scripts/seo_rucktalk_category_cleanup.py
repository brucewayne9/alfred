#!/usr/bin/env python3
"""Audit RuckTalk taxonomies, propose keep/merge/delete per term, and
(in --apply mode) execute deletions and merges.

Dry-run (default): lists every taxonomy term with post count and a proposed
action (delete if count==0, merge if 1-4, keep if 5+). Writes a timestamped
markdown report under data/seo/sites/rucktalk/audit/. Mike reviews the report,
then creates data/seo/sites/rucktalk/category_decisions.yaml with explicit
{action, merge_into} per term to be removed.

--apply mode: reads the decisions YAML and executes deletes / merges via wp-cli
on rt-wordpress. Idempotent — re-running --apply with a yaml that's already
been processed is a no-op (deleted terms just 404 the second time).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: install PyYAML (pip install pyyaml). Available in venv at /home/aialfred/alfred/venv/")
    sys.exit(2)


SSH_PREFIX = ["ssh", "server-100", "timeout", "30"]
WP_PREFIX = ["docker", "exec", "rt-wordpress", "wp"]
WP_SUFFIX = ["--allow-root", "--path=/var/www/html"]


def wp(*args: str) -> str:
    """Run wp-cli on rt-wordpress via ssh. Returns stdout."""
    cmd = SSH_PREFIX + WP_PREFIX + list(args) + WP_SUFFIX
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        print(f"WARN: wp-cli rc={out.returncode}: {' '.join(args)[:100]}", file=sys.stderr)
        if out.stderr:
            print(f"      stderr: {out.stderr.strip()[:200]}", file=sys.stderr)
    return out.stdout


def list_taxonomies() -> list[str]:
    """Return slugs of public, non-builtin taxonomies."""
    raw = wp("taxonomy", "list", "--format=csv", "--fields=name,public,_builtin")
    out = []
    for line in raw.strip().splitlines()[1:]:  # skip header
        parts = line.split(",")
        if len(parts) < 3:
            continue
        name, public, builtin = parts[0], parts[1], parts[2]
        # Skip core builtins (post_format, nav_menu, link_category) — focus on Sonaar's noise.
        if builtin.strip().strip('"').lower() in {"1", "true"}:
            continue
        out.append(name)
    return out


def list_terms(taxonomy: str) -> list[dict]:
    """Return [{term_id, slug, name, count}] for a taxonomy."""
    raw = wp("term", "list", taxonomy, "--format=csv", "--fields=term_id,slug,name,count")
    out = []
    for line in raw.strip().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            out.append(dict(
                term_id=int(parts[0]),
                slug=parts[1],
                name=parts[2].strip('"'),
                count=int(parts[3] or 0),
            ))
        except ValueError:
            continue
    return out


def propose(count: int) -> str:
    if count >= 5:
        return "keep"
    if count >= 1:
        return "merge"
    return "delete"


def audit() -> None:
    out_dir = Path("data/seo/sites/rucktalk/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"categories-{datetime.utcnow():%Y-%m-%d-%H%M}.md"

    lines = ["# RuckTalk taxonomy audit", f"_{datetime.utcnow().isoformat()}Z_", ""]
    total_terms = 0
    for tax in list_taxonomies():
        terms = list_terms(tax)
        if not terms:
            continue
        total_terms += len(terms)
        lines.append(f"## {tax}  _({len(terms)} terms)_")
        lines.append("| term_id | slug | name | count | proposed |")
        lines.append("|---:|---|---|---:|---|")
        # Sort by count desc so the survivors are at top of each block.
        for t in sorted(terms, key=lambda x: -x["count"]):
            lines.append(
                f"| {t['term_id']} | {t['slug']} | {t['name']} | {t['count']} | {propose(t['count'])} |"
            )
        lines.append("")
    lines.append(f"---")
    lines.append(f"**Total terms across all taxonomies: {total_terms}**")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Create `data/seo/sites/rucktalk/category_decisions.yaml` with explicit decisions, e.g.:")
    lines.append("")
    lines.append("```yaml")
    lines.append("category:")
    lines.append("  - { term_id: 12, slug: 'old-sonaar-stuff', action: delete }")
    lines.append("  - { term_id: 17, slug: 'warrior-mindset',  action: merge, merge_into: 5 }")
    lines.append("  - { term_id: 5,  slug: 'tactical-living',  action: keep }")
    lines.append("podcast_genre:")
    lines.append("  - { term_id: 22, slug: 'uncategorized',    action: delete }")
    lines.append("```")
    lines.append("")
    lines.append("Then run `python3 scripts/seo_rucktalk_category_cleanup.py --apply`.")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    print(f"audited {total_terms} terms across {len(list_taxonomies())} non-builtin taxonomies")


def apply_decisions() -> None:
    path = Path("data/seo/sites/rucktalk/category_decisions.yaml")
    if not path.exists():
        sys.exit(f"missing {path} — run audit (dry-run) first and create the decisions YAML")
    decisions = yaml.safe_load(path.read_text())
    if not decisions:
        sys.exit(f"{path} is empty — nothing to apply")

    for taxonomy, actions in decisions.items():
        for action in actions:
            verb = action.get("action")
            tid = action.get("term_id")
            if verb == "keep" or not verb or not tid:
                continue
            if verb == "delete":
                print(f"DELETE {taxonomy} term {tid} ({action.get('slug', '?')})")
                wp("term", "delete", taxonomy, str(tid))
            elif verb == "merge":
                target = action.get("merge_into")
                if not target:
                    print(f"SKIP merge for {tid}: no merge_into target", file=sys.stderr)
                    continue
                print(f"MERGE {taxonomy} term {tid} -> {target}")
                # Reassign post relationships, then delete the source term.
                # `term_taxonomy_id` is the join table column, not `term_id`.
                wp("db", "query",
                   f"UPDATE wp_term_relationships SET term_taxonomy_id={target} WHERE term_taxonomy_id={tid};")
                wp("term", "delete", taxonomy, str(tid))
    print("apply complete")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="execute deletions/merges from decisions YAML")
    args = p.parse_args()
    if args.apply:
        apply_decisions()
    else:
        audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
