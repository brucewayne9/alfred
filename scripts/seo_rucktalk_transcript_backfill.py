#!/usr/bin/env python3
"""Backfill _rt_transcript post_meta on rt-wordpress from existing Whisper
JSON files in /home/aialfred/rucktalk_pipeline/transcripts/.

Idempotent — skips posts that already have a non-empty _rt_transcript unless
--force is passed. Matches transcripts to posts by extracting the episode
number from the JSON filename (episode_N.json) and the post slug (episode-N-*).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

sys.path.insert(0, "/home/aialfred/alfred")
from core.seo.sites.registry import get_site_by_slug

TRANSCRIPTS_DIR = Path("/home/aialfred/rucktalk_pipeline/transcripts")
EPISODE_NUM_PAT = re.compile(r"episode[_-](\d+)", re.IGNORECASE)


def extract_episode_num(name: str) -> int | None:
    m = EPISODE_NUM_PAT.search(name)
    return int(m.group(1)) if m else None


def transcript_text_from_file(path: Path) -> str:
    """Extract the best-effort full text from a Whisper JSON output."""
    data = json.loads(path.read_text(encoding="utf-8"))
    text = (data.get("text") or "").strip()
    if text:
        return text
    return " ".join(s.get("text", "").strip() for s in data.get("segments", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing _rt_transcript values")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be backfilled, don't write")
    args = ap.parse_args()

    site = get_site_by_slug("rucktalk")
    if not site or not site.wp_app_password:
        sys.exit("rucktalk site row missing or wp_app_password not set; run seo_init_rucktalk.py first")

    auth = HTTPBasicAuth(site.wp_username, site.wp_app_password)
    rest = site.wp_rest_url.rstrip("/")

    # 1. Build {ep_num: (path, text)} from disk.
    transcripts: dict[int, tuple[Path, str]] = {}
    for path in sorted(Path(TRANSCRIPTS_DIR).glob("*.json")):
        ep = extract_episode_num(path.stem)
        if ep is None:
            print(f"SKIP {path.name}: no episode number in filename", file=sys.stderr)
            continue
        try:
            text = transcript_text_from_file(path)
        except Exception as exc:
            print(f"SKIP {path.name}: parse failed ({exc})", file=sys.stderr)
            continue
        if not text:
            print(f"SKIP {path.name}: empty text + empty segments", file=sys.stderr)
            continue
        transcripts[ep] = (path, text)

    print(f"Found {len(transcripts)} transcripts on disk: episodes {sorted(transcripts)}")

    # 2. Fetch the full list of published podcasts via WP-CLI.
    # (The 'podcast' post type isn't registered to show_in_rest, so REST returns 404.)
    try:
        result = subprocess.run(
            [
                "ssh", "server-100",
                "docker exec rt-wordpress wp post list --post_type=podcast --status=publish "
                "--fields=ID,post_name --format=json --allow-root --path=/var/www/html"
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            sys.exit(f"WP-CLI failed: {result.stderr}")
        all_posts = json.loads(result.stdout)
        # Normalize to {id, slug} for later matching
        all_posts = [{"id": p["ID"], "slug": p["post_name"]} for p in all_posts]
    except Exception as exc:
        sys.exit(f"Failed to fetch podcasts: {exc}")
    print(f"Found {len(all_posts)} published podcasts on rucktalk.com")

    # 3. For each transcript, find the matching post by slug prefix and POST.
    backfilled = 0
    for ep, (path, text) in sorted(transcripts.items()):
        candidates = [p for p in all_posts if re.match(rf"^episode[_-]{ep}\b", p["slug"], re.IGNORECASE)]
        if len(candidates) != 1:
            print(f"NO_MATCH episode {ep}: {len(candidates)} candidates "
                  f"(slugs: {[p['slug'] for p in candidates]})")
            continue
        post = candidates[0]
        post_id = post["id"]

        if args.dry_run:
            print(f"DRY-RUN episode {ep} (post {post_id}, slug={post['slug']}): "
                  f"would write {len(text)} chars")
            continue

        url = f"{rest}/alfred-seo/v1/transcript"
        resp = requests.post(
            url,
            auth=auth,
            json={"post_id": post_id, "transcript": text},
            timeout=30,
        )
        if resp.ok:
            body = resp.json()
            print(f"OK episode {ep} (post {post_id}, slug={post['slug']}): "
                  f"{body.get('chars', '?')} chars written")
            backfilled += 1
        else:
            print(f"FAIL episode {ep} (post {post_id}): HTTP {resp.status_code}: "
                  f"{resp.text[:200]}", file=sys.stderr)

    print(f"\nbackfilled {backfilled} / {len(transcripts)} transcripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
