#!/usr/bin/env python3
"""Snapshot what plugins currently emit on rucktalk.com so we can diff
before vs after the Rank Math kill (Plan 1B-1 Task 10).

Writes a timestamped markdown report under data/seo/sites/rucktalk/audit/.
"""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = "https://rucktalk.com"


def fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    return urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")


def main() -> int:
    urls = [BASE + "/", BASE + "/about/"]
    # Append newest 3 podcast episodes via REST.
    try:
        posts = json.loads(fetch(BASE + "/wp-json/wp/v2/podcast?per_page=3"))
        urls.extend(p["link"] for p in posts)
    except Exception as exc:
        print(f"WARN: could not fetch podcast list ({exc}) — auditing home + about only")

    out_dir = Path("data/seo/sites/rucktalk/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"collision-{datetime.utcnow():%Y-%m-%d-%H%M}.md"

    lines = [
        "# RuckTalk SEO collision audit",
        f"_Captured: {datetime.utcnow().isoformat()}Z_",
        "",
    ]
    for url in urls:
        try:
            html = fetch(url)
        except Exception as exc:
            lines.append(f"## {url}")
            lines.append(f"- FETCH FAILED: {exc}")
            lines.append("")
            continue

        # Match every JSON-LD block, capturing class= if present (Rank Math marks
        # its blocks `class="rank-math-schema-pro"`).
        schemas = re.findall(
            r'<script[^>]+application/ld\+json[^>]*(?:class="([^"]*)")?[^>]*>(.*?)</script>',
            html,
            re.S,
        )
        metas = re.findall(
            r'<meta\s+[^>]*name="(description|robots|generator)"[^>]*content="([^"]*)"',
            html,
        )
        ogs = re.findall(
            r'<meta\s+[^>]*property="(og:[^"]+)"[^>]*content="([^"]*)"',
            html,
        )

        lines.append(f"## {url}")
        lines.append(f"- JSON-LD blocks: {len(schemas)}")
        for cls, body in schemas:
            types = re.findall(r'"@type":"([A-Za-z]+)"', body)
            lines.append(f"  - class=`{cls or '(none)'}` types={types}")
        lines.append(f"- meta name= tags: {len(metas)}")
        for name, content in metas:
            lines.append(f"  - {name}: {content[:120]}")
        lines.append(f"- og: tags: {len(ogs)}")
        # Surface the Rank Math fingerprint comment if present.
        if "Rank Math" in html:
            lines.append("- **Rank Math HTML comment detected on page**")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
