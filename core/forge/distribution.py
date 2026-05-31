"""Forge Distribution — ready-to-post packs (no auto-posting). Roster + pack + checklist."""
from __future__ import annotations

PLATFORMS = ["TikTok", "Instagram Reels", "YouTube Shorts"]

_BASE_TAGS = ["#RodWave", "#DontLookDown", "#newmusic", "#fyp"]
_PLATFORM_TAGS = {
    "TikTok": ["#fyp", "#foryou", "#tiktokmusic"],
    "Instagram Reels": ["#reels", "#reelsinstagram", "#explore"],
    "YouTube Shorts": ["#shorts", "#youtubeshorts"],
}


def build_caption(hook: str, platform: str) -> str:
    """Paste-ready post copy: the hook + a CTA + base & platform hashtags."""
    hook = (hook or "").strip()
    tags = _BASE_TAGS + _PLATFORM_TAGS.get(platform, [])
    seen, ordered = set(), []
    for t in tags:
        k = t.lower()
        if k not in seen:
            seen.add(k); ordered.append(t)
    cta = "Don't Look Down — out June 19."
    lines = [hook] if hook else []
    lines += [cta, " ".join(ordered)]
    return "\n\n".join(lines)


def assign_posts(job_id: str, files: list[dict], accounts: list[dict], *,
                 caption: str = "", stagger_minutes: int = 20) -> list[dict]:
    """Round-robin files across accounts; each post gets platform, stagger, caption, id."""
    if not accounts or not files:
        return []
    posts = []
    for i, f in enumerate(files):
        acct = accounts[i % len(accounts)]
        platform = acct.get("platform") or "TikTok"
        posts.append({
            "post_id": f"{job_id}:{i}",
            "job_id": job_id,
            "file_name": f.get("name", ""),
            "file_path": f.get("path", ""),
            "account": acct.get("handle", ""),
            "platform": platform,
            "stagger_minutes": i * stagger_minutes,
            "caption": build_caption(caption, platform),
            "posted": False,
        })
    return posts
