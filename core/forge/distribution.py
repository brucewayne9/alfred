"""Forge Distribution — ready-to-post packs (no auto-posting). Roster + pack + checklist."""
from __future__ import annotations

import os
import json
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Roster persistence (JSON file)
# ---------------------------------------------------------------------------

DEFAULT_ACCOUNTS = [
    {"handle": "@rodwave.daily", "platform": "TikTok", "tier": "burner"},
    {"handle": "@dontlookdown.clips", "platform": "Instagram Reels", "tier": "burner"},
    {"handle": "@rodwave", "platform": "YouTube Shorts", "tier": "official"},
]


def _accounts_path() -> Path:
    return Path(os.environ.get("FORGE_ACCOUNTS_PATH", "data/forge_accounts.json"))


def get_accounts() -> list[dict]:
    """Return the roster. If no file exists yet, seed it with DEFAULT_ACCOUNTS."""
    p = _accounts_path()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return set_accounts(DEFAULT_ACCOUNTS)


def set_accounts(accounts: list[dict]) -> list[dict]:
    """Persist the roster as JSON, return it."""
    p = _accounts_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(accounts, f, indent=2)
    return accounts


# ---------------------------------------------------------------------------
# Posted-status persistence (SQLite)
# ---------------------------------------------------------------------------


def _init_dist() -> None:
    from core.forge.db import _conn

    with _conn() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS dist_posts ("
            "post_id TEXT PRIMARY KEY, "
            "posted INTEGER NOT NULL DEFAULT 0, "
            "posted_at INTEGER)"
        )


def mark_posted(post_id: str, posted: bool = True) -> None:
    from core.forge.db import _conn

    _init_dist()
    posted_int = 1 if posted else 0
    posted_at = int(time.time()) if posted else None
    with _conn() as c:
        c.execute(
            "INSERT INTO dist_posts (post_id, posted, posted_at) VALUES (?, ?, ?) "
            "ON CONFLICT(post_id) DO UPDATE SET posted = excluded.posted, "
            "posted_at = excluded.posted_at",
            (post_id, posted_int, posted_at),
        )


def posted_map(post_ids: list[str]) -> dict:
    """Return {post_id: True} for the given ids marked posted=1."""
    from core.forge.db import _conn

    if not post_ids:
        return {}
    _init_dist()
    placeholders = ",".join("?" for _ in post_ids)
    with _conn() as c:
        rows = c.execute(
            f"SELECT post_id FROM dist_posts WHERE posted = 1 AND post_id IN ({placeholders})",
            post_ids,
        ).fetchall()
    return {r["post_id"]: True for r in rows}


# ---------------------------------------------------------------------------
# Pack assembly
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Postiz drafts — push a pack into Postiz as DRAFTS (a human hits send)
# ---------------------------------------------------------------------------

_POSTIZ_SCRIPT_DIR = Path(
    os.environ.get("POSTIZ_SCRIPT_DIR",
                   "/home/aialfred/.openclaw/workspace/scripts/integrations")
)


def push_to_postiz(job_id: str, accounts: list[dict] | None = None) -> dict:
    """Push a job's pack into Postiz as DRAFTS — nothing auto-publishes.

    Each account in the roster must carry a ``postiz_id`` (its connected Postiz
    integration). Accounts without one are reported as ``skipped`` so the gap
    (un-connected Mainstay accounts) is visible, never silently swallowed.
    """
    import subprocess
    import tempfile
    from datetime import datetime, timedelta
    from core.forge import library

    pack = build_pack(job_id, accounts)
    posts = pack.get("posts", [])
    roster = {a.get("handle"): a for a in pack.get("accounts", [])}

    script = _POSTIZ_SCRIPT_DIR / "postiz.py"
    if not script.exists():
        return {"job_id": job_id, "error": f"postiz script missing: {script}",
                "pushed": [], "skipped": [], "counts": {"pushed": 0, "skipped": len(posts)}}

    pushed, skipped = [], []
    base = datetime.utcnow()
    for p in posts:
        acct = roster.get(p["account"], {})
        integration_id = acct.get("postiz_id")
        if not integration_id:
            skipped.append({"post_id": p["post_id"], "account": p["account"],
                            "reason": "account not connected to Postiz (no postiz_id)"})
            continue

        tmp = None
        try:
            data, _mime = library.read_file(p["file_path"])
            suffix = Path(p["file_name"]).suffix or ".mp4"
            fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="forge_postiz_")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            when = (base + timedelta(minutes=p.get("stagger_minutes", 0))) \
                .replace(microsecond=0).isoformat()
            proc = subprocess.run(
                ["python3", str(script), "create-draft",
                 p["caption"], integration_id, tmp, when],
                cwd=str(_POSTIZ_SCRIPT_DIR), capture_output=True, text=True, timeout=180)
            out = (proc.stdout or "").strip().splitlines()
            parsed = json.loads(out[-1]) if out else {"ok": False, "error": "no output"}
            entry = {"post_id": p["post_id"], "account": p["account"],
                     "platform": p["platform"], "ok": bool(parsed.get("ok")),
                     "draft": parsed.get("result")}
            if not parsed.get("ok"):
                entry["error"] = parsed.get("error") or proc.stderr[-300:]
            pushed.append(entry)
        except Exception as exc:
            pushed.append({"post_id": p["post_id"], "account": p["account"],
                           "ok": False, "error": str(exc)})
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    ok = sum(1 for e in pushed if e.get("ok"))
    return {
        "job_id": job_id,
        "pushed": pushed,
        "skipped": skipped,
        "counts": {"drafts_created": ok, "failed": len(pushed) - ok, "skipped": len(skipped)},
    }


def build_pack(job_id: str, accounts: list[dict] | None = None) -> dict:
    """Assemble a ready-to-post pack for a delivered job (hits Nextcloud for files)."""
    from core.forge import jobs as fj, library

    job = fj.get_job(job_id)
    if job is None:
        return {
            "job_id": job_id,
            "caption": "",
            "posts": [],
            "accounts": accounts or get_accounts(),
            "counts": {"posts": 0, "posted": 0},
            "error": "job not found",
        }

    caption = (job.get("params") or {}).get("caption", "")
    res = job.get("result") or {}
    dirs = res.get("delivered_dirs") or (
        [res["delivered_dir"]] if res.get("delivered_dir") else []
    )

    files: list[dict] = []
    for d in dirs:
        try:
            files += library.list_dir_files(d)
        except Exception:
            pass

    accounts = accounts or get_accounts()
    posts = assign_posts(job_id, files, accounts, caption=caption)

    pm = posted_map([p["post_id"] for p in posts])
    for p in posts:
        p["posted"] = pm.get(p["post_id"], False)

    return {
        "job_id": job_id,
        "caption": caption,
        "posts": posts,
        "accounts": accounts,
        "counts": {"posts": len(posts), "posted": sum(1 for p in posts if p["posted"])},
    }
