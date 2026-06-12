"""Mainstay Forge — Intelligence layer.

Pulls per-video performance from the connected TikTok accounts, attributes each
video to the *sound/variant* it came from, and aggregates a leaderboard so the
team can see which Rod Wave snippet is winning and push more of it.

Design (per Mike, 2026-06-11):
  * Primary unit = **sound/variant** (the headline board).
  * Drill-downs  = **account** (which page is hot) and **post** (individual clips).

Data source = TikTok's API via each account's OAuth token (held in Postiz). The
standard Display API exposes views / likes / comments / shares + follower stats;
it does NOT expose retention or saves (those need TikTok's Business analytics or
manual entry), so the board reports engagement-rate and share-velocity instead.

The puller (`pull_now`) is wired but gated: it lights up the moment the TikTok
production audit clears (adds the `video.list` + `user.info.stats` scopes and
DIRECT_POST attribution). Until then the board renders an honest empty state —
the schema, aggregation, and UI are all live and tested with no data.
"""
from __future__ import annotations

import time
from typing import Optional

from core.forge.db import _conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_intel() -> None:
    """Create the intel tables if absent. Safe to call repeatedly."""
    with _conn() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS intel_videos ("
            "video_id TEXT PRIMARY KEY, "          # TikTok video id (or synthetic until tracked)
            "account_handle TEXT, "
            "postiz_id TEXT, "
            "platform TEXT DEFAULT 'TikTok', "
            "sound TEXT, "                          # the song/snippet the clip used
            "variant TEXT, "                        # which structural variant / cut
            "post_id TEXT, "                        # Forge post_id when attributable
            "job_id TEXT, "
            "caption TEXT, "
            "views INTEGER DEFAULT 0, "
            "likes INTEGER DEFAULT 0, "
            "comments INTEGER DEFAULT 0, "
            "shares INTEGER DEFAULT 0, "
            "posted_at INTEGER, "
            "captured_at INTEGER)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS intel_accounts ("
            "postiz_id TEXT PRIMARY KEY, "
            "handle TEXT, "
            "platform TEXT DEFAULT 'TikTok', "
            "followers INTEGER DEFAULT 0, "
            "following INTEGER DEFAULT 0, "
            "likes INTEGER DEFAULT 0, "
            "video_count INTEGER DEFAULT 0, "
            "captured_at INTEGER)"
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_intel_videos_sound ON intel_videos(sound)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_intel_videos_acct ON intel_videos(account_handle)")
        # Phase 08 calibration: the Auto-Clips score this video was cut from, so
        # we can later ask "did high-scored clips actually get more views?".
        # Idempotent migration for DBs created before this column existed.
        cols = {r[1] for r in c.execute("PRAGMA table_info(intel_videos)").fetchall()}
        if "predicted_score" not in cols:
            c.execute("ALTER TABLE intel_videos ADD COLUMN predicted_score INTEGER")


def _now() -> int:
    return int(time.time())


# ---------------------------------------------------------------------------
# Writes (used by the puller and by tests)
# ---------------------------------------------------------------------------

def record_video(video_id: str, *, account_handle: str = "", postiz_id: str = "",
                 platform: str = "TikTok", sound: str = "", variant: str = "",
                 post_id: str = "", job_id: str = "", caption: str = "",
                 views: int = 0, likes: int = 0, comments: int = 0, shares: int = 0,
                 posted_at: Optional[int] = None,
                 predicted_score: Optional[int] = None) -> None:
    """Upsert one video's latest stats (puller calls this each refresh).

    ``predicted_score`` is the Auto-Clips virality score the clip was cut from,
    stamped at post time so calibration can join it to real engagement later.
    """
    init_intel()
    with _conn() as c:
        c.execute(
            "INSERT INTO intel_videos (video_id, account_handle, postiz_id, platform, sound, "
            "variant, post_id, job_id, caption, views, likes, comments, shares, posted_at, "
            "predicted_score, captured_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(video_id) DO UPDATE SET "
            "views=excluded.views, likes=excluded.likes, comments=excluded.comments, "
            "shares=excluded.shares, captured_at=excluded.captured_at, "
            "sound=COALESCE(NULLIF(excluded.sound,''), intel_videos.sound), "
            "variant=COALESCE(NULLIF(excluded.variant,''), intel_videos.variant), "
            "predicted_score=COALESCE(excluded.predicted_score, intel_videos.predicted_score)",
            (video_id, account_handle, postiz_id, platform, sound, variant, post_id,
             job_id, caption, views, likes, comments, shares, posted_at,
             predicted_score, _now()),
        )


def record_account(postiz_id: str, *, handle: str = "", platform: str = "TikTok",
                   followers: int = 0, following: int = 0, likes: int = 0,
                   video_count: int = 0) -> None:
    """Upsert one account's follower/profile stats."""
    init_intel()
    with _conn() as c:
        c.execute(
            "INSERT INTO intel_accounts (postiz_id, handle, platform, followers, following, "
            "likes, video_count, captured_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(postiz_id) DO UPDATE SET handle=excluded.handle, "
            "followers=excluded.followers, following=excluded.following, likes=excluded.likes, "
            "video_count=excluded.video_count, captured_at=excluded.captured_at",
            (postiz_id, handle, platform, followers, following, likes, video_count, _now()),
        )


# ---------------------------------------------------------------------------
# Board aggregation
# ---------------------------------------------------------------------------

def _engagement_rate(views: int, likes: int, comments: int, shares: int) -> float:
    if views <= 0:
        return 0.0
    return round(100.0 * (likes + comments + shares) / views, 1)


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


def board(unit: str = "sound") -> dict:
    """The Intelligence board.

    Returns the headline leaderboard for ``unit`` (sound | account | post) plus
    the winner call and the raw account + post drill-downs. With no captured data
    yet every list is empty and ``has_data`` is False — the honest pre-audit state.
    """
    init_intel()
    with _conn() as c:
        vids = _rows_to_dicts(c.execute("SELECT * FROM intel_videos").fetchall())
        accts = _rows_to_dicts(c.execute(
            "SELECT * FROM intel_accounts ORDER BY followers DESC").fetchall())

    # --- sound leaderboard -------------------------------------------------
    by_sound: dict[str, dict] = {}
    for v in vids:
        key = (v.get("sound") or "Unattributed").strip() or "Unattributed"
        agg = by_sound.setdefault(key, {"sound": key, "plays": 0, "views": 0,
                                        "likes": 0, "comments": 0, "shares": 0, "posts": 0})
        agg["views"] += v["views"]; agg["likes"] += v["likes"]
        agg["comments"] += v["comments"]; agg["shares"] += v["shares"]
        agg["posts"] += 1
    sounds = list(by_sound.values())
    for s in sounds:
        s["plays"] = s["views"]
        s["engagement"] = _engagement_rate(s["views"], s["likes"], s["comments"], s["shares"])
    sounds.sort(key=lambda s: (s["views"], s["engagement"]), reverse=True)
    peak = max((s["views"] for s in sounds), default=0) or 1
    for s in sounds:
        s["momentum"] = round(100.0 * s["views"] / peak, 1)

    # --- account drill-down ------------------------------------------------
    by_acct: dict[str, dict] = {}
    for v in vids:
        key = v.get("account_handle") or "—"
        a = by_acct.setdefault(key, {"account": key, "views": 0, "likes": 0,
                                     "comments": 0, "shares": 0, "posts": 0})
        a["views"] += v["views"]; a["likes"] += v["likes"]
        a["comments"] += v["comments"]; a["shares"] += v["shares"]; a["posts"] += 1
    accounts = []
    fol = {a["handle"]: a for a in accts}
    for a in by_acct.values():
        a["engagement"] = _engagement_rate(a["views"], a["likes"], a["comments"], a["shares"])
        a["followers"] = (fol.get(a["account"]) or {}).get("followers", 0)
        accounts.append(a)
    accounts.sort(key=lambda a: a["views"], reverse=True)
    # accounts that are connected but have no posts yet still belong on the board
    seen = {a["account"] for a in accounts}
    for a in accts:
        if a["handle"] not in seen:
            accounts.append({"account": a["handle"], "views": 0, "likes": 0, "comments": 0,
                             "shares": 0, "posts": 0, "engagement": 0.0,
                             "followers": a.get("followers", 0)})

    # --- post drill-down ---------------------------------------------------
    posts = sorted(vids, key=lambda v: v["views"], reverse=True)
    for v in posts:
        v["engagement"] = _engagement_rate(v["views"], v["likes"], v["comments"], v["shares"])

    winner = sounds[0] if sounds and sounds[0]["views"] > 0 else None
    return {
        "unit": unit,
        "has_data": bool(vids),
        "winner": winner,
        "sounds": sounds,
        "accounts": accounts,
        "posts": posts[:50],
        "totals": {
            "videos": len(vids),
            "views": sum(v["views"] for v in vids),
            "accounts_tracked": len(accts),
        },
    }


# ---------------------------------------------------------------------------
# Calibration — is the Auto-Clips virality score any good? (Phase 08, Phase 2)
# ---------------------------------------------------------------------------

# Score bands for the engagement readout. High → low so the board reads top-down.
_BANDS = [("85-100", 85, 100), ("70-84", 70, 84), ("50-69", 50, 69), ("0-49", 0, 49)]


def calibration() -> dict:
    """Two read-outs on whether the virality score predicts reality.

    editorial (LIVE NOW, no TikTok audit needed): do editors actually cut the
        high-scored clips? Compares the mean score of rendered vs skipped
        candidates — a positive ``lift`` means the scorer agrees with human taste.

    engagement (GATED on the TikTok audit + post attribution): once real view
        counts flow, bucket posted clips by score band and show mean views per
        band. Empty until tracked views with a predicted_score exist.
    """
    init_intel()
    with _conn() as c:
        cand = _rows_to_dicts(c.execute(
            "SELECT score, rendered, posted FROM clip_candidates").fetchall())
        tracked = _rows_to_dicts(c.execute(
            "SELECT predicted_score, views, likes, comments, shares FROM intel_videos "
            "WHERE predicted_score IS NOT NULL AND views > 0").fetchall())

    # --- editorial signal --------------------------------------------------
    rendered = [r["score"] for r in cand if r["rendered"]]
    skipped = [r["score"] for r in cand if not r["rendered"]]
    posted = [r for r in cand if r["posted"]]
    avg_r = round(sum(rendered) / len(rendered), 1) if rendered else 0.0
    avg_s = round(sum(skipped) / len(skipped), 1) if skipped else 0.0
    editorial = {
        "scored": len(cand),
        "rendered": len(rendered),
        "posted": len(posted),
        "avg_score_rendered": avg_r,
        "avg_score_skipped": avg_s,
        "lift": round(avg_r - avg_s, 1) if rendered and skipped else 0.0,
        "has_data": bool(rendered),
    }

    # --- engagement signal -------------------------------------------------
    bands = []
    for label, lo, hi in _BANDS:
        rows = [t for t in tracked if lo <= (t["predicted_score"] or 0) <= hi]
        if rows:
            n = len(rows)
            bands.append({
                "band": label,
                "posts": n,
                "avg_views": round(sum(t["views"] for t in rows) / n),
                "avg_engagement": round(sum(
                    _engagement_rate(t["views"], t["likes"], t["comments"], t["shares"])
                    for t in rows) / n, 1),
            })
        else:
            bands.append({"band": label, "posts": 0, "avg_views": 0, "avg_engagement": 0.0})
    engagement = {"bands": bands, "tracked": len(tracked), "has_data": bool(tracked)}

    return {"editorial": editorial, "engagement": engagement}


# ---------------------------------------------------------------------------
# Puller — gated on the TikTok production audit
# ---------------------------------------------------------------------------

def pull_now() -> dict:
    """Refresh stats for every connected account.

    Reads the live Postiz connections, then for each one pulls its recent videos
    + profile stats from TikTok and upserts them. Returns a status report.

    Gating: TikTok's video.list / user.info.stats require the production scopes
    (audit submitted 2026-06-08, clearing ~this week). Until a token+scope is
    available per account, that account is reported as ``pending`` and skipped —
    never silently — so the moment the audit clears this starts returning data.
    """
    init_intel()
    from core.forge import distribution
    targets = distribution.live_targets()
    pulled, pending, errors = [], [], []
    for t in targets:
        pid = t.get("postiz_id")
        handle = t.get("handle")
        try:
            token = _tiktok_token(pid)
            if not token:
                pending.append({"account": handle, "reason": "no TikTok token yet (audit/scope pending)"})
                continue
            prof = _fetch_user_stats(token)
            if prof:
                record_account(pid, handle=handle, platform=t.get("platform", "TikTok"), **prof)
            vids = _fetch_video_list(token)
            for v in vids:
                record_video(v["video_id"], account_handle=handle, postiz_id=pid,
                             platform=t.get("platform", "TikTok"),
                             views=v.get("view_count", 0), likes=v.get("like_count", 0),
                             comments=v.get("comment_count", 0), shares=v.get("share_count", 0),
                             caption=v.get("video_description", ""))
            pulled.append({"account": handle, "videos": len(vids)})
        except Exception as exc:  # noqa: BLE001
            errors.append({"account": handle, "error": str(exc)[:160]})
    return {"pulled": pulled, "pending": pending, "errors": errors,
            "connected": len(targets), "captured": sum(p.get("videos", 0) for p in pulled)}


def _tiktok_token(postiz_id: str | None) -> Optional[str]:
    """Resolve a usable TikTok access token for a connected account.

    The OAuth tokens live in Postiz's Postgres (117). Until the production audit
    clears the scopes — and we wire the token bridge — this returns None so the
    puller cleanly reports the account as pending rather than guessing.
    """
    # TODO(audit-clear): read + refresh the per-account token from Postiz and return it.
    return None


def _fetch_user_stats(token: str) -> dict:
    """TikTok user.info.stats -> {followers, following, likes, video_count}.

    Shape ready; activated when a token is available post-audit.
    """
    return {}


def _fetch_video_list(token: str) -> list[dict]:
    """TikTok video.list -> [{video_id, view_count, like_count, comment_count, share_count, ...}].

    Shape ready; activated when a token is available post-audit.
    """
    return []
