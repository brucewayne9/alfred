"""Sonaar theme audit for rucktalk.com — read-only via WP REST API.

Captures: active theme + version (auth-gated), all WP pages + their
templates, registered custom post types, and a markdown-formatted report
suitable for piping to docs/superpowers/audits/.

Auth-gated endpoints (/themes) need an app password; we still surface
useful info from public endpoints (/pages, /types) when the password
isn't available.

Usage:
    venv/bin/python scripts/rucktalk_redesign_audit.py > \\
        docs/superpowers/audits/2026-05-19-sonaar-theme-audit.md
"""
from __future__ import annotations

import datetime as dt
import sys

import httpx

sys.path.insert(0, "/home/aialfred/alfred")

from config.settings import settings

WP_BASE = "https://rucktalk.com/wp-json/wp/v2"
WP_USER = settings.rucktalk_wp_app_user
WP_APP_PWD = settings.rucktalk_wp_app_password


def get(path: str, params: dict | None = None, auth_required: bool = False):
    auth = (WP_USER, WP_APP_PWD) if (WP_APP_PWD and auth_required) else None
    try:
        r = httpx.get(
            f"{WP_BASE}{path}",
            params=params or {},
            auth=auth,
            timeout=30,
            follow_redirects=True,
        )
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}
    if r.status_code >= 400:
        return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}
    try:
        return r.json()
    except Exception:
        return {"_error": "non-json response"}


def section(title: str) -> None:
    print(f"\n## {title}\n")


def main() -> int:
    today = dt.date.today().isoformat()
    print(f"# Sonaar theme audit — rucktalk.com\n")
    print(f"Generated: {today}\n")
    print(
        "Pulled via WP REST API. Auth-gated endpoints (/themes) require an"
        " app password to be set in `settings.rucktalk_wp_app_password`."
    )

    # ── Active theme (auth required) ──
    section("Active theme")
    if not WP_APP_PWD:
        print("⚠️  Skipped — no WP app password configured. Set `RUCKTALK_WP_APP_PASSWORD` in /home/aialfred/alfred/config/.env.")
    else:
        themes = get("/themes", auth_required=True)
        if isinstance(themes, dict) and themes.get("_error"):
            print(f"⚠️  Failed: {themes['_error']}")
        else:
            active = [t for t in themes if t.get("status") == "active"]
            if not active:
                print("⚠️  No active theme reported.")
            for t in active:
                name = (t.get("name") or {}).get("rendered", "?")
                print(f"- **Name:** {name}")
                print(f"- **Version:** {t.get('version', '?')}")
                print(f"- **Template (parent):** {t.get('template', '?')}")
                print(f"- **Stylesheet:** {t.get('stylesheet', '?')}")
                print(f"- **Status:** {t.get('status', '?')}")
                supports = t.get("theme_supports") or {}
                if supports:
                    keys = sorted(supports.keys())
                    print(f"- **Theme supports:** {', '.join(keys)}")

    # ── Pages + templates in use ──
    section("Pages + templates in use")
    pages = get("/pages", {"per_page": 100, "_fields": "id,slug,title,template,status"})
    if isinstance(pages, dict) and pages.get("_error"):
        print(f"⚠️  Failed: {pages['_error']}")
    else:
        print("| ID | Slug | Title | Template | Status |")
        print("|---|---|---|---|---|")
        for p in pages:
            title = (p.get("title") or {}).get("rendered", "?")
            template = p.get("template") or "(default)"
            status = p.get("status", "?")
            print(f"| {p.get('id')} | `{p.get('slug')}` | {title} | `{template}` | {status} |")

    # ── Custom post types ──
    section("Registered custom post types")
    cpts = get("/types")
    if isinstance(cpts, dict) and cpts.get("_error"):
        print(f"⚠️  Failed: {cpts['_error']}")
    elif isinstance(cpts, dict):
        for slug, info in sorted(cpts.items()):
            name = info.get("name", "?")
            rest = info.get("rest_base") or "—"
            print(f"- `{slug}` — {name} (rest_base: `{rest}`)")
    else:
        print("⚠️  Unexpected response shape.")

    # ── Taxonomies ──
    section("Taxonomies")
    taxes = get("/taxonomies")
    if isinstance(taxes, dict) and taxes.get("_error"):
        print(f"⚠️  Failed: {taxes['_error']}")
    elif isinstance(taxes, dict):
        for slug, info in sorted(taxes.items()):
            name = info.get("name", "?")
            cpts_for = ", ".join(info.get("types", []) or [])
            print(f"- `{slug}` — {name} (applies to: {cpts_for or '—'})")

    # ── Categories (blog signals) ──
    section("Blog categories")
    cats = get("/categories", {"per_page": 100, "_fields": "id,slug,name,count"})
    if isinstance(cats, dict) and cats.get("_error"):
        print(f"⚠️  Failed: {cats['_error']}")
    else:
        print("| ID | Slug | Name | Posts |")
        print("|---|---|---|---|")
        for c in cats:
            print(f"| {c.get('id')} | `{c.get('slug')}` | {c.get('name')} | {c.get('count')} |")

    # ── Recent posts (just count) ──
    section("Recent published posts (last 10)")
    posts = get("/posts", {"per_page": 10, "_fields": "id,slug,title,date,categories"})
    if isinstance(posts, dict) and posts.get("_error"):
        print(f"⚠️  Failed: {posts['_error']}")
    else:
        print("| Date | Slug | Title |")
        print("|---|---|---|")
        for p in posts:
            date = (p.get("date") or "")[:10]
            title = (p.get("title") or {}).get("rendered", "?")
            print(f"| {date} | `{p.get('slug')}` | {title} |")

    # ── Recent podcast episodes (verify Sonaar CPT) ──
    section("Recent podcast episodes (Sonaar `podcast` CPT)")
    eps = get("/podcast", {"per_page": 5, "_fields": "id,slug,title,date"})
    if isinstance(eps, dict) and eps.get("_error"):
        print(f"⚠️  Failed (or `podcast` CPT not exposed in REST): {eps['_error']}")
    elif isinstance(eps, list):
        if not eps:
            print("⚠️  No podcast episodes returned.")
        else:
            print("| Date | Slug | Title |")
            print("|---|---|---|")
            for e in eps:
                date = (e.get("date") or "")[:10]
                title = (e.get("title") or {}).get("rendered", "?")
                print(f"| {date} | `{e.get('slug')}` | {title} |")

    # ── Existing menu/nav (manual checklist for server-100 follow-up) ──
    section("Menu inspection (deferred to server-100 wp-cli)")
    print("WP REST has no menus endpoint without nav-blocks. Run on server-100:")
    print("```bash")
    print("ssh server-100 \"docker exec rt-wordpress wp menu list --allow-root\"")
    print("ssh server-100 \"docker exec rt-wordpress wp menu location list --allow-root\"")
    print("```")

    # ── Sonaar override surface (deferred to server-100 grep) ──
    section("Sonaar override surface (deferred to Task 2)")
    print("Task 2 of Plan 1A greps the Sonaar parent theme PHP for actionable")
    print("hooks (`apply_filters`, `do_action`). Append findings to this doc.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
