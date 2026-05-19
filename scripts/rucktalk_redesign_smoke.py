"""Post-launch smoke tests for rucktalk.com redesign (Plan 1A Task 29).

Self-contained checker that hits production URLs over httpx and verifies
fingerprints introduced by the redesign. Run after the T3 cutover; use
`--skip-redirects` to run pre-cutover when the fitasruck.com 301 is not
yet live (the rest of the site can still be smoked early if a staging
slot is promoted).

Exit code 0 if every check passes, 1 if any fail.

Usage:
    venv/bin/python scripts/rucktalk_redesign_smoke.py
    venv/bin/python scripts/rucktalk_redesign_smoke.py --skip-redirects
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import httpx

BASE = "https://rucktalk.com"
TIMEOUT = 15.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def chk_status(label: str, url: str, expect: int = 200) -> Check:
    """200-by-default status code check (no redirect follow)."""
    try:
        r = httpx.get(url, follow_redirects=False, timeout=TIMEOUT)
        return Check(label, r.status_code == expect, f"got {r.status_code} (expected {expect})")
    except Exception as e:  # noqa: BLE001
        return Check(label, False, f"request failed: {e}")


def chk_status_follow(label: str, url: str, expect: int = 200) -> Check:
    """Status check that follows redirects (apex/www etc.)."""
    try:
        r = httpx.get(url, follow_redirects=True, timeout=TIMEOUT)
        return Check(label, r.status_code == expect, f"got {r.status_code} (expected {expect})")
    except Exception as e:  # noqa: BLE001
        return Check(label, False, f"request failed: {e}")


def chk_contains_any(label: str, url: str, needles: list[str]) -> Check:
    """Pass if ANY of the needles appears (case-insensitive) in the response body."""
    try:
        r = httpx.get(url, follow_redirects=True, timeout=TIMEOUT)
        body = r.text.lower()
        for n in needles:
            if n.lower() in body:
                return Check(label, True, f"found '{n}'")
        return Check(label, False, f"missing all of: {needles}")
    except Exception as e:  # noqa: BLE001
        return Check(label, False, f"request failed: {e}")


def chk_contains(label: str, url: str, needle: str) -> Check:
    """Pass if the needle (case-insensitive) appears in the response body."""
    return chk_contains_any(label, url, [needle])


def chk_redirect(label: str, url: str, expect_loc_contains: str) -> Check:
    """Pass if response is 3xx and Location header contains the expected substring."""
    try:
        r = httpx.get(url, follow_redirects=False, timeout=TIMEOUT)
        loc = r.headers.get("location", "")
        ok = r.status_code in (301, 302, 307, 308) and expect_loc_contains in loc
        return Check(label, ok, f"status={r.status_code} location='{loc}'")
    except Exception as e:  # noqa: BLE001
        return Check(label, False, f"request failed: {e}")


def run_checks(skip_redirects: bool) -> list[Check]:
    checks: list[Check] = []

    # ---- Page reachability ----
    checks.append(chk_status("homepage 200", f"{BASE}/"))
    checks.append(chk_status("/training/ 200", f"{BASE}/training/"))
    checks.append(chk_status("/training/free/ 200", f"{BASE}/training/free/"))
    checks.append(chk_status("/blog/ 200", f"{BASE}/blog/"))
    checks.append(chk_status("/podcast/ 200", f"{BASE}/podcast/"))

    # ---- Homepage fingerprints (locked design language) ----
    checks.append(chk_contains("hero tagline locked",        f"{BASE}/", "a guy figuring it out"))
    checks.append(chk_contains("ecosystem strip",            f"{BASE}/", "Part of the Ground Rush ecosystem"))
    checks.append(chk_contains_any("LoovaCast radio bar",    f"{BASE}/", ["rt-radio-bar", 'class="radio"']))
    checks.append(chk_contains("LumaBot mount",              f"{BASE}/", "rt-lumabot-mount"))
    checks.append(chk_contains_any("newsletter popup mount", f"{BASE}/", ["rt-popup", 'class="popup"']))

    # ---- www. subdomain (after Task 24 Cloudflare 525 fix) ----
    if not skip_redirects:
        checks.append(chk_status_follow("www.rucktalk.com reachable", "https://www.rucktalk.com/"))

    # ---- fitasruck cutover (after Task 23) ----
    if not skip_redirects:
        checks.append(chk_redirect("fitasruck.com → rucktalk.com", "https://fitasruck.com/", "rucktalk.com"))

    return checks


def render(checks: list[Check]) -> int:
    name_w = max((len(c.name) for c in checks), default=20)
    name_w = max(name_w, 20)
    detail_w = 60

    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"

    print(f"{'CHECK':<{name_w}}  {'OK':<3}  DETAIL")
    print("-" * (name_w + 3 + 3 + detail_w))
    for c in checks:
        flag = f"{GREEN}✓{RESET}" if c.ok else f"{RED}✗{RESET}"
        print(f"{c.name:<{name_w}}  {flag:<3}  {c.detail}")

    failed = [c for c in checks if not c.ok]
    total = len(checks)
    passed = total - len(failed)
    print()
    print(f"Total: {total}  ·  Passed: {passed}  ·  Failed: {len(failed)}")

    return 0 if not failed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test rucktalk.com redesign.")
    parser.add_argument(
        "--skip-redirects",
        action="store_true",
        help="Skip www.rucktalk.com + fitasruck.com 301 checks (use pre-cutover).",
    )
    args = parser.parse_args()

    checks = run_checks(skip_redirects=args.skip_redirects)
    return render(checks)


if __name__ == "__main__":
    sys.exit(main())
