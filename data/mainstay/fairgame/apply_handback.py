#!/usr/bin/env python3
"""
Apply Travis's redesigned sandbox back onto the real Fair Game app.

What it does (safely):
  1. Backs up the current app/ -> app.bak-<stamp>/
  2. For each page, strips the sandbox engine:
       - removes the  <script src="mock-api.js"></script>  line
  3. Copies the redesigned HTML + CSS (+ any new asset files) into app/
  4. Refuses to copy mock-api.js or the sandbox README into the live app
  5. Verifies every FROZEN element id still exists in each page; warns loudly if not

Usage:
    python3 apply_handback.py /path/to/travis-returned-folder
    python3 apply_handback.py /path/to/travis-returned-folder --dry-run

After it runs clean, restart the service:
    sudo systemctl restart fairgame-api.service
and smoke-test https://aialfred.groundrushcloud.com/fairgame/app/
"""
import sys, re, shutil, pathlib, time

HERE = pathlib.Path(__file__).resolve().parent
APP = HERE / "app"

FROZEN = {
    "index.html": "navRight showsGrid showsMeta statShows toast".split(),
    "show.html": ("navRight showHead inventory listings buybar qMinus qPlus qVal getBtn "
                  "payBtn doneBtn gateOverlay gateClose orderOverlay orderBody orderClose "
                  "retryClose toast").split(),
    "sell.html": ("navRight gatePane formPane donePane sellForm show viewShow face faceEcho "
                  "calc amtBuyer amtRod amtYou listBtn listAnother doneMsg doneSummary "
                  "section loadHint toast").split(),
    "account.html": ("navRight registerForm registerBtn phone email smsTarget smsVerifyBtn "
                     "smsResend smsDevCode emailTarget emailVerifyBtn emailDevCode steps "
                     "heroTitle heroLede acctRows doneBadges signOut startOver loadHint toast").split(),
    "admin.html": "kpis showsBox showsMeta ordersBox ordersMeta reloadBtn keyBtn stamp toast".split(),
}
PAGES = list(FROZEN.keys())
SKIP = {"mock-api.js", "SANDBOX-README.md", "DESIGN-HANDOFF.md"}
MOCK_LINE = re.compile(r'[ \t]*<script src="mock-api\.js"></script>\s*\n?')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        print(__doc__); sys.exit(1)
    incoming = pathlib.Path(args[0]).resolve()
    if not incoming.is_dir():
        print(f"!! not a folder: {incoming}"); sys.exit(1)
    # allow them to hand back a folder that itself contains the files, or a nested one
    if not any((incoming / p).exists() for p in PAGES):
        nested = [d for d in incoming.iterdir() if d.is_dir() and (d / "index.html").exists()]
        if nested:
            incoming = nested[0]
            print(f"   (using nested folder: {incoming.name})")

    print(f"== Fair Game handback ==\n   from: {incoming}\n   into: {APP}\n")

    problems = []
    # 1. backup
    if not dry:
        bak = HERE / f"app.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copytree(APP, bak)
        print(f"[backup] {bak.name}/")

    # 2/3. pages
    for page in PAGES:
        srcf = incoming / page
        if not srcf.exists():
            print(f"[skip ] {page} — not in handback (keeping current)"); continue
        html = srcf.read_text()
        html = MOCK_LINE.sub("", html)        # strip sandbox engine
        for fid in FROZEN[page]:
            if f'id="{fid}"' not in html and f"id='{fid}'" not in html:
                problems.append(f"{page}: missing frozen id '{fid}'")
        if dry:
            print(f"[would ] write {page}  ({'mock line removed' if 'mock-api.js' in srcf.read_text() else 'clean'})")
        else:
            (APP / page).write_text(html)
            print(f"[write ] {page}")

    # 4. css + any new assets (but never the sandbox engine / readmes)
    for f in sorted(incoming.iterdir()):
        if f.is_dir() or f.name in SKIP or f.name in PAGES:
            continue
        if f.suffix.lower() in (".html",):
            continue
        if dry:
            print(f"[would ] copy asset {f.name}")
        else:
            shutil.copy(f, APP / f.name)
            print(f"[asset ] {f.name}")

    # 5. report
    print()
    if problems:
        print("!! FROZEN-ID WARNINGS — these sections may go blank:")
        for p in problems:
            print("   -", p)
        print("\n   Fix the renamed ids in the affected page(s) and re-run, or restore from the app.bak-* backup.")
    else:
        print("OK — all frozen element ids present.")
    if not dry and not problems:
        print("\nNext: sudo systemctl restart fairgame-api.service")
        print("Then smoke-test https://aialfred.groundrushcloud.com/fairgame/app/")


if __name__ == "__main__":
    main()
