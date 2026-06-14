#!/usr/bin/env python3
"""Arena Drop Notifier — "your assets just landed" emails for the Rod Wave tour.

When we drop finished creative into an arena's DOWNLOAD folder on Nextcloud
("1. DOWNLOAD - Finished Assets From Ground Rush"), every venue contact who
signed up for THAT building through the portal gets a polished, celebratory
heads-up — so each drop becomes a little "the Rod Wave team is on it" moment
instead of the buildings chasing us for files.

Pieces it reuses (all already running):
  - WHO signed up for which arena -> data/mainstay/arena_portal/portal_state.json
    (members map: email -> {arena: idx}); same store the portal writes.
  - WHAT to watch -> each arena's DOWNLOAD subtree (recursive; assets land in
    category subfolders: Localized Ad Assets / Radio / PR / Logos / ...).
  - The mailer -> integrations.email.client, sent from Mike's Google Workspace
    address (mjohnson@groundrushinc.com) for best inbox placement + continuity
    (the venues already correspond with Mike). Replies land right back with him.

Deliverability (Mike's hard ask — stay out of spam):
  - sent from Google Workspace (strong SPF/DKIM/DMARC reputation),
  - one recipient per message (no bulk To/CC; venues never see each other),
  - multipart/alternative with a real plain-text part (text first),
  - restrained copy, single tasteful emoji, balanced text-to-link ratio.

Modes:
  --baseline   record everything currently in the DOWNLOAD folders as already
               seen, WITHOUT emailing. Run once at install so we never blast for
               pre-existing files. (Also auto-runs on first ever execution.)
  --dry-run    print what would send; touch nothing, email nobody.
  (default)    live: email contacts about new files, then record them as seen.

Cron (every 10 min):
  */10 * * * * /home/aialfred/alfred/venv/bin/python3 /home/aialfred/alfred/scripts/arena_drop_notifier.py >> /home/aialfred/alfred/data/arena_drop_notifier.log 2>&1
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

from integrations.email.client import email_client          # noqa: E402
from integrations.nextcloud import client as nc             # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
PORTAL_STATE = _ROOT / "data" / "mainstay" / "arena_portal" / "portal_state.json"
LINKS_PATH = _ROOT / "data" / "mainstay" / "tour" / "arena_folder_links.json"
NOTIFY_STATE = _ROOT / "data" / "mainstay" / "arena_portal" / "drop_notify_state.json"

DOWNLOAD_SUBFOLDER = "1. DOWNLOAD - Finished Assets From Ground Rush"
UPLOAD_SUBFOLDER = "2. UPLOAD - Your Logo + Ad Specs Here"
VENUE_SENDER = "groundrushinc"        # Mike Johnson <mjohnson@groundrushinc.com> (Workspace)
INTERNAL_SENDER = "groundrush info"   # info@groundrushlabs.com — for the heads-up to Mike
PORTAL_URL = "https://venues.groundrushlabs.com"


# ----------------------------------------------------------------- small utils

def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%SZ}] {msg}", flush=True)


def _key(path: str) -> str:
    """Decode a (possibly URL-encoded) WebDAV path to a stable comparison key."""
    return urllib.parse.unquote(path or "").rstrip("/")


def _arenas() -> list[dict]:
    try:
        return json.loads(LINKS_PATH.read_text())
    except Exception:
        return []


def _members_by_arena() -> dict[str, list[str]]:
    """idx(str) -> [emails] of everyone who picked that venue in the portal."""
    by = defaultdict(list)
    try:
        state = json.loads(PORTAL_STATE.read_text())
    except Exception:
        return by
    for email, m in (state.get("members") or {}).items():
        idx = m.get("arena")
        if idx is not None:
            by[str(idx)].append(email)
    return by


def _alert_recipients() -> list[str]:
    try:
        state = json.loads(PORTAL_STATE.read_text())
    except Exception:
        return ["mjohnson@groundrushinc.com"]
    r = state.get("alert_recipients")
    return ["mjohnson@groundrushinc.com"] if r is None else list(r)


def _load_state() -> dict:
    if NOTIFY_STATE.exists():
        try:
            return json.loads(NOTIFY_STATE.read_text())
        except Exception:
            pass
    return {"baseline_done": False, "arenas": {}}


def _save_state(state: dict) -> None:
    NOTIFY_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = NOTIFY_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(NOTIFY_STATE)


# ----------------------------------------------------------------- nextcloud walk

def _walk_files(decoded_path: str) -> list[dict]:
    """Recurse a folder and return every FILE under it (skips subfolders)."""
    out: list[dict] = []
    try:
        items = nc.list_files(decoded_path, depth=1)
    except Exception as e:
        _log(f"  ! list failed for {decoded_path}: {e}")
        return out
    base = _key(decoded_path)
    for it in items:
        k = _key(it.get("path") or "")
        # leading-slash-insensitive self-skip: WebDAV returns the folder itself
        # (URL-encoded) on special-char paths; without this it recurses into
        # itself and every file gets listed twice. (fix 2026-06-14)
        if not k or k.lstrip("/") == base.lstrip("/"):
            continue
        if it.get("is_folder"):
            out.extend(_walk_files(k))
        else:
            segs = k.split("/")
            out.append({
                "path": k,
                "name": urllib.parse.unquote(it.get("name") or segs[-1]),
                "category": segs[-2] if len(segs) >= 2 else "",
                "size": it.get("size") or 0,
                "modified": it.get("modified") or "",
            })
    return out


def _fmt_size(b: int) -> str:
    kb = (b or 0) / 1024
    return f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"


# ----------------------------------------------------------------- email build

def _build_venue_email(arena: dict, files: list[dict]) -> tuple[str, str, str]:
    city = arena.get("city", "your city")
    venue = arena.get("venue", "your venue")
    link = arena.get("link") or PORTAL_URL
    n = len(files)
    noun = "file" if n == 1 else "files"

    subject = f"Your Rod Wave tour assets just landed, {city}"

    # group by category subfolder for a clean list
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        by_cat[f.get("category") or "Assets"].append(f)

    # ---- plain-text alternative (kept first in the message; helps inbox placement)
    lines = [
        f"Just landed in your folder — {venue}.",
        "",
        f"The Mainstay marketing team just delivered {n} new {noun} to your",
        "tour download folder:",
        "",
    ]
    for cat, fs in by_cat.items():
        lines.append(f"{cat}:")
        for f in fs:
            lines.append(f"  - {f['name']} ({_fmt_size(f['size'])})")
        lines.append("")
    lines += [
        f"Open your folder: {link}",
        "",
        "Grab it, post it, light up the building. Anything you need tweaked,",
        "just reply — you'll reach us directly.",
        "",
        "The rollout's picking up speed. Let's pack the house.",
        "",
        "— The Rod Wave Tour team",
        "Mainstay Marketing",
    ]
    text = "\n".join(lines)

    # ---- html
    cat_blocks = ""
    for cat, fs in by_cat.items():
        rows = "".join(
            f'<tr><td style="padding:6px 0;font-size:14px;color:#222;line-height:1.4">'
            f'<span style="color:#f97316;font-weight:800">&#9656;</span>&nbsp;&nbsp;{f["name"]}'
            f'<span style="color:#999;font-weight:600">&nbsp;&middot; {_fmt_size(f["size"])}</span>'
            f'</td></tr>'
            for f in fs
        )
        cat_blocks += (
            f'<p style="margin:14px 0 4px;font-size:11px;letter-spacing:.12em;'
            f'text-transform:uppercase;color:#888;font-weight:700">{cat}</p>'
            f'<table width="100%" cellpadding="0" cellspacing="0"><tbody>{rows}</tbody></table>'
        )

    html = f"""\
<table width="100%" cellpadding="0" cellspacing="0" bgcolor="#f4f4f5" style="font-family:Inter,Arial,Helvetica,sans-serif">
 <tbody><tr><td align="center" style="padding:24px 12px">
  <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;border-radius:16px;overflow:hidden">

   <tbody><tr><td bgcolor="#0a0a0a" style="padding:34px 32px 28px">
     <p style="margin:0 0 12px;font-size:12px;letter-spacing:.20em;text-transform:uppercase;color:#f97316;font-weight:800">Rod Wave &middot; Don&#39;t Look Down</p>
     <h1 style="margin:0;font-size:26px;line-height:1.25;color:#ffffff;font-weight:800">Just landed in your folder. &#127915;</h1>
     <p style="margin:12px 0 0;font-size:15px;color:#cfcfcf;line-height:1.5">Fresh creative for <strong style="color:#fff">{venue}</strong>.</p>
   </td></tr>

   <tr><td style="padding:28px 32px 6px">
     <p style="margin:0 0 6px;font-size:15px;color:#222;line-height:1.6">The Mainstay marketing team just delivered <strong>{n} new {noun}</strong> to your tour download folder:</p>
     <table width="100%" cellpadding="0" cellspacing="0" bgcolor="#fff7ed" style="border:1px solid #fed7aa;border-radius:12px;margin:14px 0 22px"><tbody><tr><td style="padding:8px 20px 16px">
       {cat_blocks}
     </td></tr></tbody></table>
     <p style="margin:0 0 22px;font-size:15px;color:#222;line-height:1.6">Grab it, post it, light up the building. We built it in-house so it&#39;s on-brand and ready to run.</p>
   </td></tr>

   <tr><td align="center" style="padding:2px 32px 28px">
     <table cellpadding="0" cellspacing="0" align="center"><tbody><tr><td bgcolor="#f97316" style="border-radius:11px">
       <a href="{link}" style="display:inline-block;padding:15px 34px;color:#180a00;text-decoration:none;font-weight:800;font-size:16px" target="_blank">Open your folder &rarr;</a>
     </td></tr></tbody></table>
   </td></tr>

   <tr><td style="padding:0 32px 30px">
     <hr style="border:none;border-top:1px solid #eee;margin:0 0 20px">
     <p style="margin:0;font-size:15px;color:#222;line-height:1.55">Anything you need tweaked, just reply &mdash; you&#39;ll reach us directly. The rollout&#39;s picking up speed; let&#39;s pack the house.</p>
     <p style="margin:14px 0 0;font-size:14px;font-weight:800;color:#0a0a0a">&mdash; The Rod Wave Tour team</p>
     <p style="margin:2px 0 0;font-size:13px;color:#666">Mainstay Marketing</p>
   </td></tr>

   <tr><td bgcolor="#0a0a0a" align="center" style="padding:18px 32px">
     <p style="margin:0;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#999;font-weight:700">Mainstay Marketing</p>
   </td></tr>

  </tbody></table>
 </td></tr>
</tbody></table>"""
    return subject, html, text


# ----------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    baseline = "--baseline" in argv
    dry = "--dry-run" in argv

    state = _load_state()
    # First ever run with no recorded baseline = treat as baseline (seed, don't blast)
    if not state.get("baseline_done") and not dry:
        if not baseline:
            _log("No baseline recorded yet — seeding as baseline (no emails this run).")
        baseline = True

    members = _members_by_arena()
    arenas = _arenas()
    st_arenas = state.setdefault("arenas", {})

    total_new = total_mailed = 0

    for a in arenas:
        idx = str(a.get("idx"))
        entry = st_arenas.get(idx, {})
        recipients = members.get(idx, [])

        # DOWNLOAD = finished assets we send OUT to the building.
        dl_current = _walk_files(f"{a['folder']}/{DOWNLOAD_SUBFOLDER}")
        dl_seen = set(entry.get("seen", []))
        dl_new = [f for f in dl_current if f["path"] not in dl_seen]

        # UPLOAD = logos + ad specs coming IN from the building (or staged by us).
        up_current = _walk_files(f"{a['folder']}/{UPLOAD_SUBFOLDER}")
        up_seen = set(entry.get("seen_up", []))
        up_new = [f for f in up_current if f["path"] not in up_seen]

        if baseline:
            if dl_current or up_current:
                st_arenas[idx] = {
                    "seen": sorted(f["path"] for f in dl_current),
                    "seen_up": sorted(f["path"] for f in up_current),
                    "welcomed": sorted(recipients),
                }
                _log(f"baseline {a['city']}: seeded {len(dl_current)} download "
                     f"+ {len(up_current)} upload file(s).")
            continue

        # "welcomed" = contacts already told about this arena's folder (via a drop
        # or a backfill). First run after this feature shipped = migrating: assume
        # the members already on file are caught up, so we don't retroactively blast.
        migrating = "welcomed" not in entry
        welcomed = set(recipients) if migrating else set(entry.get("welcomed", []))
        delta_recipients = [m for m in recipients if m in welcomed]          # get the new drop only
        backfill_members = [m for m in recipients if m not in welcomed] if dl_current else []  # signed in late

        if dl_new:
            total_new += len(dl_new)
            _log(f"{a['city']} [download]: {len(dl_new)} new file(s); {len(delta_recipients)} caught-up contact(s).")
            if dry:
                for f in dl_new:
                    _log(f"   would notify -> {delta_recipients or '(none)'} : {f['category']}/{f['name']}")
            else:
                cc = _alert_recipients()
                if delta_recipients:
                    subject, html, text = _build_venue_email(a, dl_new)
                    for to in delta_recipients:
                        try:
                            r = email_client.send_email(account=VENUE_SENDER, to=to, subject=subject,
                                                        body=html, html=True, text_body=text,
                                                        cc=[c for c in cc if c != to])
                            if isinstance(r, dict) and r.get("error"):
                                _log(f"   ! send error to {to}: {r['error']}")
                            else:
                                total_mailed += 1
                                _log(f"   sent (drop) -> {to}")
                        except Exception as e:
                            _log(f"   ! send exception to {to}: {e}")
                else:
                    _log(f"   (no caught-up contacts for {a['city']} yet — internal heads-up only)")
                _internal_summary(a, dl_new, delta_recipients)

        # Welcome backfill: a contact who signs in AFTER delivery gets auto-told what's
        # ALREADY in their folder, so nothing they can grab goes unannounced.
        if backfill_members and not migrating:
            _log(f"{a['city']} [backfill]: {len(backfill_members)} new sign-in(s); folder has {len(dl_current)} file(s).")
            if dry:
                for m in backfill_members:
                    _log(f"   would backfill -> {m} : {len(dl_current)} existing file(s)")
            else:
                cc = _alert_recipients()
                subject, html, text = _build_venue_email(a, dl_current)
                subject = f"Your Rod Wave tour assets are ready, {a.get('city','your city')}"
                for to in backfill_members:
                    try:
                        r = email_client.send_email(account=VENUE_SENDER, to=to, subject=subject,
                                                    body=html, html=True, text_body=text,
                                                    cc=[c for c in cc if c != to])
                        if isinstance(r, dict) and r.get("error"):
                            _log(f"   ! backfill error to {to}: {r['error']}")
                        else:
                            total_mailed += 1
                            _log(f"   backfilled -> {to}")
                    except Exception as e:
                        _log(f"   ! backfill exception to {to}: {e}")

        if up_new:
            total_new += len(up_new)
            _log(f"{a['city']} [upload]: {len(up_new)} new file(s).")
            if dry:
                for f in up_new:
                    _log(f"   would alert (upload) -> Mike + Dre : {f['category']}/{f['name']}")
            else:
                _upload_summary(a, up_new)

        # Persist seen + welcomed every run (not just on changes) so late sign-ins
        # are tracked the moment they appear.
        if not dry:
            st_arenas[idx] = {
                "seen": sorted(dl_seen | {f["path"] for f in dl_current}),
                "seen_up": sorted(up_seen | {f["path"] for f in up_current}),
                "welcomed": sorted(welcomed | set(recipients)),
            }

    if baseline:
        state["baseline_done"] = True

    if not dry:
        _save_state(state)

    _log(f"done. new files: {total_new}, emails sent: {total_mailed}, mode: "
         f"{'dry-run' if dry else 'baseline' if baseline else 'live'}")
    return 0


def _internal_summary(arena: dict, files: list[dict], recipients: list[str]) -> None:
    """Plain heads-up to Mike + Dre the moment a drop lands — fires whether or
    not any venue contact has signed up through the portal yet."""
    to_list = _alert_recipients()
    if not to_list:
        return
    names = "\n".join(f"  - {f['category']}/{f['name']} ({_fmt_size(f['size'])})" for f in files)
    if recipients:
        lead = (f"Notified {len(recipients)} contact(s) at {arena['venue']} ({arena['city']}) "
                f"about {len(files)} new file(s) in their download folder:")
        tail = f"\n\nContacts: {', '.join(recipients)}\n"
        subj = f"[Arena drops] {arena['city']} notified — {len(files)} file(s)"
    else:
        lead = (f"{len(files)} new file(s) just landed in the {arena['venue']} ({arena['city']}) "
                f"download folder. No venue contacts have signed up through the portal yet, "
                f"so nothing went out to the building — this is a heads-up to you and Dre only:")
        tail = "\n"
        subj = f"[Arena drops] {arena['city']} — {len(files)} file(s) landed"
    body = f"{lead}\n\n{names}{tail}"
    for to in to_list:
        try:
            email_client.send_email(account=INTERNAL_SENDER, to=to,
                                    subject=subj, body=body, html=False)
        except Exception:
            pass


def _upload_summary(arena: dict, files: list[dict]) -> None:
    """Internal-only alert to Mike + Dre when logos/specs land in an UPLOAD folder.
    Uploads come IN from the building (or are staged by us), so nothing goes out
    to the venue — this just tells the team what arrived and where."""
    to_list = _alert_recipients()
    if not to_list:
        return
    folder = f"{arena['folder']}/{UPLOAD_SUBFOLDER}"
    names = "\n".join(f"  - {f['category']}/{f['name']} ({_fmt_size(f['size'])})" for f in files)
    n = len(files)
    noun = "file" if n == 1 else "files"
    body = (
        f"{n} {noun} just landed in the UPLOAD folder for {arena['venue']} "
        f"({arena['city']}):\n\n{names}\n\n"
        f"Folder: {folder}\n"
        f"Portal: {arena.get('link') or PORTAL_URL}\n"
    )
    for to in to_list:
        try:
            email_client.send_email(account=INTERNAL_SENDER, to=to,
                                    subject=f"[Arena uploads] {arena['city']} — {n} {noun} received",
                                    body=body, html=False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
