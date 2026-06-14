"""Arena Portal — the Rod Wave tour "command center" front door for venues.

ONE generic link goes to all 25 arenas (BCC). A venue contact:
  1. enters their work email      -> we email a 6-digit code
  2. enters the code              -> verified, session issued (remembered ~60 days)
  3. picks their venue (dropdown) -> email + domain locked to that arena
  4. chooses who makes the art    -> Ground Rush builds it / their in-house team
  5. drag-drops logo + ad specs   -> files land in THAT arena's Nextcloud UPLOAD folder

The domain is the durable key: once anyone @hawks.com is bound to Atlanta, the
dropdown pre-selects Atlanta for every Hawks colleague after them.

Public endpoints live under /arena/api/* (no edge auth — venues have no login).
Admin endpoints live under /arena/api/admin/* and are gated at the edge by Caddy
basic_auth (same `mainstay` login as The Rollout / Forge). The app binds
127.0.0.1 so only Caddy reaches it.
"""
from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form, Header, HTTPException

from integrations.email.client import email_client
from integrations.nextcloud import client as nc

_ROOT = Path(__file__).parent.parent.parent
STATE_PATH = _ROOT / "data" / "mainstay" / "arena_portal" / "portal_state.json"
LINKS_PATH = _ROOT / "data" / "mainstay" / "tour" / "arena_folder_links.json"

UPLOAD_SUBFOLDER = "2. UPLOAD - Your Logo + Ad Specs Here"
SENDER_ACCOUNT = "groundrush info"          # info@groundrushlabs.com — internal alerts to Mike+Dre
# Venue-facing one-time codes go from Google Workspace (groundrushinc.com): aligned
# SPF/DKIM/DMARC + established reputation clear strict corporate filters (Live Nation
# Trend Micro, AEG, etc.) that quarantine our self-hosted Mailcow sender. (2026-06-14)
CODE_SENDER_ACCOUNT = "alfred-gw"           # alfred@groundrushinc.com
CODE_TTL = 15 * 60                          # 6-digit code valid 15 min
SESSION_TTL = 60 * 24 * 3600               # remember a verified contact ~60 days
RESEND_COOLDOWN = 30                        # min seconds between code sends per email
MAX_SENDS_PER_HOUR = 8
MAX_UPLOAD_BYTES = 50 * 1024 * 1024        # 50 MB / file
ALLOWED_EXT = {
    "png", "jpg", "jpeg", "gif", "webp", "tif", "tiff", "bmp",
    "svg", "ai", "eps", "pdf", "psd", "indd",
    "zip", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv",
}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

ARENA_DEADLINE = "noon ET, Tuesday June 17"
ANNOUNCE = "Thursday, June 18 (1 PM ET)"

ADMIN_URL = "https://aialfred.groundrushcloud.com/arena/admin"
# Who gets pinged when a venue acts. Editable in the command center; falls back to
# Mike if the list was never set. An empty list (set on purpose) = alerts off.
DEFAULT_ALERTS = ["mjohnson@groundrushinc.com"]


# ---------------------------------------------------------------- arenas / state

def _arenas() -> list[dict]:
    try:
        return json.loads(LINKS_PATH.read_text())
    except Exception:
        return []


def _load() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"codes": {}, "sessions": {}, "members": {}, "domain_map": {},
            "submissions": {}, "alert_recipients": list(DEFAULT_ALERTS)}


def _save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_PATH)


def _now() -> int:
    return int(time.time())


def _domain(email: str) -> str:
    return email.split("@", 1)[1].lower().strip() if "@" in email else ""


def _arena_for_idx(idx) -> dict | None:
    for a in _arenas():
        if str(a.get("idx")) == str(idx):
            return a
    return None


def _gc(state: dict) -> None:
    """Drop expired codes + sessions so the store doesn't grow unbounded."""
    now = _now()
    state["codes"] = {e: c for e, c in state.get("codes", {}).items()
                      if c.get("exp", 0) > now}
    state["sessions"] = {t: s for t, s in state.get("sessions", {}).items()
                         if s.get("exp", 0) > now}


# ---------------------------------------------------------------- auth helper

def _session_email(state: dict, authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="not signed in")
    token = authorization.split(" ", 1)[1].strip()
    sess = state.get("sessions", {}).get(token)
    if not sess or sess.get("exp", 0) < _now():
        raise HTTPException(status_code=401, detail="session expired")
    return sess["email"]


def _member_view(state: dict, email: str) -> dict:
    """What the signed-in venue contact should see about themselves."""
    member = state.get("members", {}).get(email) or {}
    arena = _arena_for_idx(member["arena"]) if member.get("arena") else None
    sub = state.get("submissions", {}).get(str(member.get("arena"))) if member.get("arena") else None
    out = {
        "email": email,
        "domain": _domain(email),
        "locked_arena": None,
        "suggested_arena": None,
        "choice": (sub or {}).get("choice"),
        "uploads": [],
        "deadline": ARENA_DEADLINE,
        "announce": ANNOUNCE,
    }
    if arena:
        out["locked_arena"] = {"idx": arena["idx"], "city": arena["city"],
                               "venue": arena["venue"], "dates": arena["dates"]}
        out["uploads"] = _list_uploads(arena)
    else:
        # not yet locked — if their domain is known, pre-select it in the dropdown
        sug = state.get("domain_map", {}).get(_domain(email))
        if sug:
            sa = _arena_for_idx(sug)
            if sa:
                out["suggested_arena"] = sa["idx"]
    return out


def _list_uploads(arena: dict) -> list[dict]:
    path = f"{arena['folder']}/{UPLOAD_SUBFOLDER}"
    try:
        items = nc.list_files(path, depth=1)
    except Exception:
        return []
    files = []
    for it in items:
        if it.get("is_folder"):
            continue
        files.append({"name": it.get("name"), "size": it.get("size"),
                      "modified": it.get("modified")})
    return files


def _public_arenas() -> list[dict]:
    return [{"idx": a["idx"], "label": f'{a["city"]} — {a["venue"]}',
             "city": a["city"], "venue": a["venue"], "dates": a["dates"]}
            for a in _arenas()]


# ---------------------------------------------------------------- routes

def register(app: FastAPI) -> None:

    # ---- step 1: request a verification code --------------------------------
    @app.post("/arena/api/request-code")
    async def request_code(request: Request):
        body = await request.json()
        email = (body.get("email") or "").strip().lower()
        if not EMAIL_RE.match(email):
            raise HTTPException(status_code=400, detail="Please enter a valid work email.")
        state = _load()
        _gc(state)
        rec = state["codes"].get(email, {})
        now = _now()
        if rec.get("last") and now - rec["last"] < RESEND_COOLDOWN:
            raise HTTPException(status_code=429,
                                detail="A code was just sent — give it a moment, then check your inbox.")
        # rolling hourly cap
        window = [t for t in rec.get("sends", []) if now - t < 3600]
        if len(window) >= MAX_SENDS_PER_HOUR:
            raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
        code = f"{secrets.randbelow(1_000_000):06d}"
        window.append(now)
        state["codes"][email] = {"code": code, "exp": now + CODE_TTL,
                                 "last": now, "sends": window}
        _save(state)
        _send_code(email, code)
        return {"ok": True}

    # ---- step 2: verify the code -> issue session ---------------------------
    @app.post("/arena/api/verify")
    async def verify(request: Request):
        body = await request.json()
        email = (body.get("email") or "").strip().lower()
        code = (body.get("code") or "").strip()
        state = _load()
        _gc(state)
        rec = state["codes"].get(email)
        if not rec or rec.get("exp", 0) < _now():
            raise HTTPException(status_code=400, detail="That code has expired. Request a new one.")
        if code != rec.get("code"):
            raise HTTPException(status_code=400, detail="That code isn't right. Check and try again.")
        # burn the code, mint a session
        state["codes"].pop(email, None)
        token = secrets.token_urlsafe(32)
        state["sessions"][token] = {"email": email, "exp": _now() + SESSION_TTL}
        _save(state)
        return {"ok": True, "token": token, "arenas": _public_arenas(),
                "me": _member_view(state, email)}

    # ---- session re-hydrate (returning, remembered contact) -----------------
    @app.get("/arena/api/session")
    async def session(authorization: str | None = Header(default=None)):
        state = _load()
        email = _session_email(state, authorization)
        return {"ok": True, "arenas": _public_arenas(), "me": _member_view(state, email)}

    # ---- step 3: pick venue -> lock email + domain --------------------------
    @app.post("/arena/api/select")
    async def select(request: Request, authorization: str | None = Header(default=None)):
        state = _load()
        email = _session_email(state, authorization)
        body = await request.json()
        idx = body.get("arena")
        arena = _arena_for_idx(idx)
        if not arena:
            raise HTTPException(status_code=400, detail="Unknown venue.")
        dom = _domain(email)
        prev = state.get("members", {}).get(email)
        is_new = not prev or str(prev.get("arena")) != str(arena["idx"])
        state.setdefault("members", {})[email] = {
            "arena": arena["idx"], "domain": dom, "verified_at": _now(),
        }
        # bind the domain so future colleagues pre-select this venue
        if dom:
            state.setdefault("domain_map", {})[dom] = arena["idx"]
        _save(state)
        if is_new:
            _notify(state, f"\U0001F3DF️ {arena['city']} signed in — Rod Wave portal",
                    [("Venue", f"{arena['venue']} ({arena['city']})"),
                     ("Dates", arena["dates"]),
                     ("Contact", email)])
        return {"ok": True, "me": _member_view(state, email)}

    # ---- step 4: record art decision ----------------------------------------
    @app.post("/arena/api/choice")
    async def choice(request: Request, authorization: str | None = Header(default=None)):
        state = _load()
        email = _session_email(state, authorization)
        member = state.get("members", {}).get(email)
        if not member or not member.get("arena"):
            raise HTTPException(status_code=400, detail="Select your venue first.")
        body = await request.json()
        val = body.get("choice")
        if val not in ("ground_rush", "in_house"):
            raise HTTPException(status_code=400, detail="Invalid choice.")
        key = str(member["arena"])
        prev_choice = (state.get("submissions", {}).get(key) or {}).get("choice")
        state.setdefault("submissions", {})[key] = {
            "choice": val, "by": email, "at": _now(),
        }
        _save(state)
        if val != prev_choice:
            arena = _arena_for_idx(member["arena"]) or {}
            label = "Ground Rush builds the art" if val == "ground_rush" else "Their in-house team builds the art"
            _notify(state, f"\U0001F3A8 {arena.get('city','Venue')} chose: {label}",
                    [("Venue", f"{arena.get('venue','')} ({arena.get('city','')})"),
                     ("Decision", label),
                     ("Contact", email)])
        return {"ok": True, "me": _member_view(state, email)}

    # ---- step 5: upload a file into the arena's Nextcloud UPLOAD folder ------
    @app.post("/arena/api/upload")
    async def upload(authorization: str | None = Header(default=None),
                     file: UploadFile = File(...)):
        state = _load()
        email = _session_email(state, authorization)
        member = state.get("members", {}).get(email)
        if not member or not member.get("arena"):
            raise HTTPException(status_code=400, detail="Select your venue first.")
        arena = _arena_for_idx(member["arena"])
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File is over 50 MB. Please compress or split it.")
        name = _safe_name(file.filename or "upload")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=415,
                                detail="That file type isn't accepted. Use an image, PDF, vector, or zip.")
        dest = f"{arena['folder']}/{UPLOAD_SUBFOLDER}/{name}"
        try:
            nc.upload_file(dest, raw, content_type=file.content_type or "application/octet-stream")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upload failed, please retry. ({e})")
        kb = len(raw) / 1024
        size = f"{kb:.0f} KB" if kb < 1024 else f"{kb/1024:.1f} MB"
        _notify(state, f"\U0001F4CE {arena['city']} uploaded: {name}",
                [("Venue", f"{arena['venue']} ({arena['city']})"),
                 ("File", f"{name} · {size}"),
                 ("Uploaded by", email),
                 ("Folder", f'<a href="{arena.get("link","")}">open in Nextcloud</a>' if arena.get("link") else "—")])
        return {"ok": True, "uploads": _list_uploads(arena)}

    # ================= ADMIN (edge-gated by Caddy basic_auth) ================
    @app.get("/arena/api/admin/state")
    def admin_state(live: int = 1):
        state = _load()
        members = state.get("members", {})
        subs = state.get("submissions", {})
        rows = []
        for a in _arenas():
            key = str(a["idx"])
            mem = [{"email": e, "domain": m.get("domain"), "verified_at": m.get("verified_at")}
                   for e, m in members.items() if str(m.get("arena")) == key]
            sub = subs.get(key) or {}
            uploads = _list_uploads(a) if live else []
            rows.append({
                "idx": a["idx"], "city": a["city"], "venue": a["venue"], "dates": a["dates"],
                "link": a.get("link"),
                "members": sorted(mem, key=lambda x: x.get("verified_at") or 0),
                "responded": bool(mem),
                "choice": sub.get("choice"), "choice_by": sub.get("by"), "choice_at": sub.get("at"),
                "uploads": uploads, "upload_count": len(uploads),
            })
        return {
            "ok": True, "arenas": rows,
            "domain_map": state.get("domain_map", {}),
            "alert_recipients": _recipients(state),
            "totals": {
                "arenas": len(rows),
                "responded": sum(1 for r in rows if r["responded"]),
                "with_uploads": sum(1 for r in rows if r["upload_count"] > 0),
                "ground_rush": sum(1 for r in rows if r["choice"] == "ground_rush"),
                "in_house": sum(1 for r in rows if r["choice"] == "in_house"),
            },
            "deadline": ARENA_DEADLINE, "announce": ANNOUNCE,
        }

    @app.post("/arena/api/admin/bind")
    async def admin_bind(request: Request):
        body = await request.json()
        dom = (body.get("domain") or "").strip().lower()
        idx = body.get("arena")
        state = _load()
        if not dom:
            raise HTTPException(status_code=400, detail="domain required")
        if idx in (None, "", "none"):
            state.get("domain_map", {}).pop(dom, None)
        else:
            if not _arena_for_idx(idx):
                raise HTTPException(status_code=400, detail="unknown arena")
            state.setdefault("domain_map", {})[dom] = idx
        _save(state)
        return {"ok": True, "domain_map": state.get("domain_map", {})}

    @app.post("/arena/api/admin/alerts")
    async def admin_alerts(request: Request):
        """Set the alert recipient list (validated emails, de-duped)."""
        body = await request.json()
        recips = body.get("recipients")
        if not isinstance(recips, list):
            raise HTTPException(status_code=400, detail="recipients must be a list")
        clean = []
        for e in recips:
            e = (e or "").strip().lower()
            if EMAIL_RE.match(e) and e not in clean:
                clean.append(e)
        state = _load()
        state["alert_recipients"] = clean
        _save(state)
        return {"ok": True, "alert_recipients": clean}


# ---------------------------------------------------------------- alerts

def _recipients(state: dict) -> list[str]:
    """Who to ping. Falls back to Mike if the key was never set; respects an
    explicitly-emptied list (alerts off)."""
    r = state.get("alert_recipients")
    return list(DEFAULT_ALERTS) if r is None else list(r)


def _notify(state: dict, subject: str, rows: list[tuple]) -> None:
    """Fire a branded alert to every recipient. Never raises — a mail failure
    must not break a venue's action."""
    recips = _recipients(state)
    if not recips:
        return
    body = "".join(
        f'<tr><td style="padding:4px 14px 4px 0;color:#888;font-size:13px;white-space:nowrap;'
        f'vertical-align:top">{k}</td>'
        f'<td style="padding:4px 0;font-size:14px;font-weight:600;color:#0a0a0a">{v}</td></tr>'
        for k, v in rows
    )
    html = f"""\
<div style="font-family:Inter,Arial,sans-serif;max-width:500px;margin:0 auto;color:#0a0a0a">
  <p style="font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#f97316;font-weight:700;margin:0 0 10px">
    Rod Wave · Don't Look Down — Arena Portal</p>
  <table style="border-collapse:collapse;margin:0 0 20px">{body}</table>
  <a href="{ADMIN_URL}" style="display:inline-block;background:#f97316;color:#180a00;
     text-decoration:none;font-weight:800;font-size:13px;padding:11px 18px;border-radius:9px">
    Open the command center →</a>
  <p style="margin:22px 0 0;font-size:11px;color:#aaa">You're getting this because you're on the arena-portal alert list.</p>
</div>"""
    for to in recips:
        try:
            email_client.send_email(account=SENDER_ACCOUNT, to=to,
                                    subject=subject, body=html, html=True)
        except Exception:
            pass


# ---------------------------------------------------------------- helpers

def _safe_name(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ \-()]", "_", name).strip().strip(".")
    return name[:160] or "upload"


def _send_code(email: str, code: str) -> None:
    subject = f"Your Rod Wave tour portal code: {code}"
    html = f"""\
<div style="font-family:Inter,Arial,sans-serif;max-width:480px;margin:0 auto;color:#0a0a0a">
  <p style="font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#f97316;font-weight:700;margin:0 0 6px">
    Rod Wave · Don't Look Down Tour</p>
  <p style="margin:0 0 18px;font-size:15px">Here is your one-time code to open your venue's asset portal:</p>
  <div style="font-size:34px;font-weight:800;letter-spacing:.32em;background:#0a0a0a;color:#fff;
              text-align:center;padding:18px 0;border-radius:10px">{code}</div>
  <p style="margin:18px 0 0;font-size:13px;color:#555">
    Enter it on the portal to sign in. The code expires in 15 minutes.<br>
    If you didn't request this, you can ignore this email.</p>
  <p style="margin:22px 0 0;font-size:12px;color:#999">Marketing by Ground Rush</p>
</div>"""
    email_client.send_email(account=CODE_SENDER_ACCOUNT, to=email, subject=subject,
                            body=html, html=True)
