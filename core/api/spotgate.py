"""SpotGate — gated listening for sellable audio spots (radio/commercial).

The pitch: you upload a spot, pick its length tier (:30 or :60), and mint a
unique link per client to send. The client gets a clean in-browser player with a
HARD 3-listen cap counted SERVER-SIDE on the link itself — not per browser, not
per device. Refresh, switch phones, open three browsers: three total listens,
period. No download button and the raw file URL is never exposed (audio streams
through a short-lived per-play grant). After the third listen the client hits a
paywall with the Stripe payment link FOR THAT TIER (:30=$315, :60=$400). Once
they pay (Stripe webhook keyed on client_reference_id == the link token) the page
unlocks and asks "what email should I send the spot to?" — SpotGate emails them
the MP3 to keep, and pings the operator (Mike) on both paid and delivered.

Every uploaded spot is also mirrored to Nextcloud (Radio Spots/30-Second |
60-Second) as the operator's organized archive — never sent to the client.

Served behind Caddy: /spot/* (static UI) and /spot/api/* (this app, :8405).

Endpoints
  GET  /spot/api/state           ?t=token  -> player state (plays, paid, buy_url)
  POST /spot/api/play            ?t=token  -> count a listen; returns stream grant or 402
  GET  /spot/api/stream          ?t&g      -> stream audio bytes (grant-gated, no download)
  POST /spot/api/deliver         ?t=token  -> email the MP3 (paid links only)
  POST /spot/api/stripe/webhook            -> unlock link on checkout.session.completed
  --- admin (X-Admin-Token) ---
  GET  /spot/api/admin/spots                -> list spot library (tier, price, NC link)
  POST /spot/api/admin/spots                -> upload a spot (multipart: file,title,length)
  GET  /spot/api/admin/links                -> list minted links + status
  POST /spot/api/admin/links                -> mint a link for a spot {spot,label,plays?}
  POST /spot/api/admin/links/{token}/mark-paid -> manual unlock fallback
  DELETE /spot/api/admin/links/{token}      -> delete a link

Secrets/config come from systemd Environment + config/.env (Stripe key, webhook
secret, admin token, per-tier payment links, email passwords, Nextcloud creds).
"""
from __future__ import annotations

import logging
import mimetypes
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")

import stripe
from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from integrations.email.client import EmailClient
from integrations.nextcloud import client as nextcloud

logger = logging.getLogger("spotgate")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ROOT = Path("/home/aialfred/alfred/data/mainstay/spotgate")
SPOTS_DIR = ROOT / "spots"
DB_PATH = Path(os.environ.get("SPOTGATE_DB_PATH", str(ROOT / "spotgate.db")))

# Empty default -> admin API is hard-401'd if the token is unset (never matches).
ADMIN_TOKEN = os.environ.get("SPOTGATE_ADMIN_TOKEN", "")
WEBHOOK_SECRET = os.environ.get("SPOTGATE_STRIPE_WEBHOOK_SECRET", "")
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "")
PUBLIC_BASE = os.environ.get(
    "SPOTGATE_PUBLIC_BASE", "https://aialfred.groundrushcloud.com"
)
FROM_ACCOUNT = os.environ.get("SPOTGATE_EMAIL_ACCOUNT", "alfred-gw")
# Where SpotGate pings the operator (Mike) on paid / delivered events.
NOTIFY_EMAIL = os.environ.get("SPOTGATE_NOTIFY_EMAIL", "mjohnson@groundrushinc.com")
DEFAULT_PLAYS = int(os.environ.get("SPOTGATE_DEFAULT_PLAYS", "3"))

# Length tiers — each maps to its own Stripe payment link, price, and Nextcloud
# archive folder. Spots carry a tier; the paywall + price follow from it.
TIERS: dict[str, dict] = {
    "30": {
        "label": ":30",
        "name": "30-Second",
        "price_label": "$315",
        "payment_link": os.environ.get(
            "SPOTGATE_PAYMENT_LINK_30",
            "https://buy.stripe.com/28E5kC7so9SWbO456OgQE0m",
        ),
        "nc_folder": "/Radio Spots/30-Second",
    },
    "60": {
        "label": ":60",
        "name": "60-Second",
        "price_label": "$400",
        "payment_link": os.environ.get(
            "SPOTGATE_PAYMENT_LINK_60",
            "https://buy.stripe.com/6oU8wOh2Y9SWcS82YGgQE0l",
        ),
        "nc_folder": "/Radio Spots/60-Second",
    },
}
DEFAULT_TIER = "60"

ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".mp4"}
MAX_UPLOAD_BYTES = 60 * 1024 * 1024  # 60 MB — plenty for a radio spot

GRANT_TTL = 180  # seconds a stream grant stays valid (covers pause/resume/range)

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

# --------------------------------------------------------------------------- #
# Persistence (sqlite, WAL). Every connection is closed via context manager to
# avoid the fd leak that previously wedged sibling services.
# --------------------------------------------------------------------------- #

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def _init_db() -> None:
    SPOTS_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS spots(
                slug           TEXT PRIMARY KEY,
                title          TEXT NOT NULL,
                filename       TEXT NOT NULL,
                mimetype       TEXT NOT NULL,
                length_tier    TEXT NOT NULL DEFAULT '60',
                nextcloud_path TEXT,
                nextcloud_share TEXT,
                created_at     REAL NOT NULL
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS links(
                token        TEXT PRIMARY KEY,
                spot_slug    TEXT NOT NULL,
                label        TEXT,
                plays_used   INTEGER NOT NULL DEFAULT 0,
                plays_allowed INTEGER NOT NULL DEFAULT 3,
                paid_at      REAL,
                delivered_to TEXT,
                delivered_at REAL,
                created_at   REAL NOT NULL
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS grants(
                grant      TEXT PRIMARY KEY,
                token      TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        # Seed the original Pompano spot (a :60) if the library is empty and the
        # file is present, so existing links keep working after the migration.
        n = c.execute("SELECT COUNT(*) AS n FROM spots").fetchone()["n"]
        if n == 0 and (SPOTS_DIR / "pompano-music-festival.mp3").exists():
            c.execute(
                "INSERT INTO spots(slug,title,filename,mimetype,length_tier,created_at)"
                " VALUES(?,?,?,?,?,?)",
                (
                    "pompano-music-festival",
                    "Pompano Music Festival — Radio Spot",
                    "pompano-music-festival.mp3",
                    "audio/mpeg",
                    "60",
                    time.time(),
                ),
            )
            logger.info("seeded Pompano spot as a :60")
        c.commit()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="SpotGate — gated audio spots")
_init_db()


def _require_admin(token: str | None) -> None:
    # constant-time compare; empty configured token can never match.
    if not ADMIN_TOKEN or not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="admin token required")


def _get_spot(slug: str) -> dict | None:
    with _connect() as c:
        row = c.execute("SELECT * FROM spots WHERE slug=?", (slug,)).fetchone()
    return dict(row) if row else None


def _spot_or_404(slug: str) -> dict:
    spot = _get_spot(slug)
    if not spot:
        raise HTTPException(status_code=404, detail="unknown spot")
    return spot


def _tier(spot: dict) -> dict:
    return TIERS.get((spot or {}).get("length_tier") or DEFAULT_TIER, TIERS[DEFAULT_TIER])


def _link_or_404(c: sqlite3.Connection, token: str) -> sqlite3.Row:
    row = c.execute("SELECT * FROM links WHERE token=?", (token,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="invalid or expired link")
    return row


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s or "spot")[:48]


def _notify_operator(subject: str, body_html: str) -> None:
    """Ping Mike on a money/delivery event. Never raises — a notification
    failure must not break the client's purchase or delivery."""
    if not NOTIFY_EMAIL:
        return
    try:
        EmailClient().send_email(
            account=FROM_ACCOUNT,
            to=NOTIFY_EMAIL,
            subject=subject,
            body=body_html,
            html=True,
        )
    except Exception:  # noqa: BLE001
        logger.exception("operator notification failed (non-fatal)")


def _buy_url(token: str, spot: dict | None) -> str:
    # Pass the link token through Stripe so the webhook knows WHO paid, and pick
    # the payment link for the spot's length tier.
    link = _tier(spot)["payment_link"]
    sep = "&" if "?" in link else "?"
    return f"{link}{sep}client_reference_id={token}"


def _state_payload(row: sqlite3.Row) -> dict:
    spot = _get_spot(row["spot_slug"]) or {}
    tier = _tier(spot)
    paid = row["paid_at"] is not None
    used = row["plays_used"]
    allowed = row["plays_allowed"]
    left = max(0, allowed - used)
    return {
        "ok": True,
        "title": spot.get("title", row["spot_slug"]),
        "tier_label": tier["label"],
        "price_label": tier["price_label"],
        "plays_used": used,
        "plays_allowed": allowed,
        "plays_left": left,
        "paid": paid,
        "delivered": row["delivered_at"] is not None,
        "delivered_to": row["delivered_to"],
        "can_listen": paid or left > 0,
        "buy_url": _buy_url(row["token"], spot),
    }


# --------------------------------------------------------------------------- #
# Public endpoints
# --------------------------------------------------------------------------- #
_NOCACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, private", "Pragma": "no-cache"}


@app.get("/spot/api/state")
def get_state(t: str = Query(...)):
    with _connect() as c:
        row = _link_or_404(c, t)
        # Never let a browser/proxy cache the listen count — a refresh must
        # always reflect the live server-side total, never a stale "3 of 3".
        return JSONResponse(content=_state_payload(row), headers=_NOCACHE)


@app.post("/spot/api/play")
def register_play(t: str = Query(...)):
    """Count one listen against the link's hard cap (server-side, cross-device).

    Paid links don't consume the counter. On success returns a short-lived grant
    the player exchanges for the audio stream. When the cap is hit and the link
    isn't paid, returns 402 so the UI shows the paywall.
    """
    now = time.time()
    with _connect() as c:
        row = _link_or_404(c, t)
        paid = row["paid_at"] is not None
        if not paid and row["plays_used"] >= row["plays_allowed"]:
            spot = _get_spot(row["spot_slug"])
            return JSONResponse(
                status_code=402,
                content={"allowed": False, "paid": False, "buy_url": _buy_url(t, spot)},
            )
        if not paid:
            c.execute(
                "UPDATE links SET plays_used = plays_used + 1 WHERE token=?", (t,)
            )
        grant = secrets.token_urlsafe(18)
        c.execute(
            "INSERT INTO grants(grant, token, created_at) VALUES(?,?,?)",
            (grant, t, now),
        )
        # opportunistic cleanup of stale grants
        c.execute("DELETE FROM grants WHERE created_at < ?", (now - GRANT_TTL,))
        c.commit()
        row = _link_or_404(c, t)
        payload = _state_payload(row)
        payload.update({"allowed": True, "grant": grant})
        return payload


@app.get("/spot/api/stream")
def stream(t: str = Query(...), g: str = Query(...)):
    """Stream the audio bytes for a valid, fresh grant. No download affordance:
    inline disposition, no-store, and the URL is useless without a live grant."""
    now = time.time()
    with _connect() as c:
        row = c.execute(
            "SELECT * FROM grants WHERE grant=? AND token=?", (g, t)
        ).fetchone()
        if not row or (now - row["created_at"]) > GRANT_TTL:
            raise HTTPException(status_code=403, detail="no valid play grant")
        link = _link_or_404(c, t)
    spot = _spot_or_404(link["spot_slug"])
    path = SPOTS_DIR / spot["filename"]
    if not path.exists():
        raise HTTPException(status_code=500, detail="spot file missing")
    data = path.read_bytes()

    def _gen():
        yield data

    return StreamingResponse(
        _gen(),
        media_type=spot["mimetype"],
        headers={
            "Content-Length": str(len(data)),
            "Content-Disposition": "inline",
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Accept-Ranges": "none",
        },
    )


@app.post("/spot/api/deliver")
def deliver(t: str = Query(...), payload: dict = Body(...)):
    """Email the MP3 to the address the (paid) client provides — theirs to keep."""
    email = (payload or {}).get("email", "").strip()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="valid email required")
    with _connect() as c:
        row = _link_or_404(c, t)
        if row["paid_at"] is None:
            raise HTTPException(status_code=402, detail="payment required first")
    spot = _spot_or_404(row["spot_slug"])
    path = SPOTS_DIR / spot["filename"]
    if not path.exists():
        raise HTTPException(status_code=500, detail="spot file missing")

    title = spot["title"]
    body = (
        f"<p>Thank you for your purchase. Your spot — <strong>{title}</strong> — "
        f"is attached as an MP3. It's yours to use however you'd like.</p>"
        f"<p>— Ground Rush Inc</p>"
    )
    text = f"Thank you for your purchase. Your spot — {title} — is attached as an MP3.\n\n— Ground Rush Inc"
    try:
        res = EmailClient().send_email(
            account=FROM_ACCOUNT,
            to=email,
            subject=f"Your radio spot — {title}",
            body=body,
            html=True,
            text_body=text,
            attachments=[
                {
                    "filename": spot["filename"],
                    "content": path.read_bytes(),
                    "mimetype": spot["mimetype"],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("deliver email failed")
        raise HTTPException(status_code=502, detail=f"email send failed: {e}")
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=502, detail=res["error"])

    now = time.time()
    with _connect() as c:
        c.execute(
            "UPDATE links SET delivered_to=?, delivered_at=? WHERE token=?",
            (email, now, t),
        )
        c.commit()
    logger.info("delivered spot %s -> %s", row["spot_slug"], email)
    _notify_operator(
        f"📨 SpotGate DELIVERED — {title}",
        f"<p><strong>The spot was emailed to the client.</strong></p>"
        f"<ul>"
        f"<li><strong>Spot:</strong> {title}</li>"
        f"<li><strong>Client:</strong> {row['label'] or '(no label)'}</li>"
        f"<li><strong>Sent to:</strong> {email}</li>"
        f"</ul>",
    )
    return {"ok": True, "delivered_to": email}


@app.post("/spot/api/stripe/webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    """Unlock the link whose token rode in as client_reference_id when Stripe
    confirms the checkout completed (or the payment link succeeded)."""
    body = await request.body()
    event = None
    if WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                body, stripe_signature, WEBHOOK_SECRET
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("stripe webhook bad signature: %s", e)
            raise HTTPException(status_code=400, detail="bad signature")
    else:
        # No secret configured yet — accept unverified (dev only).
        import json

        event = json.loads(body or b"{}")

    # construct_event returns a StripeObject whose .get() routes through
    # __getattr__ and raises (stripe 15.x) — use item access + `in` only.
    etype = event["type"] if "type" in event else None
    data = event["data"] if "data" in event else {}
    obj = data["object"] if "object" in data else {}
    if etype in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        token = obj["client_reference_id"] if "client_reference_id" in obj else None
        if token:
            now = time.time()
            newly_paid = False
            row = None
            with _connect() as c:
                row = c.execute(
                    "SELECT * FROM links WHERE token=?", (token,)
                ).fetchone()
                if row:
                    newly_paid = row["paid_at"] is None  # only the first time
                    c.execute(
                        "UPDATE links SET paid_at=COALESCE(paid_at,?) WHERE token=?",
                        (now, token),
                    )
                    c.commit()
                    logger.info("stripe: link %s marked PAID", token)
            # Ping Mike — but only on the first transition (Stripe retries webhooks).
            if row and newly_paid:
                spot = _get_spot(row["spot_slug"]) or {}
                title = spot.get("title", row["spot_slug"])
                payer = ""
                if "customer_details" in obj and obj["customer_details"]:
                    cd = obj["customer_details"]
                    payer = (cd["email"] if "email" in cd else "") or ""
                if not payer and "customer_email" in obj:
                    payer = obj["customer_email"] or ""
                amount = ""
                if "amount_total" in obj and obj["amount_total"]:
                    amount = f"${obj['amount_total']/100:,.2f}"
                amount = amount or _tier(spot)["price_label"]
                label = row["label"] or "(no label)"
                _notify_operator(
                    f"💰 SpotGate PAID — {title}",
                    f"<p><strong>Payment received.</strong></p>"
                    f"<ul>"
                    f"<li><strong>Spot:</strong> {title} ({_tier(spot)['label']})</li>"
                    f"<li><strong>Client:</strong> {label}</li>"
                    f"<li><strong>Amount:</strong> {amount or '—'}</li>"
                    f"<li><strong>Paid by:</strong> {payer or '—'}</li>"
                    f"</ul>"
                    f"<p>They'll now be asked where to email the spot. "
                    f"You'll get a second note the moment it's delivered.</p>",
                )
    return {"received": True}


# --------------------------------------------------------------------------- #
# Admin endpoints
# --------------------------------------------------------------------------- #
@app.get("/spot/api/admin/spots")
def admin_spots(x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    with _connect() as c:
        rows = c.execute("SELECT * FROM spots ORDER BY created_at DESC").fetchall()
    out = []
    for r in rows:
        tier = TIERS.get(r["length_tier"], TIERS[DEFAULT_TIER])
        out.append(
            {
                "slug": r["slug"],
                "title": r["title"],
                "length_tier": r["length_tier"],
                "tier_label": tier["label"],
                "price_label": tier["price_label"],
                "nextcloud_share": r["nextcloud_share"],
                "created_at": r["created_at"],
            }
        )
    # Tier reference for the UI dropdown/help.
    tiers = [
        {"value": k, "label": v["label"], "name": v["name"], "price_label": v["price_label"]}
        for k, v in sorted(TIERS.items())
    ]
    return {"spots": out, "tiers": tiers}


@app.post("/spot/api/admin/spots")
async def admin_upload_spot(
    file: UploadFile = File(...),
    title: str = Form(...),
    length: str = Form(...),
    x_admin_token: str = Header(None),
):
    """Upload a spot: store it privately, register it, and mirror it to the
    operator's Nextcloud archive (Radio Spots/<tier>). Mintable immediately."""
    _require_admin(x_admin_token)
    length = (length or "").strip()
    if length not in TIERS:
        raise HTTPException(status_code=400, detail=f"length must be one of {list(TIERS)}")
    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title required")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported audio type '{ext or '?'}' (allowed: {sorted(ALLOWED_AUDIO_EXT)})",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 60 MB)")
    mimetype = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "audio/mpeg"

    # Unique slug + on-disk filename.
    slug = f"{_slugify(title)}-{secrets.token_hex(3)}"
    filename = f"{slug}{ext}"
    (SPOTS_DIR / filename).write_bytes(content)

    # Mirror to Nextcloud archive (non-fatal — the spot still works locally).
    nc_path, nc_share = None, None
    try:
        folder = TIERS[length]["nc_folder"]
        nextcloud.create_folder_safe(folder)
        nc_path = f"{folder}/{filename}"
        nextcloud.upload_file(nc_path, content, content_type=mimetype)
        try:
            share = nextcloud.create_public_share(nc_path, permissions=1)  # read-only
            nc_share = share.get("url")
        except Exception:  # noqa: BLE001
            logger.warning("nextcloud share link failed for %s (upload ok)", nc_path)
    except Exception:  # noqa: BLE001
        logger.exception("nextcloud archive failed (non-fatal) for %s", slug)
        nc_path = None

    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT INTO spots(slug,title,filename,mimetype,length_tier,nextcloud_path,nextcloud_share,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (slug, title, filename, mimetype, length, nc_path, nc_share, now),
        )
        c.commit()
    logger.info("uploaded spot %s (%s, %d bytes)", slug, length, len(content))
    tier = TIERS[length]
    return {
        "ok": True,
        "slug": slug,
        "title": title,
        "length_tier": length,
        "tier_label": tier["label"],
        "price_label": tier["price_label"],
        "nextcloud_share": nc_share,
        "archived": nc_path is not None,
    }


@app.get("/spot/api/admin/links")
def admin_links(x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    with _connect() as c:
        rows = c.execute("SELECT * FROM links ORDER BY created_at DESC").fetchall()
        spots = {r["slug"]: dict(r) for r in c.execute("SELECT * FROM spots").fetchall()}
    out = []
    for r in rows:
        spot = spots.get(r["spot_slug"], {})
        tier = TIERS.get(spot.get("length_tier"), TIERS[DEFAULT_TIER])
        out.append(
            {
                "token": r["token"],
                "spot_slug": r["spot_slug"],
                "title": spot.get("title", r["spot_slug"]),
                "tier_label": tier["label"],
                "label": r["label"],
                "plays_used": r["plays_used"],
                "plays_allowed": r["plays_allowed"],
                "paid": r["paid_at"] is not None,
                "delivered_to": r["delivered_to"],
                "url": f"{PUBLIC_BASE}/spot/?t={r['token']}",
                "created_at": r["created_at"],
            }
        )
    return {"links": out}


@app.post("/spot/api/admin/links")
def admin_create_link(
    payload: dict = Body(...), x_admin_token: str = Header(None)
):
    _require_admin(x_admin_token)
    slug = (payload or {}).get("spot", "").strip()
    _spot_or_404(slug)
    label = (payload or {}).get("label", "").strip()
    plays = int((payload or {}).get("plays", DEFAULT_PLAYS) or DEFAULT_PLAYS)
    token = secrets.token_urlsafe(12)
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT INTO links(token, spot_slug, label, plays_used, plays_allowed, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (token, slug, label, 0, plays, now),
        )
        c.commit()
    return {
        "ok": True,
        "token": token,
        "url": f"{PUBLIC_BASE}/spot/?t={token}",
        "plays_allowed": plays,
    }


@app.post("/spot/api/admin/links/{token}/mark-paid")
def admin_mark_paid(token: str, x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    now = time.time()
    with _connect() as c:
        row = _link_or_404(c, token)
        c.execute(
            "UPDATE links SET paid_at=COALESCE(paid_at,?) WHERE token=?",
            (now, token),
        )
        c.commit()
    return {"ok": True, "token": token, "paid": True}


@app.delete("/spot/api/admin/links/{token}")
def admin_delete_link(token: str, x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    with _connect() as c:
        _link_or_404(c, token)
        c.execute("DELETE FROM grants WHERE token=?", (token,))
        c.execute("DELETE FROM links WHERE token=?", (token,))
        c.commit()
    logger.info("admin deleted link %s", token)
    return {"ok": True, "deleted": token}


@app.get("/spot/api/health")
def health():
    with _connect() as c:
        n = c.execute("SELECT COUNT(*) AS n FROM spots").fetchone()["n"]
    return {"ok": True, "service": "spotgate", "spots": n}
