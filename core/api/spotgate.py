"""SpotGate — gated listening for a sellable audio spot (radio/commercial).

The pitch: you mint a unique link per client and send it. The client gets a
clean in-browser player with a HARD 3-listen cap that is counted SERVER-SIDE on
the link itself — not per browser, not per device. Refresh, switch phones, open
three browsers: it is three total listens for that link, period. There is no
download button and the raw file URL is never exposed (audio streams through a
short-lived per-play grant). After the third listen the client hits a paywall
with the operator's existing Stripe payment link. Once they pay (detected via a
Stripe webhook keyed on client_reference_id == the link token) the page unlocks
and asks "what email should I send the spot to?" — and SpotGate emails them the
MP3 to keep.

Served behind Caddy: /spot/* (static UI) and /spot/api/* (this app, :8405).

Endpoints
  GET  /spot/api/state           ?t=token  -> player state (plays, paid, buy_url)
  POST /spot/api/play            ?t=token  -> count a listen; returns stream grant or 402
  GET  /spot/api/stream          ?t&g      -> stream audio bytes (grant-gated, no download)
  POST /spot/api/deliver         ?t=token  -> email the MP3 (paid links only)
  POST /spot/api/stripe/webhook            -> unlock link on checkout.session.completed
  --- admin (X-Admin-Token) ---
  GET  /spot/api/admin/spots                -> list available spots
  GET  /spot/api/admin/links                -> list minted links + status
  POST /spot/api/admin/links                -> mint a link for a spot {spot,label,plays?}
  POST /spot/api/admin/links/{token}/mark-paid -> manual unlock fallback

Secrets/config come from the systemd Environment + config/.env (Stripe key,
webhook secret, admin token, the $400 payment link URL, email passwords).
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")

import stripe
from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from integrations.email.client import EmailClient

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
# The operator's existing $400 Stripe Payment Link (buy.stripe.com/...).
PAYMENT_LINK = os.environ.get(
    "SPOTGATE_PAYMENT_LINK_URL", "https://buy.stripe.com/6oU8wOh2Y9SWcS82YGgQE0l"
)
PUBLIC_BASE = os.environ.get(
    "SPOTGATE_PUBLIC_BASE", "https://aialfred.groundrushcloud.com"
)
FROM_ACCOUNT = os.environ.get("SPOTGATE_EMAIL_ACCOUNT", "alfred-gw")
DEFAULT_PLAYS = int(os.environ.get("SPOTGATE_DEFAULT_PLAYS", "3"))

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

# Spot registry. Files live private under SPOTS_DIR; the public never sees these
# paths. Add more spots here (or via a future admin upload) as needed.
SPOTS: dict[str, dict] = {
    "pompano-music-festival": {
        "title": "Pompano Music Festival — Radio Spot",
        "filename": "pompano-music-festival.mp3",
        "mimetype": "audio/mpeg",
        "price_label": "$400",
    },
}

GRANT_TTL = 180  # seconds a stream grant stays valid (covers pause/resume/range)

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
    with _connect() as c:
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
        c.commit()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="SpotGate — gated audio spot")
_init_db()


def _require_admin(token: str | None) -> None:
    # constant-time compare; empty configured token can never match.
    if not ADMIN_TOKEN or not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="admin token required")


def _spot_or_404(slug: str) -> dict:
    spot = SPOTS.get(slug)
    if not spot:
        raise HTTPException(status_code=404, detail="unknown spot")
    return spot


def _link_or_404(c: sqlite3.Connection, token: str) -> sqlite3.Row:
    row = c.execute("SELECT * FROM links WHERE token=?", (token,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="invalid or expired link")
    return row


def _buy_url(token: str) -> str:
    # Pass the link token through Stripe so the webhook knows WHO paid.
    sep = "&" if "?" in PAYMENT_LINK else "?"
    return f"{PAYMENT_LINK}{sep}client_reference_id={token}"


def _state_payload(row: sqlite3.Row) -> dict:
    spot = SPOTS.get(row["spot_slug"], {})
    paid = row["paid_at"] is not None
    used = row["plays_used"]
    allowed = row["plays_allowed"]
    left = max(0, allowed - used)
    return {
        "ok": True,
        "title": spot.get("title", row["spot_slug"]),
        "price_label": spot.get("price_label", ""),
        "plays_used": used,
        "plays_allowed": allowed,
        "plays_left": left,
        "paid": paid,
        "delivered": row["delivered_at"] is not None,
        "delivered_to": row["delivered_to"],
        "can_listen": paid or left > 0,
        "buy_url": _buy_url(row["token"]),
    }


# --------------------------------------------------------------------------- #
# Public endpoints
# --------------------------------------------------------------------------- #
@app.get("/spot/api/state")
def get_state(t: str = Query(...)):
    with _connect() as c:
        row = _link_or_404(c, t)
        return _state_payload(row)


@app.post("/spot/api/play")
def register_play(t: str = Query(...)):
    """Count one listen against the link's hard cap (server-side, cross-device).

    Paid links don't consume the counter. On success returns a short-lived,
    grant the player exchanges for the audio stream. When the cap is hit and the
    link isn't paid, returns 402 so the UI shows the paywall.
    """
    now = time.time()
    with _connect() as c:
        row = _link_or_404(c, t)
        paid = row["paid_at"] is not None
        if not paid and row["plays_used"] >= row["plays_allowed"]:
            return JSONResponse(
                status_code=402,
                content={"allowed": False, "paid": False, "buy_url": _buy_url(t)},
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
            with _connect() as c:
                row = c.execute(
                    "SELECT token FROM links WHERE token=?", (token,)
                ).fetchone()
                if row:
                    c.execute(
                        "UPDATE links SET paid_at=COALESCE(paid_at,?) WHERE token=?",
                        (now, token),
                    )
                    c.commit()
                    logger.info("stripe: link %s marked PAID", token)
    return {"received": True}


# --------------------------------------------------------------------------- #
# Admin endpoints
# --------------------------------------------------------------------------- #
@app.get("/spot/api/admin/spots")
def admin_spots(x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    return {
        "spots": [
            {"slug": s, "title": v["title"], "price_label": v.get("price_label", "")}
            for s, v in SPOTS.items()
        ]
    }


@app.get("/spot/api/admin/links")
def admin_links(x_admin_token: str = Header(None)):
    _require_admin(x_admin_token)
    with _connect() as c:
        rows = c.execute(
            "SELECT * FROM links ORDER BY created_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        spot = SPOTS.get(r["spot_slug"], {})
        paid = r["paid_at"] is not None
        out.append(
            {
                "token": r["token"],
                "spot_slug": r["spot_slug"],
                "title": spot.get("title", r["spot_slug"]),
                "label": r["label"],
                "plays_used": r["plays_used"],
                "plays_allowed": r["plays_allowed"],
                "paid": paid,
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
    return {"ok": True, "service": "spotgate", "spots": list(SPOTS.keys())}
