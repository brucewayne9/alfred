"""Fair Game — Rod Wave's fan-first, anti-scalper ticket marketplace (public API).

This is the single FastAPI sub-app that wires every Fair Game backend module
into one HTTP surface, mirroring core/api/arena_portal.py: public endpoints live
under /fairgame/api/*, the app binds 127.0.0.1 (the operator runs uvicorn; Caddy
fronts it), and nothing here issues a ticket barcode — Fair Game rides
Ticketmaster's rails (transfer simulated in v1 via core/fairgame/tm_transfer.py).

What it ties together:
  identity  -- one-identity-one-account dedupe + DLD-waitlist priority flag
  verify    -- SMS then email 6-digit codes (anti-bot "medium strength" gate)
  sessions  -- opaque Bearer token for a verified fan
  events    -- Rod's 35 DLD shows + per-section inventory
  access    -- the presale gate: capped-qty primary grant
  listings  -- capped resale offers (face + $15, seller +$10 / Rod +$5)
  orders    -- resale escrow state machine (hold -> released | refunded)
  admin     -- operator rollups (token-gated)

On import the app self-bootstraps: it creates the schema, seeds the shows and
demo inventory, and opens a default access wave per show so the primary buy flow
works out of the box. All of that is idempotent.

Sends (SMS/email) no-op gracefully when creds are missing (logged, not raised).
With FAIRGAME_DEV_ECHO=1 the verification code is echoed in the API response so
tests and local dev can complete the flow with no provider — NEVER set in prod.
"""
from __future__ import annotations

import hmac
import logging
import os
import re as _re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# Make the repo root importable when uvicorn loads this module directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from core.fairgame import (
    access,
    admin,
    aggregator,
    db,
    delivery,
    events,
    holdings,
    identity,
    listings,
    orders,
    seatmap,
    sessions,
    stripe_connect,
    verify,
)

logger = logging.getLogger("fairgame")

_TM_EMAIL_RE = _re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _valid_tm_email(s) -> bool:
    return bool(s) and bool(_TM_EMAIL_RE.match(s))


def _require_checkout(b: dict):
    """Pull + validate tm_email and the final-sale ack from a purchase body."""
    tm_email = (b.get("tm_email") or "").strip()
    ack = bool(b.get("final_sale_ack"))
    if not _valid_tm_email(tm_email):
        raise HTTPException(status_code=400, detail="a valid Ticketmaster email is required")
    if not ack:
        raise HTTPException(status_code=400, detail="you must acknowledge all sales are final")
    return tm_email, ack

# A wave wide enough to always be open in dev/demo (epoch 0 .. year ~2065).
_WAVE_OPEN_FROM = 0
_WAVE_OPEN_TO = 3_000_000_000
_DEFAULT_WAVE_MAX_QTY = 4

# No baked-in default: if FAIRGAME_ADMIN_TOKEN is unset the admin API is hard
# 401'd (an empty token can never match). Source the real token from the
# environment / config/.env — never a publicly-known string committed to the repo.
ADMIN_TOKEN = os.environ.get("FAIRGAME_ADMIN_TOKEN", "")


def _dev_echo() -> bool:
    return os.environ.get("FAIRGAME_DEV_ECHO") == "1"


# --------------------------------------------------------------------------- #
# Bootstrap (idempotent) — schema, shows, inventory, a default open wave/show
# --------------------------------------------------------------------------- #

def _ensure_default_waves() -> None:
    """Give every show one open, non-priority access wave if it has none."""
    with db.connect() as c:
        show_ids = [r["id"] for r in c.execute("SELECT id FROM shows").fetchall()]
        having = {
            r["show_id"]
            for r in c.execute("SELECT DISTINCT show_id FROM access_waves").fetchall()
        }
    for sid in show_ids:
        if sid in having:
            continue
        access.create_wave(
            sid,
            "General Access",
            _WAVE_OPEN_FROM,
            _WAVE_OPEN_TO,
            priority_only=False,
            max_qty_per_fan=_DEFAULT_WAVE_MAX_QTY,
        )


def bootstrap() -> None:
    db.init_db()
    events.seed_shows()
    events.seed_demo_inventory()
    _ensure_default_waves()


bootstrap()

app = FastAPI(title="Fair Game — Rod Wave Tickets")


# --------------------------------------------------------------------------- #
# Send wrappers — never raise (missing creds just log + no-op)
# --------------------------------------------------------------------------- #

def _send_sms(phone: str, code: str) -> None:
    """Fire a Klaviyo event so a Klaviyo Flow texts the code. Klaviyo SMS sends
    via Flows (not a direct API call): a Flow triggered by 'Fans First Code
    Requested' sends an SMS using {{ event.code }}. Never breaks signup."""
    try:
        from integrations import klaviyo_client

        klaviyo_client.track_event(
            "Fans First Code Requested",
            phone=phone,
            properties={"code": code, "expires_minutes": 10},
        )
    except Exception as e:  # noqa: BLE001 - send must never break the flow
        logger.warning("fairgame sms (klaviyo) send skipped/failed: %s", e)


def _send_email(email: str, code: str) -> None:
    try:
        from integrations.email.client import email_client

        email_client.send_email(
            account="alfred-gw",
            to=email,
            subject="Your Rod Wave (Fair Game) verification code",
            body=f"Your Fair Game verification code is {code}. It expires in 10 minutes.",
            html=False,
        )
    except Exception as e:  # noqa: BLE001 - send must never break the flow
        logger.warning("fairgame email send skipped/failed: %s", e)


# Where fan contact-form messages land. Placeholder inbox; override via env.
CONTACT_INBOX = os.environ.get("FAIRGAME_CONTACT_INBOX", "mjohnson@groundrushlabs.com")

_CONTACT_TOPICS = {
    "order": "A ticket I bought",
    "transfer": "Getting my ticket / transfer",
    "selling": "Selling a seat",
    "account": "Account / verification",
    "other": "Something else",
}


def _send_contact(name: str, email: str, topic: str, message: str) -> bool:
    """Email a fan's contact-form message to the support inbox (reply-to the fan).
    Returns True on apparent success; never raises."""
    try:
        from integrations.email.client import email_client

        label = _CONTACT_TOPICS.get(topic, "Contact")
        body = (
            "New Fans First contact message\n\n"
            f"From:  {name} <{email}>\n"
            f"Topic: {label}\n\n"
            f"{message}\n\n"
            "— Sent from the Fans First contact form. Reply to reach the fan directly."
        )
        res = email_client.send_email(
            account="groundrush info",  # info@groundrushlabs.com (Mailcow sender)
            to=CONTACT_INBOX,
            subject=f"[Fans First] {label} — {name}",
            body=body,
            html=False,
            reply_to=email,
        )
        if isinstance(res, dict) and res.get("error"):
            logger.warning("fairgame contact email failed: %s", res.get("error"))
            return False
        return True
    except Exception as e:  # noqa: BLE001 - send must never break the flow
        logger.warning("fairgame contact email error: %s", e)
        return False


# --------------------------------------------------------------------------- #
# Lightweight per-IP rate limiting (in-memory) for register/verify
# --------------------------------------------------------------------------- #

_RL_WINDOW = 60          # seconds
_RL_MAX = 30             # requests per IP per window on rate-limited routes
_rl_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(req: Request) -> None:
    ip = req.client.host if req.client else "?"
    now = time.time()
    hits = _rl_hits[ip]
    while hits and now - hits[0] > _RL_WINDOW:
        hits.popleft()
    if len(hits) >= _RL_MAX:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")
    hits.append(now)


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #

def _bearer(authorization: str) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return authorization.strip() or None


def _require_fan(authorization: str) -> dict:
    """Resolve a Bearer session to a fan dict, or 401."""
    tok = _bearer(authorization)
    sess = sessions.resolve(tok) if tok else None
    if not sess:
        raise HTTPException(status_code=401, detail="unauthorized")
    fan = identity.get_fan(sess["fan_id"])
    if not fan:
        raise HTTPException(status_code=401, detail="unauthorized")
    return fan


def _fan_summary(fan: dict) -> dict:
    return {
        "id": fan["id"],
        "email": fan["email"],
        "phone": fan["phone"],
        "status": fan["status"],
        "priority": fan["priority"],
    }


def _require_admin(token: str) -> None:
    # Unconfigured admin token => locked (no default secret to brute or leak).
    # Constant-time compare to avoid leaking the token via response timing.
    if not ADMIN_TOKEN or not hmac.compare_digest(token or "", ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="admin token required")


# --------------------------------------------------------------------------- #
# Shows view helpers
# --------------------------------------------------------------------------- #

def _show_card(show: dict) -> dict:
    inv = events.get_inventory(show["id"])
    listed = listings.list_active(show["id"])
    primary_min = min((i["face_price_cents"] for i in inv if i["qty_available"] > 0),
                      default=None)
    resale_min = min((l["buyer_total_cents"] for l in listed), default=None)
    return {
        "id": show["id"],
        "idx": show["idx"],
        "city": show["city"],
        "venue": show["venue"],
        "show_date": show["show_date"],
        "status": show["status"],
        "remaining": events.remaining(show["id"]),
        "min_price_cents": primary_min,
        "resale_from_cents": resale_min,
        "active_listings": len(listed),
    }


# --------------------------------------------------------------------------- #
# FANS
# --------------------------------------------------------------------------- #

@app.post("/fairgame/api/register")
async def register(req: Request):
    _rate_limit(req)
    b = await req.json()
    email = (b.get("email") or "").strip()
    phone = (b.get("phone") or "").strip()
    if not email or not phone:
        raise HTTPException(status_code=400, detail="email and phone required")
    ip = req.client.host if req.client else None
    fan = identity.upsert_fan(email, phone, b.get("device_fp"), ip)
    holder: dict = {}
    try:
        verify.start_verification(
            fan["id"], "sms",
            lambda code: (holder.update(code=code), _send_sms(fan["phone"], code)),
        )
    except verify.VerifyError as e:
        raise HTTPException(status_code=429, detail=str(e))
    out = {"fan_id": fan["id"], "sent": True, "channel": "sms"}
    if _dev_echo():
        out["dev_code"] = holder.get("code")
    return out


@app.post("/fairgame/api/verify")
async def verify_sms(req: Request):
    _rate_limit(req)
    b = await req.json()
    fid = b.get("fan_id")
    code = b.get("code", "")
    if not fid:
        raise HTTPException(status_code=400, detail="fan_id required")
    try:
        ok = verify.check_code(fid, "sms", code)
    except verify.VerifyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=400, detail="invalid code")
    fan = identity.get_fan(fid)
    if not fan:
        raise HTTPException(status_code=404, detail="fan not found")
    holder: dict = {}
    try:
        verify.start_verification(
            fid, "email",
            lambda c: (holder.update(code=c), _send_email(fan["email"], c)),
        )
    except verify.VerifyError as e:
        raise HTTPException(status_code=429, detail=str(e))
    out = {"verified_sms": True, "channel": "email"}
    if _dev_echo():
        out["dev_code"] = holder.get("code")
    return out


@app.post("/fairgame/api/verify-email")
async def verify_email(req: Request):
    _rate_limit(req)
    b = await req.json()
    fid = b.get("fan_id")
    code = b.get("code", "")
    if not fid:
        raise HTTPException(status_code=400, detail="fan_id required")
    try:
        ok = verify.check_code(fid, "email", code)
    except verify.VerifyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=400, detail="invalid code")
    with db.connect() as c:
        c.execute("UPDATE fans SET status='verified', updated_at=? WHERE id=?",
                  (int(time.time()), fid))
    fan = identity.get_fan(fid)
    ip = req.client.host if req.client else None
    tok = sessions.issue(fid, b.get("device_fp"), ip)
    return {"token": tok, "fan": _fan_summary(fan)}


@app.get("/fairgame/api/me")
async def me(authorization: str = Header(default="")):
    fan = _require_fan(authorization)
    return {"fan": _fan_summary(fan)}


# --------------------------------------------------------------------------- #
# MY ACTIVITY — a fan's own listings + purchases (the "My Tickets" surface)
# --------------------------------------------------------------------------- #

@app.get("/fairgame/api/my/listings")
async def my_listings(authorization: str = Header(default="")):
    """Seats the signed-in fan has listed (any status). Seller-scoped."""
    fan = _require_fan(authorization)
    return {"listings": listings.list_by_seller(fan["id"])}


@app.get("/fairgame/api/my/orders")
async def my_orders(authorization: str = Header(default="")):
    """Tickets the signed-in fan has bought, with live escrow state. Buyer-scoped."""
    fan = _require_fan(authorization)
    return {"orders": orders.list_by_buyer(fan["id"])}


# --------------------------------------------------------------------------- #
# SHOWS
# --------------------------------------------------------------------------- #

@app.get("/fairgame/api/shows")
async def shows():
    return {"shows": [_show_card(s) for s in events.list_shows()]}


@app.get("/fairgame/api/shows/{show_id}")
async def show_detail(show_id: str):
    show = events.get_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="show not found")
    card = _show_card(show)
    card["inventory"] = events.get_inventory(show_id)
    card["listings"] = listings.list_active(show_id)
    card["seatmap"] = seatmap.status(show_id)
    return card


# --------------------------------------------------------------------------- #
# SEAT MAPS — venue geometry (open layer; availability overlays later)
# --------------------------------------------------------------------------- #

@app.get("/fairgame/api/shows/{show_id}/seatmap")
async def show_seatmap(show_id: str, full: int = 0):
    """Section-level map for first paint (default), or the complete map with
    every seat when ?full=1. 404 when no map is ingested for this show yet."""
    if not events.get_show(show_id):
        raise HTTPException(status_code=404, detail="show not found")
    data = seatmap.full(show_id) if full else seatmap.overview(show_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=seatmap.status(show_id).get("reason", "no seatmap"))
    return data


@app.get("/fairgame/api/shows/{show_id}/seatmap/section/{section_id}")
async def show_seatmap_section(show_id: str, section_id: str):
    """One section's rows + seats, lazy-loaded on zoom. 404 if not found."""
    sec = seatmap.section(show_id, section_id)
    if sec is None:
        raise HTTPException(status_code=404, detail="section not found")
    return sec


@app.get("/fairgame/api/shows/{show_id}/seatmap/holdings")
async def show_seatmap_holdings(show_id: str):
    """Green/red/grey overlay: which sections Rod holds + their status.
    404 when no map is ingested for this show."""
    if not events.get_show(show_id):
        raise HTTPException(status_code=404, detail="show not found")
    data = holdings.overlay(show_id)
    if data is None:
        raise HTTPException(status_code=404,
                            detail=seatmap.status(show_id).get("reason", "no seatmap"))
    return data


# --------------------------------------------------------------------------- #
# ACCESS (primary capped buy)
# --------------------------------------------------------------------------- #

@app.post("/fairgame/api/access")
async def grant(req: Request, authorization: str = Header(default="")):
    # The acting fan is ALWAYS the Bearer session — never a body-supplied fan_id
    # (that would let an unauthenticated caller grant access AS any fan and drain
    # inventory). This is the presale gate; verified fans only.
    fan = _require_fan(authorization)
    if fan["status"] != "verified":
        raise HTTPException(status_code=403, detail="fan is not verified")
    fid = fan["id"]
    b = await req.json()
    tm_email, ack = _require_checkout(b)
    show_id = b.get("show_id")
    qty = int(b.get("qty", 1))
    if not show_id:
        raise HTTPException(status_code=400, detail="show_id required")
    try:
        g = access.grant_access(fid, show_id, qty, tm_email=tm_email, final_sale_ack=ack)
    except access.AccessError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"grant": g}


# --------------------------------------------------------------------------- #
# RESALE — exchange, listings, escrow orders
# --------------------------------------------------------------------------- #

@app.get("/fairgame/api/exchange/{show_id}")
async def exchange(show_id: str):
    return {"show_id": show_id, "listings": listings.list_active(show_id)}


@app.post("/fairgame/api/listings")
async def create_listing(req: Request, authorization: str = Header(default="")):
    fan = _require_fan(authorization)
    b = await req.json()
    show_id = b.get("show_id")
    section = b.get("section")
    try:
        face = int(b.get("face_price_cents"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="face_price_cents must be an integer")
    if not show_id or not section:
        raise HTTPException(status_code=400, detail="show_id and section required")
    try:
        lst = listings.create_listing(fan["id"], show_id, section, face)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"listing": lst}


@app.post("/fairgame/api/buy")
async def buy(req: Request, authorization: str = Header(default="")):
    fan = _require_fan(authorization)
    b = await req.json()
    tm_email, ack = _require_checkout(b)
    listing_id = b.get("listing_id")
    if not listing_id:
        raise HTTPException(status_code=400, detail="listing_id required")
    try:
        order = orders.create_order(fan["id"], listing_id, tm_email=tm_email, final_sale_ack=ack)
    except orders.OrderError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"order": order}


def _is_admin(token: str) -> bool:
    """True if the supplied admin token is configured and matches (const-time)."""
    return bool(ADMIN_TOKEN) and hmac.compare_digest(token or "", ADMIN_TOKEN)


def _require_order_actor(order_id: str, authorization: str, admin_token: str):
    """Load the order and assert the caller is its buyer (session) or an admin.

    Escrow terminal transitions and order reads move money / leak PII, so they
    are never world-callable. Returns the order row.
    """
    order = orders.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="order not found")
    if _is_admin(admin_token):
        return order
    # Otherwise the caller must be the order's own buyer, via a Bearer session.
    fan = _require_fan(authorization)
    if fan["id"] != order.get("buyer_fan_id"):
        raise HTTPException(status_code=403, detail="forbidden")
    return order


@app.post("/fairgame/api/orders/{order_id}/confirm")
async def confirm_order(
    order_id: str,
    authorization: str = Header(default=""),
    x_fairgame_admin: str = Header(default=""),
):
    # Buyer confirms receipt of the transfer, or an admin/TM webhook settles it.
    # Never let an anonymous client force-release a seller's escrowed funds.
    _require_order_actor(order_id, authorization, x_fairgame_admin)
    try:
        order = orders.confirm_transfer(order_id)
    except orders.OrderError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"order": order}


@app.post("/fairgame/api/orders/{order_id}/fail")
async def fail_order(order_id: str, x_fairgame_admin: str = Header(default="")):
    # Refund / fraud-wall is admin (or TM webhook) only — never the buyer and
    # never anonymous, so nobody can force-refund a legitimately settled seat.
    _require_admin(x_fairgame_admin)
    try:
        order = orders.fail_transfer(order_id)
    except orders.OrderError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"order": order}


@app.get("/fairgame/api/orders/{order_id}")
async def get_order(
    order_id: str,
    authorization: str = Header(default=""),
    x_fairgame_admin: str = Header(default=""),
):
    # Order rows carry buyer_fan_id / amount / payment refs — buyer or admin only.
    order = _require_order_actor(order_id, authorization, x_fairgame_admin)
    return {"order": order}


# --------------------------------------------------------------------------- #
# DISCOVER — the "everything else" aggregator tab (camouflage)
# We don't hold this inventory: index a real feed, redirect out with our
# affiliate sub-ID, charge a small "verified ticket" service fee. SPIKE.
# --------------------------------------------------------------------------- #

@app.get("/fairgame/api/discover")
async def discover(q: str = "", seg: str = "", city: str = "", size: int = 24):
    segment = seg.strip() or None
    return aggregator.search(q.strip(), segment, city.strip() or None, size)


@app.get("/fairgame/api/discover/out")
async def discover_out(u: str, src: str = "partner"):
    # Outbound redirect to the originating marketplace, carrying our affiliate
    # sub-ID. Only http(s) targets are allowed so this can't be used as an
    # open redirect to arbitrary schemes.
    if not (u.startswith("https://") or u.startswith("http://")):
        raise HTTPException(status_code=400, detail="invalid target")
    dest = aggregator.affiliate_url(u, src)
    logger.info("fairgame discover redirect -> %s (src=%s)", src, dest)
    return RedirectResponse(dest, status_code=302)


@app.post("/fairgame/api/discover/unlock")
async def discover_unlock(req: Request):
    """The $1 Discover unlock. Sim mode (no Stripe key) grants instantly so the
    product is demoable; live mode returns a Stripe checkout URL to complete."""
    amount = aggregator.SERVICE_FEE_CENTS  # 100 = $1
    if stripe_connect.is_sim():
        return {"unlocked": True, "sim": True, "amount_cents": amount}
    # Live: create a $1 checkout. Buyer completes payment, returns unlocked.
    url = stripe_connect.create_unlock_checkout(amount)
    return {"unlocked": False, "checkout_url": url, "amount_cents": amount}


@app.post("/fairgame/api/contact")
async def contact(req: Request):
    """Fan contact form -> support inbox. Honeypot + light gibberish guard,
    per-IP rate-limited. Never reveals the inbox address."""
    _rate_limit(req)
    try:
        data = await req.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid body")
    # Honeypot: bots fill the hidden 'company' field — accept silently, drop it.
    if (data.get("company") or "").strip():
        return {"ok": True}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    topic = (data.get("topic") or "other").strip()
    message = (data.get("message") or data.get("msg") or "").strip()
    if not name or not _valid_tm_email(email) or len(message) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please add your name, a valid email, and a message.",
        )
    if len(name) > 120 or len(email) > 200 or len(message) > 4000:
        raise HTTPException(status_code=400, detail="That's a bit long — please trim it down.")
    if not _send_contact(name, email, topic, message):
        raise HTTPException(status_code=502, detail="Could not send right now. Please try again.")
    logger.info("fairgame contact from %s <%s> topic=%s", name, email, topic)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# ADMIN (header-token gated)
# --------------------------------------------------------------------------- #

@app.get("/fairgame/api/admin/stats")
async def admin_stats(x_fairgame_admin: str = Header(default="")):
    _require_admin(x_fairgame_admin)
    return admin.stats()


@app.get("/fairgame/api/admin/fans")
async def admin_fans(x_fairgame_admin: str = Header(default=""), limit: int = 100):
    _require_admin(x_fairgame_admin)
    return {"fans": admin.list_fans(limit)}


@app.get("/fairgame/api/admin/orders")
async def admin_orders(x_fairgame_admin: str = Header(default=""), limit: int = 100):
    _require_admin(x_fairgame_admin)
    return {"orders": admin.list_orders(limit)}


@app.get("/fairgame/api/admin/shows/{show_id}/inventory")
async def admin_show_inventory(show_id: str, x_fairgame_admin: str = Header(default="")):
    _require_admin(x_fairgame_admin)
    show = events.get_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="show not found")
    return {"inventory": events.get_inventory(show_id)}


@app.patch("/fairgame/api/admin/inventory/{inv_id}")
async def admin_patch_inventory(inv_id: str, req: Request, x_fairgame_admin: str = Header(default="")):
    _require_admin(x_fairgame_admin)
    b = await req.json()
    kwargs: dict = {}
    for field in ("face_price_cents", "qty_available", "qty_total"):
        if field in b:
            v = b[field]
            if not isinstance(v, int) or v < 0:
                raise HTTPException(status_code=400, detail=f"{field} must be a non-negative integer")
            kwargs[field] = v
    row = events.update_inventory(inv_id, **kwargs)
    if row is None:
        raise HTTPException(status_code=404, detail="inventory row not found")
    return {"inventory": row}


@app.post("/fairgame/api/admin/shows/{show_id}/inventory")
async def admin_add_inventory(show_id: str, req: Request, x_fairgame_admin: str = Header(default="")):
    _require_admin(x_fairgame_admin)
    show = events.get_show(show_id)
    if not show:
        raise HTTPException(status_code=404, detail="show not found")
    b = await req.json()
    section = (b.get("section") or "").strip()
    if not section:
        raise HTTPException(status_code=400, detail="section is required")
    qty = b.get("qty")
    face_price_cents = b.get("face_price_cents")
    for fname, fv in (("qty", qty), ("face_price_cents", face_price_cents)):
        if not isinstance(fv, int) or fv < 0:
            raise HTTPException(status_code=400, detail=f"{fname} must be a non-negative integer")
    row = events.add_inventory(show_id, section, qty, face_price_cents)
    return {"inventory": row}


@app.get("/fairgame/api/admin/delivery")
async def admin_delivery_queue(x_fairgame_admin: str = Header(default="")):
    """Return the operator delivery queue — all purchases needing a TM transfer."""
    _require_admin(x_fairgame_admin)
    return {"queue": delivery.queue()}


@app.post("/fairgame/api/admin/delivery/mark")
async def admin_delivery_mark(req: Request, x_fairgame_admin: str = Header(default="")):
    """Mark a purchase as delivered (operator has completed the TM transfer).

    Body: {kind: 'order'|'grant', id: str}
    """
    _require_admin(x_fairgame_admin)
    b = await req.json()
    kind = (b.get("kind") or "").strip()
    item_id = (b.get("id") or "").strip()
    if not kind or not item_id:
        raise HTTPException(status_code=400, detail="kind and id are required")
    try:
        result = delivery.mark_delivered(kind, item_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"item": result}
