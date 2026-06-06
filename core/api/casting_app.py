# core/api/casting_app.py
"""Standalone, isolated entrypoint for the Central Casting BETA.

Runs as its own process (separate from alfred.service) so beta code can never
block or crash the main single-worker app. Mounts ONLY: auth + casting + SPA.
Deliberately does NOT use core.api.main's lifespan (which runs GPU-cleanup and
watcher loops that must not run twice)."""
from __future__ import annotations
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

logger = logging.getLogger("central_casting_beta")

app = FastAPI(title="Central Casting (beta)")

# Rate limiter — required so the @limiter.limit decorators below work, mirroring
# core.api.main's setup. Without app.state.limiter + SlowAPIMiddleware the
# decorated routes raise at request time.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(
    status_code=429, content={"detail": "Rate limit exceeded"}))
app.add_middleware(SlowAPIMiddleware)

# --- auth (so login + require_auth work) ---
# main.py defines these endpoints INLINE on its app (no reusable auth router),
# so we mirror them precisely here on top of the SAME core.security.auth
# primitives. The frontend login overlay (frontend/src/api/auth.ts) calls the
# bare /auth/* paths below (NOT /api/auth/*). require_auth/get_current_user read
# the Bearer header or alfred_token cookie straight off the request, so no extra
# app wiring is needed for the casting routes' guard to be satisfied.
from core.security.auth import (
    create_access_token,
    get_current_user,
    get_user_auth_methods,
    require_auth,
    setup_initial_user,
    verify_totp,
    verify_user,
)


class LoginRequest(BaseModel):
    username: str
    password: str


class TOTPLoginRequest(BaseModel):
    username: str
    code: str


@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, req: LoginRequest):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if user.get("totp_enabled"):
        return JSONResponse({
            "requires_2fa": True,
            "username": user["username"],
            "message": "Please enter your 2FA code",
        })
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    response = JSONResponse({"token": token, "username": user["username"], "role": user["role"]})
    response.set_cookie("alfred_token", token, httponly=True, secure=True, samesite="lax", max_age=86400)
    return response


@app.post("/auth/2fa/login")
@limiter.limit("5/minute")
async def totp_login(request: Request, req: TOTPLoginRequest):
    """Complete login with TOTP verification (step 2 of 2FA login)."""
    if not verify_totp(req.username, req.code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")
    from core.security.auth import _load_users
    users = _load_users()
    user = users.get(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    token = create_access_token({"sub": req.username, "role": user["role"]})
    response = JSONResponse({"token": token, "username": req.username, "role": user["role"]})
    response.set_cookie("alfred_token", token, httponly=True, secure=True, samesite="lax", max_age=86400)
    return response


@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    if user is None:
        return {"authenticated": False}
    return {"authenticated": True, "username": user.get("sub", user.get("username")), "role": user.get("role")}


@app.get("/auth/auto")
async def auto_login(request: Request):
    """Auto-login for trusted local network clients (mirrors main.py)."""
    import ipaddress
    client_ip = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    trusted = False
    try:
        addr = ipaddress.ip_address(client_ip)
        private_nets = [
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("::1/128"),
            ipaddress.ip_network("75.43.156.0/24"),  # Ground Rush LAN
        ]
        trusted = any(addr in net for net in private_nets)
    except ValueError:
        trusted = False

    if not trusted:
        return JSONResponse({"auto_login": False})

    from core.security.auth import _load_users
    users = _load_users()
    admin_user = None
    for username, user_data in users.items():
        if user_data.get("role") == "admin":
            admin_user = username
            break
    if not admin_user:
        return JSONResponse({"auto_login": False})

    token = create_access_token({"sub": admin_user, "role": "admin"})
    response = JSONResponse({"auto_login": True, "username": admin_user})
    response.set_cookie("alfred_token", token, httponly=True, secure=False, samesite="lax", max_age=86400)
    return response


@app.post("/auth/logout")
async def logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("alfred_token")
    return response


@app.get("/auth/methods")
async def auth_methods(username: str = None):
    """Get available auth methods for a user (for login page)."""
    if not username:
        return {"exists": False, "totp_enabled": False, "has_passkeys": False}
    return get_user_auth_methods(username)


# Ensure an initial admin user exists (same call main.py makes in its lifespan).
# Cheap, idempotent, no background loops.
try:
    _pwd = setup_initial_user()
    if _pwd:
        logger.info("Initial admin user created — check logs for password")
except Exception as _e:  # noqa: BLE001
    logger.warning("setup_initial_user failed: %s", _e)


# --- casting (forced ON) ---
from core.casting.db import init_db
from core.casting.api_router import register as register_casting


@app.get("/api/casting/enabled")
async def _casting_enabled():
    return {"enabled": True}


init_db()
register_casting(app)


# --- serve the built SPA (index + assets, with SPA fallback) ---
# Mounted LAST so it never shadows /api/* or /auth/* routes above.
_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="spa")
else:
    logger.warning("frontend dist not found at %s", _dist)
