"""Authentication and security layer for Alfred."""

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.settings import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

USERS_FILE = Path(settings.base_dir) / "config" / "users.json"


def _ensure_secret_key():
    """Generate and persist a secret key if not set."""
    if settings.secret_key and settings.secret_key != "":
        return settings.secret_key
    key = secrets.token_hex(32)
    env_path = Path(settings.base_dir) / "config" / ".env"
    with open(env_path, "a") as f:
        f.write(f"\nSECRET_KEY={key}\n")
    settings.secret_key = key
    logger.info("Generated new SECRET_KEY")
    return key


def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))
    USERS_FILE.chmod(0o600)


def create_user(username: str, password: str, role: str = "admin") -> bool:
    """Create a new user."""
    users = _load_users()
    if username in users:
        return False
    users[username] = {
        "password_hash": pwd_context.hash(password),
        "role": role,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    logger.info(f"User created: {username} ({role})")
    return True


def verify_user(username: str, password: str) -> dict | None:
    """Verify username and password. Returns user dict or None."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    if not pwd_context.verify(password, user["password_hash"]):
        return None
    return {"username": username, "role": user["role"]}


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    secret = _ensure_secret_key()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret, algorithm=settings.algorithm)


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    secret = _ensure_secret_key()
    try:
        payload = jwt.decode(token, secret, algorithms=[settings.algorithm])
        return payload
    except JWTError:
        return None


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict | None:
    """Get current user from JWT token. Returns None if no auth (allows unauthenticated for now)."""
    # Check Bearer token
    if credentials:
        payload = decode_token(credentials.credentials)
        if payload:
            return payload

    # Check cookie
    token = request.cookies.get("alfred_token")
    if token:
        payload = decode_token(token)
        if payload:
            return payload

    # During initial setup, allow unauthenticated access
    users = _load_users()
    if not users:
        return {"username": "setup", "role": "admin"}

    return None


async def require_auth(user: dict | None = Depends(get_current_user)) -> dict:
    """Require authentication. Raises 401 if not authenticated."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def setup_initial_user():
    """Create the initial admin user if none exists."""
    users = _load_users()
    if not users:
        password = secrets.token_urlsafe(16)
        create_user("bruce", password, "admin")
        logger.info(f"Initial admin user created. Username: bruce")
        logger.info(f"Initial password (change after first login): {password}")
        return password
    return None
