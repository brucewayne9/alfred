"""Authentication and security layer for Alfred with 2FA and Passkey support."""

import json
import logging
import secrets
import base64
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pyotp
import qrcode

from config.settings import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

USERS_FILE = Path(settings.base_dir) / "config" / "users.json"
RP_ID = "aialfred.groundrushcloud.com"  # Relying Party ID for WebAuthn
RP_NAME = "Alfred AI Assistant"


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
        "totp_secret": None,
        "totp_enabled": False,
        "passkeys": [],  # List of registered WebAuthn credentials
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
    return {
        "username": username,
        "role": user["role"],
        "totp_enabled": user.get("totp_enabled", False),
        "has_passkeys": len(user.get("passkeys", [])) > 0,
    }


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
    """Get current user from JWT token. Returns None if no auth."""
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


# ==================== TOTP 2FA ====================

def setup_totp(username: str) -> dict:
    """Generate a new TOTP secret for a user. Returns secret and QR code."""
    users = _load_users()
    if username not in users:
        return {"error": "User not found"}

    # Generate new secret
    secret = pyotp.random_base32()
    users[username]["totp_secret"] = secret
    users[username]["totp_enabled"] = False  # Not enabled until verified
    _save_users(users)

    # Generate provisioning URI
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(name=username, issuer_name=RP_NAME)

    # Generate QR code as base64
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "secret": secret,
        "qr_code": f"data:image/png;base64,{qr_base64}",
        "uri": uri,
    }


def verify_totp(username: str, code: str) -> bool:
    """Verify a TOTP code for a user."""
    users = _load_users()
    user = users.get(username)
    if not user or not user.get("totp_secret"):
        return False

    totp = pyotp.TOTP(user["totp_secret"])
    return totp.verify(code, valid_window=1)  # Allow 30 second window


def enable_totp(username: str, code: str) -> dict:
    """Enable TOTP after verifying the code."""
    if not verify_totp(username, code):
        return {"error": "Invalid code"}

    users = _load_users()
    users[username]["totp_enabled"] = True
    _save_users(users)
    logger.info(f"TOTP enabled for user: {username}")
    return {"success": True, "message": "2FA enabled"}


def disable_totp(username: str, code: str) -> dict:
    """Disable TOTP after verifying the code."""
    if not verify_totp(username, code):
        return {"error": "Invalid code"}

    users = _load_users()
    users[username]["totp_enabled"] = False
    users[username]["totp_secret"] = None
    _save_users(users)
    logger.info(f"TOTP disabled for user: {username}")
    return {"success": True, "message": "2FA disabled"}


def is_totp_enabled(username: str) -> bool:
    """Check if TOTP is enabled for a user."""
    users = _load_users()
    user = users.get(username)
    return user.get("totp_enabled", False) if user else False


# ==================== PASSKEYS (WebAuthn) ====================

# Store challenges temporarily (in production, use Redis or similar)
_challenges: dict[str, bytes] = {}


def get_passkey_registration_options(username: str) -> dict:
    """Generate WebAuthn registration options."""
    from webauthn import generate_registration_options
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria,
        UserVerificationRequirement,
        ResidentKeyRequirement,
    )

    users = _load_users()
    user = users.get(username)
    if not user:
        return {"error": "User not found"}

    # Get existing credential IDs to exclude
    exclude_credentials = []
    for passkey in user.get("passkeys", []):
        exclude_credentials.append({
            "id": base64.urlsafe_b64decode(passkey["credential_id"]),
            "type": "public-key",
        })

    user_id = username.encode()

    options = generate_registration_options(
        rp_id=RP_ID,
        rp_name=RP_NAME,
        user_id=user_id,
        user_name=username,
        user_display_name=username,
        exclude_credentials=exclude_credentials if exclude_credentials else None,
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.PREFERRED,
            resident_key=ResidentKeyRequirement.PREFERRED,
            # Allow cross-platform authenticators (phones, security keys)
            authenticator_attachment=None,  # None = allow both platform and cross-platform
        ),
    )

    # Store challenge for verification
    _challenges[username] = options.challenge

    # Convert to JSON-serializable format
    return {
        "challenge": base64.urlsafe_b64encode(options.challenge).decode().rstrip("="),
        "rp": {"id": options.rp.id, "name": options.rp.name},
        "user": {
            "id": base64.urlsafe_b64encode(options.user.id).decode().rstrip("="),
            "name": options.user.name,
            "displayName": options.user.display_name,
        },
        "pubKeyCredParams": [{"type": "public-key", "alg": p.alg} for p in options.pub_key_cred_params],
        "timeout": options.timeout,
        "authenticatorSelection": {
            "userVerification": options.authenticator_selection.user_verification.value,
            "residentKey": options.authenticator_selection.resident_key.value,
        },
        "attestation": options.attestation.value,
    }


def verify_passkey_registration(username: str, credential: dict) -> dict:
    """Verify and store a new passkey registration."""
    from webauthn import verify_registration_response
    from webauthn.helpers.structs import RegistrationCredential, AuthenticatorAttestationResponse

    users = _load_users()
    user = users.get(username)
    if not user:
        return {"error": "User not found"}

    challenge = _challenges.get(username)
    if not challenge:
        return {"error": "No pending registration"}

    try:
        # Reconstruct the credential object with proper types
        attestation_response = AuthenticatorAttestationResponse(
            client_data_json=base64.urlsafe_b64decode(credential["response"]["clientDataJSON"] + "=="),
            attestation_object=base64.urlsafe_b64decode(credential["response"]["attestationObject"] + "=="),
        )

        reg_credential = RegistrationCredential(
            id=credential["id"],
            raw_id=base64.urlsafe_b64decode(credential["rawId"] + "=="),
            response=attestation_response,
        )

        verification = verify_registration_response(
            credential=reg_credential,
            expected_challenge=challenge,
            expected_rp_id=RP_ID,
            expected_origin=f"https://{RP_ID}",
        )

        # Store the credential with transports for cross-device support
        transports = credential.get("response", {}).get("transports", ["internal", "hybrid"])
        passkey = {
            "credential_id": base64.urlsafe_b64encode(verification.credential_id).decode(),
            "public_key": base64.urlsafe_b64encode(verification.credential_public_key).decode(),
            "sign_count": verification.sign_count,
            "transports": transports,  # e.g., ["internal", "hybrid"] for cross-device
            "created": datetime.now(timezone.utc).isoformat(),
            "name": credential.get("name", f"Passkey {len(user.get('passkeys', [])) + 1}"),
        }

        if "passkeys" not in user:
            user["passkeys"] = []
        user["passkeys"].append(passkey)
        users[username] = user
        _save_users(users)

        # Clean up challenge
        del _challenges[username]

        logger.info(f"Passkey registered for user: {username}")
        return {"success": True, "message": "Passkey registered"}

    except Exception as e:
        logger.error(f"Passkey registration failed: {e}")
        return {"error": str(e)}


def get_passkey_login_options(username: str = None) -> dict:
    """Generate WebAuthn authentication options."""
    from webauthn import generate_authentication_options
    from webauthn.helpers.structs import UserVerificationRequirement

    allow_credentials = []

    if username:
        users = _load_users()
        user = users.get(username)
        if user:
            for passkey in user.get("passkeys", []):
                allow_credentials.append({
                    "id": base64.urlsafe_b64decode(passkey["credential_id"] + "=="),
                    "type": "public-key",
                    "transports": passkey.get("transports", ["internal", "hybrid"]),
                })

    options = generate_authentication_options(
        rp_id=RP_ID,
        allow_credentials=allow_credentials if allow_credentials else None,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    # Store challenge
    challenge_key = username or "anonymous"
    _challenges[challenge_key] = options.challenge

    return {
        "challenge": base64.urlsafe_b64encode(options.challenge).decode().rstrip("="),
        "rpId": RP_ID,
        "timeout": options.timeout,
        "userVerification": options.user_verification.value,
        "allowCredentials": [
            {
                "id": base64.urlsafe_b64encode(c["id"]).decode().rstrip("="),
                "type": "public-key",
                "transports": c.get("transports", ["internal", "hybrid"]),
            }
            for c in allow_credentials
        ] if allow_credentials else [],
    }


def verify_passkey_login(credential: dict, username: str = None) -> dict:
    """Verify a passkey login attempt."""
    from webauthn import verify_authentication_response
    from webauthn.helpers.structs import AuthenticationCredential, AuthenticatorAssertionResponse

    users = _load_users()

    # Find the user with this credential
    credential_id = credential["id"]
    found_user = None
    found_passkey = None

    for uname, user in users.items():
        for passkey in user.get("passkeys", []):
            # Compare credential IDs
            if passkey["credential_id"].rstrip("=") == credential_id.rstrip("="):
                found_user = uname
                found_passkey = passkey
                break
        if found_user:
            break

    if not found_user or not found_passkey:
        return {"error": "Passkey not found"}

    challenge_key = username or "anonymous"
    challenge = _challenges.get(challenge_key)
    if not challenge:
        return {"error": "No pending login"}

    try:
        # Construct proper response object
        user_handle = None
        if credential["response"].get("userHandle"):
            user_handle = base64.urlsafe_b64decode(credential["response"]["userHandle"] + "==")

        assertion_response = AuthenticatorAssertionResponse(
            client_data_json=base64.urlsafe_b64decode(credential["response"]["clientDataJSON"] + "=="),
            authenticator_data=base64.urlsafe_b64decode(credential["response"]["authenticatorData"] + "=="),
            signature=base64.urlsafe_b64decode(credential["response"]["signature"] + "=="),
            user_handle=user_handle,
        )

        auth_credential = AuthenticationCredential(
            id=credential["id"],
            raw_id=base64.urlsafe_b64decode(credential["rawId"] + "=="),
            response=assertion_response,
        )

        verification = verify_authentication_response(
            credential=auth_credential,
            expected_challenge=challenge,
            expected_rp_id=RP_ID,
            expected_origin=f"https://{RP_ID}",
            credential_public_key=base64.urlsafe_b64decode(found_passkey["public_key"] + "=="),
            credential_current_sign_count=found_passkey["sign_count"],
        )

        # Update sign count
        found_passkey["sign_count"] = verification.new_sign_count
        _save_users(users)

        # Clean up challenge
        del _challenges[challenge_key]

        user_data = users[found_user]
        logger.info(f"Passkey login successful for user: {found_user}")

        return {
            "success": True,
            "username": found_user,
            "role": user_data["role"],
        }

    except Exception as e:
        logger.error(f"Passkey login failed: {e}")
        return {"error": str(e)}


def list_passkeys(username: str) -> list:
    """List all passkeys for a user."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return []

    return [
        {
            "id": p["credential_id"][:20] + "...",
            "name": p.get("name", "Passkey"),
            "created": p.get("created", "Unknown"),
        }
        for p in user.get("passkeys", [])
    ]


def delete_passkey(username: str, credential_id_prefix: str) -> dict:
    """Delete a passkey by credential ID prefix."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return {"error": "User not found"}

    passkeys = user.get("passkeys", [])
    original_count = len(passkeys)

    user["passkeys"] = [
        p for p in passkeys
        if not p["credential_id"].startswith(credential_id_prefix)
    ]

    if len(user["passkeys"]) == original_count:
        return {"error": "Passkey not found"}

    _save_users(users)
    logger.info(f"Passkey deleted for user: {username}")
    return {"success": True, "message": "Passkey deleted"}


def has_passkeys(username: str) -> bool:
    """Check if user has any registered passkeys."""
    users = _load_users()
    user = users.get(username)
    return len(user.get("passkeys", [])) > 0 if user else False


def get_user_auth_methods(username: str) -> dict:
    """Get available auth methods for a user."""
    users = _load_users()
    user = users.get(username)
    if not user:
        return {"exists": False}

    return {
        "exists": True,
        "totp_enabled": user.get("totp_enabled", False),
        "has_passkeys": len(user.get("passkeys", [])) > 0,
        "passkey_count": len(user.get("passkeys", [])),
    }
