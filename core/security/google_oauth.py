"""Google OAuth2 flow for Gmail and Calendar access."""

import json
import logging
from pathlib import Path

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest

from config.settings import settings

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

TOKEN_FILE = Path(settings.base_dir) / "config" / "google_token.json"


def get_auth_url() -> str:
    """Generate the Google OAuth authorization URL."""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.google_redirect_uri],
            }
        },
        scopes=SCOPES,
    )
    flow.redirect_uri = settings.google_redirect_uri

    auth_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    return auth_url


def handle_callback(authorization_code: str) -> bool:
    """Exchange authorization code for tokens and save them."""
    try:
        import requests as _requests

        # Exchange code for tokens directly to avoid scope-change errors
        # when Google returns additional previously-granted scopes.
        token_resp = _requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": authorization_code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_data = token_resp.json()

        if "error" in token_data:
            logger.error(f"Google token exchange failed: {token_data}")
            return False

        creds = Credentials(
            token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=SCOPES,
        )
        _save_credentials(creds)
        logger.info("Google OAuth tokens saved successfully")
        return True
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {e}")
        return False


def get_credentials() -> Credentials | None:
    """Get valid Google credentials, refreshing if needed."""
    if not TOKEN_FILE.exists():
        return None

    try:
        data = json.loads(TOKEN_FILE.read_text())
        creds = Credentials.from_authorized_user_info(data, SCOPES)

        if creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
            _save_credentials(creds)
            logger.info("Google token refreshed")

        if creds.valid:
            return creds
    except Exception as e:
        logger.error(f"Failed to load Google credentials: {e}")

    return None


def is_connected() -> bool:
    """Check if Google services are connected."""
    creds = get_credentials()
    return creds is not None and creds.valid


def _save_credentials(creds: Credentials):
    """Save credentials to file."""
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    TOKEN_FILE.write_text(json.dumps(data))
    TOKEN_FILE.chmod(0o600)
