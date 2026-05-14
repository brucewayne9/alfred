# integrations/google_seo/oauth.py
"""Google OAuth helper for Search Console + GA4. PageSpeed Insights uses an API key, not OAuth.

Flow:
1. First-time: run `python -m integrations.google_seo.oauth authorize` from a TTY.
   Opens a browser, Mike consents, refresh token saved to seo_oauth_token_path.
2. Steady-state: clients import get_credentials() which loads + auto-refreshes.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

SCOPES: list[str] = [
    "https://www.googleapis.com/auth/webmasters.readonly",  # Search Console
    "https://www.googleapis.com/auth/analytics.readonly",   # GA4
]


def _config() -> dict:
    from config.settings import settings
    if not settings.seo_google_oauth_client_id or not settings.seo_google_oauth_client_secret:
        raise RuntimeError(
            "SEO_GOOGLE_OAUTH_CLIENT_ID / SEO_GOOGLE_OAUTH_CLIENT_SECRET missing in config/.env. "
            "See docs/seo/OAUTH_SETUP.md."
        )
    return {
        "client_id": settings.seo_google_oauth_client_id,
        "client_secret": settings.seo_google_oauth_client_secret,
        "token_path": settings.seo_oauth_token_path,
    }


def _client_secrets_dict(client_id: str, client_secret: str) -> dict:
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:0/"],
        }
    }


def get_credentials() -> Credentials:
    """Load + refresh credentials. Raises if no token file present yet."""
    cfg = _config()
    token_path = Path(cfg["token_path"])
    if not token_path.exists():
        raise RuntimeError(
            f"OAuth token not found at {token_path}. Run `python -m integrations.google_seo.oauth authorize` first."
        )
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        else:
            raise RuntimeError("OAuth token invalid + no refresh token — re-run authorize.")
    return creds


def authorize_interactive() -> None:
    """Run the installed-app flow on a TTY. Saves refresh token to disk."""
    cfg = _config()
    flow = InstalledAppFlow.from_client_config(
        _client_secrets_dict(cfg["client_id"], cfg["client_secret"]),
        SCOPES,
    )
    creds = flow.run_local_server(port=0, open_browser=True)
    token_path = Path(cfg["token_path"])
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    print(f"Token saved to {token_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "authorize":
        authorize_interactive()
    else:
        print("Usage: python -m integrations.google_seo.oauth authorize")
        sys.exit(1)
