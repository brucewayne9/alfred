"""
PayPal Tracking API client.

Used by the Roen Telegram bot's /orders flow to push USPS (UPS / FedEx
stubbed in) tracking numbers onto PayPal transactions so seller
protection events fire and the buyer sees tracking in their PayPal app.

Per CLAUDE.md feedback `never-read-env-at-import`: this module does NOT
read os.environ at import. Construct a client with explicit credentials
via `get_client_from_env(env_dict)` from the bot's runtime config loader.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_LIVE_BASE = "https://api-m.paypal.com"
_SANDBOX_BASE = "https://api-m.sandbox.paypal.com"


class PayPalTrackingError(RuntimeError):
    """Raised when PayPal Tracking API rejects a call."""


@dataclass
class TrackerResult:
    transaction_id: str
    tracking_number: str
    carrier: str
    status: str  # "SHIPPED" / "ON_HOLD" / "DELIVERED" / "CANCELLED"
    raw: dict


# Carrier codes PayPal accepts in trackers-batch payload.
# Reference: https://developer.paypal.com/api/tracking/v1/#definition-carrier
_CARRIER_CODES = {
    "USPS": "USPS",
    "UPS": "UPS",
    "FEDEX": "FEDEX",
    "FED EX": "FEDEX",
    "FED-EX": "FEDEX",
    "DHL": "DHL",
}


def detect_carrier(tracking: str) -> str:
    """
    Best-effort carrier detection from tracking-number format.
    Returns one of USPS / UPS / FEDEX. Defaults to USPS — Roen primary carrier.

    USPS — 20-22 digits, common prefixes 94/93/92/82, or 13-char Intl (e.g. EA..US).
    UPS  — `1Z` + 16 alphanumeric (18 chars total).
    FedEx — 12, 15, 20, or 22 pure digits (less reliable; USPS overlaps).
    """
    t = re.sub(r"\s+", "", tracking).upper()
    if t.startswith("1Z") and len(t) == 18 and t[2:].isalnum():
        return "UPS"
    if re.match(r"^[A-Z]{2}\d{9}[A-Z]{2}$", t):  # EA123456789US
        return "USPS"
    if t.isdigit():
        if len(t) in (20, 22) and t[:2] in ("94", "93", "92", "82", "70"):
            return "USPS"
        if len(t) in (12, 15):
            return "FEDEX"
        return "USPS"  # default — most Roen shipments are USPS
    return "USPS"


def normalize_tracking(tracking: str) -> str:
    """Strip whitespace and uppercase. PayPal's API is picky about formatting."""
    return re.sub(r"\s+", "", tracking).upper()


class PayPalTrackingClient:
    """
    Lightweight client for PayPal's `/v1/shipping/trackers-batch` endpoint.

    Token caching is in-memory with a safety margin; the OAuth token's
    `expires_in` is ~9h on live so we refresh proactively a few minutes
    before expiry to avoid mid-call expiry races.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        env: str = "live",
        *,
        session: Optional[requests.Session] = None,
        timeout: int = 20,
    ):
        if not client_id or not client_secret:
            raise ValueError("PayPal client_id and client_secret are required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.base = _LIVE_BASE if env.lower() == "live" else _SANDBOX_BASE
        self.timeout = timeout
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    # ---------- auth ----------

    def _refresh_token(self) -> str:
        r = self._session.post(
            f"{self.base}/v1/oauth2/token",
            headers={"Accept": "application/json", "Accept-Language": "en_US"},
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise PayPalTrackingError(f"OAuth refresh failed {r.status_code}: {r.text[:200]}")
        data = r.json()
        self._token = data["access_token"]
        # refresh 5 min before expiry to avoid races
        self._token_expires_at = time.time() + max(60, int(data.get("expires_in", 32400)) - 300)
        return self._token

    def _auth_header(self) -> dict:
        with self._lock:
            if not self._token or time.time() >= self._token_expires_at:
                self._refresh_token()
            return {"Authorization": f"Bearer {self._token}"}

    # ---------- tracking ----------

    def add_tracker(
        self,
        transaction_id: str,
        tracking_number: str,
        carrier: str = "USPS",
        *,
        notify_buyer: bool = True,
        status: str = "SHIPPED",
    ) -> TrackerResult:
        """
        POST a single tracker to PayPal.

        `transaction_id` is PayPal's capture/transaction ID for the order — NOT
        the WooCommerce order ID. The Roen WC integration stores PayPal's
        transaction ID on the WC order meta as `_paypal_transaction_id` (set by
        the WC PayPal Payments plugin on successful capture).
        """
        carrier_code = _CARRIER_CODES.get(carrier.upper(), carrier.upper())
        if carrier_code not in {"USPS", "UPS", "FEDEX", "DHL"}:
            raise ValueError(f"Unsupported carrier: {carrier!r}")
        if status not in {"SHIPPED", "ON_HOLD", "DELIVERED", "CANCELLED"}:
            raise ValueError(f"Invalid tracker status: {status!r}")

        body = {
            "trackers": [
                {
                    "transaction_id": transaction_id,
                    "tracking_number": normalize_tracking(tracking_number),
                    "status": status,
                    "carrier": carrier_code,
                    "notify_buyer": notify_buyer,
                }
            ]
        }

        r = self._session.post(
            f"{self.base}/v1/shipping/trackers-batch",
            headers={
                **self._auth_header(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )

        if r.status_code in (200, 201, 207):
            data = r.json()
            trackers = data.get("tracker_identifiers") or data.get("trackers") or []
            errors = data.get("errors") or []
            if errors:
                raise PayPalTrackingError(
                    f"PayPal accepted batch but returned errors: {errors}"
                )
            return TrackerResult(
                transaction_id=transaction_id,
                tracking_number=normalize_tracking(tracking_number),
                carrier=carrier_code,
                status=status,
                raw=data,
            )

        # Treat 422 (already exists) as success — idempotent re-ship.
        if r.status_code == 422 and "TRANSACTION_ALREADY_TRACKED" in r.text:
            logger.info(
                "PayPal already has tracking for %s — treating as success",
                transaction_id,
            )
            return TrackerResult(
                transaction_id=transaction_id,
                tracking_number=normalize_tracking(tracking_number),
                carrier=carrier_code,
                status=status,
                raw={"already_tracked": True},
            )

        raise PayPalTrackingError(
            f"trackers-batch failed {r.status_code}: {r.text[:300]}"
        )


def get_client_from_env(env: dict) -> PayPalTrackingClient:
    """
    Construct a client from a Roen-bot env dict.

    Expects ROEN_BOT_PAYPAL_CLIENT_ID / _CLIENT_SECRET / _ENV in the dict.
    Raises KeyError if missing — caller should surface to the operator.
    """
    return PayPalTrackingClient(
        client_id=env["ROEN_BOT_PAYPAL_CLIENT_ID"],
        client_secret=env["ROEN_BOT_PAYPAL_CLIENT_SECRET"],
        env=env.get("ROEN_BOT_PAYPAL_ENV", "live"),
    )
