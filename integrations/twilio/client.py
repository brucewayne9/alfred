"""Twilio client for SMS and Voice communications."""

import hashlib
import hmac
import logging
import os
from base64 import b64encode
from urllib.parse import urlencode

from twilio.rest import Client
from twilio.request_validator import RequestValidator
from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")

logger = logging.getLogger(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Initialize client
_client = None
_validator = None


def _get_client() -> Client:
    """Get or create Twilio client."""
    global _client
    if _client is None:
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            raise ValueError("Twilio credentials not configured")
        _client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    return _client


def _get_validator() -> RequestValidator:
    """Get or create request validator."""
    global _validator
    if _validator is None:
        if not TWILIO_AUTH_TOKEN:
            raise ValueError("Twilio auth token not configured")
        _validator = RequestValidator(TWILIO_AUTH_TOKEN)
    return _validator


def validate_twilio_signature(url: str, params: dict, signature: str) -> bool:
    """Validate that a request came from Twilio.

    Args:
        url: The full URL of the webhook
        params: The POST parameters from the request
        signature: The X-Twilio-Signature header value

    Returns:
        True if the signature is valid, False otherwise
    """
    try:
        validator = _get_validator()
        return validator.validate(url, params, signature)
    except Exception as e:
        logger.error(f"Twilio signature validation error: {e}")
        return False


def send_sms(to: str, message: str, from_number: str = None) -> dict:
    """Send an SMS message.

    Args:
        to: Recipient phone number (E.164 format, e.g., +1234567890)
        message: The message text
        from_number: Optional sender number (defaults to TWILIO_PHONE_NUMBER)

    Returns:
        dict with status, sid, and any error info
    """
    try:
        client = _get_client()
        from_num = from_number or TWILIO_PHONE_NUMBER

        if not from_num:
            return {"success": False, "error": "No sender phone number configured"}

        # Ensure proper format
        if not to.startswith("+"):
            to = f"+1{to.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')}"

        msg = client.messages.create(
            body=message,
            from_=from_num,
            to=to
        )

        logger.info(f"SMS sent to {to}: {msg.sid}")
        return {
            "success": True,
            "sid": msg.sid,
            "status": msg.status,
            "to": to,
            "message": f"SMS sent to {to}"
        }

    except Exception as e:
        logger.error(f"SMS send failed: {e}")
        return {"success": False, "error": str(e)}


def make_call(to: str, message: str = None, twiml_url: str = None, from_number: str = None) -> dict:
    """Make an outbound voice call using Kokoro TTS.

    Args:
        to: Recipient phone number (E.164 format)
        message: Text to speak (uses Kokoro TTS via Alfred API)
        twiml_url: URL returning TwiML (alternative to message)
        from_number: Optional caller ID (defaults to TWILIO_PHONE_NUMBER)

    Returns:
        dict with call sid and status
    """
    try:
        client = _get_client()
        from_num = from_number or TWILIO_PHONE_NUMBER

        if not from_num:
            return {"success": False, "error": "No caller phone number configured"}

        # Ensure proper format
        if not to.startswith("+"):
            to = f"+1{to.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')}"

        call_params = {
            "from_": from_num,
            "to": to,
        }

        if twiml_url:
            call_params["url"] = twiml_url
        elif message:
            # Use Kokoro TTS via Alfred API endpoint
            # The endpoint generates TwiML with Play using Kokoro-generated audio
            from urllib.parse import quote
            base_url = "https://aialfred.groundrushcloud.com"
            encoded_message = quote(message, safe='')
            call_params["url"] = f"{base_url}/webhooks/twilio/voice/outbound?message={encoded_message}"
        else:
            return {"success": False, "error": "Must provide message or twiml_url"}

        call = client.calls.create(**call_params)

        logger.info(f"Call initiated to {to}: {call.sid}")
        return {
            "success": True,
            "sid": call.sid,
            "status": call.status,
            "to": to,
            "message": f"Call initiated to {to}"
        }

    except Exception as e:
        logger.error(f"Call failed: {e}")
        return {"success": False, "error": str(e)}


def get_call_status(call_sid: str) -> dict:
    """Get the status of a call.

    Args:
        call_sid: The SID of the call to check

    Returns:
        dict with call status info
    """
    try:
        client = _get_client()
        call = client.calls(call_sid).fetch()

        return {
            "success": True,
            "sid": call.sid,
            "status": call.status,
            "duration": call.duration,
            "direction": call.direction,
            "from": call.from_,
            "to": call.to,
            "start_time": str(call.start_time) if call.start_time else None,
            "end_time": str(call.end_time) if call.end_time else None,
        }

    except Exception as e:
        logger.error(f"Get call status failed: {e}")
        return {"success": False, "error": str(e)}


def get_messages(limit: int = 10, to: str = None, from_: str = None) -> dict:
    """Get recent SMS messages.

    Args:
        limit: Maximum messages to return
        to: Filter by recipient
        from_: Filter by sender

    Returns:
        dict with list of messages
    """
    try:
        client = _get_client()

        params = {"limit": limit}
        if to:
            params["to"] = to
        if from_:
            params["from_"] = from_

        messages = client.messages.list(**params)

        return {
            "success": True,
            "messages": [
                {
                    "sid": m.sid,
                    "from": m.from_,
                    "to": m.to,
                    "body": m.body,
                    "status": m.status,
                    "direction": m.direction,
                    "date_sent": str(m.date_sent) if m.date_sent else None,
                }
                for m in messages
            ]
        }

    except Exception as e:
        logger.error(f"Get messages failed: {e}")
        return {"success": False, "error": str(e)}


def health_check() -> dict:
    """Check Twilio connection status."""
    try:
        client = _get_client()
        account = client.api.accounts(TWILIO_ACCOUNT_SID).fetch()

        return {
            "success": True,
            "status": "connected",
            "account_name": account.friendly_name,
            "account_status": account.status,
            "phone_number": TWILIO_PHONE_NUMBER,
        }

    except Exception as e:
        logger.error(f"Twilio health check failed: {e}")
        return {"success": False, "status": "disconnected", "error": str(e)}
