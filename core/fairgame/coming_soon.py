"""Coming-soon waitlist signup → Klaviyo.

Two steps, because Klaviyo splits identity from consent:
  1. Upsert the profile  (email, first/last name, phone) — create, or PATCH the
     existing profile on a 409 duplicate.
  2. Subscribe the email to the waitlist list (id from FAIRGAME_WAITLIST_LIST_ID)
     with marketing consent. When that list is set to Double Opt-In in Klaviyo,
     the subscription job sends the confirmation email automatically, so fans
     confirm before we ever contact them.

Phone is stored as phone_number when it parses to E.164, else as a phone_raw
property so an odd number never fails the signup.

Stdlib only (urllib) so the 109 deploy needs no extra packages.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

_BASE = "https://a.klaviyo.com/api"
_REVISION = "2024-10-15"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_E164_RE = re.compile(r"^\+\d{8,15}$")


def configured() -> bool:
    return bool(os.environ.get("KLAVIYO_PRIVATE_KEY") and os.environ.get("FAIRGAME_WAITLIST_LIST_ID"))


def _norm_phone(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw or "")
    if not digits:
        return ""
    if not digits.startswith("+"):
        d = re.sub(r"\D", "", digits)
        if len(d) == 10:
            digits = "+1" + d
        elif len(d) == 11 and d.startswith("1"):
            digits = "+" + d
        else:
            digits = "+" + d
    return digits


def _req(method: str, path: str, body: dict):
    """Returns (status, parsed_json_or_None, error_text_or_None)."""
    key = os.environ.get("KLAVIYO_PRIVATE_KEY", "")
    req = urllib.request.Request(
        f"{_BASE}{path}",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Klaviyo-API-Key {key}",
            "revision": _REVISION,
            "content-type": "application/json",
            "accept": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode() or "{}"
            return r.status, json.loads(raw), None
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode()
            return e.code, json.loads(raw), raw[:400]
        except Exception:  # noqa: BLE001
            return e.code, None, str(e)
    except Exception as e:  # noqa: BLE001
        return 0, None, str(e)


def _profile_attrs(first, last, email, phone) -> dict:
    attrs: dict = {"email": email}
    if (first or "").strip():
        attrs["first_name"] = first.strip()[:60]
    if (last or "").strip():
        attrs["last_name"] = last.strip()[:60]
    phone_n = _norm_phone(phone)
    if phone_n:
        if _E164_RE.match(phone_n):
            attrs["phone_number"] = phone_n
        else:
            attrs["properties"] = {"phone_raw": (phone or "").strip()[:40]}
    return attrs


def _upsert_profile(first, last, email, phone):
    """Create the profile, or PATCH the existing one on duplicate. Returns (ok, err)."""
    attrs = _profile_attrs(first, last, email, phone)
    status, data, err = _req("POST", "/profiles/", {"data": {"type": "profile", "attributes": attrs}})
    if status in (200, 201):
        return True, None
    if status == 409 and data:
        # Already exists — pull the id and update names/phone.
        dup_id = (
            data.get("errors", [{}])[0].get("meta", {}).get("duplicate_profile_id")
            if data.get("errors") else None
        )
        if dup_id:
            # email is an identifier on PATCH; keep it out of the writable attrs is fine to leave in.
            st2, _, err2 = _req(
                "PATCH", f"/profiles/{dup_id}/",
                {"data": {"type": "profile", "id": dup_id, "attributes": attrs}},
            )
            if st2 in (200, 201):
                return True, None
            return False, f"patch {st2}: {err2}"
    return False, f"create {status}: {err}"


def _subscribe_email(email):
    """Subscribe the email to the waitlist list (double opt-in honored). Returns (ok, err)."""
    list_id = os.environ.get("FAIRGAME_WAITLIST_LIST_ID", "")
    body = {
        "data": {
            "type": "profile-subscription-bulk-create-job",
            "attributes": {
                "custom_source": "Fans First Coming Soon",
                "profiles": {
                    "data": [{
                        "type": "profile",
                        "attributes": {
                            "email": email,
                            "subscriptions": {"email": {"marketing": {"consent": "SUBSCRIBED"}}},
                        },
                    }]
                },
            },
            "relationships": {"list": {"data": {"type": "list", "id": list_id}}},
        }
    }
    status, _, err = _req("POST", "/profile-subscription-bulk-create-jobs/", body)
    if status in (200, 201, 202):
        return True, None
    return False, f"subscribe {status}: {err}"


def subscribe(first_name: str, last_name: str, email: str, phone: str):
    """Capture a waitlist signup. Returns (ok: bool, error: str|None)."""
    if not configured():
        return False, "Waitlist isn't configured yet."
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        return False, "A valid email is required."

    ok, err = _upsert_profile(first_name, last_name, email, phone)
    if not ok:
        return False, err
    return _subscribe_email(email)
