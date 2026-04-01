#!/usr/bin/env python3
"""Auto-refresh Meta long-lived access token (60-day expiry).

Reads current token from openclaw.json, exchanges for a new long-lived token,
and writes it back. Run via cron every 30 days.
"""
import json
import urllib.request
import urllib.parse
import sys
from datetime import datetime

OPENCLAW_JSON = "/home/aialfred/.openclaw/openclaw.json"

def refresh():
    with open(OPENCLAW_JSON) as f:
        cfg = json.load(f)

    env = cfg["env"]["vars"]
    current_token = env.get("META_ACCESS_TOKEN", "")
    app_id = env.get("META_APP_ID", "")
    app_secret = env.get("META_APP_SECRET", "")

    if not all([current_token, app_id, app_secret]):
        print("ERROR: Missing META_ACCESS_TOKEN, META_APP_ID, or META_APP_SECRET in openclaw.json")
        sys.exit(1)

    params = urllib.parse.urlencode({
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": current_token,
    })
    url = f"https://graph.facebook.com/v21.0/oauth/access_token?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"ERROR: Token refresh failed: {e}")
        sys.exit(1)

    new_token = data.get("access_token")
    expires_in = data.get("expires_in", 0)

    if not new_token:
        print(f"ERROR: No token in response: {data}")
        sys.exit(1)

    env["META_ACCESS_TOKEN"] = new_token

    with open(OPENCLAW_JSON, "w") as f:
        json.dump(cfg, f, indent=2)

    days = expires_in // 86400
    print(f"{datetime.now().isoformat()} — Meta token refreshed. Expires in {days} days.")

if __name__ == "__main__":
    refresh()
