"""Shared config loader for OpenClaw integrations.

Usage in any integration script:
    from openclaw_env import env

    API_KEY = env("MY_API_KEY")
    # Checks os.environ first, then ~/.openclaw/openclaw.json env.vars
"""
import os, json

_openclaw_vars = None

def _load_openclaw_vars():
    global _openclaw_vars
    if _openclaw_vars is not None:
        return _openclaw_vars
    _openclaw_vars = {}
    for path in [
        os.path.expanduser("~/.openclaw/openclaw.json"),
        "/home/brucewayne9/.openclaw/openclaw.json",
    ]:
        try:
            with open(path) as f:
                data = json.load(f)
                _openclaw_vars = data.get("env", {}).get("vars", {})
                return _openclaw_vars
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return _openclaw_vars

def env(key, default=""):
    """Get config value: os.environ first, then openclaw.json, then default."""
    val = os.environ.get(key, "")
    if val:
        return val
    return _load_openclaw_vars().get(key, default)
