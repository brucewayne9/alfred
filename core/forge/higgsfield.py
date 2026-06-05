"""Higgsfield CLI bridge for Forge — AI image/video on the Mainstay Ultra subscription.

Forge shells out to the official ``higgsfield`` CLI (device-login auth that draws from
the account's *subscription* credits — NOT the metered developer API). This mirrors the
``run_comfyui_cloud`` contract: each generator returns a local file path on success or
``None`` on failure, so renderers keep ComfyUI as a free fallback.

Verified live 2026-06-05 on dharmic@mainstaymusicgroup.com (ultra, 6060 cr):

    higgsfield generate create <job_set_type> --prompt "..." [--image PATH] \
        [--aspect_ratio 9:16] [--resolution 4k] --wait --json
    -> [{"status":"completed","result_url":"https://...cloudfront.net/x.png", ...}]

Models / cost (per generation):
    image  nano_banana_2 (Nano Banana Pro 4K) = 2 cr
           params: prompt(req), aspect_ratio[auto,1:1,9:16,16:9,...], resolution[1k,2k,4k], input_images[]
    video  kling3_0 (Kling v3.0) = 10 cr
           params: prompt(req), aspect_ratio[16:9,9:16,1:1], duration(int 5), mode[std,pro,4k], sound[on,off]
           start frame via --image / --start-image (local path auto-uploaded)

Auth tokens are short-lived. On an auth failure we log a clear marker and return ``None``
so callers fall back to ComfyUI; the watchdog can surface a "re-run higgsfield auth login"
ping to Mike.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Default model picks (the premium look Mike asked for).
DEFAULT_IMAGE_MODEL = "nano_banana_2"  # Nano Banana Pro (4K, flawless text)
DEFAULT_VIDEO_MODEL = "kling3_0"       # Kling v3.0 (image-to-video)

# Candidate locations for the globally-installed CLI binary (npm/nvm global bin).
_CLI_CANDIDATES = (
    "higgsfield",  # PATH
    os.path.expanduser("~/.nvm/versions/node/v22.22.0/bin/higgsfield"),
    "/usr/local/bin/higgsfield",
    os.path.expanduser("~/.npm-global/bin/higgsfield"),
)


def _cli_path() -> str | None:
    """Resolve the higgsfield CLI binary, or None if not installed."""
    which = shutil.which("higgsfield")
    if which:
        return which
    for cand in _CLI_CANDIDATES:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def is_available() -> bool:
    """True if the CLI binary is present (does not check auth/credits)."""
    return _cli_path() is not None


def _run_cli(args: list[str], timeout: int) -> str | None:
    """Run a higgsfield CLI subcommand; return stdout on success, None on failure.

    Logs a distinct AUTH-EXPIRED marker when the failure looks like an expired token
    so the watchdog can prompt Mike to re-run ``higgsfield auth login``.
    """
    cli = _cli_path()
    if not cli:
        logger.error("higgsfield CLI not found — install with: npm i -g @higgsfield/cli")
        return None
    env = os.environ.copy()
    # Ensure the node bin dir is on PATH so the CLI's own resolution works under systemd.
    env["PATH"] = env.get("PATH", "") + os.pathsep + str(Path(cli).parent)
    try:
        proc = subprocess.run(
            [cli, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("higgsfield CLI timed out after %ss: %s", timeout, " ".join(args[:3]))
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("higgsfield CLI invocation error: %s", str(exc)[:200])
        return None

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        logger.error("higgsfield CLI failed (rc=%s): %s", proc.returncode, err[-400:])
        low = err.lower()
        if any(tok in low for tok in ("auth", "login", "unauthor", "token", "expired", "401")):
            logger.error("HIGGSFIELD_AUTH_EXPIRED — run `higgsfield auth login` (token is short-lived)")
        return None
    return proc.stdout


def account_status(timeout: int = 30) -> str | None:
    """Return the raw `account status` line (email — plan, N credits), or None."""
    out = _run_cli(["account", "status"], timeout)
    return out.strip() if out else None


def credits_remaining(timeout: int = 30) -> int | None:
    """Parse available subscription credits from `account status`, or None."""
    out = account_status(timeout)
    if not out:
        return None
    m = re.search(r"(\d[\d,]*)\s*credits", out)
    return int(m.group(1).replace(",", "")) if m else None


def estimate_cost(model: str, prompt: str, *, image: str | Path | None = None,
                  timeout: int = 60) -> int | None:
    """Estimate credits for a generation WITHOUT spending any (built-in cost gate)."""
    args = ["generate", "cost", model, "--prompt", prompt, "--json"]
    if image:
        args += ["--image", str(image)]
    out = _run_cli(args, timeout)
    if not out:
        return None
    out = out.strip()
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            for k in ("credits", "cost", "total", "total_credits"):
                if isinstance(data.get(k), (int, float)):
                    return int(data[k])
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d+)", out)  # falls back to "2 credits"
    return int(m.group(1)) if m else None


def _result_url(stdout: str) -> str | None:
    """Pull the completed result_url out of `generate create --json` output."""
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError) as exc:
        logger.error("higgsfield JSON parse failed: %s | %.200s", exc, stdout)
        return None
    jobs = data if isinstance(data, list) else [data]
    # Prefer an explicitly completed job, then any job carrying a result_url.
    for prefer_completed in (True, False):
        for job in jobs:
            if not isinstance(job, dict):
                continue
            if prefer_completed and job.get("status") != "completed":
                continue
            url = job.get("result_url") or job.get("url")
            if url:
                return url
    logger.error("higgsfield returned no result_url: %.300s", stdout)
    return None


def _download(url: str, out_path: str | Path) -> Path | None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.stream("GET", url, timeout=180, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
    except Exception as exc:
        logger.error("higgsfield asset download failed: %s", str(exc)[:200])
        return None
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    logger.error("higgsfield download produced empty file: %s", out_path)
    return None


def _generate(model: str, prompt: str, out_path: str | Path, *,
              media: str | Path | None = None, media_flag: str = "--image",
              params: dict | None = None, timeout: int = 600) -> Path | None:
    args = ["generate", "create", model, "--prompt", prompt, "--wait", "--json"]
    if media:
        args += [media_flag, str(media)]
    for key, val in (params or {}).items():
        if val is not None:
            args += [f"--{key}", str(val)]
    out = _run_cli(args, timeout)
    if not out:
        return None
    url = _result_url(out)
    if not url:
        return None
    return _download(url, out_path)


def generate_image(prompt: str, out_path: str | Path, *,
                   model: str = DEFAULT_IMAGE_MODEL,
                   aspect_ratio: str | None = "9:16",
                   resolution: str | None = "4k",
                   reference_image: str | Path | None = None,
                   timeout: int = 300) -> Path | None:
    """Generate a still (default Nano Banana Pro 4K, 9:16). Returns local path or None.

    Pass ``reference_image`` (local path) to condition on an uploaded image.
    """
    return _generate(
        model, prompt, out_path,
        media=reference_image, media_flag="--image",
        params={"aspect_ratio": aspect_ratio, "resolution": resolution},
        timeout=timeout,
    )


def generate_video(prompt: str, out_path: str | Path, *,
                   model: str = DEFAULT_VIDEO_MODEL,
                   start_image: str | Path | None = None,
                   aspect_ratio: str | None = "9:16",
                   duration: int | None = None,
                   mode: str | None = None,
                   sound: str | None = None,
                   timeout: int = 900) -> Path | None:
    """Generate a clip (default Kling v3.0). ``start_image`` is the image-to-video frame.

    Returns local path or None. Kling needs a start frame for image-to-video; a local
    path is auto-uploaded by the CLI.
    """
    return _generate(
        model, prompt, out_path,
        media=start_image, media_flag="--start-image",
        params={
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "mode": mode,
            "sound": sound,
        },
        timeout=timeout,
    )
