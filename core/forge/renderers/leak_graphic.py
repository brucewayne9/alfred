"""Leak-Graphic renderer — a Rod-Wave-style fake album cover via ComfyUI + Remotion.

The "leak graphic" is the single-image post format (fake album cover / engagement
bait). We generate the cover ART from the caption via ComfyUI Cloud (cloud-first,
local GPU fallback handled inside run_comfyui_cloud), then bake the title +
tracklist text over it using the Remotion `LeakCardRig` still composition.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
REMOTION = Path("/home/aialfred/remotion")


def _node_env() -> dict:
    """Env for the Remotion subprocess with a modern Node on PATH.

    Under systemd the service PATH has no nvm, so `npx` falls back to the
    system Node (v12), which Remotion can't run. Prepend the newest nvm
    Node >= 18 so the renderer works regardless of ambient PATH.
    """
    env = dict(os.environ)
    candidates = []
    nvm = Path("/home/aialfred/.nvm/versions/node")
    if nvm.is_dir():
        for d in nvm.iterdir():
            m = re.match(r"v(\d+)\.", d.name)
            if m and int(m.group(1)) >= 18 and (d / "bin" / "node").exists():
                candidates.append((int(m.group(1)), d / "bin"))
    if candidates:
        best = max(candidates)[1]
        env["PATH"] = f"{best}:{env.get('PATH', '')}"
    return env


def build_prompt(caption: str, override: str | None = None) -> str:
    """Turn the user's caption into a cover-art prompt (no literal text in the image).

    Args:
        caption: The post caption used to theme the cover art.
        override: Optional vessel-mood string (e.g. from Remix). When non-empty,
            the override drives the primary visual theme; caption is included as
            secondary context. When absent, caption alone drives the theme.
    """
    caption = (caption or "").strip()
    base = (
        "cinematic melancholic rap album cover, moody dramatic low-key lighting, "
        "deep shadows, film grain, 35mm, emotional portrait energy, lonely night mood, "
        "muted gold and charcoal palette, vertical poster composition, high detail, no text"
    )
    vessel = (override or "").strip()
    if vessel:
        suffix = f", vessel mood: {vessel}"
        if caption:
            suffix += f", themed around: {caption}"
        return f"{base}{suffix}"
    return f"{base}, themed around: {caption}" if caption else base


def _normalize_tracklist(tracklist) -> list[str]:
    """Coerce a tracklist input (str | list | None) into a clean list (cap 7)."""
    default = ["Intro", "Same Old Pain", "25", "Don't Look Down", "Houston"]
    if tracklist is None:
        items = default
    elif isinstance(tracklist, str):
        parts = tracklist.split("\n") if "\n" in tracklist else tracklist.split(",")
        items = [p.strip() for p in parts if p.strip()]
    elif isinstance(tracklist, (list, tuple)):
        items = [str(p).strip() for p in tracklist if str(p).strip()]
    else:
        items = []
    if not items:
        items = default
    return items[:7]


def render(params: dict, out_path: str | Path) -> Path:
    """Render the texted leak-graphic cover to `out_path` (PNG). Raises on failure.

    Pipeline: ComfyUI-Cloud cover art (1080x1920) -> copy into Remotion public ->
    Remotion `LeakCardRig` still with title + tracklist baked over the art.
    """
    out_path = Path(out_path).resolve()  # absolute: subprocess runs with cwd=REMOTION
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="forge_leak_render_"))

    # 1. Generate cover art via ComfyUI Cloud (same lazy-import pattern as kinetic)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.rucktalk_common import run_comfyui_cloud  # lazy: heavy import

    prompt = build_prompt(params.get("caption", ""), override=params.get("vessel_prompt"))
    result = run_comfyui_cloud(prompt, width=1080, height=1920)  # 9:16 portrait
    if not result:
        raise RuntimeError("ComfyUI Cloud returned no image")
    src = Path(result)
    if not src.exists():
        raise RuntimeError(f"ComfyUI image missing on disk: {src}")
    art = work / "art.png"
    art.write_bytes(src.read_bytes())

    # 2. Parse text inputs
    title = (params.get("title") or params.get("caption") or "Don't Look Down").strip()
    tracklist = _normalize_tracklist(params.get("tracklist"))
    tag = params.get("tag", "LEAKED")

    # 3. Copy art into Remotion public dir under a unique name (always cleaned up)
    pub_name = f"forge_leak_{uuid.uuid4().hex}.png"
    pub_path = REMOTION / "public" / pub_name
    try:
        shutil.copyfile(art, pub_path)

        # 4. Props for LeakCardRig
        props = {
            "bgImage": pub_name,
            "title": title,
            "tracklist": tracklist,
            "tag": tag,
        }
        props_path = work / "props.json"
        props_path.write_text(json.dumps(props))

        # 5. Render the still via Remotion
        rr = subprocess.run(
            ["npx", "remotion", "still", "src/index.ts", "LeakCardRig",
             str(out_path), f"--props={props_path}"],
            cwd=str(REMOTION), capture_output=True, text=True, timeout=600,
            env=_node_env())
        if rr.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"remotion still failed: {rr.stderr[-800:]}")
    finally:
        try:
            pub_path.unlink()
        except OSError:
            pass
        shutil.rmtree(work, ignore_errors=True)

    return Path(out_path)
