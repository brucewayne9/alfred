"""Mainstay Forge — default job handlers + registration.

register_default_handlers() wires the built-in handlers into the queue. The demo
'echo' handler proves the queue end-to-end; real format renderers (kinetic-lyric,
montage, leak-graphic) will register here in later plans.

ACTIVATION (apply manually to core/api/main.py once that file is settled — do NOT
auto-apply; main.py currently carries unrelated uncommitted work). Alongside the
other register(app) calls:

    from core.api.forge import register as _register_forge
    from core.forge.handlers import register_default_handlers
    _register_forge(app)
    register_default_handlers()

and in the lifespan startup / shutdown:

    import asyncio
    from core.forge import jobs as _forge_jobs
    forge_worker = asyncio.create_task(_forge_jobs.worker_loop())   # startup, before yield
    forge_worker.cancel()                                           # shutdown, after yield
"""
from core.forge import jobs as forge_jobs


def _echo_handler(params: dict) -> dict:
    return {"echo": params}


def _run_remix_format(render, params: dict, *, fmt: str, default_subfolder: str) -> dict:
    """Render R remix looks, stealth-multiply each by `variations`, deliver per-look."""
    import tempfile, time, shutil
    from pathlib import Path
    from core.forge.remix import build_remixes
    from core.forge.multiply import multiply
    from core.forge import delivery

    remixes = build_remixes(params, int(params.get("remix", 1) or 1))
    n = int(params.get("variations", 18) or 18)
    base = (params.get("subfolder") or default_subfolder).strip()
    stamp = int(time.time())
    ext = ".png" if fmt == "leak_graphic" else ".mp4"
    total_delivered, look_dirs = 0, []
    work = Path(tempfile.mkdtemp(prefix=f"forge_{fmt}_"))
    try:
        for rp in remixes:
            ri = rp.get("remix_index", 0)
            label = f"{fmt}_{stamp}_look{ri:02d}" if len(remixes) > 1 else f"{fmt}_{stamp}"
            master = render(rp, work / f"{label}_master{ext}")
            variants = multiply(master, n, work / f"v{ri}", base_name=label,
                                 allow_flip=(fmt != "leak_graphic")) if n else []
            dest = f"{base}/{label}"
            look_delivered = 0
            try:
                delivery.deliver(master, dest, filename=master.name)
                look_delivered += 1
            except Exception:  # noqa: BLE001
                pass
            for v in variants:
                try:
                    delivery.deliver(v, dest); look_delivered += 1
                except Exception:  # noqa: BLE001
                    pass
            # Only record a delivered dir if something actually landed — otherwise
            # the Library points at a folder Nextcloud never created (→ 404/500).
            if look_delivered:
                look_dirs.append(f"Content/Mainstay-RodWave/{dest}")
                total_delivered += look_delivered
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"format": fmt, "remix_looks": len(remixes), "variations_each": n,
            "delivered": total_delivered, "delivered_dirs": look_dirs}


def _leak_graphic_handler(params: dict) -> dict:
    from core.forge.renderers.leak_graphic import render
    return _run_remix_format(render, params, fmt="leak_graphic",
                             default_subfolder="Viral Album Videos/Processed")


def _kinetic_lyric_handler(params: dict) -> dict:
    from core.forge.renderers.kinetic_lyric import render
    return _run_remix_format(render, params, fmt="kinetic_lyric",
                             default_subfolder="Viral Music Verticals/Kinetic Lyric")


def _film_montage_handler(params: dict) -> dict:
    from core.forge.renderers.film_montage import render
    return _run_remix_format(render, params, fmt="film_montage",
                             default_subfolder="Viral Music Verticals/Film Montage")


def register_default_handlers() -> None:
    """Register all built-in Forge job handlers into the queue."""
    forge_jobs.register_handler("echo", _echo_handler)
    forge_jobs.register_handler("leak_graphic", _leak_graphic_handler)
    forge_jobs.register_handler("kinetic_lyric", _kinetic_lyric_handler)
    forge_jobs.register_handler("film_montage", _film_montage_handler)
