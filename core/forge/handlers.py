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
import shutil
import tempfile
import time
from pathlib import Path

from core.forge import jobs as forge_jobs


def _echo_handler(params: dict) -> dict:
    return {"echo": params}


def _leak_graphic_handler(params: dict) -> dict:
    """Forge it (Leak Graphic): ComfyUI-Cloud cover art -> N variants -> Nextcloud."""
    from core.forge.renderers.leak_graphic import render
    from core.forge.multiply import multiply
    from core.forge import delivery

    label = f"leak_{int(time.time())}"
    work = Path(tempfile.mkdtemp(prefix="forge_"))
    master = render(params, work / f"{label}_master.png")

    n = int(params.get("variations", 18) or 18)
    variants = multiply(master, n, work / "variants", base_name=label)

    # "Viral Album Videos / Processed" (dashboard label) -> real WebDAV subpath
    subfolder = (params.get("subfolder") or "Viral Album Videos/Processed").replace(" / ", "/").strip()
    dest = f"{subfolder}/{label}"

    delivered = []
    try:
        delivered.append(delivery.deliver(master, dest, filename=f"{label}_master.png"))
    except Exception as e:  # noqa: BLE001 — record but keep going
        delivered.append(f"ERROR master: {e}")
    for v in variants:
        try:
            delivered.append(delivery.deliver(v, dest))
        except Exception:  # noqa: BLE001
            pass

    return {
        "format": "leak_graphic",
        "master": str(master),
        "variant_count": len(variants),
        "delivered_dir": f"Content/Mainstay-RodWave/{dest}",
        "delivered": len([d for d in delivered if not str(d).startswith("ERROR")]),
    }


def _kinetic_lyric_handler(params: dict) -> dict:
    """Forge it (Kinetic Lyric): hook audio -> word-timed lyric vertical -> N variants -> Nextcloud."""
    import tempfile, time
    from pathlib import Path
    from core.forge.renderers.kinetic_lyric import render
    from core.forge.multiply import multiply
    from core.forge import delivery

    label = f"kinetic_{int(time.time())}"
    work = Path(tempfile.mkdtemp(prefix="forge_kin_"))
    try:
        master = render(params, work / f"{label}_master.mp4")

        n = int(params.get("variations", 18) or 18)
        variants = multiply(master, n, work / "variants", base_name=label) if n else []

        subfolder = (params.get("subfolder") or "Viral Music Verticals/Kinetic Lyric").strip()
        dest = f"{subfolder}/{label}"
        delivered = []
        try:
            delivered.append(delivery.deliver(master, dest, filename=f"{label}_master.mp4"))
        except Exception as e:  # noqa: BLE001
            delivered.append(f"ERROR master: {e}")
        for v in variants:
            try:
                delivered.append(delivery.deliver(v, dest))
            except Exception:  # noqa: BLE001
                pass
        result = {
            "format": "kinetic_lyric",
            "master": str(master),
            "variant_count": len(variants),
            "delivered_dir": f"Content/Mainstay-RodWave/{dest}",
            "delivered": len([d for d in delivered if not str(d).startswith("ERROR")]),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _film_montage_handler(params: dict) -> dict:
    """Forge it (Film Montage): hook audio + b-roll sources -> branded vertical -> N variants -> Nextcloud."""
    import tempfile, time
    from pathlib import Path
    from core.forge.renderers.film_montage import render
    from core.forge.multiply import multiply
    from core.forge import delivery

    label = f"montage_{int(time.time())}"
    work = Path(tempfile.mkdtemp(prefix="forge_mon_"))
    try:
        master = render(params, work / f"{label}_master.mp4")

        n = int(params.get("variations", 18) or 18)
        variants = multiply(master, n, work / "variants", base_name=label) if n else []

        subfolder = (params.get("subfolder") or "Viral Music Verticals/Film Montage").strip()
        dest = f"{subfolder}/{label}"
        delivered = []
        try:
            delivered.append(delivery.deliver(master, dest, filename=f"{label}_master.mp4"))
        except Exception as e:  # noqa: BLE001
            delivered.append(f"ERROR master: {e}")
        for v in variants:
            try:
                delivered.append(delivery.deliver(v, dest))
            except Exception:  # noqa: BLE001
                pass
        result = {
            "format": "film_montage",
            "master": str(master),
            "variant_count": len(variants),
            "delivered_dir": f"Content/Mainstay-RodWave/{dest}",
            "delivered": len([d for d in delivered if not str(d).startswith("ERROR")]),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def register_default_handlers() -> None:
    """Register all built-in Forge job handlers into the queue."""
    forge_jobs.register_handler("echo", _echo_handler)
    forge_jobs.register_handler("leak_graphic", _leak_graphic_handler)
    forge_jobs.register_handler("kinetic_lyric", _kinetic_lyric_handler)
    forge_jobs.register_handler("film_montage", _film_montage_handler)
