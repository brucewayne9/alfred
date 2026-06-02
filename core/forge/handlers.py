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
            forge_jobs.check_cancel()  # stop between looks if the user hit Stop
            ri = rp.get("remix_index", 0)
            label = f"{fmt}_{stamp}_look{ri:02d}" if len(remixes) > 1 else f"{fmt}_{stamp}"
            master = render(rp, work / f"{label}_master{ext}")
            forge_jobs.check_cancel()  # and again after the (slow) render, before delivery
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


def _ingest_transcribe_handler(params: dict) -> dict:
    """Thin wrapper around ingest.transcribe_handler.

    Lazy import keeps faster-whisper and CUDA initialisation out of module
    import time, consistent with the renderer-handler pattern above.
    """
    from core.forge import ingest
    return ingest.transcribe_handler(params)


def _topic_clip_handler(params: dict) -> dict:
    """Custom handler for topic_clip jobs — structural variant loop.

    Loops _build_variant_assemblies, renders each variant via render(), then
    passes each master through multiply() for pixel-level Tier-2 anti-suppression.
    Does NOT route through _run_remix_format / build_remixes (that varies vessel
    mood — wrong for structural segment variation).

    allow_flip=False is mandatory: captions and the Mainstay logo must never be
    mirrored.

    Result dict emits:
      format          — "topic_clip"
      variant_count   — number of structural variants built
      variations_each — stealth copies per structural variant
      delivered       — total files delivered to Nextcloud
      delivered_dirs  — list of Nextcloud dir paths (for Library + Postiz to work)
    """
    import tempfile
    import time
    import shutil
    from pathlib import Path
    from core.forge.renderers.topic_clip import render, _build_variant_assemblies, enforce_duration
    from core.forge.multiply import multiply
    from core.forge import delivery

    segments = params.get("segments") or []
    variant_count = int(params.get("variant_count", 3) or 3)
    variant_count = max(1, min(10, variant_count))
    # stealth copies per structural variant (default 3 — research open Q3)
    stealth = int(params.get("variations", 3) or 3)
    base = (params.get("subfolder") or "Intelligent Clips").strip()
    stamp = int(time.time())

    variants = _build_variant_assemblies(enforce_duration(segments), variant_count)
    work = Path(tempfile.mkdtemp(prefix="forge_topic_clip_"))
    total_delivered, var_dirs = 0, []
    try:
        for i, var in enumerate(variants):
            forge_jobs.check_cancel()
            label = f"topic_clip_{stamp}_v{i:02d}"
            # Build per-variant params: override segments + variant_index for render()
            rp = dict(params)
            rp["variant_index"] = i
            rp["segments"] = var["segments"]
            master = render(rp, work / f"{label}_master.mp4")
            forge_jobs.check_cancel()
            copies = (
                multiply(master, stealth, work / f"v{i}", base_name=label, allow_flip=False)
                if stealth else []
            )
            dest = f"{base}/{label}"
            look_delivered = 0
            try:
                delivery.deliver(master, dest, filename=master.name)
                look_delivered += 1
            except Exception:  # noqa: BLE001
                pass
            for c in copies:
                try:
                    delivery.deliver(c, dest)
                    look_delivered += 1
                except Exception:  # noqa: BLE001
                    pass
            if look_delivered:
                var_dirs.append(f"Content/Mainstay-RodWave/{dest}")
                total_delivered += look_delivered
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return {
        "format": "topic_clip",
        "variant_count": len(variants),
        "variations_each": stealth,
        "delivered": total_delivered,
        "delivered_dirs": var_dirs,
    }


def register_default_handlers() -> None:
    """Register all built-in Forge job handlers into the queue."""
    forge_jobs.register_handler("echo", _echo_handler)
    forge_jobs.register_handler("leak_graphic", _leak_graphic_handler)
    forge_jobs.register_handler("kinetic_lyric", _kinetic_lyric_handler)
    forge_jobs.register_handler("film_montage", _film_montage_handler)
    forge_jobs.register_handler("ingest_transcribe", _ingest_transcribe_handler)
    forge_jobs.register_handler("topic_clip", _topic_clip_handler)
