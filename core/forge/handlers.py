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


def register_default_handlers() -> None:
    """Register all built-in Forge job handlers into the queue."""
    forge_jobs.register_handler("echo", _echo_handler)
