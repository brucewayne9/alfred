from fastapi import FastAPI


def register(app: FastAPI) -> None:
    # Import settings at call time (not module top) to honor the no-env-at-import rule.
    from config.settings import settings

    # Always-on probe so the frontend can ask whether casting is enabled.
    @app.get("/api/casting/enabled")
    async def casting_enabled():
        from config.settings import settings
        return {"enabled": bool(settings.casting_enabled)}

    # Heavy registration (DB init + full router) only when the flag is set.
    if settings.casting_enabled:
        from core.casting.db import init_db
        from core.casting.api_router import register as _register
        init_db()
        _register(app)
