from fastapi import FastAPI
from core.casting.db import init_db
from core.casting.api_router import register as _register


def register(app: FastAPI) -> None:
    init_db()
    _register(app)
