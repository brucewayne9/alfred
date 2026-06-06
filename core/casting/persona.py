# core/casting/persona.py
from __future__ import annotations
import requests
from config.settings import settings
from core.casting.archetypes import ARCHETYPES
from core.casting.models import PersonaDraft

_SYSTEM = (
    "You are a radio program director. Write a tight first-person PERSONA BRIEF for an AI radio host. "
    "Cover: who they are, their voice/tone, their lanes (topics), how they open and close, and their "
    "delivery quirks. Keep it under 320 words. Output ONLY the persona text, no preamble. "
    "The host emits delivery tags ([neutral],[fired],[serious],[amused],[thoughtful],[wry],[intimate]) "
    "and {laugh} in their scripts, so describe WHEN each mood shows up."
)

def _archetype_summary(archetype_id: str | None) -> str:
    for a in ARCHETYPES:
        if a["id"] == archetype_id:
            return a["summary"]
    return ""

def _build_prompt(*, name: str, brief: str, archetype_summary: str) -> str:
    parts = [f"Host name: {name}.", f"Operator brief: {brief}."]
    if archetype_summary:
        parts.append(f"Start from this archetype and remix it: {archetype_summary}.")
    return " ".join(parts)

def draft_persona(*, name: str, brief: str, archetype_id: str | None = None) -> PersonaDraft:
    user_msg = _build_prompt(name=name, brief=brief, archetype_summary=_archetype_summary(archetype_id))
    body = {
        "model": settings.casting_model,
        "think": False,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "options": {"num_predict": 700, "temperature": 0.8},
    }
    resp = requests.post(f"{settings.casting_ollama_url}/api/chat", json=body, timeout=120)
    resp.raise_for_status()
    content = (resp.json().get("message", {}) or {}).get("content", "").strip()
    tags = [archetype_id] if archetype_id else []
    return PersonaDraft(persona_prompt=content, archetype_tags=tags)
