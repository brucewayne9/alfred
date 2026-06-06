from __future__ import annotations
import shutil, tempfile
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from config.settings import settings
from core.casting import db, voice, persona as persona_mod, preview as preview_mod
from core.casting.mood_pack import MOOD_PACK, MOODS
from core.casting.archetypes import ARCHETYPES
from core.casting.models import (DJCreate, DJOut, PersonaBrief, PersonaDraft,
                                 AssignmentCreate, AssignmentOut)

def _preview_path(dj_id: int) -> Path:
    return Path(settings.casting_previews_dir) / f"{dj_id}.wav"

def _render_preview_cached(dj_id: int) -> str:
    """Render the DJ's neutral clip to the cache path and return it. Caller must
    ensure the DJ has a neutral clip first. Registers the neutral clip into the
    Qwen resources dir first so 105:7860 can resolve it by name (not path)."""
    out = str(_preview_path(dj_id))
    voice.register_to_engine(dj_id, ["neutral"])
    preview_mod.render_preview(
        voice_name=voice.engine_voice_name(dj_id, "neutral"), out_path=out)
    return out

def _dj_out(row: dict) -> dict:
    return DJOut(
        id=row["id"], name=row["name"], role=row["role"], status=row["status"],
        persona_prompt=row["persona_prompt"], archetype_tags=row["archetype_tags"],
        expertise=row["expertise"], voice_source=row["voice_source"],
        moods_present=row["moods_present"], avatar=row.get("avatar"),
    ).model_dump()

def register(app: FastAPI, auth_dep=None) -> None:
    from core.security.auth import require_auth
    guard = auth_dep or require_auth

    @app.get("/api/casting/moodpack")
    async def moodpack(_user=Depends(guard)):
        return {"moods": [{"mood": m, **MOOD_PACK[m]} for m in MOODS]}

    @app.get("/api/casting/archetypes")
    async def archetypes(_user=Depends(guard)):
        return ARCHETYPES

    @app.get("/api/casting/djs")
    async def list_djs(_user=Depends(guard)):
        return [_dj_out(r) for r in db.list_djs()]

    @app.post("/api/casting/djs")
    async def create_dj(body: DJCreate, _user=Depends(guard)):
        dj_id = db.create_dj(name=body.name, role=body.role, persona_prompt=body.persona_prompt,
                             archetype_tags=body.archetype_tags, expertise=body.expertise,
                             voice_source=body.voice_source)
        return _dj_out(db.get_dj(dj_id))

    @app.get("/api/casting/djs/{dj_id}")
    async def get_dj(dj_id: int, _user=Depends(guard)):
        row = db.get_dj(dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        return _dj_out(row)

    @app.post("/api/casting/persona/draft", response_model=PersonaDraft)
    async def draft(body: PersonaBrief, _user=Depends(guard)):
        return persona_mod.draft_persona(name=body.name, brief=body.brief,
                                         archetype_id=body.archetype_id)

    @app.post("/api/casting/djs/{dj_id}/voice/{mood}")
    async def upload_mood(dj_id: int, mood: str, file: UploadFile = File(...), _user=Depends(guard)):
        if mood not in MOODS:
            raise HTTPException(400, f"unknown mood '{mood}'")
        if not db.get_dj(dj_id):
            raise HTTPException(404, "DJ not found")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        ok, reason = voice.validate_clip(tmp_path)
        if not ok:
            Path(tmp_path).unlink(missing_ok=True)
            raise HTTPException(422, reason)
        voice.store_mood(dj_id=dj_id, mood=mood, src_path=tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        db.set_mood_present(dj_id, mood)
        # neutral present => at least a working DJ
        row = db.get_dj(dj_id)
        if "neutral" in row["moods_present"] and row["status"] == "draft":
            db.set_status(dj_id, "ready")
        # eagerly render+cache the preview off the neutral clip (best-effort)
        if mood == "neutral":
            try:
                _render_preview_cached(dj_id)
            except Exception:
                pass
        return _dj_out(db.get_dj(dj_id))

    @app.get("/api/casting/djs/{dj_id}/preview")
    async def preview(dj_id: int, _user=Depends(guard)):
        row = db.get_dj(dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        cached = _preview_path(dj_id)
        if cached.exists():
            return FileResponse(str(cached), media_type="audio/wav", filename=f"preview_{dj_id}.wav")
        if "neutral" in row["moods_present"]:
            out = _render_preview_cached(dj_id)
            return FileResponse(out, media_type="audio/wav", filename=f"preview_{dj_id}.wav")
        raise HTTPException(404, "no preview yet")

    @app.post("/api/casting/djs/{dj_id}/preview")
    async def rerender_preview(dj_id: int, _user=Depends(guard)):
        # force re-render (re-roll): overwrite the cached file
        row = db.get_dj(dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        if "neutral" not in row["moods_present"]:
            raise HTTPException(422, "need a neutral clip before preview")
        out = _render_preview_cached(dj_id)
        return FileResponse(out, media_type="audio/wav", filename=f"preview_{dj_id}.wav")

    @app.get("/api/casting/assignments")
    async def list_assignments(station_id: int | None = None, _user=Depends(guard)):
        return [AssignmentOut(id=a["id"], dj_id=a["dj_id"], dj_name=a["dj_name"],
                              station_id=a["station_id"], slot=a["slot"],
                              effective_at=a["effective_at"], applied=bool(a["applied"])).model_dump()
                for a in db.list_assignments(station_id)]

    @app.post("/api/casting/assignments")
    async def create_assignment(body: AssignmentCreate, _user=Depends(guard)):
        row = db.get_dj(body.dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        if row["status"] == "draft":
            raise HTTPException(422, "DJ is still a draft; add a neutral clip first")
        # datetime-local sends "YYYY-MM-DDTHH:MM" (no seconds). Normalize to a
        # canonical seconds-precision ISO string so lexical compares stay sound.
        try:
            eff = datetime.fromisoformat(body.effective_at).isoformat(timespec="seconds")
        except ValueError:
            raise HTTPException(422, "bad effective_at")
        aid = db.create_assignment(dj_id=body.dj_id, station_id=body.station_id,
                                   slot=body.slot, effective_at=eff)
        return {"id": aid}
