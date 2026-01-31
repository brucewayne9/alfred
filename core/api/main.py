"""Alfred API Server - Main FastAPI application with auth, integrations, and tool calling."""

import logging
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.brain.router import ask, classify_query, ModelTier, get_system_prompt
from core.memory.store import store_memory, recall, store_conversation
from core.memory.conversations import (
    init_db as init_conversations_db,
    create_conversation, list_conversations as list_convos,
    get_conversation, add_message, archive_conversation, update_title,
    list_archived_conversations, restore_conversation, delete_conversation_permanently,
    search_conversations, create_project, list_projects, get_project,
    update_project, delete_project, add_reference, list_references,
    get_reference, update_reference, delete_reference, search_references,
    move_conversation_to_project, list_conversations_by_project, get_project_context,
    UPLOADS_DIR,
)
from core.security.auth import (
    create_user, verify_user, create_access_token, get_current_user,
    require_auth, setup_initial_user,
)
from core.security.google_oauth import get_auth_url, handle_callback, is_connected
from config.settings import settings

# Register all tools on import
from core.tools.definitions import register_all
register_all()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("alfred")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Alfred is starting up...")
    logger.info(f"Local LLM: {settings.ollama_model} at {settings.ollama_host}")
    logger.info(f"Cloud LLM: {settings.anthropic_model}")
    # Initialize conversation history database
    init_conversations_db()
    logger.info("Conversation history database initialized")
    # Create initial admin user if needed
    pwd = setup_initial_user()
    if pwd:
        logger.info("Initial admin user created — check logs for password")
    # Pre-warm voice models in background thread so startup isn't blocked
    import asyncio
    loop = asyncio.get_event_loop()
    def _warmup_voice():
        try:
            from interfaces.voice.stt import warmup as stt_warmup
            stt_warmup(model_size=settings.whisper_model)
        except Exception as e:
            logger.warning(f"STT warmup failed: {e}")
        try:
            from interfaces.voice.tts import warmup as tts_warmup
            tts_warmup()
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")
    loop.run_in_executor(None, _warmup_voice)
    yield
    logger.info("Alfred is shutting down...")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Alfred", version="0.3.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aialfred.groundrushcloud.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Static files for PWA icons
_static_dir = Path(__file__).parent.parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/manifest.json")
async def pwa_manifest():
    return JSONResponse({
        "name": "Alfred",
        "short_name": "Alfred",
        "description": "Personal AI Assistant",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0a0a0a",
        "theme_color": "#0a0a0a",
        "orientation": "any",
        "icons": [
            {"src": "/static/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/sw.js")
async def service_worker():
    return Response(content=SERVICE_WORKER_JS, media_type="application/javascript",
                    headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


# ==================== Models ====================

class ChatRequest(BaseModel):
    message: str
    tier: str | None = None
    session_id: str = "default"
    # Optional image data for vision queries
    image_path: str | None = None
    image_base64: str | None = None
    image_media_type: str | None = None
    # Optional document path for analysis
    document_path: str | None = None
    # Optional TTS settings (for Shortcuts/API clients)
    voice: str | None = None  # Voice ID (e.g., "Gwen_Stacy", "design:Natalie")
    tts_backend: str | None = None  # TTS backend ("kokoro", "qwen3")

class ChatResponse(BaseModel):
    response: str
    tier: str
    timestamp: str
    images: list[dict] | None = None  # [{base64, filename, download_url}]
    ui_action: dict | None = None  # {action, value} for frontend to execute

class MemoryRequest(BaseModel):
    text: str
    category: str = "general"

class RecallRequest(BaseModel):
    query: str
    category: str = "general"
    n_results: int = 5

class LoginRequest(BaseModel):
    username: str
    password: str

class SetupRequest(BaseModel):
    username: str
    password: str

class AddServerRequest(BaseModel):
    name: str
    host: str
    username: str = "root"
    port: int = 22
    key_path: str = ""
    description: str = ""

class ConversationTitleRequest(BaseModel):
    title: str


class ProjectRequest(BaseModel):
    name: str
    description: str = ""
    color: str = "#3b82f6"


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None


class ReferenceNoteRequest(BaseModel):
    title: str
    content: str


class ReferenceUpdateRequest(BaseModel):
    title: str | None = None
    content: str | None = None


class ConversationProjectRequest(BaseModel):
    project_id: str | None = None


# Conversation history per session
_sessions: dict[str, list[dict]] = {}


def get_session_messages(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        msgs = [{"role": "system", "content": get_system_prompt()}]
        # Try to restore from SQLite
        conv = get_conversation(session_id)
        if conv:
            for m in conv["messages"]:
                msgs.append({"role": m["role"], "content": m["content"]})
        _sessions[session_id] = msgs
    return _sessions[session_id]


# ==================== Auth Endpoints ====================

@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, req: LoginRequest):
    user = verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    response = JSONResponse({"token": token, "username": user["username"], "role": user["role"]})
    response.set_cookie("alfred_token", token, httponly=True, secure=True, samesite="lax", max_age=86400)
    return response


@app.post("/auth/setup")
async def setup(req: SetupRequest):
    """Create the first admin user (only works when no users exist)."""
    from core.security.auth import _load_users
    users = _load_users()
    if users:
        raise HTTPException(status_code=400, detail="Setup already complete")
    create_user(req.username, req.password, "admin")
    token = create_access_token({"sub": req.username, "role": "admin"})
    response = JSONResponse({"token": token, "username": req.username, "message": "Setup complete"})
    response.set_cookie("alfred_token", token, httponly=True, secure=True, samesite="lax", max_age=86400)
    return response


@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    if user is None:
        return {"authenticated": False}
    return {"authenticated": True, "username": user.get("sub", user.get("username")), "role": user.get("role")}


@app.post("/auth/logout")
async def logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("alfred_token")
    return response


@app.post("/auth/change-password")
async def change_password(req: LoginRequest, user: dict = Depends(require_auth)):
    """Change password. The LoginRequest.password field is the new password."""
    from core.security.auth import _load_users, _save_users, pwd_context
    users = _load_users()
    username = user.get("sub", user.get("username"))
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username]["password_hash"] = pwd_context.hash(req.password)
    _save_users(users)
    return {"message": "Password changed"}


# ==================== Google OAuth ====================

@app.get("/auth/google")
async def google_auth(user: dict = Depends(require_auth)):
    """Start Google OAuth flow."""
    if not settings.google_client_id:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")
    auth_url = get_auth_url()
    return RedirectResponse(auth_url)


@app.get("/auth/google/callback")
async def google_callback(code: str = None, error: str = None):
    """Handle Google OAuth callback."""
    if error:
        return HTMLResponse(f"<h2>Google Authorization Failed</h2><p>{error}</p><a href='/'>Back to Alfred</a>")
    if not code:
        return HTMLResponse("<h2>No authorization code received</h2><a href='/'>Back to Alfred</a>")

    success = handle_callback(code)
    if success:
        return HTMLResponse(
            "<h2>Google Connected Successfully</h2>"
            "<p>Gmail and Calendar are now linked to Alfred.</p>"
            "<script>setTimeout(()=>window.location='/',2000)</script>"
        )
    return HTMLResponse("<h2>Connection Failed</h2><p>Check logs for details.</p><a href='/'>Back to Alfred</a>")


# ==================== Integration Status ====================

@app.get("/integrations/status")
async def integration_status(user: dict = Depends(require_auth)):
    """Check which integrations are connected."""
    google = is_connected()

    # Check server config
    from integrations.servers.manager import list_servers
    servers = []
    try:
        servers = list_servers()
    except Exception:
        pass

    # Check CRM connection
    crm_connected = False
    try:
        from integrations.base_crm.client import is_connected as crm_is_connected
        crm_connected = crm_is_connected()
    except Exception:
        pass

    # Check n8n connection
    n8n_connected = False
    n8n_workflow_count = 0
    try:
        from integrations.n8n.client import is_connected as n8n_is_connected, list_workflows as n8n_list
        n8n_connected = n8n_is_connected()
        if n8n_connected:
            n8n_workflow_count = len(n8n_list(limit=100))
    except Exception:
        pass

    # Check Nextcloud connection
    nextcloud_connected = False
    try:
        from integrations.nextcloud.client import is_connected as nc_is_connected
        nextcloud_connected = nc_is_connected()
    except Exception:
        pass

    # Check Stripe connection
    stripe_connected = False
    try:
        from integrations.stripe.client import is_connected as stripe_is_connected
        stripe_connected = stripe_is_connected()
    except Exception:
        pass

    return {
        "google": {
            "connected": google,
            "services": ["Gmail", "Calendar"] if google else [],
        },
        "servers": {
            "count": len(servers),
            "names": [s["name"] for s in servers],
        },
        "base_crm": {
            "configured": bool(settings.base_crm_api_key),
            "connected": crm_connected,
            "label": "Twenty CRM",
        },
        "n8n": {
            "configured": bool(settings.n8n_api_key),
            "connected": n8n_connected,
            "workflow_count": n8n_workflow_count,
            "label": "n8n Automations",
        },
        "nextcloud": {
            "configured": bool(settings.nextcloud_url),
            "connected": nextcloud_connected,
            "label": "Nextcloud",
        },
        "stripe": {
            "configured": bool(settings.stripe_api_key),
            "connected": stripe_connected,
            "label": "Stripe Payments",
        },
        "ollama": {
            "model": settings.ollama_model,
            "host": settings.ollama_host,
        },
        "anthropic": {
            "configured": bool(settings.anthropic_api_key and settings.anthropic_api_key != "sk-ant-CHANGEME"),
            "model": settings.anthropic_model,
        },
        "claude_code": {
            "configured": _check_claude_code_cli(),
            "label": "Claude Code (Max)",
        },
        "tts": {
            "backend": settings.tts_model,
            "qwen3_available": _check_qwen3_available(),
        },
    }


def _check_qwen3_available() -> bool:
    """Check if Qwen3-TTS server is available."""
    try:
        import requests
        resp = requests.get("http://localhost:7860/docs", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _check_claude_code_cli() -> bool:
    """Check if Claude Code CLI is available (for Max subscription users)."""
    import shutil
    import os
    from pathlib import Path
    # Check standard which first
    if shutil.which("claude"):
        return True
    # Also check common nvm/npm paths since uvicorn may not inherit full PATH
    common_paths = [
        Path.home() / ".nvm/versions/node/v20.12.2/bin/claude",
        Path.home() / ".local/bin/claude",
        Path("/usr/local/bin/claude"),
    ]
    for path in common_paths:
        if path.exists() and os.access(path, os.X_OK):
            return True
    return False


@app.get("/settings/tts")
async def get_tts_settings(user: dict = Depends(require_auth)):
    """Get current TTS settings."""
    return {
        "backend": settings.tts_model,
        "qwen3_available": _check_qwen3_available(),
    }


def _safe_env_update(key: str, value: str) -> None:
    """Safely update a key in .env file with atomic write and backup."""
    env_path = Path("/home/aialfred/alfred/config/.env")
    backup_path = Path("/home/aialfred/alfred/config/.env.backup")

    if env_path.exists():
        original = env_path.read_text()
        original_lines = len(original.strip().split('\n'))

        # Update or append the key
        pattern = rf"^{re.escape(key)}=.*$"
        if re.search(pattern, original, re.MULTILINE):
            new_content = re.sub(pattern, f"{key}={value}", original, flags=re.MULTILINE)
        else:
            new_content = original.rstrip() + f"\n{key}={value}\n"

        # Safety check: new content shouldn't be drastically shorter
        new_lines = len(new_content.strip().split('\n'))
        if new_lines < original_lines - 1:
            logger.error(f"ENV update aborted: would reduce from {original_lines} to {new_lines} lines")
            raise HTTPException(status_code=500, detail="ENV update failed safety check")

        # Atomic write: write to temp, then rename
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', dir=env_path.parent, delete=False) as tmp:
            tmp.write(new_content)
            tmp_path = Path(tmp.name)

        # Update backup
        backup_path.write_text(original)

        # Atomic rename
        tmp_path.rename(env_path)
    else:
        env_path.write_text(f"{key}={value}\n")


@app.put("/settings/tts")
async def set_tts_settings(backend: str, user: dict = Depends(require_auth)):
    """Set TTS backend (kokoro, qwen3, piper)."""
    if backend not in ["kokoro", "qwen3", "piper"]:
        raise HTTPException(status_code=400, detail="Invalid TTS backend")

    _safe_env_update("TTS_MODEL", backend)

    # Update runtime settings
    settings.tts_model = backend

    return {"backend": backend, "message": f"TTS backend set to {backend}"}


# Kokoro voice options
KOKORO_VOICES = [
    {"id": "bm_daniel", "name": "Daniel", "desc": "British Male"},
    {"id": "bm_george", "name": "George", "desc": "British Male"},
    {"id": "bm_lewis", "name": "Lewis", "desc": "British Male"},
    {"id": "bf_emma", "name": "Emma", "desc": "British Female"},
    {"id": "bf_isabella", "name": "Isabella", "desc": "British Female"},
    {"id": "am_adam", "name": "Adam", "desc": "American Male"},
    {"id": "am_michael", "name": "Michael", "desc": "American Male"},
    {"id": "af_heart", "name": "Heart", "desc": "American Female"},
    {"id": "af_bella", "name": "Bella", "desc": "American Female"},
    {"id": "af_nicole", "name": "Nicole", "desc": "American Female"},
    {"id": "af_sarah", "name": "Sarah", "desc": "American Female"},
    {"id": "af_sky", "name": "Sky", "desc": "American Female"},
]


@app.get("/settings/voices")
async def get_voices(user: dict = Depends(require_auth)):
    """Get available voices for the current TTS backend."""
    import requests as req
    # Read directly from env file to avoid caching issues
    env_path = Path("/home/aialfred/alfred/config/.env")
    backend = settings.tts_model
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("TTS_MODEL="):
                backend = line.split("=", 1)[1].strip()
                break

    if backend == "qwen3":
        voices = []

        # List cloned voices from Qwen3 resources directory
        resources_dir = Path("/home/aialfred/qwen3-tts/resources")
        if resources_dir.exists():
            for f in resources_dir.iterdir():
                if f.suffix in [".mp3", ".wav"]:
                    voice_id = f.stem
                    voices.append({"id": voice_id, "name": voice_id, "desc": "Cloned voice", "type": "clone"})

        # List designed voices from Qwen3 API
        try:
            resp = req.get("http://localhost:7860/voice_design/voices", timeout=5)
            if resp.status_code == 200:
                for v in resp.json().get("voices", []):
                    voices.append({
                        "id": f"design:{v['name']}",
                        "name": f"{v['name']} (designed)",
                        "desc": v.get("description", "Designed voice")[:50],
                        "type": "design"
                    })
        except Exception as e:
            logger.warning(f"Could not fetch designed voices: {e}")

        if not voices:
            voices = [{"id": "demo_speaker0", "name": "Demo Speaker", "desc": "Default voice", "type": "clone"}]
        # Read current voice from env file
        current_voice = settings.tts_voice
        for line in env_path.read_text().splitlines():
            if line.startswith("TTS_VOICE="):
                current_voice = line.split("=", 1)[1].strip()
                break
        return {"backend": backend, "voices": voices, "current": current_voice}
    else:
        # Kokoro voices - read current voice from env file
        current_voice = settings.tts_voice
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TTS_VOICE="):
                    current_voice = line.split("=", 1)[1].strip()
                    break
        return {"backend": backend, "voices": KOKORO_VOICES, "current": current_voice}


@app.put("/settings/voice")
async def set_voice(voice: str, user: dict = Depends(require_auth)):
    """Set the TTS voice."""
    _safe_env_update("TTS_VOICE", voice)

    # Update runtime settings
    settings.tts_voice = voice

    return {"voice": voice, "message": f"Voice set to {voice}"}


# ==================== Server Management ====================

@app.post("/servers/add")
async def add_server(req: AddServerRequest, user: dict = Depends(require_auth)):
    """Register a server for management."""
    from integrations.servers.manager import add_server as _add
    _add(req.name, req.host, req.username, req.port, req.key_path, req.description)
    return {"message": f"Server '{req.name}' registered", "name": req.name}


# ==================== File Upload/Download ====================

@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Upload a document for analysis (PDF, DOCX, XLSX, CSV, TXT, MD, JSON)."""
    from core.tools.files import save_upload, parse_document
    allowed_ext = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".md", ".json"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    path = save_upload(content, file.filename)
    # Parse the document
    result = parse_document(path)

    # Index to LightRAG for long-term memory
    lightrag_status = "not_indexed"
    if result["text"] and not result["error"]:
        try:
            from integrations.lightrag.client import upload_text
            lr_result = await upload_text(
                result["text"],
                description=f"Document: {file.filename}"
            )
            lightrag_status = "indexed" if lr_result["success"] else f"failed: {lr_result.get('error', 'unknown')}"
            logger.info(f"LightRAG indexing: {lightrag_status}")
        except Exception as e:
            lightrag_status = f"failed: {e}"
            logger.warning(f"LightRAG indexing failed: {e}")

    return {
        "path": path,
        "filename": file.filename,
        "text_preview": result["text"][:2000] + ("..." if len(result["text"]) > 2000 else ""),
        "metadata": result["metadata"],
        "error": result["error"],
        "lightrag_status": lightrag_status,
    }


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Upload an image for vision analysis."""
    from core.tools.files import save_upload
    import base64
    allowed_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {ext}")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 20MB)")
    path = save_upload(content, file.filename)
    # Return base64 for inline display
    b64 = base64.b64encode(content).decode("utf-8")
    media_type = f"image/{ext.lstrip('.')}"
    if ext == ".jpg":
        media_type = "image/jpeg"
    return {
        "path": path,
        "filename": file.filename,
        "base64": b64,
        "media_type": media_type,
        "size_bytes": len(content),
    }


@app.get("/download/{filename}")
async def download_file(filename: str, user: dict = Depends(require_auth)):
    """Download a generated document."""
    from core.tools.files import GENERATED_DIR
    # Security: only allow alphanumeric, dots, underscores, hyphens
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if not all(c in safe_chars for c in filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = GENERATED_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Determine content type
    ext = path.suffix.lower()
    content_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
    }
    ct = content_types.get(ext, "application/octet-stream")
    return Response(
        content=path.read_bytes(),
        media_type=ct,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ==================== Conversation History Endpoints ====================

@app.get("/conversations")
async def conversations_list(limit: int = 50, offset: int = 0, project_id: str | None = None, user: dict = Depends(require_auth)):
    """List conversations ordered by most recent, optionally filtered by project."""
    return list_convos(limit=limit, offset=offset, project_id=project_id)


@app.post("/conversations")
async def conversations_create(user: dict = Depends(require_auth)):
    """Create a new conversation."""
    return create_conversation()


# Static routes MUST come before dynamic {conv_id} routes
@app.get("/conversations/search")
async def conversations_search(q: str, limit: int = 20, user: dict = Depends(require_auth)):
    """Search conversations using full-text search."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    return search_conversations(q, limit)


@app.get("/conversations/archived")
async def conversations_archived_list(limit: int = 50, offset: int = 0, user: dict = Depends(require_auth)):
    """List archived conversations."""
    return list_archived_conversations(limit=limit, offset=offset)


# Dynamic routes with {conv_id} parameter
@app.get("/conversations/{conv_id}")
async def conversations_get(conv_id: str, user: dict = Depends(require_auth)):
    """Get full conversation with messages."""
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.delete("/conversations/{conv_id}")
async def conversations_delete(conv_id: str, user: dict = Depends(require_auth)):
    """Archive (soft-delete) a conversation."""
    if not archive_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"archived": True}


@app.put("/conversations/{conv_id}/title")
async def conversations_rename(conv_id: str, req: ConversationTitleRequest, user: dict = Depends(require_auth)):
    """Rename a conversation."""
    if not update_title(conv_id, req.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"updated": True}


@app.put("/conversations/{conv_id}/project")
async def conversations_move_to_project(conv_id: str, req: ConversationProjectRequest, user: dict = Depends(require_auth)):
    """Move a conversation to a project or remove from project."""
    if not move_conversation_to_project(conv_id, req.project_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"updated": True, "project_id": req.project_id}


@app.post("/conversations/{conv_id}/restore")
async def conversations_restore(conv_id: str, user: dict = Depends(require_auth)):
    """Restore an archived conversation."""
    if not restore_conversation(conv_id):
        raise HTTPException(status_code=404, detail="Archived conversation not found")
    return {"restored": True}


@app.delete("/conversations/{conv_id}/permanent")
async def conversations_delete_permanent(conv_id: str, user: dict = Depends(require_auth)):
    """Permanently delete a conversation."""
    if not delete_conversation_permanently(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


# ==================== Projects ====================

@app.get("/projects")
async def projects_list(user: dict = Depends(require_auth)):
    """List all projects."""
    return list_projects()


@app.post("/projects")
async def projects_create(req: ProjectRequest, user: dict = Depends(require_auth)):
    """Create a new project."""
    return create_project(req.name, req.description, req.color)


@app.get("/projects/{project_id}")
async def projects_get(project_id: str, user: dict = Depends(require_auth)):
    """Get project details."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.put("/projects/{project_id}")
async def projects_update(project_id: str, req: ProjectUpdateRequest, user: dict = Depends(require_auth)):
    """Update a project."""
    if not update_project(project_id, req.name, req.description, req.color):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"updated": True}


@app.delete("/projects/{project_id}")
async def projects_delete(project_id: str, user: dict = Depends(require_auth)):
    """Delete a project and all its references."""
    if not delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": True}


@app.get("/projects/{project_id}/references")
async def projects_references_list(project_id: str, user: dict = Depends(require_auth)):
    """List all references for a project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return list_references(project_id)


@app.post("/projects/{project_id}/references")
async def projects_references_add_note(project_id: str, req: ReferenceNoteRequest, user: dict = Depends(require_auth)):
    """Add a note reference to a project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return add_reference(project_id, "note", req.title, req.content)


@app.post("/projects/{project_id}/references/upload")
async def projects_references_upload(project_id: str, file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Upload a file reference to a project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type
    allowed_ext = {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Save file
    import uuid as uuid_mod
    file_id = uuid_mod.uuid4().hex[:12]
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._-")[:50]
    relative_path = f"{project_id}/{file_id}_{safe_name}"
    file_path = UPLOADS_DIR / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)

    # Extract text for searchability
    extracted_text = ""
    mime_type = file.content_type or "application/octet-stream"

    if ext == ".pdf":
        try:
            import PyPDF2
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.warning(f"PDF text extraction failed: {e}")
    elif ext == ".docx":
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(content))
            extracted_text = "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            logger.warning(f"DOCX text extraction failed: {e}")
    elif ext in {".txt", ".md"}:
        try:
            extracted_text = content.decode("utf-8", errors="replace")
        except Exception:
            pass

    ref = add_reference(
        project_id,
        "file",
        file.filename,
        extracted_text or None,
        relative_path,
        mime_type,
        len(content),
    )
    return ref


@app.get("/projects/{project_id}/references/search")
async def projects_references_search(project_id: str, q: str, limit: int = 20, user: dict = Depends(require_auth)):
    """Search references within a project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    return search_references(project_id, q, limit)


@app.get("/projects/{project_id}/conversations")
async def projects_conversations_list(project_id: str, limit: int = 50, offset: int = 0, user: dict = Depends(require_auth)):
    """List conversations in a project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return list_conversations_by_project(project_id, limit, offset)


# ==================== References ====================

@app.get("/references/{ref_id}")
async def references_get(ref_id: int, user: dict = Depends(require_auth)):
    """Get reference details."""
    ref = get_reference(ref_id)
    if not ref:
        raise HTTPException(status_code=404, detail="Reference not found")
    return ref


@app.put("/references/{ref_id}")
async def references_update(ref_id: int, req: ReferenceUpdateRequest, user: dict = Depends(require_auth)):
    """Update a reference."""
    if not update_reference(ref_id, req.title, req.content):
        raise HTTPException(status_code=404, detail="Reference not found")
    return {"updated": True}


@app.delete("/references/{ref_id}")
async def references_delete(ref_id: int, user: dict = Depends(require_auth)):
    """Delete a reference."""
    if not delete_reference(ref_id):
        raise HTTPException(status_code=404, detail="Reference not found")
    return {"deleted": True}


@app.get("/references/{ref_id}/download")
async def references_download(ref_id: int, user: dict = Depends(require_auth)):
    """Download a file reference."""
    ref = get_reference(ref_id)
    if not ref or not ref.get("file_path"):
        raise HTTPException(status_code=404, detail="File reference not found")
    file_path = UPLOADS_DIR / ref["file_path"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    return Response(
        content=file_path.read_bytes(),
        media_type=ref.get("file_type", "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{ref["title"]}"'}
    )


# ==================== Core Endpoints ====================

@app.get("/")
async def root():
    return HTMLResponse(content=CHAT_HTML, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/health")
async def health():
    return {
        "status": "online",
        "name": "Alfred",
        "version": "0.3.0",
        "timestamp": datetime.now().isoformat(),
        "llm_local": settings.ollama_model,
        "llm_cloud": settings.anthropic_model,
    }


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest, user: dict = Depends(require_auth)):
    """Send a message to Alfred."""
    messages = get_session_messages(req.session_id)
    messages.append({"role": "user", "content": req.message})

    # Handle image vision request
    if req.image_base64 and req.image_media_type:
        from core.brain.router import query_claude_vision
        response = await query_claude_vision(
            req.message or "Describe this image in detail.",
            req.image_base64,
            req.image_media_type
        )
        messages.append({"role": "assistant", "content": response})
        store_conversation(req.message, response, req.session_id)
        try:
            add_message(req.session_id, "user", req.message, "cloud")
            add_message(req.session_id, "assistant", response, "cloud")
        except Exception as e:
            logger.warning(f"Failed to persist vision message: {e}")
        return ChatResponse(response=response, tier="cloud", timestamp=datetime.now().isoformat())

    # Handle document analysis request
    if req.document_path:
        from core.tools.files import parse_document
        doc_result = parse_document(req.document_path)
        if doc_result["error"]:
            response = f"Error reading document: {doc_result['error']}"
        else:
            # Include document content in context
            doc_text = doc_result["text"][:8000]  # Limit for context window
            if len(doc_result["text"]) > 8000:
                doc_text += "\n... (document truncated)"
            augmented_msg = f"[Document content from {doc_result['metadata']['filename']}]:\n{doc_text}\n\n{req.message}"
            messages[-1]["content"] = augmented_msg

    # Check memory for relevant context
    memories = recall(req.message)
    if memories:
        context = "\n".join(f"- {m['text']}" for m in memories[:3])
        messages[-1]["content"] = (
            f"[Relevant context from memory:\n{context}]\n\n" + messages[-1]["content"]
        )

    # Inject project context if conversation belongs to a project
    conv = get_conversation(req.session_id)
    if conv and conv.get("project_id"):
        project_context = get_project_context(conv["project_id"])
        if project_context:
            # Prepend project context to the system message
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n[Project Context]\n{project_context}"
            else:
                messages.insert(0, {"role": "system", "content": f"[Project Context]\n{project_context}"})

    tier = ModelTier(req.tier) if req.tier else classify_query(req.message)
    result = await ask(req.message, messages=messages, tier=tier)

    # Handle both old string return and new dict return
    if isinstance(result, dict):
        response = result["response"]
        images = result.get("images")
        ui_action = result.get("ui_action")
    else:
        response = result
        images = None
        ui_action = None

    messages.append({"role": "assistant", "content": response})
    store_conversation(req.message, response, req.session_id)

    # Persist to SQLite conversation history
    try:
        add_message(req.session_id, "user", req.message, tier.value)
        add_message(req.session_id, "assistant", response, tier.value)
    except Exception as e:
        logger.warning(f"Failed to persist conversation message: {e}")

    # Keep session history manageable (last 20 exchanges)
    if len(messages) > 42:
        _sessions[req.session_id] = [messages[0]] + messages[-40:]

    return ChatResponse(
        response=response,
        tier=tier.value,
        timestamp=datetime.now().isoformat(),
        images=images,
        ui_action=ui_action,
    )


@app.post("/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatRequest, user: dict = Depends(require_auth)):
    """Stream a response from Alfred."""
    messages = get_session_messages(req.session_id)
    messages.append({"role": "user", "content": req.message})

    tier = ModelTier(req.tier) if req.tier else classify_query(req.message)

    async def generate():
        full_response = []
        stream = await ask(req.message, messages=messages, tier=tier, stream=True)
        async for chunk in stream:
            full_response.append(chunk)
            yield chunk
        response_text = "".join(full_response)
        messages.append({"role": "assistant", "content": response_text})
        store_conversation(req.message, response_text, req.session_id)
        # Persist to SQLite conversation history
        try:
            add_message(req.session_id, "user", req.message, tier.value)
            add_message(req.session_id, "assistant", response_text, tier.value)
        except Exception as e:
            logger.warning(f"Failed to persist streamed message: {e}")

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/voice/voices")
async def get_all_voices(user: dict = Depends(require_auth)):
    """Get all available voices for both TTS backends.

    Returns voices grouped by backend for easy selection in Shortcuts.
    """
    import requests as req

    result = {
        "kokoro": KOKORO_VOICES,
        "qwen3": [],
        "current_backend": settings.tts_model,
        "current_voice": settings.tts_voice,
    }

    # Get Qwen3 voices
    resources_dir = Path("/home/aialfred/qwen3-tts/resources")
    if resources_dir.exists():
        for f in resources_dir.iterdir():
            if f.suffix in [".mp3", ".wav"]:
                voice_id = f.stem
                result["qwen3"].append({
                    "id": voice_id,
                    "name": voice_id.replace("_", " "),
                    "desc": "Cloned voice",
                    "type": "clone"
                })

    # Get designed voices from Qwen3 API
    try:
        resp = req.get("http://localhost:7860/voice_design/voices", timeout=5)
        if resp.status_code == 200:
            for v in resp.json().get("voices", []):
                result["qwen3"].append({
                    "id": f"design:{v['name']}",
                    "name": f"{v['name']} (designed)",
                    "desc": v.get("description", "Designed voice")[:50],
                    "type": "design"
                })
    except Exception:
        pass

    return result


@app.post("/voice/transcribe")
@limiter.limit("20/minute")
async def transcribe_audio(request: Request, audio: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Transcribe audio to text."""
    import asyncio
    from interfaces.voice.stt import transcribe
    audio_data = await audio.read()
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, lambda: transcribe(audio_data, model_size=settings.whisper_model))
    return {"text": text}


@app.post("/voice/speak")
@limiter.limit("30/minute")
async def text_to_speech(request: Request, req: ChatRequest, user: dict = Depends(require_auth)):
    """Convert text to speech audio.

    Optional parameters:
    - voice: Voice ID (e.g., "Gwen_Stacy", "design:Natalie", "bm_daniel")
    - tts_backend: TTS backend ("kokoro", "qwen3")
    """
    import asyncio
    from interfaces.voice.tts import speak_with_options
    loop = asyncio.get_event_loop()
    audio_data = await loop.run_in_executor(
        None,
        lambda: speak_with_options(req.message, voice_id=req.voice, backend=req.tts_backend)
    )
    return Response(content=audio_data, media_type="audio/wav")


@app.post("/voice/chat/fast")
@limiter.limit("20/minute")
async def fast_voice_chat(request: Request, req: ChatRequest, user: dict = Depends(require_auth)):
    """Ultra-fast voice chat - streams first sentence immediately.

    Returns audio for the first sentence ASAP while generating the rest.
    Subsequent sentences are returned in the 'more_audio' field.

    Optional parameters:
    - voice: Voice ID (e.g., "Gwen_Stacy", "design:Natalie")
    - tts_backend: TTS backend ("kokoro", "qwen3")
    """
    import asyncio
    from interfaces.voice.tts import speak_with_options

    query = req.message
    conv_id = req.session_id  # Use session_id as conversation_id
    voice_id = req.voice
    tts_backend = req.tts_backend

    # Get conversation context
    messages = None
    if conv_id:
        conv = get_conversation(conv_id)
        if conv:
            messages = [{"role": "system", "content": get_system_prompt(query)}]
            for msg in conv["messages"][-10:]:  # Last 10 messages for context
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})

    # Stream from LLM and capture first sentence
    tier = classify_query(query)

    if tier == ModelTier.CLAUDE_CODE:
        # Claude Code doesn't stream well, use regular path
        result = await ask(query, messages=messages, tier=tier, stream=False)
        response_text = result["response"] if isinstance(result, dict) else result
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            lambda: speak_with_options(response_text, voice_id=voice_id, backend=tts_backend)
        )
        return Response(content=audio_data, media_type="audio/wav")

    # Stream from local LLM
    stream = await ask(query, messages=messages, tier=tier, stream=True)

    # Collect text until first sentence boundary
    buffer = ""
    first_sentence = ""
    remaining = []
    sentence_end = re.compile(r'[.!?]\s*$')

    async for chunk in stream:
        buffer += chunk
        # Check for sentence boundary
        if not first_sentence and sentence_end.search(buffer):
            # Found first sentence!
            first_sentence = buffer.strip()
            buffer = ""
        elif first_sentence:
            # Collecting remaining text
            remaining.append(chunk)

    # If no sentence boundary found, use whole buffer
    if not first_sentence:
        first_sentence = buffer.strip()

    full_response = first_sentence + "".join(remaining)

    # Save to conversation
    if conv_id:
        add_message(conv_id, "user", query, "local")
        add_message(conv_id, "assistant", full_response, "local")

    # Generate TTS for first sentence (fast response)
    loop = asyncio.get_event_loop()
    first_audio = await loop.run_in_executor(
        None,
        lambda: speak_with_options(first_sentence, voice_id=voice_id, backend=tts_backend)
    )

    # Return first sentence audio immediately
    # Client can request remaining audio separately if needed
    return Response(
        content=first_audio,
        media_type="audio/wav",
        headers={
            "X-Full-Response": full_response[:500],  # Truncated for header
            "X-Has-More": "true" if remaining else "false",
        }
    )


# Pre-loaded acknowledgment audio cache
_ACK_CACHE = {}

def _load_acknowledgments():
    """Load pre-generated acknowledgment audio into memory."""
    global _ACK_CACHE
    cache_dir = Path("/home/aialfred/alfred/data/audio_cache")
    for wav_file in cache_dir.glob("*.wav"):
        _ACK_CACHE[wav_file.stem] = wav_file.read_bytes()
    logger.info(f"Loaded {len(_ACK_CACHE)} acknowledgment audio clips")

# Load on module import
_load_acknowledgments()


@app.post("/voice/chat/hybrid")
@limiter.limit("20/minute")
async def hybrid_voice_chat(request: Request, req: ChatRequest, user: dict = Depends(require_auth)):
    """Hybrid voice chat - instant acknowledgment + smart response.

    Returns a multipart response:
    1. Acknowledgment audio (instant, <100ms)
    2. Full response audio (smart model, streamed)

    Optional parameters:
    - voice: Voice ID (e.g., "Gwen_Stacy", "design:Natalie")
    - tts_backend: TTS backend ("kokoro", "qwen3")
    """
    import asyncio
    import random
    from interfaces.voice.tts import speak_with_options

    query = req.message
    conv_id = req.session_id  # Use session_id as conversation_id
    voice_id = req.voice
    tts_backend = req.tts_backend

    # Pick a random acknowledgment
    ack_keys = list(_ACK_CACHE.keys()) or ["one_moment"]
    ack_audio = _ACK_CACHE.get(random.choice(ack_keys), b"")

    async def generate_audio_stream():
        # First, yield acknowledgment immediately
        if ack_audio:
            # Send acknowledgment with a marker
            yield b"--ACKNOWLEDGE--"
            yield ack_audio
            yield b"--END_ACK--"

        # Now process with smart model (mistral-large)
        messages = None
        if conv_id:
            conv = get_conversation(conv_id)
            if conv:
                messages = [{"role": "system", "content": get_system_prompt(query)}]
                for msg in conv["messages"][-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": query})

        # Use configured model from settings
        import ollama as ollama_client
        smart_model = settings.ollama_model

        if messages is None:
            messages = [
                {"role": "system", "content": get_system_prompt(query)},
                {"role": "user", "content": query},
            ]

        # Get response from smart model
        response = ollama_client.chat(model=smart_model, messages=messages)
        response_text = response["message"]["content"]

        # Save to conversation
        if conv_id:
            add_message(conv_id, "user", query, "local")
            add_message(conv_id, "assistant", response_text, "local")

        # Generate TTS for response
        loop = asyncio.get_event_loop()
        response_audio = await loop.run_in_executor(
            None,
            lambda: speak_with_options(response_text, voice_id=voice_id, backend=tts_backend)
        )

        # Send response audio
        yield b"--RESPONSE--"
        yield response_audio
        yield b"--END_RESPONSE--"

    return StreamingResponse(
        generate_audio_stream(),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "wav"}
    )


def _needs_acknowledgment(query: str) -> tuple[bool, list[str]]:
    """Determine if a query needs an acknowledgment and which ones are appropriate.

    Returns (needs_ack, appropriate_ack_keys).
    - Conversational queries (greetings, how are you) → no ack
    - Task queries (search, check, look up, create) → ack needed
    - Complex queries → ack needed
    """
    q = query.lower().strip()

    # Conversational/greeting patterns - NO acknowledgment needed
    no_ack_patterns = [
        # Greetings
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "what's up", "whats up", "how are you", "how's it going", "hows it going",
        "yo", "sup", "howdy",
        # Simple questions about Alfred
        "who are you", "what are you", "what can you do", "help me",
        # Thank you / pleasantries
        "thank you", "thanks", "great", "perfect", "awesome", "cool", "ok", "okay",
        "goodbye", "bye", "good night", "see you", "later",
        # Very short queries (likely conversational)
    ]

    # Check for greeting patterns
    for pattern in no_ack_patterns:
        if q == pattern or q.startswith(pattern + " ") or q.startswith(pattern + ","):
            return False, []

    # Very short queries are likely conversational
    if len(q.split()) <= 3 and not any(kw in q for kw in ["search", "find", "check", "look", "get", "send", "create"]):
        return False, []

    # Task-oriented patterns - acknowledgment needed
    task_keywords = [
        "search", "find", "look up", "lookup", "check", "fetch", "get me",
        "send", "email", "message", "create", "add", "schedule", "remind",
        "calendar", "crm", "contact", "upload", "download", "server",
        "analyze", "summarize", "remember", "recall", "what did",
    ]

    # Pick appropriate ack types based on query
    for kw in task_keywords:
        if kw in q:
            # Searching/checking queries → "checking", "let_me_check"
            if any(x in q for x in ["search", "find", "check", "look", "recall", "what did"]):
                return True, ["checking", "let_me_check", "one_moment"]
            # Action queries → "right_away", "certainly"
            if any(x in q for x in ["send", "create", "add", "schedule", "upload"]):
                return True, ["right_away", "certainly"]
            # Default task
            return True, ["one_moment", "certainly"]

    # Long queries likely need processing time
    if len(q.split()) > 10:
        return True, ["one_moment", "certainly"]

    # Default: skip ack for shorter, unclear queries
    return False, []


@app.post("/voice/chat/ack")
async def get_acknowledgment(request: Request, req: ChatRequest = None, user: dict = Depends(require_auth)):
    """Get a contextually appropriate acknowledgment audio clip.

    Pass the query in ChatRequest.message to get smart acknowledgment selection.
    Returns empty audio for conversational queries that don't need an ack.
    """
    import random
    if not _ACK_CACHE:
        _load_acknowledgments()

    # Get query from request body if provided
    query = ""
    if req and req.message:
        query = req.message

    # Determine if we need an acknowledgment
    needs_ack, appropriate_keys = _needs_acknowledgment(query)

    if not needs_ack:
        # Return empty audio for conversational queries
        return Response(content=b"", media_type="audio/wav")

    # Filter to available keys
    available_keys = [k for k in appropriate_keys if k in _ACK_CACHE]
    if not available_keys:
        available_keys = list(_ACK_CACHE.keys())

    if not available_keys:
        return Response(content=b"", media_type="audio/wav")

    return Response(content=_ACK_CACHE[random.choice(available_keys)], media_type="audio/wav")


@app.post("/memory/store")
async def store(req: MemoryRequest, user: dict = Depends(require_auth)):
    """Store information in Alfred's memory."""
    doc_id = store_memory(req.text, req.category)
    return {"stored": True, "id": doc_id}


@app.post("/memory/recall")
async def recall_memories(req: RecallRequest, user: dict = Depends(require_auth)):
    """Recall information from Alfred's memory."""
    memories = recall(req.query, req.category, req.n_results)
    return {"memories": memories, "count": len(memories)}


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    """WebSocket endpoint for real-time chat."""
    # Authenticate via query param or cookie
    token = ws.query_params.get("token") or ws.cookies.get("alfred_token")
    if not token:
        await ws.close(code=4001, reason="Authentication required")
        return
    from core.security.auth import decode_token
    payload = decode_token(token)
    if not payload:
        await ws.close(code=4001, reason="Invalid token")
        return
    await ws.accept()
    session_id = f"ws_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"WebSocket connected: {session_id} (user: {payload.get('sub', 'unknown')})")

    try:
        while True:
            data = await ws.receive_text()
            messages = get_session_messages(session_id)
            messages.append({"role": "user", "content": data})

            tier = classify_query(data)
            stream = await ask(data, messages=messages, tier=tier, stream=True)

            full_response = []
            async for chunk in stream:
                full_response.append(chunk)
                await ws.send_text(chunk)

            await ws.send_text("\n[END]")
            response_text = "".join(full_response)
            messages.append({"role": "assistant", "content": response_text})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")


# ==================== Wake Word WebSocket ====================

# Wake word detection state
_wakeword_model = None
_wakeword_initialized = False

def get_wakeword_model():
    """Get or initialize the wake word model."""
    global _wakeword_model, _wakeword_initialized
    if _wakeword_initialized:
        return _wakeword_model
    try:
        from interfaces.voice.wakeword import get_model
        _wakeword_model = get_model()
        _wakeword_initialized = True
        if _wakeword_model:
            logger.info("Wake word model loaded successfully")
        return _wakeword_model
    except Exception as e:
        logger.error(f"Failed to load wake word model: {e}")
        _wakeword_initialized = True
        return None


@app.websocket("/ws/wakeword")
async def websocket_wakeword(ws: WebSocket):
    """WebSocket endpoint for wake word detection streaming."""
    import numpy as np

    # Authenticate
    token = ws.query_params.get("token") or ws.cookies.get("alfred_token")
    if not token:
        await ws.close(code=4001, reason="Authentication required")
        return
    from core.security.auth import decode_token
    payload = decode_token(token)
    if not payload:
        await ws.close(code=4001, reason="Invalid token")
        return

    await ws.accept()
    logger.info(f"Wake word WebSocket connected: {payload.get('sub', 'unknown')}")

    # Get wake word model
    model = get_wakeword_model()
    if not model:
        await ws.send_json({"type": "error", "message": "Wake word model not available"})
        await ws.close()
        return

    await ws.send_json({"type": "ready", "message": "Wake word detection active"})

    # Audio buffer for processing
    audio_buffer = np.array([], dtype=np.int16)
    chunk_size = 1280  # 80ms at 16kHz
    threshold = 0.35  # Lowered for custom trained model
    log_counter = 0  # For periodic debug logging

    try:
        while True:
            # Receive binary audio data
            data = await ws.receive_bytes()

            # Handle keep-alive pings (empty data)
            if len(data) == 0:
                continue

            # Convert to numpy array (expecting 16-bit PCM)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            # Process in chunks
            while len(audio_buffer) >= chunk_size:
                process_chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]

                # Run prediction
                try:
                    prediction = model.predict(process_chunk)
                    for model_name, score in prediction.items():
                        # Log high scores for debugging (every 50th chunk to avoid spam)
                        log_counter += 1
                        if score > 0.1 or (log_counter % 50 == 0 and score > 0.01):
                            logger.debug(f"Wake word score: {score:.3f}")

                        if score >= threshold:
                            logger.info(f"Wake word detected! Score: {score:.3f}")
                            await ws.send_json({
                                "type": "detected",
                                "score": float(score),
                                "model": model_name
                            })
                            # Reset model state
                            model.reset()
                            audio_buffer = np.array([], dtype=np.int16)
                            break
                except Exception as e:
                    logger.error(f"Wake word prediction error: {e}")

    except WebSocketDisconnect:
        logger.info("Wake word WebSocket disconnected")
    except Exception as e:
        logger.error(f"Wake word WebSocket error: {e}")


# ==================== HTML UI ====================

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>Alfred</title>
    <link rel="manifest" href="/manifest.json">
    <link rel="apple-touch-icon" href="/static/apple-touch-icon.png">
    <meta name="theme-color" content="#0a0a0a">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Alfred">
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.18/dist/bundle.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0a; color: #e0e0e0; height: 100vh;
            display: flex; flex-direction: column;
            padding-top: env(safe-area-inset-top);
            padding-bottom: env(safe-area-inset-bottom);
        }
        header {
            padding: 12px 16px; border-bottom: 1px solid #2f2f2f;
            display: flex; align-items: center; gap: 12px;
            background: #0a0a0a;
        }
        header h1 { font-size: 16px; font-weight: 500; color: #e0e0e0; }
        header .status { display: none; }
        .header-right { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .header-btn {
            background: transparent; border: none; color: #8e8e8e; padding: 8px 12px;
            border-radius: 8px; cursor: pointer; font-size: 13px;
        }
        .header-btn:hover { background: #2f2f2f; color: #e0e0e0; }
        #hamburger-btn {
            background: none; border: none; color: #8e8e8e; font-size: 20px;
            cursor: pointer; padding: 8px; border-radius: 8px; line-height: 1;
        }
        #hamburger-btn:hover { background: #2f2f2f; color: #e0e0e0; }
        #chat {
            flex: 1; overflow-y: auto; padding: 24px 48px 120px;
            display: flex; flex-direction: column; gap: 24px;
            width: 100%;
        }
        #chat.welcome-state { display: none; }

        /* Welcome Screen */
        #welcome-screen {
            flex: 1; display: flex; flex-direction: column;
            justify-content: center; align-items: center;
            padding: 24px 48px; display: none;
            width: 100%; height: 100%;
        }
        #welcome-screen.visible { display: flex; }
        #welcome-screen h2 {
            font-size: 32px; font-weight: 500; color: #e0e0e0;
            margin-bottom: 32px;
        }
        #welcome-input-container {
            width: 100%; max-width: 100%; padding: 0 10%;
        }
        #welcome-input-box {
            background: #2f2f2f; border-radius: 24px;
            padding: 12px 16px; display: flex; flex-direction: column;
            border: 1px solid #424242;
        }
        #welcome-input {
            background: transparent; border: none; color: #e0e0e0;
            font-size: 16px; outline: none; resize: none;
            min-height: 24px; max-height: 200px; padding: 4px 0;
        }
        #welcome-input::placeholder { color: #8e8e8e; }
        #welcome-input-actions {
            display: flex; align-items: center; justify-content: space-between;
            margin-top: 8px; padding-top: 8px;
        }
        #welcome-input-left { display: flex; align-items: center; gap: 8px; }
        #welcome-input-right { display: flex; align-items: center; gap: 8px; }
        .welcome-btn {
            background: transparent; border: none; color: #8e8e8e;
            cursor: pointer; padding: 8px; border-radius: 8px;
            font-size: 18px; display: flex; align-items: center; justify-content: center;
        }
        .welcome-btn:hover { background: #424242; color: #e0e0e0; }
        .mic-logo-btn {
            padding: 0; width: 36px; height: 36px; border-radius: 50%;
            overflow: hidden; background: transparent;
        }
        .mic-logo-btn img {
            width: 100%; height: 100%; object-fit: cover; border-radius: 50%;
        }
        .mic-logo-btn:hover { background: transparent; transform: scale(1.1); }
        .mic-logo-btn.recording {
            box-shadow: 0 0 12px rgba(232, 110, 44, 0.8);
            animation: pulse-glow 1s infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 8px rgba(232, 110, 44, 0.6); }
            50% { box-shadow: 0 0 16px rgba(232, 110, 44, 1); }
        }
        #welcome-send-btn {
            background: #fff; color: #000; border: none;
            width: 36px; height: 36px; border-radius: 50%;
            cursor: pointer; display: flex; align-items: center; justify-content: center;
            font-size: 16px;
        }
        #welcome-send-btn:hover { background: #d1d1d1; }
        #welcome-send-btn:disabled { background: #424242; color: #8e8e8e; cursor: not-allowed; }

        /* Message Styles - ChatGPT style */
        .msg { max-width: 100%; line-height: 1.6; }
        .msg.user {
            align-self: flex-end; max-width: 60%;
            background: #2f2f2f; color: #e0e0e0;
            padding: 12px 18px; border-radius: 20px;
        }
        .msg.alfred {
            align-self: flex-start; background: transparent; border: none;
            padding: 0; width: 100%;
        }
        .msg .label { display: none; }
        .msg.user .label { display: none; }
        .msg.alfred .label { display: none; }
        .msg .content { white-space: pre-wrap; font-family: inherit; font-size: 15px; }
        .msg .content code { background: #2f2f2f; padding: 2px 6px; border-radius: 4px; font-family: 'SF Mono', Monaco, monospace; font-size: 14px; }
        .msg .content strong { color: #fff; }
        .msg .content pre { background: #1e1e1e; border-radius: 8px; padding: 12px 16px; overflow-x: auto; margin: 12px 0; }
        .msg .content pre code { background: transparent; padding: 0; }

        /* Alfred message action buttons */
        .msg-actions {
            display: flex; align-items: center; gap: 4px;
            margin-top: 8px; opacity: 0; transition: opacity 0.2s;
        }
        .msg.alfred:hover .msg-actions { opacity: 1; }
        .msg-action-btn {
            background: transparent; border: none; color: #8e8e8e;
            cursor: pointer; padding: 6px 8px; border-radius: 6px;
            font-size: 14px; display: flex; align-items: center; justify-content: center;
        }
        .msg-action-btn:hover { background: #2f2f2f; color: #e0e0e0; }
        .msg-action-btn.active { color: #10b981; }
        .msg-action-btn svg { width: 18px; height: 18px; }

        /* Thinking indicator */
        #thinking {
            display: none; align-self: flex-start; padding: 8px 0;
            background: transparent; border: none; border-radius: 0;
            margin: 0;
        }
        #thinking.visible { display: flex; align-items: center; gap: 12px; }
        #thinking .label { font-size: 14px; color: #8e8e8e; }
        .morph-shape {
            width: 24px; height: 24px;
            background: linear-gradient(135deg, #4a9eff, #e86e2c);
            animation: morph 2s ease-in-out infinite, spin 3s linear infinite, colorShift 4s ease-in-out infinite;
        }
        @keyframes morph {
            0%, 100% { border-radius: 50%; }           /* circle */
            25% { border-radius: 0; }                  /* square */
            50% { border-radius: 50% 0 50% 0; }        /* diamond */
            75% { border-radius: 50% 50% 0 50%; }      /* leaf */
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        @keyframes colorShift {
            0%, 100% { background: linear-gradient(135deg, #4a9eff, #e86e2c); }
            50% { background: linear-gradient(135deg, #e86e2c, #4ade80); }
        }
        #input-area {
            position: fixed; bottom: 0; left: 0; right: 0;
            padding: 16px 24px 24px;
            display: flex; justify-content: center;
            background: linear-gradient(transparent, #0a0a0a 20%);
            padding-bottom: max(24px, env(safe-area-inset-bottom));
        }
        #input-area.hidden { display: none; }
        #input-box {
            background: #2f2f2f; border-radius: 24px;
            padding: 12px 20px; display: flex; flex-direction: column;
            border: 1px solid #424242; width: 100%;
            margin: 0 10%;
        }
        #input {
            background: transparent; border: none; color: #e0e0e0;
            font-size: 15px; outline: none; resize: none;
            min-height: 24px; max-height: 200px; padding: 4px 0;
            flex: 1;
        }
        #input::placeholder { color: #8e8e8e; }
        #input:focus { border-color: transparent; }
        #input-actions {
            display: flex; align-items: center; justify-content: space-between;
            margin-top: 8px;
        }
        #input-left { display: flex; align-items: center; gap: 4px; }
        #input-right { display: flex; align-items: center; gap: 8px; }
        button {
            padding: 10px 20px; border-radius: 8px; border: none;
            background: #2563eb; color: white; font-size: 15px;
            cursor: pointer; font-weight: 500;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        #send-btn {
            background: #fff; color: #000; border: none;
            width: 36px; height: 36px; border-radius: 50%;
            cursor: pointer; display: flex; align-items: center; justify-content: center;
            font-size: 16px; padding: 0;
        }
        #send-btn:hover { background: #d1d1d1; }
        #send-btn:disabled { background: #424242; color: #8e8e8e; cursor: not-allowed; }
        #mic-btn {
            background: transparent; width: 36px; height: 36px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            padding: 0; overflow: hidden; border: none; transition: all 0.2s;
            color: #8e8e8e; font-size: 18px;
        }
        #mic-btn img { width: 28px; height: 28px; object-fit: cover; border-radius: 50%; }
        #mic-btn:hover { background: #424242; color: #e0e0e0; }
        #mic-btn.recording { background: #dc2626; color: #fff; animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }
        @keyframes glow-listen { 0%,100% { box-shadow: 0 0 8px rgba(74,222,128,0.4); } 50% { box-shadow: 0 0 16px rgba(74,222,128,0.7); } }
        @keyframes glow-hear { 0%,100% { box-shadow: 0 0 10px rgba(232,110,44,0.5); } 50% { box-shadow: 0 0 20px rgba(232,110,44,0.9); } }
        @keyframes glow-process { 0% { box-shadow: 0 0 8px rgba(96,165,250,0.4); } 50% { box-shadow: 0 0 18px rgba(96,165,250,0.8); } 100% { box-shadow: 0 0 8px rgba(96,165,250,0.4); } }
        @keyframes glow-speak { 0%,100% { box-shadow: 0 0 8px rgba(167,139,250,0.4); } 50% { box-shadow: 0 0 16px rgba(167,139,250,0.7); } }
        #mic-btn.vad-listen { border-color: #4ade80; animation: glow-listen 2s ease-in-out infinite; }
        #mic-btn.vad-hear { border-color: #e86e2c; animation: glow-hear 0.6s ease-in-out infinite; }
        #mic-btn.vad-process { border-color: #60a5fa; animation: glow-process 1s linear infinite; }
        #mic-btn.vad-speak { border-color: #a78bfa; animation: glow-speak 1.5s ease-in-out infinite; }
        .tier-badge {
            font-size: 10px; padding: 2px 6px; border-radius: 4px;
            display: inline-block; margin-top: 4px;
        }
        .tier-local { background: #166534; color: #4ade80; }
        .tier-cloud { background: #1e3a5f; color: #60a5fa; }
        .tier-claude-code { background: #5b21b6; color: #c4b5fd; }
        .speak-btn {
            background: none; border: none; color: #666; cursor: pointer;
            font-size: 16px; padding: 2px 6px; border-radius: 4px;
            margin-left: 8px; vertical-align: middle;
        }
        .speak-btn:hover { color: #4a9eff; background: #1a1a2e; }
        .speak-btn.speaking { color: #4ade80; animation: pulse 1s infinite; }
        .stop-btn {
            display: none; background: none; border: none; color: #666; cursor: pointer;
            font-size: 14px; padding: 2px 6px; border-radius: 4px;
            margin-left: 4px; vertical-align: middle;
        }
        .stop-btn.visible { display: inline-block; }
        .stop-btn:hover { color: #f87171; background: #1a1a1a; }
        .upload-btn.has-file { color: #10b981; }
        #file-input { display: none; }
        #pending-file {
            display: none; padding: 8px 16px; background: #2f2f2f;
            border-radius: 12px; font-size: 13px;
            align-items: center; gap: 10px;
            position: fixed; bottom: 100px; left: 10%; right: 10%;
            z-index: 10; border: 1px solid #424242;
        }
        #pending-file.visible { display: flex; }
        #pending-file img { max-width: 80px; max-height: 60px; border-radius: 4px; }
        #pending-file .file-info { flex: 1; color: #ccc; }
        #pending-file .file-name { font-weight: 500; }
        #pending-file .file-meta { font-size: 11px; color: #666; margin-top: 2px; }
        #pending-file .remove-file {
            background: none; border: none; color: #666; cursor: pointer;
            font-size: 18px; padding: 4px;
        }
        #pending-file .remove-file:hover { color: #f87171; }
        .msg .download-btn {
            display: inline-block; margin-top: 8px; padding: 6px 12px;
            background: #2563eb; color: white; border-radius: 6px;
            text-decoration: none; font-size: 12px;
        }
        .msg .download-btn:hover { background: #1d4ed8; }
        .msg .inline-image { max-width: 100%; max-height: 300px; border-radius: 8px; margin-top: 8px; }
        .mode-toggle {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 7px 14px; border-radius: 20px;
            border: 1px solid #2a2a2a; background: #161616;
            color: #555; cursor: pointer; font-size: 12px;
            font-weight: 500; transition: all 0.3s ease;
            -webkit-tap-highlight-color: transparent;
            letter-spacing: 0.2px;
        }
        .mode-toggle:hover { border-color: #444; color: #999; }
        .mode-toggle .dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: #333; transition: all 0.3s ease;
        }
        #auto-speak-btn.active {
            border-color: rgba(74,222,128,0.3); background: #0a1f0e; color: #4ade80;
        }
        #auto-speak-btn.active .dot { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
        #handsfree-btn.active {
            border-color: rgba(232,110,44,0.3); background: #1a0f08; color: #e86e2c;
        }
        #handsfree-btn.active .dot { background: #e86e2c; box-shadow: 0 0 6px #e86e2c; }
        #wakeword-btn.active {
            border-color: rgba(96,165,250,0.3); background: #0a1020; color: #60a5fa;
        }
        #wakeword-btn.active .dot { background: #60a5fa; box-shadow: 0 0 6px #60a5fa; animation: pulse 2s infinite; }
        #wakeword-status {
            display: none; padding: 4px 16px; text-align: center;
            font-size: 11px; color: #60a5fa; background: rgba(96,165,250,0.1);
            border-radius: 4px; margin: 0 8px;
        }
        #wakeword-status.visible { display: inline-block; }
        #vad-status {
            display: none; padding: 8px 16px; text-align: center;
            font-size: 12px; color: #8e8e8e;
            letter-spacing: 0.3px;
            position: fixed; bottom: 90px; left: 50%;
            transform: translateX(-50%);
            background: #2f2f2f; border-radius: 20px;
            border: 1px solid #424242;
        }
        #vad-status.visible { display: block; }
        #vad-status .vad-dot {
            display: inline-block; width: 6px; height: 6px; border-radius: 50%;
            margin-right: 6px; vertical-align: middle;
        }
        #vad-status.state-listen { color: #4ade80; }
        #vad-status.state-listen .vad-dot { background: #4ade80; animation: pulse 2s infinite; }
        #vad-status.state-hear { color: #e86e2c; }
        #vad-status.state-hear .vad-dot { background: #e86e2c; animation: pulse 0.5s infinite; }
        #vad-status.state-process { color: #60a5fa; }
        #vad-status.state-process .vad-dot { background: #60a5fa; animation: pulse 0.8s infinite; }
        #vad-status.state-speak { color: #a78bfa; }
        #vad-status.state-speak .vad-dot { background: #a78bfa; animation: pulse 1.2s infinite; }

        /* Login overlay */
        #login-overlay {
            position: fixed; top:0; left:0; width:100%; height:100%;
            background: #0a0a0a; z-index: 100;
            display: flex; align-items: center; justify-content: center;
        }
        #login-overlay.hidden { display: none; }
        .login-box {
            background: #111; border: 1px solid #333; border-radius: 12px;
            padding: 32px; width: 360px;
        }
        .login-box h2 { margin-bottom: 24px; color: #fff; font-size: 22px; text-align: center; }
        .login-box input {
            width: 100%; padding: 10px 14px; margin-bottom: 12px;
            border-radius: 6px; border: 1px solid #333; background: #1a1a1a;
            color: #e0e0e0; font-size: 14px; outline: none;
        }
        .login-box input:focus { border-color: #4a9eff; }
        .login-box button { width: 100%; margin-top: 8px; }
        .login-error { color: #f87171; font-size: 13px; margin-bottom: 8px; }

        /* Settings panel */
        #settings-panel {
            position: fixed; top:0; right:-420px; width:400px; height:100%;
            background: #111; border-left: 1px solid #333; z-index: 90;
            transition: right 0.3s; padding: 24px; overflow-y: auto;
        }
        #settings-panel.open { right: 0; }
        #settings-panel h2 { font-size: 18px; margin-bottom: 20px; color: #fff; }
        .setting-section { margin-bottom: 24px; }
        .setting-section h3 {
            font-size: 14px; color: #888; text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 10px;
        }
        .status-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 0; border-bottom: 1px solid #1a1a1a;
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px;
        }
        .status-dot.green { background: #4ade80; }
        .status-dot.red { background: #f87171; }
        .status-dot.yellow { background: #facc15; }
        .connect-btn {
            padding: 4px 10px; font-size: 12px; border-radius: 4px;
            background: #2563eb; border: none; color: white; cursor: pointer;
        }
        .connect-btn:hover { background: #1d4ed8; }
        .tts-option {
            flex: 1; min-width: 120px; padding: 12px; border-radius: 8px;
            background: #1a1a1a; border: 2px solid #333; cursor: pointer;
            text-align: left; transition: all 0.2s;
        }
        .tts-option:hover { border-color: #555; }
        .tts-option.active { border-color: #3b82f6; background: #1e3a5f; }
        .tts-option .tts-name { display: block; font-weight: 600; color: #e0e0e0; font-size: 14px; }
        .tts-option .tts-desc { display: block; font-size: 11px; color: #888; margin-top: 4px; }
        .close-btn {
            position: absolute; top: 16px; right: 16px; background: none;
            border: none; color: #888; font-size: 24px; cursor: pointer;
        }
        .close-btn:hover { color: #fff; }
        #settings-overlay {
            position: fixed; top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.5); z-index: 85; display: none;
        }
        #settings-overlay.open { display: block; }

        /* History drawer */
        #history-overlay {
            position: fixed; top:0; left:0; width:100%; height:100%;
            background: rgba(0,0,0,0.5); z-index: 85; display: none;
        }
        #history-overlay.open { display: block; }
        #history-panel {
            position: fixed; top:0; left:-320px; width:300px; height:100%;
            background: #111; border-right: 1px solid #333; z-index: 90;
            transition: left 0.3s; display: flex; flex-direction: column;
        }
        #history-panel.open { left: 0; }
        .history-header {
            padding: 12px 16px; border-bottom: 1px solid #222;
            display: flex; align-items: center; justify-content: space-between;
        }
        .history-header h2 { font-size: 16px; color: #fff; }
        #new-chat-btn {
            padding: 6px 14px; font-size: 13px; border-radius: 6px;
            background: #2563eb; border: none; color: white; cursor: pointer;
        }
        #new-chat-btn:hover { background: #1d4ed8; }

        /* Search bar */
        #search-container { padding: 8px 12px; border-bottom: 1px solid #222; }
        #search-input {
            width: 100%; padding: 8px 12px; border-radius: 6px;
            border: 1px solid #333; background: #1a1a1a; color: #e0e0e0;
            font-size: 13px; outline: none;
        }
        #search-input:focus { border-color: #4a9eff; }
        #search-input::placeholder { color: #666; }
        #search-results { display: none; padding: 8px; border-bottom: 1px solid #222; max-height: 200px; overflow-y: auto; }
        #search-results.visible { display: block; }
        .search-result { padding: 8px 10px; border-radius: 6px; cursor: pointer; margin-bottom: 4px; }
        .search-result:hover { background: #1a1a1a; }
        .search-result-title { font-size: 13px; color: #e0e0e0; }
        .search-result-snippet { font-size: 11px; color: #888; margin-top: 2px; }
        .search-result-snippet mark { background: #3b5998; color: #fff; padding: 0 2px; border-radius: 2px; }

        /* Sidebar sections */
        .sidebar-content { flex: 1; overflow-y: auto; }
        #sidebar-footer {
            padding: 12px 16px; border-top: 1px solid #2f2f2f;
            background: #0a0a0a;
        }
        #sidebar-settings-btn {
            display: flex; align-items: center; gap: 10px;
            width: 100%; padding: 10px 12px; border-radius: 8px;
            background: transparent; border: none; color: #8e8e8e;
            font-size: 14px; cursor: pointer; text-align: left;
        }
        #sidebar-settings-btn:hover { background: #2f2f2f; color: #e0e0e0; }
        #sidebar-settings-btn .settings-icon { font-size: 18px; }
        .sidebar-section { border-bottom: 1px solid #1a1a1a; }
        .section-header {
            padding: 10px 12px; display: flex; align-items: center; gap: 8px;
            cursor: pointer; user-select: none; color: #888; font-size: 12px;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .section-header:hover { color: #ccc; }
        .section-header .arrow { transition: transform 0.2s; font-size: 10px; }
        .section-header.collapsed .arrow { transform: rotate(-90deg); }
        .section-content { display: block; }
        .section-header.collapsed + .section-content { display: none; }
        .section-badge {
            background: #3b82f6; color: #fff; font-size: 11px;
            padding: 2px 6px; border-radius: 10px; margin-left: auto;
            min-width: 18px; text-align: center;
        }

        /* Projects */
        .project-item {
            padding: 8px 12px; border-radius: 6px; cursor: pointer;
            margin: 2px 8px; display: flex; align-items: center; gap: 8px;
        }
        .project-item:hover { background: #1a1a1a; }
        .project-item.active { background: #1e3a5f; }
        .project-color { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
        .project-name { font-size: 13px; color: #e0e0e0; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .project-count { font-size: 11px; color: #666; }
        #add-project-btn {
            margin: 8px 12px; padding: 6px 12px; font-size: 12px;
            background: transparent; border: 1px dashed #333; color: #666;
            border-radius: 6px; cursor: pointer; width: calc(100% - 24px);
        }
        #add-project-btn:hover { border-color: #4a9eff; color: #4a9eff; }

        /* Conversation list */
        #conversation-list { padding: 4px 0; }
        .conv-item {
            padding: 10px 12px; border-radius: 8px; cursor: pointer;
            margin: 2px 8px; transition: background 0.15s;
        }
        .conv-item:hover { background: #1a1a1a; }
        .conv-item.active { background: #1e3a5f; }
        .conv-item-title {
            font-size: 14px; color: #e0e0e0; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis;
        }
        .conv-item-meta {
            font-size: 11px; color: #666; margin-top: 3px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .conv-item-preview {
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            flex: 1; margin-right: 8px;
        }
        .conv-actions { display: flex; gap: 2px; visibility: hidden; }
        .conv-item:hover .conv-actions { visibility: visible; }
        .conv-action-btn {
            background: none; border: none; color: #555; cursor: pointer;
            font-size: 12px; padding: 2px 4px; border-radius: 3px;
        }
        .conv-action-btn:hover { color: #fff; background: #333; }
        .conv-action-btn.delete:hover { color: #f87171; }
        @media (pointer: coarse) { .conv-actions { visibility: visible !important; } }

        /* Project view header */
        #project-view-header {
            display: none; padding: 12px; border-bottom: 1px solid #222;
            background: #0f0f0f;
        }
        #project-view-header.visible { display: block; }
        .project-view-title { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
        .project-view-name { font-size: 15px; font-weight: 500; color: #fff; }
        .project-view-desc { font-size: 12px; color: #888; margin-bottom: 8px; }
        .project-tabs { display: flex; gap: 4px; }
        .project-tab {
            padding: 6px 12px; font-size: 12px; border-radius: 4px;
            background: transparent; border: none; color: #888; cursor: pointer;
        }
        .project-tab:hover { color: #ccc; }
        .project-tab.active { background: #222; color: #fff; }

        /* References list */
        #references-list { display: none; padding: 8px; }
        #references-list.visible { display: block; }
        .ref-item {
            padding: 10px 12px; border-radius: 6px; margin-bottom: 4px;
            background: #1a1a1a; border: 1px solid #222;
        }
        .ref-item:hover { border-color: #333; }
        .ref-item-header { display: flex; align-items: center; gap: 8px; }
        .ref-type-icon { font-size: 14px; }
        .ref-title { font-size: 13px; color: #e0e0e0; flex: 1; }
        .ref-actions { display: flex; gap: 4px; }
        .ref-action-btn {
            background: none; border: none; color: #666; cursor: pointer;
            font-size: 12px; padding: 2px 4px;
        }
        .ref-action-btn:hover { color: #fff; }
        .ref-preview { font-size: 11px; color: #888; margin-top: 4px; max-height: 40px; overflow: hidden; }
        .ref-meta { font-size: 10px; color: #555; margin-top: 4px; }
        #add-ref-btn {
            margin-top: 8px; padding: 8px; font-size: 12px; width: 100%;
            background: transparent; border: 1px dashed #333; color: #666;
            border-radius: 6px; cursor: pointer;
        }
        #add-ref-btn:hover { border-color: #4a9eff; color: #4a9eff; }

        /* Modal */
        .modal-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 100; display: none;
            align-items: center; justify-content: center;
        }
        .modal-overlay.visible { display: flex; }
        .modal {
            background: #111; border: 1px solid #333; border-radius: 12px;
            padding: 24px; width: 90%; max-width: 500px; max-height: 80vh;
            overflow-y: auto;
        }
        .modal h3 { font-size: 16px; color: #fff; margin-bottom: 16px; }
        .modal-input {
            width: 100%; padding: 10px 12px; margin-bottom: 12px;
            border-radius: 6px; border: 1px solid #333; background: #1a1a1a;
            color: #e0e0e0; font-size: 14px; outline: none;
        }
        .modal-input:focus { border-color: #4a9eff; }
        .modal-textarea {
            width: 100%; padding: 10px 12px; margin-bottom: 12px;
            border-radius: 6px; border: 1px solid #333; background: #1a1a1a;
            color: #e0e0e0; font-size: 14px; outline: none; resize: vertical;
            min-height: 100px; font-family: inherit;
        }
        .modal-textarea:focus { border-color: #4a9eff; }
        .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }
        .modal-btn {
            padding: 8px 16px; border-radius: 6px; font-size: 13px;
            cursor: pointer; border: none;
        }
        .modal-btn-primary { background: #2563eb; color: white; }
        .modal-btn-primary:hover { background: #1d4ed8; }
        .modal-btn-secondary { background: #333; color: #ccc; }
        .modal-btn-secondary:hover { background: #444; }
        .color-picker { display: flex; gap: 8px; margin-bottom: 12px; }
        .color-option {
            width: 24px; height: 24px; border-radius: 4px; cursor: pointer;
            border: 2px solid transparent;
        }
        .color-option:hover { opacity: 0.8; }
        .color-option.selected { border-color: #fff; }

        /* File drop zone */
        .file-drop-zone {
            border: 2px dashed #333; border-radius: 8px; padding: 24px;
            text-align: center; color: #666; margin-bottom: 12px;
            transition: all 0.2s;
        }
        .file-drop-zone.dragover { border-color: #4a9eff; background: rgba(74,158,255,0.1); }
        .file-drop-zone input { display: none; }

        /* Responsive: tablet / landscape phone */
        @media (max-width: 768px) {
            .msg { max-width: 90%; }
            #settings-panel { width: 100%; right: -100%; }
            #history-panel { width: 85%; left: -85%; }
            .login-box { width: 90%; max-width: 360px; }
            header { padding: 12px 16px; gap: 8px; }
            #input-area { padding: 12px 16px; }
        }

        /* Responsive: portrait phone */
        @media (max-width: 480px) {
            .msg { max-width: 85%; font-size: 14px; }
            .msg.user { padding: 10px 14px; max-width: 85%; }
            header { padding: 10px 12px; gap: 6px; }
            header h1 { font-size: 16px; }
            .header-right { gap: 4px; }
            .header-btn { padding: 6px 10px; font-size: 12px; }
            .mode-toggle { padding: 5px 10px; font-size: 11px; gap: 5px; }
            .mode-toggle .dot { width: 5px; height: 5px; }
            #vad-status { padding: 6px 12px; font-size: 11px; bottom: 80px; left: 5%; right: 5%; transform: none; }
            #input-area { padding: 12px 12px 16px; }
            #input-box { padding: 10px 14px; margin: 0 2%; }
            #input { font-size: 14px; }
            #welcome-screen { padding: 16px; }
            #welcome-screen h2 { font-size: 22px; }
            #welcome-input-container { padding: 0 2%; }
            #welcome-input-box { padding: 10px 14px; }
            #chat { padding: 16px 12px 100px; gap: 16px; }
            #history-panel { width: 100%; left: -100%; }
            .login-box { padding: 24px 20px; }
            #pending-file { bottom: 85px; left: 5%; right: 5%; }
            .msg-actions { opacity: 1; }
        }
    </style>
</head>
<body>
    <!-- Login Overlay -->
    <div id="login-overlay">
        <div class="login-box">
            <h2>Alfred</h2>
            <div id="login-error" class="login-error" style="display:none"></div>
            <input id="login-user" type="text" placeholder="Username" autocomplete="username">
            <input id="login-pass" type="password" placeholder="Password" autocomplete="current-password"
                onkeydown="if(event.key==='Enter')doLogin()">
            <button onclick="doLogin()">Sign In</button>
        </div>
    </div>

    <!-- History Drawer -->
    <div id="history-overlay" onclick="toggleHistory()"></div>
    <div id="history-panel">
        <div class="history-header">
            <h2 id="sidebar-title">Conversations</h2>
            <button id="new-chat-btn" onclick="newConversation()">New Chat</button>
        </div>

        <!-- Search -->
        <div id="search-container">
            <input type="text" id="search-input" placeholder="Search conversations..." oninput="debounceSearch()">
        </div>
        <div id="search-results"></div>

        <!-- Project View Header (shown when viewing a project) -->
        <div id="project-view-header">
            <div class="project-view-title">
                <span class="project-color" id="project-view-color"></span>
                <span class="project-view-name" id="project-view-name"></span>
                <button class="conv-action-btn" onclick="exitProjectView()" title="Back">&#8592;</button>
                <button class="conv-action-btn" onclick="editCurrentProject()" title="Edit">&#9998;</button>
            </div>
            <div class="project-view-desc" id="project-view-desc"></div>
            <div class="project-tabs">
                <button class="project-tab active" id="tab-chats" onclick="showProjectTab('chats')">Chats</button>
                <button class="project-tab" id="tab-refs" onclick="showProjectTab('refs')">References</button>
            </div>
        </div>

        <!-- References List (in project view) -->
        <div id="references-list"></div>

        <div class="sidebar-content">
            <!-- Projects Section -->
            <div class="sidebar-section" id="projects-section">
                <div class="section-header" onclick="toggleSection('projects')">
                    <span class="arrow">&#9660;</span>
                    <span>Projects</span>
                </div>
                <div class="section-content" id="projects-content">
                    <div id="project-list"></div>
                    <button id="add-project-btn" onclick="showProjectModal()">+ New Project</button>
                </div>
            </div>

            <!-- Recent Chats Section -->
            <div class="sidebar-section" id="chats-section">
                <div class="section-header" onclick="toggleSection('chats')">
                    <span class="arrow">&#9660;</span>
                    <span>Recent Chats</span>
                </div>
                <div class="section-content" id="chats-content">
                    <div id="conversation-list"></div>
                </div>
            </div>

            <!-- Archived Section -->
            <div class="sidebar-section" id="archived-section">
                <div class="section-header collapsed" onclick="toggleSection('archived')">
                    <span class="arrow">&#9660;</span>
                    <span>Archived</span>
                    <span id="archived-count" class="section-badge"></span>
                </div>
                <div class="section-content" id="archived-content">
                    <div id="archived-list"></div>
                </div>
            </div>
        </div>

        <!-- Settings Button at Bottom -->
        <div id="sidebar-footer">
            <button id="sidebar-settings-btn" onclick="toggleSettings(); toggleHistory();">
                <span class="settings-icon">&#9881;</span>
                <span>Settings</span>
            </button>
        </div>
    </div>

    <!-- Project Modal -->
    <div class="modal-overlay" id="project-modal">
        <div class="modal">
            <h3 id="project-modal-title">New Project</h3>
            <input type="text" class="modal-input" id="project-name-input" placeholder="Project name">
            <textarea class="modal-textarea" id="project-desc-input" placeholder="Description (optional)"></textarea>
            <div class="color-picker" id="color-picker">
                <div class="color-option selected" style="background:#3b82f6" data-color="#3b82f6"></div>
                <div class="color-option" style="background:#10b981" data-color="#10b981"></div>
                <div class="color-option" style="background:#f59e0b" data-color="#f59e0b"></div>
                <div class="color-option" style="background:#ef4444" data-color="#ef4444"></div>
                <div class="color-option" style="background:#8b5cf6" data-color="#8b5cf6"></div>
                <div class="color-option" style="background:#ec4899" data-color="#ec4899"></div>
            </div>
            <div class="modal-actions">
                <button class="modal-btn modal-btn-secondary" onclick="hideProjectModal()">Cancel</button>
                <button class="modal-btn modal-btn-primary" onclick="saveProject()">Save</button>
            </div>
        </div>
    </div>

    <!-- Reference Modal -->
    <div class="modal-overlay" id="reference-modal">
        <div class="modal">
            <h3 id="reference-modal-title">Add Note</h3>
            <input type="text" class="modal-input" id="ref-title-input" placeholder="Title">
            <textarea class="modal-textarea" id="ref-content-input" placeholder="Note content (Markdown supported)" style="min-height:150px"></textarea>
            <div class="file-drop-zone" id="file-drop-zone" style="display:none">
                <div>Drop file here or click to upload</div>
                <div style="font-size:11px;margin-top:8px">PDF, TXT, MD, DOCX, PNG, JPG (max 10MB)</div>
                <input type="file" id="ref-file-input" accept=".pdf,.txt,.md,.docx,.png,.jpg,.jpeg,.gif,.webp">
            </div>
            <div class="modal-actions">
                <button class="modal-btn modal-btn-secondary" onclick="hideReferenceModal()">Cancel</button>
                <button class="modal-btn modal-btn-primary" onclick="saveReference()">Save</button>
            </div>
        </div>
    </div>

    <!-- Move to Project Modal -->
    <div class="modal-overlay" id="move-modal">
        <div class="modal">
            <h3>Move to Project</h3>
            <div id="move-project-list"></div>
            <div class="modal-actions">
                <button class="modal-btn modal-btn-secondary" onclick="hideMoveModal()">Cancel</button>
            </div>
        </div>
    </div>

    <!-- Settings Panel -->
    <div id="settings-overlay" onclick="toggleSettings()"></div>
    <div id="settings-panel">
        <button class="close-btn" onclick="toggleSettings()">&times;</button>
        <h2>Settings</h2>

        <div class="setting-section">
            <h3>Integrations</h3>
            <div id="integration-list">Loading...</div>
        </div>

        <div class="setting-section">
            <h3>LLM</h3>
            <div id="llm-info">Loading...</div>
        </div>

        <div class="setting-section">
            <h3>Text-to-Speech</h3>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">
                <button id="tts-kokoro" class="tts-option" onclick="setTTS('kokoro')">
                    <span class="tts-name">Kokoro</span>
                    <span class="tts-desc">Fast, local</span>
                </button>
                <button id="tts-qwen3" class="tts-option" onclick="setTTS('qwen3')">
                    <span class="tts-name">Qwen3</span>
                    <span class="tts-desc">High quality, cloning</span>
                </button>
            </div>
            <div style="margin-bottom:8px">
                <label style="font-size:12px;color:#888;display:block;margin-bottom:4px">Voice</label>
                <select id="tts-voice" onchange="setVoice(this.value)" style="width:100%;padding:8px 12px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#e0e0e0;font-size:13px">
                    <option value="">Loading voices...</option>
                </select>
            </div>
            <div id="tts-status" style="font-size:12px;color:#888;margin-top:8px"></div>
        </div>

        <div class="setting-section">
            <h3>Account</h3>
            <div id="account-info"></div>
            <div style="margin-top:12px">
                <input id="new-pass" type="password" placeholder="New password" style="width:100%;padding:8px 12px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#e0e0e0;font-size:13px;margin-bottom:8px;outline:none">
                <button class="header-btn" style="width:100%" onclick="changePassword()">Change Password</button>
                <div id="pass-msg" style="font-size:12px;margin-top:6px;display:none"></div>
            </div>
            <button class="header-btn" style="margin-top:12px;width:100%" onclick="doLogout()">Sign Out</button>
        </div>
    </div>

    <header>
        <button id="hamburger-btn" onclick="toggleHistory()" title="Conversation history">&#9776;</button>
        <h1>Alfred</h1>
        <span class="status" id="header-status">Online</span>
        <div class="header-right">
            <button id="wakeword-btn" class="mode-toggle" onclick="toggleWakeWord()" title="'Hey Alfred' wake word detection"><span class="dot"></span>Hey Alfred</button>
            <span id="wakeword-status">Listening for "Hey Alfred"...</span>
            <button id="handsfree-btn" class="mode-toggle" onclick="toggleHandsFree()" title="Hands-free voice conversation"><span class="dot"></span>Hands-free</button>
            <button id="auto-speak-btn" class="mode-toggle" onclick="toggleAutoSpeak()" title="Auto-speak responses"><span class="dot"></span>Auto-speak</button>
            <button class="header-btn" onclick="toggleSettings()">&#9881;</button>
        </div>
    </header>

    <!-- Welcome Screen (shown when no messages) -->
    <div id="welcome-screen" class="visible">
        <h2>What can I help with?</h2>
        <div id="welcome-input-container">
            <div id="welcome-input-box">
                <textarea id="welcome-input" placeholder="Ask anything" rows="1"
                    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendFromWelcome()}"
                    oninput="autoResizeWelcome(this)"></textarea>
                <div id="welcome-input-actions">
                    <div id="welcome-input-left">
                        <button class="welcome-btn upload-btn" onclick="document.getElementById('file-input').click()" title="Attach">&#43;</button>
                    </div>
                    <div id="welcome-input-right">
                        <button class="welcome-btn mic-logo-btn" id="welcome-mic-btn" onclick="toggleMic()" title="Voice input"><img src="/static/gr-logo.jpeg" alt="Mic"></button>
                        <button id="welcome-send-btn" onclick="sendFromWelcome()" title="Send" disabled>&#9650;</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Chat Area (shown after first message) -->
    <div id="chat" class="welcome-state">
        <div id="thinking">
            <div class="morph-shape"></div>
            <div class="label">Alfred is thinking...</div>
        </div>
    </div>

    <div id="vad-status"><span class="vad-dot"></span><span id="vad-status-text">Listening...</span></div>
    <div id="pending-file">
        <img id="pending-preview" src="" alt="">
        <div class="file-info">
            <div class="file-name" id="pending-name"></div>
            <div class="file-meta" id="pending-meta"></div>
        </div>
        <button class="remove-file" onclick="clearPendingFile()" title="Remove">&times;</button>
    </div>

    <!-- Bottom Input (shown after first message) -->
    <div id="input-area" class="hidden">
        <div id="input-box">
            <textarea id="input" placeholder="Ask anything" rows="1"
                onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"
                oninput="autoResizeInput(this)"></textarea>
            <div id="input-actions">
                <div id="input-left">
                    <button class="welcome-btn upload-btn" onclick="document.getElementById('file-input').click()" title="Attach">&#43;</button>
                </div>
                <div id="input-right">
                    <button class="welcome-btn mic-logo-btn" id="mic-btn" onclick="toggleMic()" title="Voice input"><img src="/static/gr-logo.jpeg" alt="Mic"></button>
                    <button id="send-btn" onclick="send()" title="Send">&#9650;</button>
                </div>
            </div>
        </div>
        <input type="file" id="file-input" accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.md,.json,.jpg,.jpeg,.png,.gif,.webp" onchange="handleFileSelect(event)">
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        let mediaRecorder = null;
        let isRecording = false;
        let authToken = localStorage.getItem('alfred_token') || '';
        let currentConversationId = null;
        let loadingHistory = false;

        // ==================== State ====================
        let currentProjectId = null;
        let editingProjectId = null;
        let editingRefId = null;
        let projects = [];
        let searchTimeout = null;
        let isWelcomeState = true;

        // ==================== Welcome Screen ====================

        function showWelcomeState() {
            isWelcomeState = true;
            document.getElementById('welcome-screen').classList.add('visible');
            document.getElementById('chat').classList.add('welcome-state');
            document.getElementById('input-area').classList.add('hidden');
        }

        function showChatState() {
            isWelcomeState = false;
            document.getElementById('welcome-screen').classList.remove('visible');
            document.getElementById('chat').classList.remove('welcome-state');
            document.getElementById('input-area').classList.remove('hidden');
        }

        function autoResizeWelcome(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 200) + 'px';
            // Enable/disable send button
            const btn = document.getElementById('welcome-send-btn');
            btn.disabled = !el.value.trim();
        }

        function autoResizeInput(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 200) + 'px';
        }

        function sendFromWelcome() {
            const welcomeInput = document.getElementById('welcome-input');
            const text = welcomeInput.value.trim();
            if (!text) return;
            // Transfer to main input and send
            document.getElementById('input').value = text;
            welcomeInput.value = '';
            showChatState();
            send();
        }

        // Enable welcome send button on input
        document.getElementById('welcome-input')?.addEventListener('input', function() {
            document.getElementById('welcome-send-btn').disabled = !this.value.trim();
        });

        // ==================== Conversation History ====================

        function toggleHistory() {
            document.getElementById('history-panel').classList.toggle('open');
            document.getElementById('history-overlay').classList.toggle('open');
        }

        function toggleSection(section) {
            const header = document.querySelector(`#${section}-section .section-header`);
            header.classList.toggle('collapsed');
        }

        function formatRelativeDate(isoStr) {
            const d = new Date(isoStr);
            const now = new Date();
            const diffMs = now - d;
            const diffMin = Math.floor(diffMs / 60000);
            if (diffMin < 1) return 'just now';
            if (diffMin < 60) return diffMin + 'm ago';
            const diffHr = Math.floor(diffMin / 60);
            if (diffHr < 24) return diffHr + 'h ago';
            const diffDay = Math.floor(diffHr / 24);
            if (diffDay < 7) return diffDay + 'd ago';
            return d.toLocaleDateString();
        }

        function authHeaders(extra) {
            const h = authToken ? {'Authorization': 'Bearer ' + authToken} : {};
            return extra ? Object.assign(h, extra) : h;
        }

        async function loadConversations() {
            try {
                const url = currentProjectId ? `/conversations?project_id=${currentProjectId}` : '/conversations';
                const resp = await fetch(url, {headers: authHeaders()});
                const convs = await resp.json();
                const listEl = document.getElementById('conversation-list');
                if (!convs.length) {
                    listEl.innerHTML = '<div style="padding:16px;color:#666;font-size:13px;text-align:center">No conversations yet</div>';
                    return;
                }
                listEl.innerHTML = convs.map(c => {
                    const active = c.id === currentConversationId ? ' active' : '';
                    const title = c.title || 'New conversation';
                    const preview = c.last_message || '';
                    return `<div class="conv-item${active}" data-id="${c.id}" onclick="switchConversation('${c.id}')">
                        <div class="conv-item-title">${escapeHtml(title)}</div>
                        <div class="conv-item-meta">
                            <span class="conv-item-preview">${escapeHtml(preview)}</span>
                            <span>${formatRelativeDate(c.updated_at)}</span>
                            <div class="conv-actions">
                                <button class="conv-action-btn" onclick="event.stopPropagation();showMoveModal('${c.id}')" title="Move to project">&#128193;</button>
                                <button class="conv-action-btn" onclick="event.stopPropagation();archiveConversation('${c.id}')" title="Archive">&#128451;</button>
                                <button class="conv-action-btn delete" onclick="event.stopPropagation();deleteConversationPermanently('${c.id}')" title="Delete">&#10005;</button>
                            </div>
                        </div>
                    </div>`;
                }).join('');
            } catch(e) {
                console.error('Failed to load conversations:', e);
            }
        }

        async function loadArchivedConversations() {
            try {
                const resp = await fetch('/conversations/archived', {headers: authHeaders()});
                const convs = await resp.json();
                const listEl = document.getElementById('archived-list');

                // Update archived count badge
                const badge = document.getElementById('archived-count');
                if (badge) {
                    badge.textContent = convs.length || '';
                    badge.style.display = convs.length ? 'inline-block' : 'none';
                }

                if (!convs.length) {
                    listEl.innerHTML = '<div style="padding:16px;color:#666;font-size:13px;text-align:center">No archived chats</div>';
                    return;
                }
                listEl.innerHTML = convs.map(c => {
                    const title = c.title || 'Untitled conversation';
                    return `<div class="conv-item" data-id="${c.id}">
                        <div class="conv-item-title" style="color:#888">${escapeHtml(title)}</div>
                        <div class="conv-item-meta">
                            <span>${formatRelativeDate(c.updated_at)}</span>
                            <div class="conv-actions" style="visibility:visible">
                                <button class="conv-action-btn" onclick="event.stopPropagation();restoreConversation('${c.id}')" title="Restore">&#8634;</button>
                                <button class="conv-action-btn delete" onclick="event.stopPropagation();deleteConversationPermanently('${c.id}')" title="Delete permanently">&#10005;</button>
                            </div>
                        </div>
                    </div>`;
                }).join('');
            } catch(e) {
                console.error('Failed to load archived:', e);
            }
        }

        async function newConversation() {
            try {
                const resp = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
                const data = await resp.json();
                currentConversationId = data.id;
                // If in project view, assign to project
                if (currentProjectId) {
                    await fetch(`/conversations/${data.id}/project`, {
                        method: 'PUT',
                        headers: authHeaders({'Content-Type': 'application/json'}),
                        body: JSON.stringify({project_id: currentProjectId})
                    });
                }
                clearChat();
                await loadConversations();
                toggleHistory();
            } catch(e) {
                console.error('Failed to create conversation:', e);
            }
        }

        async function switchConversation(convId) {
            if (convId === currentConversationId) {
                toggleHistory();
                return;
            }
            try {
                const resp = await fetch('/conversations/' + convId, {headers: authHeaders()});
                if (!resp.ok) return;
                const data = await resp.json();
                currentConversationId = convId;
                clearChat();
                loadingHistory = true;
                if (data.messages && data.messages.length > 0) {
                    showChatState();
                    data.messages.forEach(m => {
                        if (m.role === 'user') addMsg(m.content, 'user', null, true);
                        else if (m.role === 'assistant') addMsg(m.content, 'alfred', m.tier, true);
                    });
                }
                loadingHistory = false;
                await loadConversations();
                toggleHistory();
            } catch(e) {
                console.error('Failed to load conversation:', e);
            }
        }

        async function archiveConversation(convId) {
            try {
                await fetch('/conversations/' + convId, {method: 'DELETE', headers: authHeaders()});
                if (convId === currentConversationId) {
                    currentConversationId = null;
                    clearChat();
                    const cr = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
                    currentConversationId = (await cr.json()).id;
                }
                await loadConversations();
                await loadArchivedConversations();
            } catch(e) {
                console.error('Failed to archive:', e);
            }
        }

        async function restoreConversation(convId) {
            try {
                await fetch(`/conversations/${convId}/restore`, {method: 'POST', headers: authHeaders()});
                await loadConversations();
                await loadArchivedConversations();
            } catch(e) {
                console.error('Failed to restore:', e);
            }
        }

        async function deleteConversationPermanently(convId) {
            if (!confirm('Permanently delete this conversation? This cannot be undone.')) return;
            try {
                await fetch(`/conversations/${convId}/permanent`, {method: 'DELETE', headers: authHeaders()});
                await loadArchivedConversations();
            } catch(e) {
                console.error('Failed to delete permanently:', e);
            }
        }

        function clearChat() {
            chat.innerHTML = '<div id="thinking"><div class="morph-shape"></div><div class="label">Alfred is thinking...</div></div>';
            msgTexts = {};
            msgCounter = 0;
            showWelcomeState();
        }

        async function initConversations() {
            await loadProjects();
            await loadConversations();
            await loadArchivedConversations();
            // Always start with a fresh new chat
            const cr = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
            currentConversationId = (await cr.json()).id;
            document.getElementById('messages').innerHTML = '';
            // Ensure menu stays closed on load
            document.getElementById('history-panel').classList.remove('open');
            document.getElementById('history-overlay').classList.remove('open');
        }

        // ==================== Search ====================

        function debounceSearch() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(doSearch, 300);
        }

        async function doSearch() {
            const query = document.getElementById('search-input').value.trim();
            const resultsEl = document.getElementById('search-results');
            if (query.length < 2) {
                resultsEl.classList.remove('visible');
                return;
            }
            try {
                const resp = await fetch(`/conversations/search?q=${encodeURIComponent(query)}`, {headers: authHeaders()});
                const results = await resp.json();
                if (!results.length) {
                    resultsEl.innerHTML = '<div style="padding:12px;color:#666;font-size:12px;text-align:center">No results found</div>';
                } else {
                    resultsEl.innerHTML = results.map(r => `
                        <div class="search-result" onclick="switchConversation('${r.id}')">
                            <div class="search-result-title">${escapeHtml(r.title || 'Untitled')}</div>
                            <div class="search-result-snippet">${r.snippet || ''}</div>
                        </div>
                    `).join('');
                }
                resultsEl.classList.add('visible');
            } catch(e) {
                console.error('Search failed:', e);
            }
        }

        // ==================== Projects ====================

        async function loadProjects() {
            try {
                const resp = await fetch('/projects', {headers: authHeaders()});
                projects = await resp.json();
                const listEl = document.getElementById('project-list');
                if (!projects.length) {
                    listEl.innerHTML = '';
                    return;
                }
                listEl.innerHTML = projects.map(p => `
                    <div class="project-item${currentProjectId === p.id ? ' active' : ''}" onclick="viewProject('${p.id}')">
                        <span class="project-color" style="background:${p.color}"></span>
                        <span class="project-name">${escapeHtml(p.name)}</span>
                        <span class="project-count">${p.conversation_count || 0}</span>
                    </div>
                `).join('');
            } catch(e) {
                console.error('Failed to load projects:', e);
            }
        }

        async function viewProject(projectId) {
            currentProjectId = projectId;
            const project = projects.find(p => p.id === projectId);
            if (!project) return;

            // Show project view header
            document.getElementById('project-view-header').classList.add('visible');
            document.getElementById('project-view-color').style.background = project.color;
            document.getElementById('project-view-name').textContent = project.name;
            document.getElementById('project-view-desc').textContent = project.description || '';

            // Hide sidebar sections, show project chats
            document.getElementById('projects-section').style.display = 'none';
            document.getElementById('archived-section').style.display = 'none';
            document.querySelector('#chats-section .section-header').style.display = 'none';

            showProjectTab('chats');
            await loadConversations();
            await loadProjects();
        }

        function exitProjectView() {
            currentProjectId = null;
            document.getElementById('project-view-header').classList.remove('visible');
            document.getElementById('references-list').classList.remove('visible');
            document.getElementById('projects-section').style.display = '';
            document.getElementById('archived-section').style.display = '';
            document.querySelector('#chats-section .section-header').style.display = '';
            loadConversations();
            loadProjects();
        }

        function showProjectTab(tab) {
            document.getElementById('tab-chats').classList.toggle('active', tab === 'chats');
            document.getElementById('tab-refs').classList.toggle('active', tab === 'refs');
            document.getElementById('conversation-list').parentElement.style.display = tab === 'chats' ? '' : 'none';
            document.getElementById('references-list').classList.toggle('visible', tab === 'refs');
            if (tab === 'refs') loadReferences();
        }

        function showProjectModal(projectId = null) {
            editingProjectId = projectId;
            document.getElementById('project-modal-title').textContent = projectId ? 'Edit Project' : 'New Project';
            if (projectId) {
                const p = projects.find(pr => pr.id === projectId);
                if (p) {
                    document.getElementById('project-name-input').value = p.name;
                    document.getElementById('project-desc-input').value = p.description || '';
                    selectColor(p.color);
                }
            } else {
                document.getElementById('project-name-input').value = '';
                document.getElementById('project-desc-input').value = '';
                selectColor('#3b82f6');
            }
            document.getElementById('project-modal').classList.add('visible');
        }

        function hideProjectModal() {
            document.getElementById('project-modal').classList.remove('visible');
            editingProjectId = null;
        }

        function selectColor(color) {
            document.querySelectorAll('.color-option').forEach(el => {
                el.classList.toggle('selected', el.dataset.color === color);
            });
        }

        document.getElementById('color-picker').addEventListener('click', e => {
            if (e.target.classList.contains('color-option')) {
                selectColor(e.target.dataset.color);
            }
        });

        async function saveProject() {
            const name = document.getElementById('project-name-input').value.trim();
            if (!name) return alert('Please enter a project name');
            const description = document.getElementById('project-desc-input').value.trim();
            const color = document.querySelector('.color-option.selected')?.dataset.color || '#3b82f6';

            try {
                if (editingProjectId) {
                    await fetch(`/projects/${editingProjectId}`, {
                        method: 'PUT',
                        headers: authHeaders({'Content-Type': 'application/json'}),
                        body: JSON.stringify({name, description, color})
                    });
                } else {
                    await fetch('/projects', {
                        method: 'POST',
                        headers: authHeaders({'Content-Type': 'application/json'}),
                        body: JSON.stringify({name, description, color})
                    });
                }
                hideProjectModal();
                await loadProjects();
                if (currentProjectId === editingProjectId) {
                    viewProject(currentProjectId);
                }
            } catch(e) {
                console.error('Failed to save project:', e);
            }
        }

        function editCurrentProject() {
            if (currentProjectId) showProjectModal(currentProjectId);
        }

        async function deleteProject(projectId) {
            if (!confirm('Delete this project and all its references? Conversations will be kept.')) return;
            try {
                await fetch(`/projects/${projectId}`, {method: 'DELETE', headers: authHeaders()});
                if (currentProjectId === projectId) exitProjectView();
                await loadProjects();
            } catch(e) {
                console.error('Failed to delete project:', e);
            }
        }

        // ==================== References ====================

        async function loadReferences() {
            if (!currentProjectId) return;
            try {
                const resp = await fetch(`/projects/${currentProjectId}/references`, {headers: authHeaders()});
                const refs = await resp.json();
                const listEl = document.getElementById('references-list');
                let html = '';
                if (!refs.length) {
                    html = '<div style="padding:16px;color:#666;font-size:13px;text-align:center">No references yet</div>';
                } else {
                    html = refs.map(r => {
                        const icon = r.type === 'note' ? '&#128221;' : '&#128196;';
                        const preview = r.content ? r.content.substring(0, 100) + (r.content.length > 100 ? '...' : '') : '';
                        const meta = r.type === 'file' ? `${r.file_type || 'file'} - ${formatFileSize(r.file_size)}` : '';
                        return `<div class="ref-item">
                            <div class="ref-item-header">
                                <span class="ref-type-icon">${icon}</span>
                                <span class="ref-title">${escapeHtml(r.title)}</span>
                                <div class="ref-actions">
                                    ${r.type === 'note' ? `<button class="ref-action-btn" onclick="editReference(${r.id})" title="Edit">&#9998;</button>` : ''}
                                    ${r.type === 'file' ? `<button class="ref-action-btn" onclick="downloadReference(${r.id})" title="Download">&#8681;</button>` : ''}
                                    <button class="ref-action-btn" onclick="deleteReference(${r.id})" title="Delete">&#10005;</button>
                                </div>
                            </div>
                            ${preview ? `<div class="ref-preview">${escapeHtml(preview)}</div>` : ''}
                            ${meta ? `<div class="ref-meta">${meta}</div>` : ''}
                        </div>`;
                    }).join('');
                }
                html += `<button id="add-ref-btn" onclick="showReferenceModal()">+ Add Reference</button>`;
                listEl.innerHTML = html;
            } catch(e) {
                console.error('Failed to load references:', e);
            }
        }

        function formatFileSize(bytes) {
            if (!bytes) return '';
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
            return (bytes/1024/1024).toFixed(1) + ' MB';
        }

        function showReferenceModal(refId = null, isFile = false) {
            editingRefId = refId;
            document.getElementById('reference-modal-title').textContent = refId ? 'Edit Note' : (isFile ? 'Upload File' : 'Add Note');
            document.getElementById('ref-title-input').value = '';
            document.getElementById('ref-content-input').value = '';
            document.getElementById('ref-content-input').style.display = isFile ? 'none' : '';
            document.getElementById('file-drop-zone').style.display = isFile ? 'block' : 'none';
            if (refId) {
                fetch(`/references/${refId}`, {headers: authHeaders()})
                    .then(r => r.json())
                    .then(ref => {
                        document.getElementById('ref-title-input').value = ref.title;
                        document.getElementById('ref-content-input').value = ref.content || '';
                    });
            }
            document.getElementById('reference-modal').classList.add('visible');
        }

        function hideReferenceModal() {
            document.getElementById('reference-modal').classList.remove('visible');
            editingRefId = null;
        }

        async function saveReference() {
            const title = document.getElementById('ref-title-input').value.trim();
            const content = document.getElementById('ref-content-input').value;
            const fileInput = document.getElementById('ref-file-input');

            if (fileInput.files.length > 0) {
                // File upload
                const form = new FormData();
                form.append('file', fileInput.files[0]);
                try {
                    await fetch(`/projects/${currentProjectId}/references/upload`, {
                        method: 'POST',
                        headers: authHeaders(),
                        body: form
                    });
                    hideReferenceModal();
                    fileInput.value = '';
                    await loadReferences();
                } catch(e) {
                    console.error('Upload failed:', e);
                    alert('Upload failed');
                }
            } else if (title) {
                // Note
                try {
                    if (editingRefId) {
                        await fetch(`/references/${editingRefId}`, {
                            method: 'PUT',
                            headers: authHeaders({'Content-Type': 'application/json'}),
                            body: JSON.stringify({title, content})
                        });
                    } else {
                        await fetch(`/projects/${currentProjectId}/references`, {
                            method: 'POST',
                            headers: authHeaders({'Content-Type': 'application/json'}),
                            body: JSON.stringify({title, content})
                        });
                    }
                    hideReferenceModal();
                    await loadReferences();
                } catch(e) {
                    console.error('Save failed:', e);
                }
            }
        }

        async function editReference(refId) {
            showReferenceModal(refId, false);
        }

        async function deleteReference(refId) {
            if (!confirm('Delete this reference?')) return;
            try {
                await fetch(`/references/${refId}`, {method: 'DELETE', headers: authHeaders()});
                await loadReferences();
            } catch(e) {
                console.error('Delete failed:', e);
            }
        }

        function downloadReference(refId) {
            window.open(`/references/${refId}/download`, '_blank');
        }

        // File drop zone
        const dropZone = document.getElementById('file-drop-zone');
        dropZone.addEventListener('click', () => document.getElementById('ref-file-input').click());
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                document.getElementById('ref-file-input').files = e.dataTransfer.files;
                dropZone.querySelector('div').textContent = e.dataTransfer.files[0].name;
            }
        });

        // ==================== Move to Project ====================

        let movingConvId = null;

        function showMoveModal(convId) {
            movingConvId = convId;
            const listEl = document.getElementById('move-project-list');
            let html = `<div class="project-item" onclick="moveToProject(null)">
                <span class="project-name">No Project</span>
            </div>`;
            html += projects.map(p => `
                <div class="project-item" onclick="moveToProject('${p.id}')">
                    <span class="project-color" style="background:${p.color}"></span>
                    <span class="project-name">${escapeHtml(p.name)}</span>
                </div>
            `).join('');
            listEl.innerHTML = html;
            document.getElementById('move-modal').classList.add('visible');
        }

        function hideMoveModal() {
            document.getElementById('move-modal').classList.remove('visible');
            movingConvId = null;
        }

        async function moveToProject(projectId) {
            if (!movingConvId) return;
            try {
                await fetch(`/conversations/${movingConvId}/project`, {
                    method: 'PUT',
                    headers: authHeaders({'Content-Type': 'application/json'}),
                    body: JSON.stringify({project_id: projectId})
                });
                hideMoveModal();
                await loadConversations();
                await loadProjects();
            } catch(e) {
                console.error('Move failed:', e);
            }
        }

        // ==================== Auth ====================

        (async function checkAuth() {
            try {
                const resp = await fetch('/auth/me', {headers: authHeaders()});
                const data = await resp.json();
                if (data.authenticated) {
                    document.getElementById('login-overlay').classList.add('hidden');
                    loadIntegrations();
                    initConversations();
                    // Auto-enable Hey Alfred wake word on startup
                    setTimeout(() => {
                        toggleWakeWord();
                        // Force menu closed after everything initializes
                        document.getElementById('history-panel').classList.remove('open');
                        document.getElementById('history-overlay').classList.remove('open');
                    }, 500);
                    // Extra delayed close in case async operations open it
                    setTimeout(() => {
                        document.getElementById('history-panel').classList.remove('open');
                        document.getElementById('history-overlay').classList.remove('open');
                    }, 1200);
                }
            } catch(e) {}
        })();

        async function doLogin() {
            const user = document.getElementById('login-user').value;
            const pass = document.getElementById('login-pass').value;
            const errEl = document.getElementById('login-error');
            errEl.style.display = 'none';

            try {
                const resp = await fetch('/auth/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username: user, password: pass})
                });
                if (!resp.ok) {
                    errEl.textContent = 'Invalid credentials';
                    errEl.style.display = 'block';
                    return;
                }
                const data = await resp.json();
                authToken = data.token;
                localStorage.setItem('alfred_token', authToken);
                document.getElementById('login-overlay').classList.add('hidden');
                loadIntegrations();
                initConversations();
                // Auto-enable Hey Alfred wake word on startup
                setTimeout(() => {
                    toggleWakeWord();
                    // Force menu closed after everything initializes
                    document.getElementById('history-panel').classList.remove('open');
                    document.getElementById('history-overlay').classList.remove('open');
                }, 500);
                // Extra delayed close in case async operations open it
                setTimeout(() => {
                    document.getElementById('history-panel').classList.remove('open');
                    document.getElementById('history-overlay').classList.remove('open');
                }, 1200);
            } catch(e) {
                errEl.textContent = 'Connection error';
                errEl.style.display = 'block';
            }
        }

        async function doLogout() {
            await fetch('/auth/logout', {method: 'POST'});
            localStorage.removeItem('alfred_token');
            authToken = '';
            location.reload();
        }

        async function changePassword() {
            const newPass = document.getElementById('new-pass').value;
            const msgEl = document.getElementById('pass-msg');
            if (!newPass || newPass.length < 6) {
                msgEl.textContent = 'Password must be at least 6 characters';
                msgEl.style.color = '#f87171';
                msgEl.style.display = 'block';
                return;
            }
            try {
                const resp = await fetch('/auth/change-password', {
                    method: 'POST',
                    headers: authHeaders({'Content-Type': 'application/json'}),
                    body: JSON.stringify({username: 'bruce', password: newPass})
                });
                if (resp.ok) {
                    msgEl.textContent = 'Password changed';
                    msgEl.style.color = '#4ade80';
                    document.getElementById('new-pass').value = '';
                } else {
                    msgEl.textContent = 'Failed to change password';
                    msgEl.style.color = '#f87171';
                }
            } catch(e) {
                msgEl.textContent = 'Connection error';
                msgEl.style.color = '#f87171';
            }
            msgEl.style.display = 'block';
        }

        function toggleSettings() {
            document.getElementById('settings-panel').classList.toggle('open');
            document.getElementById('settings-overlay').classList.toggle('open');
        }

        async function loadIntegrations() {
            try {
                const resp = await fetch('/integrations/status', {headers: authHeaders()});
                const data = await resp.json();

                // Integration list
                let html = '';
                // Google
                const gConnected = data.google?.connected;
                html += `<div class="status-item">
                    <span><span class="status-dot ${gConnected?'green':'red'}"></span>Google (Gmail + Calendar)</span>
                    ${gConnected ? '<span style="color:#4ade80;font-size:12px">Connected</span>' :
                    '<a href="/auth/google" class="connect-btn">Connect</a>'}
                </div>`;
                // Servers
                const sCount = data.servers?.count || 0;
                html += `<div class="status-item">
                    <span><span class="status-dot ${sCount>0?'green':'yellow'}"></span>Servers</span>
                    <span style="color:#888;font-size:12px">${sCount} registered</span>
                </div>`;
                // CRM
                const crmConnected = data.base_crm?.connected;
                const crmLabel = data.base_crm?.label || 'CRM';
                html += `<div class="status-item">
                    <span><span class="status-dot ${crmConnected?'green':'red'}"></span>${crmLabel}</span>
                    <span style="color:#888;font-size:12px">${crmConnected?'Connected':'Not connected'}</span>
                </div>`;
                // n8n
                const n8nConnected = data.n8n?.connected;
                const n8nCount = data.n8n?.workflow_count || 0;
                html += `<div class="status-item">
                    <span><span class="status-dot ${n8nConnected?'green':'red'}"></span>n8n Automations</span>
                    <span style="color:#888;font-size:12px">${n8nConnected ? n8nCount + ' workflows' : 'Not connected'}</span>
                </div>`;
                // Nextcloud
                const ncConnected = data.nextcloud?.connected;
                html += `<div class="status-item">
                    <span><span class="status-dot ${ncConnected?'green':'red'}"></span>Nextcloud</span>
                    <span style="color:#888;font-size:12px">${ncConnected?'Connected':'Not connected'}</span>
                </div>`;
                // Stripe
                const stripeConnected = data.stripe?.connected;
                html += `<div class="status-item">
                    <span><span class="status-dot ${stripeConnected?'green':'red'}"></span>Stripe Payments</span>
                    <span style="color:#888;font-size:12px">${stripeConnected?'Connected':'Not connected'}</span>
                </div>`;

                document.getElementById('integration-list').innerHTML = html;

                // LLM info
                let llmHtml = `<div class="status-item">
                    <span><span class="status-dot green"></span>Local: ${data.ollama?.model || 'N/A'}</span>
                </div>`;
                // Check Claude Code CLI first (Max subscription), then API key
                const claudeCodeConfigured = data.claude_code?.configured;
                const apiConfigured = data.anthropic?.configured;
                if (claudeCodeConfigured) {
                    llmHtml += `<div class="status-item">
                        <span><span class="status-dot green"></span>Cloud: Claude Code</span>
                        <span style="color:#888;font-size:12px">Max subscription</span>
                    </div>`;
                } else if (apiConfigured) {
                    llmHtml += `<div class="status-item">
                        <span><span class="status-dot green"></span>Cloud: ${data.anthropic?.model || 'N/A'}</span>
                        <span style="color:#888;font-size:12px">API key</span>
                    </div>`;
                } else {
                    llmHtml += `<div class="status-item">
                        <span><span class="status-dot red"></span>Cloud: Not configured</span>
                        <span style="color:#888;font-size:12px">No CLI or API key</span>
                    </div>`;
                }
                document.getElementById('llm-info').innerHTML = llmHtml;

                // Account
                const me = await (await fetch('/auth/me', {headers: authHeaders()})).json();
                document.getElementById('account-info').innerHTML =
                    `<div class="status-item"><span>User: ${me.username || 'N/A'}</span><span style="color:#888;font-size:12px">${me.role || ''}</span></div>`;

                // TTS settings
                const tts = data.tts || {};
                currentTtsBackend = tts.backend || 'kokoro';
                document.querySelectorAll('.tts-option').forEach(b => b.classList.remove('active'));
                const activeBtn = document.getElementById('tts-' + currentTtsBackend);
                if (activeBtn) activeBtn.classList.add('active');
                const ttsStatus = document.getElementById('tts-status');
                if (ttsStatus) {
                    if (tts.backend === 'qwen3' && !tts.qwen3_available) {
                        ttsStatus.innerHTML = '<span style="color:#f87171">⚠ Qwen3 server not available</span>';
                    } else if (tts.backend === 'qwen3') {
                        ttsStatus.innerHTML = '<span style="color:#4ade80">✓ Qwen3 server running</span>';
                    } else {
                        ttsStatus.textContent = '';
                    }
                }
                // Load available voices
                await loadVoices();
            } catch(e) {
                document.getElementById('integration-list').innerHTML = '<span style="color:#f87171">Failed to load</span>';
            }
        }

        async function setTTS(backend) {
            try {
                const resp = await fetch('/settings/tts?backend=' + backend, {
                    method: 'PUT',
                    headers: authHeaders()
                });
                if (resp.ok) {
                    currentTtsBackend = backend;
                    document.querySelectorAll('.tts-option').forEach(b => b.classList.remove('active'));
                    document.getElementById('tts-' + backend)?.classList.add('active');
                    const ttsStatus = document.getElementById('tts-status');
                    if (backend === 'qwen3') {
                        ttsStatus.innerHTML = '<span style="color:#888">Checking Qwen3 server...</span>';
                        const check = await fetch('/settings/tts', {headers: authHeaders()});
                        const data = await check.json();
                        if (data.qwen3_available) {
                            ttsStatus.innerHTML = '<span style="color:#4ade80">✓ Qwen3 server running</span>';
                        } else {
                            ttsStatus.innerHTML = '<span style="color:#f87171">⚠ Qwen3 server not available</span>';
                        }
                    } else {
                        ttsStatus.textContent = '';
                    }
                    // Reload voices for new backend
                    await loadVoices();
                }
            } catch(e) {
                console.error('Failed to set TTS:', e);
            }
        }

        async function loadVoices() {
            try {
                const resp = await fetch('/settings/voices', {headers: authHeaders()});
                const data = await resp.json();
                const select = document.getElementById('tts-voice');
                if (select && data.voices) {
                    select.innerHTML = data.voices.map(v =>
                        `<option value="${v.id}" ${v.id === data.current ? 'selected' : ''}>${v.name} (${v.desc})</option>`
                    ).join('');
                }
            } catch(e) {
                console.error('Failed to load voices:', e);
            }
        }

        async function setVoice(voice) {
            try {
                await fetch('/settings/voice?voice=' + encodeURIComponent(voice), {
                    method: 'PUT',
                    headers: authHeaders()
                });
            } catch(e) {
                console.error('Failed to set voice:', e);
            }
        }

        let autoSpeak = localStorage.getItem('alfred_auto_speak') === 'true';
        let currentAudio = null;
        let currentTtsBackend = 'kokoro';  // Track TTS backend for acknowledgment logic
        let msgTexts = {};
        let msgCounter = 0;
        if (autoSpeak) document.getElementById('auto-speak-btn')?.classList.add('active');

        function addMsg(text, role, tier, noAutoSpeak = false) {
            // Show chat state when adding messages
            if (isWelcomeState) showChatState();

            const div = document.createElement('div');
            div.className = `msg ${role}`;
            const mid = ++msgCounter;
            msgTexts[mid] = text;

            if (role === 'alfred') {
                // Alfred message: no bubble, just text + action buttons
                div.innerHTML = `
                    <div class="content">${renderText(text)}</div>
                    <div class="msg-actions">
                        <button class="msg-action-btn copy-btn" data-mid="${mid}" title="Copy">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
                        </button>
                        <button class="msg-action-btn thumbs-up-btn" data-mid="${mid}" title="Good response">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3"/></svg>
                        </button>
                        <button class="msg-action-btn thumbs-down-btn" data-mid="${mid}" title="Bad response">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3zm7-13h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17"/></svg>
                        </button>
                        <button class="msg-action-btn speak-btn" data-mid="${mid}" title="Read aloud">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg>
                        </button>
                        <button class="msg-action-btn regenerate-btn" data-mid="${mid}" title="Regenerate">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>
                        </button>
                    </div>
                `;
            } else {
                // User message: bubble style
                div.innerHTML = `<div class="content">${renderText(text)}</div>`;
            }

            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            if (role === 'alfred' && autoSpeak && !loadingHistory && !noAutoSpeak) {
                const btn = div.querySelector('.speak-btn');
                if (btn) speakText(btn);
            }
            return div;
        }

        function addMsgHtml(text, role, tier, noAutoSpeak = false, extraHtml = '') {
            // Show chat state when adding messages
            if (isWelcomeState) showChatState();

            const div = document.createElement('div');
            div.className = `msg ${role}`;
            const mid = ++msgCounter;
            msgTexts[mid] = text;

            if (role === 'alfred') {
                div.innerHTML = `
                    <div class="content">${renderText(text)}${extraHtml}</div>
                    <div class="msg-actions">
                        <button class="msg-action-btn copy-btn" data-mid="${mid}" title="Copy">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
                        </button>
                        <button class="msg-action-btn thumbs-up-btn" data-mid="${mid}" title="Good response">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 00-3-3l-4 9v11h11.28a2 2 0 002-1.7l1.38-9a2 2 0 00-2-2.3zM7 22H4a2 2 0 01-2-2v-7a2 2 0 012-2h3"/></svg>
                        </button>
                        <button class="msg-action-btn thumbs-down-btn" data-mid="${mid}" title="Bad response">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 003 3l4-9V2H5.72a2 2 0 00-2 1.7l-1.38 9a2 2 0 002 2.3zm7-13h2.67A2.31 2.31 0 0122 4v7a2.31 2.31 0 01-2.33 2H17"/></svg>
                        </button>
                        <button class="msg-action-btn speak-btn" data-mid="${mid}" title="Read aloud">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/></svg>
                        </button>
                        <button class="msg-action-btn regenerate-btn" data-mid="${mid}" title="Regenerate">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>
                        </button>
                    </div>
                `;
            } else {
                div.innerHTML = `<div class="content">${renderText(text)}${extraHtml}</div>`;
            }

            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            if (role === 'alfred' && autoSpeak && !loadingHistory && !noAutoSpeak) {
                const btn = div.querySelector('.speak-btn');
                if (btn) speakText(btn);
            }
            return div;
        }

        document.addEventListener('click', function(e) {
            const speakBtn = e.target.closest('.speak-btn');
            if (speakBtn) { speakText(speakBtn); return; }
            const stopBtn = e.target.closest('.stop-btn');
            if (stopBtn) { stopAudio(); return; }
            const copyBtn = e.target.closest('.copy-btn');
            if (copyBtn) { copyMessage(copyBtn); return; }
            const thumbsUpBtn = e.target.closest('.thumbs-up-btn');
            if (thumbsUpBtn) { thumbsUpBtn.classList.toggle('active'); return; }
            const thumbsDownBtn = e.target.closest('.thumbs-down-btn');
            if (thumbsDownBtn) { thumbsDownBtn.classList.toggle('active'); return; }
        });

        function copyMessage(btn) {
            const mid = btn.getAttribute('data-mid');
            const text = msgTexts[mid];
            if (!text) return;
            navigator.clipboard.writeText(text).then(() => {
                btn.classList.add('active');
                setTimeout(() => btn.classList.remove('active'), 2000);
            });
        }

        function stopAudio() {
            if (currentAudio) { currentAudio.pause(); currentAudio = null; }
            document.querySelectorAll('.speak-btn.speaking').forEach(b => b.classList.remove('speaking'));
            document.querySelectorAll('.stop-btn.visible').forEach(b => b.classList.remove('visible'));
        }

        function toggleAutoSpeak() {
            autoSpeak = !autoSpeak;
            localStorage.setItem('alfred_auto_speak', autoSpeak);
            document.getElementById('auto-speak-btn').classList.toggle('active', autoSpeak);
        }

        function executeUiAction(uiAction) {
            // Execute a UI action from the API response
            if (!uiAction || !uiAction.action) return;
            console.log('Executing UI action:', uiAction);

            if (uiAction.action === 'set_auto_speak') {
                autoSpeak = uiAction.value;
                localStorage.setItem('alfred_auto_speak', String(autoSpeak));
                document.getElementById('auto-speak-btn').classList.toggle('active', autoSpeak);
                console.log('UI Action: auto-speak set to', autoSpeak);
            }
            if (uiAction.action === 'set_hands_free') {
                if (uiAction.value) {
                    if (!handsFreeActive) toggleHandsFree();
                } else {
                    if (handsFreeActive) {
                        handsFreeActive = false;
                        document.getElementById('handsfree-btn').classList.remove('active');
                        if (vadInstance) vadInstance.pause();
                        setVadState('idle');
                    }
                }
                console.log('UI Action: hands-free set to', uiAction.value);
            }
        }

        async function speakText(btn) {
            const mid = btn.getAttribute('data-mid');
            const text = msgTexts[mid];
            if (!text) return;
            const stopBtn = btn.parentElement.querySelector('.stop-btn');
            // Toggle off if this button is already speaking
            if (btn.classList.contains('speaking')) {
                stopAudio();
                return;
            }
            // Stop any other audio first
            stopAudio();
            btn.classList.add('speaking');
            if (stopBtn) stopBtn.classList.add('visible');
            try {
                const resp = await fetch('/voice/speak', {
                    method: 'POST',
                    headers: authHeaders({'Content-Type': 'application/json'}),
                    body: JSON.stringify({message: text})
                });
                if (!resp.ok) throw new Error('TTS failed');
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                currentAudio = new Audio(url);
                currentAudio.onended = () => {
                    btn.classList.remove('speaking');
                    if (stopBtn) stopBtn.classList.remove('visible');
                    currentAudio = null;
                    URL.revokeObjectURL(url);
                };
                currentAudio.onerror = () => {
                    btn.classList.remove('speaking');
                    if (stopBtn) stopBtn.classList.remove('visible');
                    currentAudio = null;
                };
                currentAudio.play();
            } catch(e) {
                console.error('TTS error:', e);
                btn.classList.remove('speaking');
                if (stopBtn) stopBtn.classList.remove('visible');
            }
        }

        // ==================== File Upload ====================

        let pendingFile = null;  // {type: 'image'|'document', path, base64?, media_type?, filename, text_preview?}

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            const ext = file.name.split('.').pop().toLowerCase();
            const imageExts = ['jpg','jpeg','png','gif','webp'];
            const docExts = ['pdf','doc','docx','xls','xlsx','csv','txt','md','json'];

            if (imageExts.includes(ext)) {
                uploadImage(file);
            } else if (docExts.includes(ext)) {
                uploadDocument(file);
            } else {
                alert('Unsupported file type');
            }
            e.target.value = '';
        }

        async function uploadImage(file) {
            const form = new FormData();
            form.append('file', file);
            try {
                const resp = await fetch('/upload/image', {method:'POST', headers:authHeaders(), body:form});
                if (!resp.ok) throw new Error(await resp.text());
                const data = await resp.json();
                pendingFile = {type:'image', path:data.path, base64:data.base64, media_type:data.media_type, filename:data.filename};
                showPendingFile(data.filename, `Image (${Math.round(data.size_bytes/1024)}KB)`, `data:${data.media_type};base64,${data.base64}`);
            } catch(e) { alert('Upload failed: ' + e.message); }
        }

        async function uploadDocument(file) {
            const form = new FormData();
            form.append('file', file);
            try {
                const resp = await fetch('/upload/document', {method:'POST', headers:authHeaders(), body:form});
                if (!resp.ok) throw new Error(await resp.text());
                const data = await resp.json();
                pendingFile = {type:'document', path:data.path, filename:data.filename, text_preview:data.text_preview, metadata:data.metadata};
                const meta = data.metadata;
                let info = meta.extension.toUpperCase();
                if (meta.pages) info += `, ${meta.pages} pages`;
                if (meta.rows) info += `, ${meta.rows} rows`;
                showPendingFile(data.filename, info, null);
            } catch(e) { alert('Upload failed: ' + e.message); }
        }

        function showPendingFile(name, meta, previewUrl) {
            const container = document.getElementById('pending-file');
            const preview = document.getElementById('pending-preview');
            document.getElementById('pending-name').textContent = name;
            document.getElementById('pending-meta').textContent = meta;
            if (previewUrl) {
                preview.src = previewUrl;
                preview.style.display = 'block';
            } else {
                preview.style.display = 'none';
            }
            container.classList.add('visible');
            // Mark both upload buttons as having a file
            document.querySelectorAll('.upload-btn').forEach(btn => btn.classList.add('has-file'));
        }

        function clearPendingFile() {
            pendingFile = null;
            document.getElementById('pending-file').classList.remove('visible');
            document.querySelectorAll('.upload-btn').forEach(btn => btn.classList.remove('has-file'));
        }

        function escapeHtml(t) {
            const d = document.createElement('div');
            d.textContent = t;
            return d.innerHTML;
        }

        function renderText(t) {
            let s = escapeHtml(t);
            s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
            s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            // Handle download links: [Download: filename](/download/filename)
            s = s.replace(/\[Download: ([^\]]+)\]\(([^)]+)\)/g, '<a class="download-btn" href="$2" download>📥 $1</a>');
            return s;
        }

        async function send() {
            const msg = input.value.trim();
            if (!msg && !pendingFile) return;
            input.value = '';
            input.style.height = 'auto';

            // Build user message with file context
            let userMsgDisplay = msg;
            let payload = {message: msg, session_id: currentConversationId || 'default'};

            if (pendingFile) {
                if (pendingFile.type === 'image') {
                    userMsgDisplay = msg ? `[Image: ${pendingFile.filename}] ${msg}` : `[Image: ${pendingFile.filename}]`;
                    payload.image_path = pendingFile.path;
                    payload.image_base64 = pendingFile.base64;
                    payload.image_media_type = pendingFile.media_type;
                    // Show image in chat
                    const imgHtml = `<img class="inline-image" src="data:${pendingFile.media_type};base64,${pendingFile.base64}" alt="${pendingFile.filename}">`;
                    addMsgHtml(userMsgDisplay, 'user', null, true, imgHtml);
                } else {
                    userMsgDisplay = msg ? `[Document: ${pendingFile.filename}] ${msg}` : `Analyze this document: ${pendingFile.filename}`;
                    payload.document_path = pendingFile.path;
                    payload.message = userMsgDisplay;
                    addMsg(userMsgDisplay, 'user', null, true);
                }
                clearPendingFile();
            } else {
                addMsg(msg, 'user');
            }

            document.getElementById('send-btn').disabled = true;

            // Show thinking indicator (move to end of chat, after user's message)
            const thinking = document.getElementById('thinking');
            chat.appendChild(thinking);  // Move to end
            thinking.classList.add('visible');
            chat.scrollTop = chat.scrollHeight;

            // Ensure a conversation exists
            if (!currentConversationId) {
                try {
                    const cr = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
                    const cd = await cr.json();
                    currentConversationId = cd.id;
                } catch(e) {}
            }

            try {
                payload.session_id = currentConversationId || 'default';
                const resp = await fetch('/chat', {
                    method: 'POST',
                    headers: authHeaders({'Content-Type': 'application/json'}),
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                // Check for UI actions in response
                if (data.ui_action) {
                    executeUiAction(data.ui_action);
                }
                const displayResponse = data.response;
                // Check for generated images
                if (data.images && data.images.length > 0) {
                    let imagesHtml = '';
                    data.images.forEach(img => {
                        imagesHtml += `<img class="inline-image" src="data:image/png;base64,${img.base64}" alt="${img.filename}">`;
                        if (img.download_url) {
                            imagesHtml += `<a class="download-btn" href="${img.download_url}" download>📥 ${img.filename}</a>`;
                        }
                    });
                    addMsgHtml(displayResponse, 'alfred', data.tier, false, imagesHtml);
                } else {
                    addMsg(displayResponse, 'alfred', data.tier);
                }
                // Refresh conversation list to update title/preview
                loadConversations();
            } catch(e) {
                addMsg('Connection error. Please try again.', 'alfred');
            }
            // Hide thinking indicator
            thinking.classList.remove('visible');
            document.getElementById('send-btn').disabled = false;
            input.focus();
        }

        async function toggleMic() {
            // Cut off any audio Alfred is currently speaking
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
                const speaking = document.querySelector('.speak-btn.speaking');
                if (speaking) speaking.classList.remove('speaking');
            }
            const mainBtn = document.getElementById('mic-btn');
            const welcomeBtn = document.getElementById('welcome-mic-btn');
            if (isRecording) {
                mediaRecorder.stop();
                mainBtn?.classList.remove('recording');
                welcomeBtn?.classList.remove('recording');
                isRecording = false;
            } else {
                const stream = await navigator.mediaDevices.getUserMedia({audio: true});
                mediaRecorder = new MediaRecorder(stream);
                const chunks = [];
                mediaRecorder.ondataavailable = e => chunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    const blob = new Blob(chunks, {type: 'audio/webm'});
                    const form = new FormData();
                    form.append('audio', blob, 'recording.webm');
                    try {
                        const resp = await fetch('/voice/transcribe', {method:'POST', headers: authHeaders(), body: form});
                        const data = await resp.json();
                        if (data.text) {
                            // Put transcription in the appropriate input
                            if (isWelcomeState) {
                                document.getElementById('welcome-input').value = data.text;
                                sendFromWelcome();
                            } else {
                                input.value = data.text;
                                send();
                            }
                        }
                    } catch(e) { console.error('Transcription failed:', e); }
                };
                mediaRecorder.start();
                mainBtn?.classList.add('recording');
                welcomeBtn?.classList.add('recording');
                isRecording = true;
            }
        }

        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // ==================== Screen Wake Lock ====================

        let wakeLock = null;

        async function acquireWakeLock() {
            if (wakeLock) return; // Already have it
            try {
                if ('wakeLock' in navigator) {
                    wakeLock = await navigator.wakeLock.request('screen');
                    console.log('Screen wake lock acquired');
                    wakeLock.addEventListener('release', () => {
                        console.log('Screen wake lock released');
                        wakeLock = null;
                    });
                }
            } catch (err) {
                console.log('Wake lock request failed:', err.message);
            }
        }

        function releaseWakeLock() {
            // Only release if neither wake word nor hands-free is active
            if (wakeWordActive || handsFreeActive) return;
            if (wakeLock) {
                wakeLock.release();
                wakeLock = null;
                console.log('Screen wake lock released manually');
            }
        }

        // Re-acquire wake lock when page becomes visible again
        document.addEventListener('visibilitychange', async () => {
            if (document.visibilityState === 'visible' && (wakeWordActive || handsFreeActive)) {
                await acquireWakeLock();
            }
        });

        // ==================== Wake Word Detection ====================

        let wakeWordActive = false;
        let wakeWordWs = null;
        let wakeWordStream = null;
        let wakeWordContext = null;
        let wakeWordProcessor = null;

        async function toggleWakeWord() {
            const btn = document.getElementById('wakeword-btn');
            const status = document.getElementById('wakeword-status');

            if (wakeWordActive) {
                // Stop wake word detection
                wakeWordActive = false;
                btn.classList.remove('active');
                status.classList.remove('visible');
                if (wakeWordWs) {
                    wakeWordWs.close();
                    wakeWordWs = null;
                }
                if (wakeWordStream) {
                    wakeWordStream.getTracks().forEach(t => t.stop());
                    wakeWordStream = null;
                }
                if (wakeWordContext) {
                    wakeWordContext.close();
                    wakeWordContext = null;
                }
                releaseWakeLock(); // Release if hands-free also inactive
                return;
            }

            // Start wake word detection
            try {
                btn.classList.add('active');
                status.classList.add('visible');
                status.textContent = 'Starting...';
                await acquireWakeLock(); // Keep screen on

                // Get microphone access
                wakeWordStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });

                // Create audio context for resampling
                wakeWordContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000
                });

                const source = wakeWordContext.createMediaStreamSource(wakeWordStream);
                wakeWordProcessor = wakeWordContext.createScriptProcessor(1024, 1, 1);

                // Connect to WebSocket
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const token = localStorage.getItem('alfred_token');
                wakeWordWs = new WebSocket(`${wsProtocol}//${window.location.host}/ws/wakeword?token=${token}`);

                wakeWordWs.onopen = () => {
                    console.log('Wake word WebSocket connected');
                    status.textContent = 'Listening for "Hey Alfred"...';
                    wakeWordActive = true;
                    // Keep-alive ping every 30 seconds to prevent timeout
                    wakeWordWs._pingInterval = setInterval(() => {
                        if (wakeWordWs && wakeWordWs.readyState === WebSocket.OPEN) {
                            try { wakeWordWs.send(new ArrayBuffer(0)); } catch(e) {}
                        }
                    }, 30000);
                };

                wakeWordWs.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'detected') {
                        console.log('Wake word detected!', data.score);
                        status.textContent = 'Wake word detected!';

                        // Visual feedback
                        btn.style.animation = 'pulse 0.3s ease-in-out 3';
                        setTimeout(() => btn.style.animation = '', 1000);

                        // Trigger hands-free mode if not already active
                        if (!handsFreeActive) {
                            toggleHandsFree();
                        }

                        // Reset status after a moment
                        setTimeout(() => {
                            if (wakeWordActive) {
                                status.textContent = 'Listening for "Hey Alfred"...';
                            }
                        }, 2000);
                    } else if (data.type === 'error') {
                        console.error('Wake word error:', data.message);
                        status.textContent = 'Error: ' + data.message;
                    }
                };

                wakeWordWs.onclose = () => {
                    console.log('Wake word WebSocket closed');
                    if (wakeWordWs && wakeWordWs._pingInterval) {
                        clearInterval(wakeWordWs._pingInterval);
                    }
                    wakeWordWs = null;  // Clear reference immediately
                    if (wakeWordActive) {
                        // Reconnect after a short delay
                        status.textContent = 'Reconnecting...';
                        setTimeout(() => {
                            if (wakeWordActive) {
                                // Stop and restart wake word
                                wakeWordActive = false;
                                toggleWakeWord();  // This will restart it
                            }
                        }, 1000);
                    }
                };

                wakeWordWs.onerror = (err) => {
                    console.error('Wake word WebSocket error:', err);
                    status.textContent = 'Connection error';
                };

                // Process audio and send to WebSocket
                wakeWordProcessor.onaudioprocess = (e) => {
                    if (!wakeWordActive || !wakeWordWs || wakeWordWs.readyState !== WebSocket.OPEN) return;

                    const inputData = e.inputBuffer.getChannelData(0);

                    // Convert float32 to int16
                    const int16Data = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        const s = Math.max(-1, Math.min(1, inputData[i]));
                        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }

                    // Send to WebSocket
                    wakeWordWs.send(int16Data.buffer);
                };

                source.connect(wakeWordProcessor);
                wakeWordProcessor.connect(wakeWordContext.destination);

            } catch (err) {
                console.error('Wake word init failed:', err);
                btn.classList.remove('active');
                status.classList.remove('visible');
                wakeWordActive = false;
                alert('Wake word detection failed to start. Check microphone permissions.');
            }
        }

        // ==================== Hands-free VAD Mode ====================

        let handsFreeActive = false;
        let vadInstance = null;
        let vadState = 'idle'; // idle, listen, hear, process, speak
        let vadProcessing = false;

        function cutOffAlfred() {
            if (currentAudio) {
                console.log('Interrupting Alfred');
                currentAudio.pause();
                currentAudio = null;
                const s = document.querySelector('.speak-btn.speaking');
                if (s) s.classList.remove('speaking');
                // Reset to listen state after interruption
                if (handsFreeActive) setVadState('listen');
            }
        }

        // Allow interrupting Alfred by pressing Escape or clicking the chat area
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && currentAudio) {
                cutOffAlfred();
            }
        });
        document.getElementById('chat')?.addEventListener('click', (e) => {
            // Don't interrupt if clicking buttons
            if (e.target.closest('button')) return;
            if (currentAudio && vadState === 'speak') {
                cutOffAlfred();
            }
        });

        function setVadState(state) {
            vadState = state;
            const statusEl = document.getElementById('vad-status');
            const textEl = document.getElementById('vad-status-text');
            const micBtn = document.getElementById('mic-btn');
            micBtn.classList.remove('vad-listen','vad-hear','vad-process','vad-speak');
            statusEl.classList.remove('state-listen','state-hear','state-process','state-speak');
            if (state === 'idle' || !handsFreeActive) {
                statusEl.classList.remove('visible');
                return;
            }
            statusEl.classList.add('visible', 'state-' + state);
            micBtn.classList.add('vad-' + state);
            const labels = {listen:'Listening...', hear:'Hearing you...', process:'Processing...', speak:'Alfred speaking...'};
            textEl.textContent = labels[state] || '';
        }

        function float32ToWav(samples, sampleRate) {
            const buf = new ArrayBuffer(44 + samples.length * 2);
            const v = new DataView(buf);
            function w(off, s) { for(let i=0;i<s.length;i++) v.setUint8(off+i,s.charCodeAt(i)); }
            w(0,'RIFF'); v.setUint32(4,36+samples.length*2,true);
            w(8,'WAVE'); w(12,'fmt '); v.setUint32(16,16,true);
            v.setUint16(20,1,true); v.setUint16(22,1,true);
            v.setUint32(24,sampleRate,true); v.setUint32(28,sampleRate*2,true);
            v.setUint16(32,2,true); v.setUint16(34,16,true);
            w(36,'data'); v.setUint32(40,samples.length*2,true);
            for(let i=0;i<samples.length;i++){
                const s=Math.max(-1,Math.min(1,samples[i]));
                v.setInt16(44+i*2, s<0?s*0x8000:s*0x7FFF, true);
            }
            return new Blob([buf],{type:'audio/wav'});
        }

        function playVadTTS(text) {
            setVadState('speak');
            // Explicitly ensure VAD is running so user can interrupt Alfred
            if (vadInstance) {
                try {
                    vadInstance.start();
                    console.log('VAD started for interrupt detection during speech');
                } catch(e) {
                    console.log('VAD start failed:', e);
                }
            } else {
                console.log('No VAD instance available');
            }
            fetch('/voice/speak', {
                method:'POST',
                headers: authHeaders({'Content-Type':'application/json'}),
                body: JSON.stringify({message: text})
            }).then(r => r.ok ? r.blob() : null).then(blob => {
                if (!blob || !handsFreeActive) {
                    if (handsFreeActive && vadInstance) vadInstance.start();
                    setVadState(handsFreeActive?'listen':'idle');
                    return;
                }
                const url = URL.createObjectURL(blob);
                currentAudio = new Audio(url);
                currentAudio.onplay = () => {
                    console.log('Alfred speaking - VAD state:', vadState, 'VAD running:', !!vadInstance);
                };
                currentAudio.onended = () => {
                    console.log('Alfred finished speaking');
                    currentAudio = null; URL.revokeObjectURL(url);
                    if (handsFreeActive) {
                        // Resume VAD after speech ends
                        if (vadInstance) vadInstance.start();
                        setVadState('listen');
                    }
                };
                currentAudio.onerror = () => {
                    currentAudio = null;
                    if (handsFreeActive) {
                        if (vadInstance) vadInstance.start();
                        setVadState('listen');
                    }
                };
                currentAudio.play();
            }).catch(() => {
                if (handsFreeActive) {
                    if (vadInstance) vadInstance.start();
                    setVadState('listen');
                }
            });
        }

        async function processVadAudio(audioFloat32) {
            if (vadProcessing) return;
            vadProcessing = true;
            setVadState('process');
            try {
                const wavBlob = float32ToWav(audioFloat32, 16000);
                const form = new FormData();
                form.append('audio', wavBlob, 'vad_recording.wav');
                const trResp = await fetch('/voice/transcribe', {method:'POST', headers:authHeaders(), body:form});
                const trData = await trResp.json();
                const text = (trData.text || '').trim();
                if (!text || text.length < 2) {
                    vadProcessing = false;
                    setVadState('listen');
                    return;
                }

                // Check for dismissal phrases
                const dismissalPhrases = [
                    "that's all for now", "thats all for now", "that is all for now",
                    "that's all", "thats all", "that is all",
                    "goodbye alfred", "goodbye", "good bye", "bye",
                    "thanks alfred", "thank you alfred", "thanks that's all",
                    "that will be all", "that'll be all", "thatll be all",
                    "i'm done", "im done", "i am done", "done for now",
                    "go to sleep", "you can go", "dismiss", "dismissed",
                    "see you later", "talk to you later", "later alfred",
                    "all for now", "nothing else", "no that's it", "no thats it"
                ];
                const textLower = text.toLowerCase().trim();
                const isDismissal = dismissalPhrases.some(phrase => textLower.includes(phrase));
                console.log('Transcribed:', textLower, '| Dismissal detected:', isDismissal);

                if (isDismissal) {
                    console.log('Dismissal phrase detected, ending hands-free mode');
                    addMsg(text, 'user');
                    const farewellMsg = "Very good, sir. I'm here if you need me.";
                    addMsg(farewellMsg, 'alfred', 'local');
                    vadProcessing = false;

                    // Function to disable hands-free
                    function disableHandsFree() {
                        console.log('Disabling hands-free mode');
                        handsFreeActive = false;
                        document.getElementById('handsfree-btn').classList.remove('active');
                        if (vadInstance) vadInstance.pause();
                        setVadState('idle');
                        releaseWakeLock(); // Release if wake word also inactive
                        // Update wake word status if active
                        const wwStatus = document.getElementById('wakeword-status');
                        if (wwStatus && wakeWordActive) {
                            wwStatus.textContent = 'Listening for "Hey Alfred"...';
                            wwStatus.classList.add('visible');
                        }
                    }

                    // Play farewell TTS then disable hands-free
                    setVadState('speak');
                    fetch('/voice/speak', {
                        method:'POST',
                        headers: authHeaders({'Content-Type':'application/json'}),
                        body: JSON.stringify({message: farewellMsg})
                    }).then(r => r.ok ? r.blob() : null).then(blob => {
                        if (blob) {
                            const url = URL.createObjectURL(blob);
                            currentAudio = new Audio(url);
                            currentAudio.onended = () => {
                                currentAudio = null;
                                URL.revokeObjectURL(url);
                                disableHandsFree();
                            };
                            currentAudio.onerror = () => {
                                disableHandsFree();
                            };
                            currentAudio.play().catch(() => disableHandsFree());
                        } else {
                            disableHandsFree();
                        }
                    }).catch(() => {
                        disableHandsFree();
                    });
                    return;
                }

                // Check for voice control commands
                const autoSpeakOffPhrases = [
                    "turn off auto speak", "turn off autospeak", "turn off auto-speak",
                    "disable auto speak", "disable auto-speak", "disable autospeak",
                    "stop auto speak", "stop auto-speak", "mute", "mute yourself",
                    "stop speaking", "be quiet", "quiet mode", "silent mode",
                    "no more speaking", "stop talking"
                ];
                const autoSpeakOnPhrases = [
                    "turn on auto speak", "turn on autospeak", "turn on auto-speak",
                    "enable auto speak", "enable auto-speak", "enable autospeak",
                    "start auto speak", "unmute", "unmute yourself", "start speaking",
                    "speak mode", "voice mode", "talk to me", "start talking"
                ];
                const handsFreeOffPhrases = [
                    "turn off hands free", "turn off handsfree", "turn off hands-free",
                    "disable hands free", "disable hands-free", "disable handsfree",
                    "stop hands free", "stop listening", "pause listening"
                ];

                // Handle auto-speak toggle
                if (autoSpeakOffPhrases.some(phrase => textLower.includes(phrase))) {
                    addMsg(text, 'user');
                    autoSpeak = false;
                    localStorage.setItem('alfred_auto_speak', 'false');
                    document.getElementById('auto-speak-btn').classList.remove('active');
                    addMsg("Auto-speak disabled, sir. I'll stay quiet unless you ask me to speak.", 'alfred', 'local');
                    vadProcessing = false;
                    setVadState('listen');
                    return;
                }
                if (autoSpeakOnPhrases.some(phrase => textLower.includes(phrase))) {
                    addMsg(text, 'user');
                    autoSpeak = true;
                    localStorage.setItem('alfred_auto_speak', 'true');
                    document.getElementById('auto-speak-btn').classList.add('active');
                    const confirmMsg = "Auto-speak enabled, sir. I'll read my responses aloud.";
                    addMsg(confirmMsg, 'alfred', 'local');
                    playVadTTS(confirmMsg);
                    vadProcessing = false;
                    return;
                }
                // Handle hands-free off (similar to dismissal but explicit)
                if (handsFreeOffPhrases.some(phrase => textLower.includes(phrase))) {
                    addMsg(text, 'user');
                    const confirmMsg = "Hands-free mode disabled, sir.";
                    addMsg(confirmMsg, 'alfred', 'local');
                    vadProcessing = false;
                    handsFreeActive = false;
                    document.getElementById('handsfree-btn').classList.remove('active');
                    if (vadInstance) vadInstance.pause();
                    setVadState('idle');
                    releaseWakeLock(); // Release if wake word also inactive
                    return;
                }

                addMsg(text, 'user');
                if (!currentConversationId) {
                    try {
                        const cr = await fetch('/conversations', {method:'POST', headers:authHeaders()});
                        currentConversationId = (await cr.json()).id;
                    } catch(e){}
                }

                // HYBRID APPROACH: Smart acknowledgment while processing
                // - Skips ack for conversational queries ("how are you", greetings)
                // - Plays ack for task queries ("search email", "check calendar")
                // - Skip for Qwen3 (different voice, adds latency)
                setVadState('speak');
                // Explicitly ensure VAD keeps running so user can interrupt Alfred
                if (vadInstance) {
                    try { vadInstance.start(); } catch(e) {}
                }
                let ackPromise = Promise.resolve();
                if (currentTtsBackend === 'kokoro') {
                    // Pass the query so backend can decide if ack is appropriate
                    ackPromise = fetch('/voice/chat/ack', {
                        method:'POST',
                        headers: authHeaders({'Content-Type':'application/json'}),
                        body: JSON.stringify({message: text})
                    }).then(r => r.ok ? r.blob() : null).then(blob => {
                        // Only play if we got audio (empty blob = no ack needed)
                        if (blob && blob.size > 100 && handsFreeActive) {
                            const url = URL.createObjectURL(blob);
                            const ackAudio = new Audio(url);
                            ackAudio.onended = () => URL.revokeObjectURL(url);
                            ackAudio.play().catch(() => {});
                        }
                    }).catch(() => {});
                }

                // Process with smart model in parallel
                const chatResp = await fetch('/chat', {
                    method:'POST',
                    headers: authHeaders({'Content-Type':'application/json'}),
                    body: JSON.stringify({message:text, session_id: currentConversationId||'default'})
                });
                const chatData = await chatResp.json();
                loadingHistory = true;
                addMsg(chatData.response, 'alfred', chatData.tier);
                loadingHistory = false;
                loadConversations();
                vadProcessing = false;

                // Wait a moment for ack to finish, then play full response
                await ackPromise;
                setTimeout(() => playVadTTS(chatData.response), 300);
            } catch(e) {
                console.error('Hands-free error:', e);
                vadProcessing = false;
                if (handsFreeActive) {
                    // Resume VAD on error
                    if (vadInstance) vadInstance.start();
                    setVadState('listen');
                }
            }
        }

        async function toggleHandsFree() {
            const btn = document.getElementById('handsfree-btn');
            if (handsFreeActive) {
                handsFreeActive = false;
                btn.classList.remove('active');
                if (vadInstance) vadInstance.pause();
                cutOffAlfred();
                setVadState('idle');
                releaseWakeLock(); // Release if wake word also inactive
                return;
            }
            handsFreeActive = true;
            btn.classList.add('active');
            await acquireWakeLock(); // Keep screen on
            if (!autoSpeak) {
                autoSpeak = true;
                localStorage.setItem('alfred_auto_speak', 'true');
                document.getElementById('auto-speak-btn').classList.add('active');
            }
            cutOffAlfred();
            try {
                if (!vadInstance) {
                    setVadState('process');
                    vadInstance = await vad.MicVAD.new({
                        positiveSpeechThreshold: 0.5,  // Lower threshold for easier interruption
                        negativeSpeechThreshold: 0.35,
                        minSpeechFrames: 2,  // Faster detection
                        preSpeechPadFrames: 6,
                        redemptionFrames: 4,
                        // Disable echo cancellation with headphones - it can filter user voice
                        additionalAudioConstraints: {
                            echoCancellation: false,
                            noiseSuppression: false,
                            autoGainControl: true
                        },
                        onSpeechStart: () => {
                            console.log('VAD onSpeechStart - state:', vadState, 'hasAudio:', !!currentAudio);
                            if (!handsFreeActive) return;
                            // Interrupt Alfred if he's speaking
                            if (currentAudio && vadState === 'speak') {
                                console.log('Interrupting Alfred via voice!');
                                cutOffAlfred();
                                return; // Don't process this speech (likely just interruption)
                            }
                            if (!vadProcessing) setVadState('hear');
                        },
                        onSpeechEnd: (audio) => {
                            if (!handsFreeActive || vadProcessing) return;
                            processVadAudio(audio);
                        },
                        onVADMisfire: () => {
                            if (handsFreeActive && vadState === 'hear') setVadState('listen');
                        }
                    });
                }
                vadInstance.start();
                setVadState('listen');
            } catch(e) {
                console.error('VAD init failed:', e);
                handsFreeActive = false;
                btn.classList.remove('active');
                setVadState('idle');
                alert('Hands-free failed to start. Check mic permissions.');
            }
        }

        // Register service worker for PWA
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').then(reg => {
                console.log('SW registered:', reg.scope);
            }).catch(err => console.log('SW registration failed:', err));
        }
    </script>
</body>
</html>"""


SERVICE_WORKER_JS = """
const CACHE_NAME = 'alfred-v30';
const PRECACHE_URLS = ['/', '/manifest.json', '/static/icon-192.png', '/static/icon-512.png'];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS)).then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        ).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    // Network-first for API calls, cache-first for static assets
    if (url.pathname.startsWith('/chat') || url.pathname.startsWith('/auth') ||
        url.pathname.startsWith('/conversations') || url.pathname.startsWith('/voice') ||
        url.pathname.startsWith('/integrations') || url.pathname.startsWith('/memory') ||
        url.pathname.startsWith('/projects') || url.pathname.startsWith('/references')) {
        event.respondWith(fetch(event.request).catch(() => caches.match(event.request)));
    } else {
        event.respondWith(
            caches.match(event.request).then(cached => cached || fetch(event.request).then(resp => {
                if (resp.ok) {
                    const clone = resp.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
                }
                return resp;
            }))
        );
    }
});
""".strip()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
