"""Alfred API Server - Main FastAPI application with auth, integrations, and tool calling."""

import logging
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

class ChatResponse(BaseModel):
    response: str
    tier: str
    timestamp: str
    images: list[dict] | None = None  # [{base64, filename, download_url}]

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
        "ollama": {
            "model": settings.ollama_model,
            "host": settings.ollama_host,
        },
        "anthropic": {
            "configured": bool(settings.anthropic_api_key and settings.anthropic_api_key != "sk-ant-CHANGEME"),
            "model": settings.anthropic_model,
        },
    }


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
async def conversations_list(limit: int = 50, offset: int = 0, user: dict = Depends(require_auth)):
    """List conversations ordered by most recent."""
    return list_convos(limit=limit, offset=offset)


@app.post("/conversations")
async def conversations_create(user: dict = Depends(require_auth)):
    """Create a new conversation."""
    return create_conversation()


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

    tier = ModelTier(req.tier) if req.tier else classify_query(req.message)
    result = await ask(req.message, messages=messages, tier=tier)

    # Handle both old string return and new dict return
    if isinstance(result, dict):
        response = result["response"]
        images = result.get("images")
    else:
        response = result
        images = None

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
    """Convert text to speech audio."""
    import asyncio
    from interfaces.voice.tts import speak
    loop = asyncio.get_event_loop()
    audio_data = await loop.run_in_executor(None, lambda: speak(req.message))
    return Response(content=audio_data, media_type="audio/wav")


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
    threshold = 0.5

    try:
        while True:
            # Receive binary audio data
            data = await ws.receive_bytes()

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
            padding: 16px 24px; border-bottom: 1px solid #222;
            display: flex; align-items: center; gap: 12px;
        }
        header h1 { font-size: 20px; font-weight: 600; color: #fff; }
        header .status { font-size: 12px; color: #4ade80; }
        .header-right { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .header-btn {
            background: #222; border: 1px solid #333; color: #ccc; padding: 6px 12px;
            border-radius: 6px; cursor: pointer; font-size: 13px;
        }
        .header-btn:hover { background: #333; color: #fff; }
        #hamburger-btn {
            background: none; border: none; color: #ccc; font-size: 22px;
            cursor: pointer; padding: 4px 8px; border-radius: 4px; line-height: 1;
        }
        #hamburger-btn:hover { background: #222; color: #fff; }
        #chat {
            flex: 1; overflow-y: auto; padding: 24px;
            display: flex; flex-direction: column; gap: 16px;
        }
        .msg { max-width: 75%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; }
        .msg.user {
            align-self: flex-end; background: #1e3a5f; color: #e0e0e0;
            border-bottom-right-radius: 4px;
        }
        .msg.alfred {
            align-self: flex-start; background: #1a1a1a; border: 1px solid #333;
            border-bottom-left-radius: 4px;
        }
        .msg .label { font-size: 11px; color: #888; margin-bottom: 4px; }
        .msg .content { white-space: pre-wrap; font-family: inherit; }
        .msg .content code { background: #222; padding: 1px 5px; border-radius: 3px; font-family: monospace; font-size: 13px; }
        .msg .content strong { color: #fff; }

        /* Thinking indicator */
        #thinking {
            display: none; align-self: flex-start; padding: 16px 20px;
            background: #1a1a1a; border: 1px solid #333; border-radius: 12px;
            margin: 8px 0;
        }
        #thinking.visible { display: flex; align-items: center; gap: 12px; }
        #thinking .label { font-size: 11px; color: #888; }
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
            padding: 16px 24px; border-top: 1px solid #222;
            display: flex; gap: 12px; align-items: center;
        }
        #input {
            flex: 1; padding: 12px 16px; border-radius: 8px;
            border: 1px solid #333; background: #111; color: #e0e0e0;
            font-size: 15px; outline: none; resize: none;
            min-height: 44px; max-height: 120px;
        }
        #input:focus { border-color: #4a9eff; }
        button {
            padding: 10px 20px; border-radius: 8px; border: none;
            background: #2563eb; color: white; font-size: 15px;
            cursor: pointer; font-weight: 500;
        }
        button:hover { background: #1d4ed8; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        #mic-btn {
            background: transparent; width: 44px; height: 44px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            padding: 0; overflow: hidden; border: 2px solid #444; transition: all 0.2s;
        }
        #mic-btn img { width: 100%; height: 100%; object-fit: cover; border-radius: 50%; }
        #mic-btn:hover { border-color: #e86e2c; }
        #mic-btn.recording { border-color: #dc2626; box-shadow: 0 0 12px rgba(220,38,38,0.6); animation: pulse 1s infinite; }
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
        #upload-btn {
            background: #222; width: 44px; height: 44px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; border: 1px solid #333; color: #888; cursor: pointer;
            transition: all 0.2s;
        }
        #upload-btn:hover { border-color: #4a9eff; color: #4a9eff; }
        #upload-btn.has-file { border-color: #4ade80; color: #4ade80; }
        #file-input { display: none; }
        #pending-file {
            display: none; padding: 8px 16px; background: #1a1a1a;
            border-radius: 8px; margin: 8px 24px; font-size: 13px;
            align-items: center; gap: 10px;
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
            display: none; padding: 6px 24px; text-align: center;
            font-size: 12px; color: #888; border-top: 1px solid #1a1a1a;
            letter-spacing: 0.5px; text-transform: uppercase;
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
            padding: 16px; border-bottom: 1px solid #222;
            display: flex; align-items: center; justify-content: space-between;
        }
        .history-header h2 { font-size: 16px; color: #fff; }
        #new-chat-btn {
            padding: 6px 14px; font-size: 13px; border-radius: 6px;
            background: #2563eb; border: none; color: white; cursor: pointer;
        }
        #new-chat-btn:hover { background: #1d4ed8; }
        #conversation-list {
            flex: 1; overflow-y: auto; padding: 8px;
        }
        .conv-item {
            padding: 10px 12px; border-radius: 8px; cursor: pointer;
            margin-bottom: 4px; transition: background 0.15s;
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
        .conv-delete-btn {
            background: none; border: none; color: #555; cursor: pointer;
            font-size: 14px; padding: 4px 6px; border-radius: 3px; visibility: hidden;
        }
        .conv-item:hover .conv-delete-btn { visibility: visible; }
        .conv-delete-btn:hover { color: #f87171; background: #1a1a1a; }
        @media (pointer: coarse) { .conv-delete-btn { visibility: visible !important; } }

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
            .msg { max-width: 95%; padding: 10px 12px; font-size: 14px; }
            header { padding: 10px 12px; gap: 6px; }
            header h1 { font-size: 17px; }
            .header-right { gap: 4px; }
            .header-btn { padding: 5px 8px; font-size: 12px; }
            .mode-toggle { padding: 5px 10px; font-size: 11px; gap: 5px; }
            .mode-toggle .dot { width: 5px; height: 5px; }
            #vad-status { padding: 4px 12px; font-size: 11px; }
            #input-area { padding: 10px 12px; gap: 8px; }
            #input { font-size: 14px; padding: 10px 12px; }
            #mic-btn { width: 38px; height: 38px; font-size: 17px; }
            #send-btn { padding: 8px 14px; font-size: 14px; }
            #chat { padding: 16px 12px; gap: 12px; }
            #history-panel { width: 100%; left: -100%; }
            .login-box { padding: 24px 20px; }
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
            <h2>Conversations</h2>
            <button id="new-chat-btn" onclick="newConversation()">New Chat</button>
        </div>
        <div id="conversation-list"></div>
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
            <button class="header-btn" onclick="toggleSettings()">Settings</button>
        </div>
    </header>
    <div id="chat">
        <div class="msg alfred">
            <div class="label">Alfred</div>
            <div class="content">Good day, sir. How can I assist you?</div>
        </div>
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
    <div id="input-area">
        <button id="mic-btn" onclick="toggleMic()" title="Voice input"><img src="/static/gr-mic.jpeg" alt="Mic"></button>
        <button id="upload-btn" onclick="document.getElementById('file-input').click()" title="Upload file">&#128206;</button>
        <input type="file" id="file-input" accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt,.md,.json,.jpg,.jpeg,.png,.gif,.webp" onchange="handleFileSelect(event)">
        <textarea id="input" placeholder="Ask Alfred anything..." rows="1"
            onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
        <button id="send-btn" onclick="send()">Send</button>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        let mediaRecorder = null;
        let isRecording = false;
        let authToken = localStorage.getItem('alfred_token') || '';
        let currentConversationId = null;
        let loadingHistory = false;

        // ==================== Conversation History ====================

        function toggleHistory() {
            document.getElementById('history-panel').classList.toggle('open');
            document.getElementById('history-overlay').classList.toggle('open');
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
                const resp = await fetch('/conversations', {headers: authHeaders()});
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
                            <button class="conv-delete-btn" onclick="event.stopPropagation();deleteConversation('${c.id}')" title="Delete">&#10005;</button>
                        </div>
                    </div>`;
                }).join('');
            } catch(e) {
                console.error('Failed to load conversations:', e);
            }
        }

        async function newConversation() {
            try {
                const resp = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
                const data = await resp.json();
                currentConversationId = data.id;
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
                // Reload messages (suppress auto-speak for historical messages)
                loadingHistory = true;
                data.messages.forEach(m => {
                    if (m.role === 'user') addMsg(m.content, 'user', null, true);
                    else if (m.role === 'assistant') addMsg(m.content, 'alfred', m.tier, true);
                });
                loadingHistory = false;
                await loadConversations();
                toggleHistory();
            } catch(e) {
                console.error('Failed to load conversation:', e);
            }
        }

        async function deleteConversation(convId) {
            if (!confirm('Delete this conversation?')) return;
            try {
                await fetch('/conversations/' + convId, {method: 'DELETE', headers: authHeaders()});
                if (convId === currentConversationId) {
                    currentConversationId = null;
                    clearChat();
                    // Start a fresh conversation
                    try {
                        const cr = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
                        const cd = await cr.json();
                        currentConversationId = cd.id;
                    } catch(e2) {}
                }
                await loadConversations();
            } catch(e) {
                console.error('Failed to delete conversation:', e);
            }
        }

        function clearChat() {
            chat.innerHTML = '<div class="msg alfred"><div class="label">Alfred</div><div class="content">Good day, sir. How can I assist you?</div></div><div id="thinking"><div class="morph-shape"></div><div class="label">Alfred is thinking...</div></div>';
            msgTexts = {};
            msgCounter = 0;
        }

        async function initConversations() {
            await loadConversations();
            // Auto-load most recent or create new
            const resp = await fetch('/conversations?limit=1', {headers: authHeaders()});
            const convs = await resp.json();
            if (convs.length) {
                await switchConversation(convs[0].id);
            } else {
                const cr = await fetch('/conversations', {method: 'POST', headers: authHeaders()});
                const data = await cr.json();
                currentConversationId = data.id;
                await loadConversations();
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

                document.getElementById('integration-list').innerHTML = html;

                // LLM info
                let llmHtml = `<div class="status-item">
                    <span><span class="status-dot green"></span>Local: ${data.ollama?.model || 'N/A'}</span>
                </div>`;
                const aConfigured = data.anthropic?.configured;
                llmHtml += `<div class="status-item">
                    <span><span class="status-dot ${aConfigured?'green':'red'}"></span>Cloud: ${data.anthropic?.model || 'N/A'}</span>
                    <span style="color:#888;font-size:12px">${aConfigured?'Active':'No key'}</span>
                </div>`;
                document.getElementById('llm-info').innerHTML = llmHtml;

                // Account
                const me = await (await fetch('/auth/me', {headers: authHeaders()})).json();
                document.getElementById('account-info').innerHTML =
                    `<div class="status-item"><span>User: ${me.username || 'N/A'}</span><span style="color:#888;font-size:12px">${me.role || ''}</span></div>`;
            } catch(e) {
                document.getElementById('integration-list').innerHTML = '<span style="color:#f87171">Failed to load</span>';
            }
        }

        let autoSpeak = localStorage.getItem('alfred_auto_speak') === 'true';
        let currentAudio = null;
        let msgTexts = {};
        let msgCounter = 0;
        if (autoSpeak) document.getElementById('auto-speak-btn')?.classList.add('active');

        function addMsg(text, role, tier, noAutoSpeak = false) {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            let label = role === 'user' ? 'You' : 'Alfred';
            let badge = tier ? `<span class="tier-badge tier-${tier}">${tier}</span>` : '';
            let voiceBtns = '';
            if (role === 'alfred') {
                const mid = ++msgCounter;
                msgTexts[mid] = text;
                voiceBtns = `<button class="speak-btn" data-mid="${mid}" title="Read aloud">&#128264;</button><button class="stop-btn" data-mid="${mid}" title="Stop">&#9632;</button>`;
            }
            div.innerHTML = `<div class="label">${label} ${badge}${voiceBtns}</div><div class="content">${renderText(text)}</div>`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            if (role === 'alfred' && autoSpeak && !loadingHistory && !noAutoSpeak) {
                const btn = div.querySelector('.speak-btn');
                if (btn) speakText(btn);
            }
            return div;
        }

        function addMsgHtml(text, role, tier, noAutoSpeak = false, extraHtml = '') {
            const div = document.createElement('div');
            div.className = `msg ${role}`;
            let label = role === 'user' ? 'You' : 'Alfred';
            let badge = tier ? `<span class="tier-badge tier-${tier}">${tier}</span>` : '';
            let voiceBtns = '';
            if (role === 'alfred') {
                const mid = ++msgCounter;
                msgTexts[mid] = text;
                voiceBtns = `<button class="speak-btn" data-mid="${mid}" title="Read aloud">&#128264;</button><button class="stop-btn" data-mid="${mid}" title="Stop">&#9632;</button>`;
            }
            div.innerHTML = `<div class="label">${label} ${badge}${voiceBtns}</div><div class="content">${renderText(text)}${extraHtml}</div>`;
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
        });

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

        async function speakText(btn) {
            const mid = btn.getAttribute('data-mid');
            const text = msgTexts[mid];
            if (!text) return;
            // Stop any current audio
            stopAudio();
            // Toggle off if clicking same button
            if (btn.classList.contains('speaking')) { btn.classList.remove('speaking'); return; }
            btn.classList.add('speaking');
            // Show stop button for this message
            const stopBtn = btn.parentElement.querySelector('.stop-btn');
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
            document.getElementById('upload-btn').classList.add('has-file');
        }

        function clearPendingFile() {
            pendingFile = null;
            document.getElementById('pending-file').classList.remove('visible');
            document.getElementById('upload-btn').classList.remove('has-file');
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
                // Check for generated images
                if (data.images && data.images.length > 0) {
                    let imagesHtml = '';
                    data.images.forEach(img => {
                        imagesHtml += `<img class="inline-image" src="data:image/png;base64,${img.base64}" alt="${img.filename}">`;
                        if (img.download_url) {
                            imagesHtml += `<a class="download-btn" href="${img.download_url}" download>📥 ${img.filename}</a>`;
                        }
                    });
                    addMsgHtml(data.response, 'alfred', data.tier, false, imagesHtml);
                } else {
                    addMsg(data.response, 'alfred', data.tier);
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
            const btn = document.getElementById('mic-btn');
            if (isRecording) {
                mediaRecorder.stop();
                btn.classList.remove('recording');
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
                        if (data.text) { input.value = data.text; send(); }
                    } catch(e) { console.error('Transcription failed:', e); }
                };
                mediaRecorder.start();
                btn.classList.add('recording');
                isRecording = true;
            }
        }

        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
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
                return;
            }

            // Start wake word detection
            try {
                btn.classList.add('active');
                status.classList.add('visible');
                status.textContent = 'Starting...';

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
                    if (wakeWordActive) {
                        // Reconnect after a short delay
                        setTimeout(() => {
                            if (wakeWordActive && !wakeWordWs) {
                                toggleWakeWord();
                                toggleWakeWord();
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
                currentAudio.pause();
                currentAudio = null;
                const s = document.querySelector('.speak-btn.speaking');
                if (s) s.classList.remove('speaking');
            }
        }

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
            fetch('/voice/speak', {
                method:'POST',
                headers: authHeaders({'Content-Type':'application/json'}),
                body: JSON.stringify({message: text})
            }).then(r => r.ok ? r.blob() : null).then(blob => {
                if (!blob || !handsFreeActive) { setVadState(handsFreeActive?'listen':'idle'); return; }
                const url = URL.createObjectURL(blob);
                currentAudio = new Audio(url);
                currentAudio.onended = () => {
                    currentAudio = null; URL.revokeObjectURL(url);
                    if (handsFreeActive) setVadState('listen');
                };
                currentAudio.onerror = () => {
                    currentAudio = null;
                    if (handsFreeActive) setVadState('listen');
                };
                currentAudio.play();
            }).catch(() => { if (handsFreeActive) setVadState('listen'); });
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

                addMsg(text, 'user');
                if (!currentConversationId) {
                    try {
                        const cr = await fetch('/conversations', {method:'POST', headers:authHeaders()});
                        currentConversationId = (await cr.json()).id;
                    } catch(e){}
                }
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
                // Play TTS non-blocking — VAD stays active so user can interrupt
                playVadTTS(chatData.response);
            } catch(e) {
                console.error('Hands-free error:', e);
                vadProcessing = false;
                if (handsFreeActive) setVadState('listen');
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
                return;
            }
            handsFreeActive = true;
            btn.classList.add('active');
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
                        positiveSpeechThreshold: 0.8,
                        negativeSpeechThreshold: 0.3,
                        minSpeechFrames: 4,
                        preSpeechPadFrames: 8,
                        redemptionFrames: 6,
                        onSpeechStart: () => {
                            if (!handsFreeActive) return;
                            // Interrupt Alfred if he's speaking
                            if (currentAudio && vadState === 'speak') cutOffAlfred();
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
const CACHE_NAME = 'alfred-v11';
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
        url.pathname.startsWith('/integrations') || url.pathname.startsWith('/memory')) {
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
