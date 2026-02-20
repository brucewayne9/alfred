"""LightRAG integration for document memory and knowledge graph queries."""

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/aialfred/alfred/config/.env")

logger = logging.getLogger(__name__)

# LightRAG configuration from environment
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://75.43.156.117:9621")
LIGHTRAG_USER = os.getenv("LIGHTRAG_USER", "")
LIGHTRAG_PASS = os.getenv("LIGHTRAG_PASS", "")

# Token cache
_token_cache = {"token": None, "expires": None}

# Circuit breaker: skip LightRAG for a cooldown period after failures
# This prevents every chat request from waiting 30s for an unreachable server
_circuit_breaker = {"failures": 0, "last_failure": None, "cooldown_until": None}
CIRCUIT_BREAKER_THRESHOLD = 2  # Open circuit after 2 consecutive failures
CIRCUIT_BREAKER_COOLDOWN = timedelta(hours=1)  # Stay open for 1 hour (server is down)


def _circuit_is_open() -> bool:
    """Check if the circuit breaker is open (LightRAG is considered down)."""
    cb = _circuit_breaker
    if cb["cooldown_until"] and datetime.now() < cb["cooldown_until"]:
        return True
    # Reset if cooldown has passed
    if cb["cooldown_until"] and datetime.now() >= cb["cooldown_until"]:
        cb["failures"] = 0
        cb["cooldown_until"] = None
    return False


def _record_failure():
    """Record a failure and potentially open the circuit."""
    cb = _circuit_breaker
    cb["failures"] += 1
    cb["last_failure"] = datetime.now()
    if cb["failures"] >= CIRCUIT_BREAKER_THRESHOLD:
        cb["cooldown_until"] = datetime.now() + CIRCUIT_BREAKER_COOLDOWN
        logger.warning(
            f"LightRAG circuit breaker OPEN after {cb['failures']} failures. "
            f"Will retry after {CIRCUIT_BREAKER_COOLDOWN.total_seconds()}s"
        )


def _record_success():
    """Reset circuit breaker on success."""
    _circuit_breaker["failures"] = 0
    _circuit_breaker["cooldown_until"] = None


async def _get_token(timeout: float = 5) -> str:
    """Get or refresh the auth token.

    Args:
        timeout: HTTP timeout in seconds (default 5s for fast-path calls).
    """
    now = datetime.now()

    # Return cached token if still valid
    if _token_cache["token"] and _token_cache["expires"] and now < _token_cache["expires"]:
        return _token_cache["token"]

    # Login to get new token
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{LIGHTRAG_URL}/login",
            data={"username": LIGHTRAG_USER, "password": LIGHTRAG_PASS},
        )
        if resp.status_code == 200:
            data = resp.json()
            _token_cache["token"] = data["access_token"]
            # Token typically expires in 24h, refresh at 23h
            _token_cache["expires"] = now + timedelta(hours=23)
            _record_success()
            return _token_cache["token"]
        else:
            _record_failure()
            raise Exception(f"LightRAG login failed: {resp.text}")


async def _auth_headers(timeout: float = 5) -> dict:
    """Get authorization headers."""
    token = await _get_token(timeout=timeout)
    return {"Authorization": f"Bearer {token}"}


async def health_check() -> dict:
    """Check if LightRAG is healthy."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{LIGHTRAG_URL}/health")
            if resp.status_code == 200:
                return {"healthy": True, "details": resp.json()}
            return {"healthy": False, "error": resp.text}
    except Exception as e:
        return {"healthy": False, "error": str(e)}


async def upload_text(text: str, description: str = "") -> dict:
    """Upload text content to LightRAG for indexing.

    Args:
        text: The text content to index
        description: Optional description of the content

    Returns:
        dict with success status and details
    """
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/documents/text",
                headers=headers,
                json={"text": text, "description": description},
            )
            if resp.status_code == 200:
                return {"success": True, "result": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG upload_text failed: {e}")
        return {"success": False, "error": str(e)}


async def upload_file(file_path: str) -> dict:
    """Upload a file to LightRAG for indexing.

    Args:
        file_path: Path to the file to upload

    Returns:
        dict with success status and details
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=120) as client:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "application/octet-stream")}
                resp = await client.post(
                    f"{LIGHTRAG_URL}/documents/upload",
                    headers=headers,
                    files=files,
                )
            if resp.status_code == 200:
                return {"success": True, "result": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG upload_file failed: {e}")
        return {"success": False, "error": str(e)}


async def query(
    query_text: str,
    mode: str = "hybrid",
    only_need_context: bool = False,
    top_k: int = 10,
) -> dict:
    """Query the LightRAG knowledge graph.

    Args:
        query_text: The question or query
        mode: Search mode - 'naive', 'local', 'global', 'hybrid' (default)
        only_need_context: If True, return only context chunks without LLM response
        top_k: Number of results to return

    Returns:
        dict with response and context
    """
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/query",
                headers=headers,
                json={
                    "query": query_text,
                    "mode": mode,
                    "only_need_context": only_need_context,
                    "top_k": top_k,
                },
            )
            if resp.status_code == 200:
                return {"success": True, "result": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG query failed: {e}")
        return {"success": False, "error": str(e)}


async def query_context(query_text: str, top_k: int = 5) -> dict:
    """Get relevant context from LightRAG without LLM response.

    Useful for feeding context to Alfred's LLM.
    """
    return await query(query_text, mode="hybrid", only_need_context=True, top_k=top_k)


async def query_context_fast(query_text: str, top_k: int = 5) -> dict:
    """Fast context query with short timeout for the chat hot path.

    Unlike query_context, this uses a 5s timeout to avoid blocking
    chat responses when LightRAG is slow or unreachable.
    """
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/query",
                headers=headers,
                json={
                    "query": query_text,
                    "mode": "hybrid",
                    "only_need_context": True,
                    "top_k": top_k,
                },
            )
            if resp.status_code == 200:
                _record_success()
                return {"success": True, "result": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG fast query failed: {e}")
        _record_failure()
        return {"success": False, "error": str(e)}


async def list_documents(limit: int = 100, offset: int = 0) -> dict:
    """List documents in LightRAG."""
    try:
        headers = await _auth_headers()
        # LightRAG requires page_size >= 10
        page_size = max(10, limit)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/documents/paginated",
                headers=headers,
                json={"page": offset // page_size + 1, "page_size": page_size},
            )
            if resp.status_code == 200:
                return {"success": True, "documents": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG list_documents failed: {e}")
        return {"success": False, "error": str(e)}


async def get_document_status() -> dict:
    """Get document processing status counts."""
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{LIGHTRAG_URL}/documents/status_counts",
                headers=headers,
            )
            if resp.status_code == 200:
                return {"success": True, "status": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG get_document_status failed: {e}")
        return {"success": False, "error": str(e)}


async def delete_document(doc_id: str) -> dict:
    """Delete a document from LightRAG."""
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.delete(
                f"{LIGHTRAG_URL}/documents/delete_document",
                headers=headers,
                params={"doc_id": doc_id},
            )
            if resp.status_code == 200:
                return {"success": True, "result": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG delete_document failed: {e}")
        return {"success": False, "error": str(e)}


async def search_graph(label: str) -> dict:
    """Search the knowledge graph for entities matching a label."""
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{LIGHTRAG_URL}/graph/label/search",
                headers=headers,
                params={"q": label},  # API expects 'q' parameter
            )
            if resp.status_code == 200:
                return {"success": True, "entities": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG search_graph failed: {e}")
        return {"success": False, "error": str(e)}


async def get_popular_entities(limit: int = 20) -> dict:
    """Get the most connected entities in the knowledge graph."""
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{LIGHTRAG_URL}/graph/label/popular",
                headers=headers,
                params={"limit": limit},
            )
            if resp.status_code == 200:
                return {"success": True, "entities": resp.json()}
            return {"success": False, "error": resp.text}
    except Exception as e:
        logger.error(f"LightRAG get_popular_entities failed: {e}")
        return {"success": False, "error": str(e)}


async def get_knowledge_context(query: str, top_k: int = 5) -> str:
    """Get relevant knowledge context for a query.

    This is the main function Alfred uses to enrich responses
    with knowledge from the graph. Uses a circuit breaker and short
    timeout to avoid blocking chat responses when LightRAG is down.

    Args:
        query: The user's query
        top_k: Number of context chunks to retrieve

    Returns:
        String of relevant context, or empty string if none found
    """
    # Circuit breaker: skip if LightRAG has been failing
    if _circuit_is_open():
        return ""

    try:
        # Use a short timeout (5s) for the context enrichment path
        # so chat responses are not blocked when LightRAG is unreachable
        result = await query_context_fast(query, top_k=top_k)
        if result.get("success") and result.get("result"):
            context = result["result"]
            if isinstance(context, dict):
                # Extract text from context response
                chunks = context.get("chunks", [])
                if chunks:
                    return "\n\n".join([c.get("content", "") for c in chunks if c.get("content")])
                # Or it might be a direct response
                return context.get("response", "")
            elif isinstance(context, str):
                return context
        return ""
    except Exception as e:
        logger.debug(f"Knowledge context retrieval failed: {e}")
        _record_failure()
        return ""


def reset_circuit_breaker() -> dict:
    """Reset the LightRAG circuit breaker to closed state.
    Safe to call anytime — no-op if breaker is already closed.
    Also clears token cache to force fresh authentication."""
    cb = _circuit_breaker
    was_open = bool(cb["cooldown_until"] and datetime.now() < cb["cooldown_until"])
    cb["failures"] = 0
    cb["last_failure"] = None
    cb["cooldown_until"] = None
    _token_cache["token"] = None
    _token_cache["expires"] = None
    logger.info(f"Circuit breaker reset (was_open={was_open})")
    return {
        "reset": True,
        "was_open": was_open,
        "message": "Circuit breaker reset — LightRAG will be retried" if was_open else "All breakers healthy — no action needed",
    }


def get_circuit_breaker_status() -> dict:
    """Get current circuit breaker state for diagnostics."""
    cb = _circuit_breaker
    return {
        "is_open": _circuit_is_open(),
        "failures": cb["failures"],
        "threshold": CIRCUIT_BREAKER_THRESHOLD,
        "cooldown_hours": CIRCUIT_BREAKER_COOLDOWN.total_seconds() / 3600,
        "last_failure": cb["last_failure"].isoformat() if cb["last_failure"] else None,
        "cooldown_until": cb["cooldown_until"].isoformat() if cb["cooldown_until"] else None,
    }


def is_configured() -> bool:
    """Check if LightRAG is properly configured."""
    return bool(LIGHTRAG_URL and LIGHTRAG_USER and LIGHTRAG_PASS)


async def is_connected() -> bool:
    """Check if LightRAG is accessible."""
    if not is_configured():
        return False
    health = await health_check()
    return health.get("healthy", False)
