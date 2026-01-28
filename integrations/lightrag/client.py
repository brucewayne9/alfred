"""LightRAG integration for document memory and knowledge graph queries."""

import logging
from pathlib import Path
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

# LightRAG configuration
LIGHTRAG_URL = "http://75.43.156.117:9621"
LIGHTRAG_USER = "brucewayne9"
LIGHTRAG_PASS = "AlwaysGive100%"

# Token cache
_token_cache = {"token": None, "expires": None}


async def _get_token() -> str:
    """Get or refresh the auth token."""
    now = datetime.now()

    # Return cached token if still valid
    if _token_cache["token"] and _token_cache["expires"] and now < _token_cache["expires"]:
        return _token_cache["token"]

    # Login to get new token
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{LIGHTRAG_URL}/login",
            data={"username": LIGHTRAG_USER, "password": LIGHTRAG_PASS},
        )
        if resp.status_code == 200:
            data = resp.json()
            _token_cache["token"] = data["access_token"]
            # Token typically expires in 24h, refresh at 23h
            _token_cache["expires"] = now + timedelta(hours=23)
            return _token_cache["token"]
        else:
            raise Exception(f"LightRAG login failed: {resp.text}")


async def _auth_headers() -> dict:
    """Get authorization headers."""
    token = await _get_token()
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


async def list_documents(limit: int = 100, offset: int = 0) -> dict:
    """List documents in LightRAG."""
    try:
        headers = await _auth_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{LIGHTRAG_URL}/documents/paginated",
                headers=headers,
                json={"page": offset // limit + 1, "page_size": limit},
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
                params={"label": label},
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
