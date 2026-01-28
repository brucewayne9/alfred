"""Memory store using ChromaDB for RAG and conversation history."""

import logging
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

_client = None
_collections = {}


def get_client() -> chromadb.ClientAPI:
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path="/home/aialfred/alfred/data/chromadb",
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialized")
    return _client


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a named collection."""
    if name not in _collections:
        client = get_client()
        _collections[name] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{name}' ready ({_collections[name].count()} docs)")
    return _collections[name]


def store_memory(
    text: str,
    category: str = "general",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Store a piece of information in long-term memory."""
    collection = get_collection(f"memory_{category}")
    doc_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    meta = metadata or {}
    meta["timestamp"] = datetime.now().isoformat()
    meta["category"] = category

    collection.add(
        documents=[text],
        metadatas=[meta],
        ids=[doc_id],
    )
    logger.info(f"Stored memory: {doc_id}")
    return doc_id


def recall(
    query: str,
    category: str = "general",
    n_results: int = 5,
) -> list[dict]:
    """Recall relevant memories based on a query."""
    collection = get_collection(f"memory_{category}")

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    memories = []
    for i, doc in enumerate(results["documents"][0]):
        memories.append({
            "text": doc,
            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            "distance": results["distances"][0][i] if results["distances"] else 0,
        })

    return memories


def store_conversation(user_msg: str, assistant_msg: str, session_id: str = "default"):
    """Store a conversation exchange."""
    collection = get_collection("conversations")
    timestamp = datetime.now().isoformat()
    doc_id = f"conv_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    collection.add(
        documents=[f"User: {user_msg}\nAlfred: {assistant_msg}"],
        metadatas=[{
            "session_id": session_id,
            "timestamp": timestamp,
            "user_message": user_msg[:500],
        }],
        ids=[doc_id],
    )


def search_conversations(query: str, n_results: int = 5) -> list[dict]:
    """Search past conversations."""
    return recall(query, category="conversations", n_results=n_results)
