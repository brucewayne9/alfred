"""SQLite-based conversation history persistence."""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "conversations.db"

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
    return _conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_archived INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tier TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
    """)
    conn.commit()


def create_conversation(title: str = "") -> dict:
    conn = _get_conn()
    conv_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now),
    )
    conn.commit()
    return {"id": conv_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations(limit: int = 50, offset: int = 0) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT c.id, c.title, c.created_at, c.updated_at,
                  (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
           FROM conversations c
           WHERE c.is_archived = 0
           ORDER BY c.updated_at DESC
           LIMIT ? OFFSET ?""",
        (limit, offset),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "last_message": (r["last_message"] or "")[:100],
        }
        for r in rows
    ]


def get_conversation(conv_id: str) -> dict | None:
    conn = _get_conn()
    conv = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND is_archived = 0",
        (conv_id,),
    ).fetchone()
    if not conv:
        return None
    msgs = conn.execute(
        "SELECT role, content, tier, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,),
    ).fetchall()
    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "messages": [
            {"role": m["role"], "content": m["content"], "tier": m["tier"], "created_at": m["created_at"]}
            for m in msgs
        ],
    }


def add_message(conv_id: str, role: str, content: str, tier: str | None = None) -> None:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, tier, created_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, role, content, tier, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conv_id),
    )
    # Auto-set title from first user message if title is empty
    if role == "user":
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if row and not row["title"]:
            title = content[:80].split("\n")[0]
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id)
            )
    conn.commit()


def archive_conversation(conv_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE conversations SET is_archived = 1 WHERE id = ? AND is_archived = 0",
        (conv_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def update_title(conv_id: str, title: str) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND is_archived = 0",
        (title, conv_id),
    )
    conn.commit()
    return cur.rowcount > 0
