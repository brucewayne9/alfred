"""SQLite-based conversation history persistence with FTS5 search and project support."""

import sqlite3
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "conversations.db"
UPLOADS_DIR = Path(__file__).parent.parent.parent / "data" / "uploads"

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
    # Core tables
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

    # Add project_id column to conversations if not exists
    try:
        conn.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT REFERENCES projects(id)")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Projects table
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            color TEXT DEFAULT '#3b82f6',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            file_path TEXT,
            file_type TEXT,
            file_size INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_refs_project ON project_references(project_id);
    """)

    # FTS5 for messages search
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            content='messages',
            content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
            INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)

    # FTS5 for references search
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS references_fts USING fts5(
            title,
            content,
            content='project_references',
            content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS refs_ai AFTER INSERT ON project_references BEGIN
            INSERT INTO references_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS refs_ad AFTER DELETE ON project_references BEGIN
            INSERT INTO references_fts(references_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
        END;

        CREATE TRIGGER IF NOT EXISTS refs_au AFTER UPDATE ON project_references BEGIN
            INSERT INTO references_fts(references_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
            INSERT INTO references_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
        END;
    """)

    # Backfill existing messages into FTS
    conn.execute("""
        INSERT OR IGNORE INTO messages_fts(rowid, content)
        SELECT id, content FROM messages WHERE id NOT IN (SELECT rowid FROM messages_fts)
    """)

    # Backfill existing references into FTS
    conn.execute("""
        INSERT OR IGNORE INTO references_fts(rowid, title, content)
        SELECT id, title, content FROM project_references WHERE id NOT IN (SELECT rowid FROM references_fts)
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


def list_conversations(limit: int = 50, offset: int = 0, project_id: str | None = None) -> list[dict]:
    conn = _get_conn()
    if project_id is not None:
        # Filter by specific project, only show conversations with messages
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at, c.project_id,
                      (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
               FROM conversations c
               WHERE c.is_archived = 0 AND c.project_id = ?
                 AND EXISTS (SELECT 1 FROM messages WHERE conversation_id = c.id)
               ORDER BY c.updated_at DESC
               LIMIT ? OFFSET ?""",
            (project_id, limit, offset),
        ).fetchall()
    else:
        # Only show conversations that have at least one message
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at, c.project_id,
                      (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
               FROM conversations c
               WHERE c.is_archived = 0
                 AND EXISTS (SELECT 1 FROM messages WHERE conversation_id = c.id)
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
            "project_id": r["project_id"],
            "last_message": (r["last_message"] or "")[:100],
        }
        for r in rows
    ]


def get_conversation(conv_id: str) -> dict | None:
    conn = _get_conn()
    conv = conn.execute(
        "SELECT id, title, created_at, updated_at, project_id FROM conversations WHERE id = ? AND is_archived = 0",
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
        "project_id": conv["project_id"],
        "messages": [
            {"role": m["role"], "content": m["content"], "tier": m["tier"], "created_at": m["created_at"]}
            for m in msgs
        ],
    }


def add_message(conv_id: str, role: str, content: str, tier: str | None = None) -> None:
    # Prevent saving empty messages which cause API errors
    if not content or (isinstance(content, str) and not content.strip()):
        return

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


def list_archived_conversations(limit: int = 50, offset: int = 0) -> list[dict]:
    """List archived conversations."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT c.id, c.title, c.created_at, c.updated_at,
                  (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
           FROM conversations c
           WHERE c.is_archived = 1
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


def restore_conversation(conv_id: str) -> bool:
    """Restore an archived conversation."""
    conn = _get_conn()
    cur = conn.execute(
        "UPDATE conversations SET is_archived = 0 WHERE id = ? AND is_archived = 1",
        (conv_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_conversation_permanently(conv_id: str) -> bool:
    """Permanently delete a conversation and its messages."""
    conn = _get_conn()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    return cur.rowcount > 0


# ==================== Conversation Search ====================

def search_conversations(query: str, limit: int = 20) -> list[dict]:
    """Search conversations using FTS5. Returns matching conversations with snippets."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT DISTINCT
            c.id, c.title, c.created_at, c.updated_at, c.project_id,
            snippet(messages_fts, 0, '<mark>', '</mark>', '...', 32) as snippet
        FROM messages_fts
        JOIN messages m ON messages_fts.rowid = m.id
        JOIN conversations c ON m.conversation_id = c.id
        WHERE messages_fts MATCH ? AND c.is_archived = 0
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "project_id": r["project_id"],
            "snippet": r["snippet"],
        }
        for r in rows
    ]


# ==================== Projects ====================

def create_project(name: str, description: str = "", color: str = "#3b82f6") -> dict:
    """Create a new project."""
    conn = _get_conn()
    project_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO projects (id, name, description, color, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, name, description, color, now, now),
    )
    conn.commit()
    # Create uploads directory for this project
    project_uploads = UPLOADS_DIR / project_id
    project_uploads.mkdir(parents=True, exist_ok=True)
    return {"id": project_id, "name": name, "description": description, "color": color, "created_at": now, "updated_at": now}


def list_projects() -> list[dict]:
    """List all projects with reference counts."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT p.id, p.name, p.description, p.color, p.created_at, p.updated_at,
                  (SELECT COUNT(*) FROM project_references WHERE project_id = p.id) as ref_count,
                  (SELECT COUNT(*) FROM conversations WHERE project_id = p.id AND is_archived = 0) as conv_count
           FROM projects p
           ORDER BY p.updated_at DESC"""
    ).fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "color": r["color"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "reference_count": r["ref_count"],
            "conversation_count": r["conv_count"],
        }
        for r in rows
    ]


def get_project(project_id: str) -> dict | None:
    """Get a project with reference count."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT p.id, p.name, p.description, p.color, p.created_at, p.updated_at,
                  (SELECT COUNT(*) FROM project_references WHERE project_id = p.id) as ref_count,
                  (SELECT COUNT(*) FROM conversations WHERE project_id = p.id AND is_archived = 0) as conv_count
           FROM projects p WHERE p.id = ?""",
        (project_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "color": row["color"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "reference_count": row["ref_count"],
        "conversation_count": row["conv_count"],
    }


def update_project(project_id: str, name: str | None = None, description: str | None = None, color: str | None = None) -> bool:
    """Update a project."""
    conn = _get_conn()
    updates = []
    params = []
    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if description is not None:
        updates.append("description = ?")
        params.append(description)
    if color is not None:
        updates.append("color = ?")
        params.append(color)
    if not updates:
        return False
    updates.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(project_id)
    cur = conn.execute(f"UPDATE projects SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    return cur.rowcount > 0


def delete_project(project_id: str) -> bool:
    """Delete a project and all its references."""
    conn = _get_conn()
    # Get file paths to delete
    files = conn.execute(
        "SELECT file_path FROM project_references WHERE project_id = ? AND file_path IS NOT NULL",
        (project_id,),
    ).fetchall()
    # Delete files
    for f in files:
        try:
            file_path = UPLOADS_DIR / f["file_path"]
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
    # Delete project directory
    try:
        project_dir = UPLOADS_DIR / project_id
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir)
    except Exception:
        pass
    # Unassign conversations from this project
    conn.execute("UPDATE conversations SET project_id = NULL WHERE project_id = ?", (project_id,))
    # Delete references (cascade should handle this, but explicit is safer)
    conn.execute("DELETE FROM project_references WHERE project_id = ?", (project_id,))
    cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    return cur.rowcount > 0


# ==================== Project References ====================

def add_reference(
    project_id: str,
    ref_type: str,
    title: str,
    content: str | None = None,
    file_path: str | None = None,
    file_type: str | None = None,
    file_size: int | None = None,
) -> dict:
    """Add a note or file reference to a project."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO project_references
           (project_id, type, title, content, file_path, file_type, file_size, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project_id, ref_type, title, content, file_path, file_type, file_size, now, now),
    )
    ref_id = cur.lastrowid
    # Update project's updated_at
    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
    conn.commit()
    return {
        "id": ref_id,
        "project_id": project_id,
        "type": ref_type,
        "title": title,
        "content": content,
        "file_path": file_path,
        "file_type": file_type,
        "file_size": file_size,
        "created_at": now,
        "updated_at": now,
    }


def list_references(project_id: str) -> list[dict]:
    """List all references for a project."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT id, project_id, type, title, content, file_path, file_type, file_size, created_at, updated_at
           FROM project_references WHERE project_id = ? ORDER BY created_at DESC""",
        (project_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "project_id": r["project_id"],
            "type": r["type"],
            "title": r["title"],
            "content": r["content"][:500] if r["content"] else None,  # Truncate for list
            "file_path": r["file_path"],
            "file_type": r["file_type"],
            "file_size": r["file_size"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def get_reference(ref_id: int) -> dict | None:
    """Get a single reference with full content."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT id, project_id, type, title, content, file_path, file_type, file_size, created_at, updated_at
           FROM project_references WHERE id = ?""",
        (ref_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "type": row["type"],
        "title": row["title"],
        "content": row["content"],
        "file_path": row["file_path"],
        "file_type": row["file_type"],
        "file_size": row["file_size"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def update_reference(ref_id: int, title: str | None = None, content: str | None = None) -> bool:
    """Update a reference (typically a note)."""
    conn = _get_conn()
    updates = []
    params = []
    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if content is not None:
        updates.append("content = ?")
        params.append(content)
    if not updates:
        return False
    updates.append("updated_at = ?")
    now = datetime.now(timezone.utc).isoformat()
    params.append(now)
    params.append(ref_id)
    cur = conn.execute(f"UPDATE project_references SET {', '.join(updates)} WHERE id = ?", params)
    # Update project's updated_at
    if cur.rowcount > 0:
        row = conn.execute("SELECT project_id FROM project_references WHERE id = ?", (ref_id,)).fetchone()
        if row:
            conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, row["project_id"]))
    conn.commit()
    return cur.rowcount > 0


def delete_reference(ref_id: int) -> bool:
    """Delete a reference (and file if applicable)."""
    conn = _get_conn()
    row = conn.execute("SELECT project_id, file_path FROM project_references WHERE id = ?", (ref_id,)).fetchone()
    if not row:
        return False
    # Delete file if exists
    if row["file_path"]:
        try:
            file_path = UPLOADS_DIR / row["file_path"]
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass
    cur = conn.execute("DELETE FROM project_references WHERE id = ?", (ref_id,))
    # Update project's updated_at
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, row["project_id"]))
    conn.commit()
    return cur.rowcount > 0


def search_references(project_id: str, query: str, limit: int = 20) -> list[dict]:
    """Search references within a project using FTS5."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT
            pr.id, pr.project_id, pr.type, pr.title, pr.file_path, pr.file_type, pr.file_size,
            pr.created_at, pr.updated_at,
            snippet(references_fts, 0, '<mark>', '</mark>', '...', 16) as title_snippet,
            snippet(references_fts, 1, '<mark>', '</mark>', '...', 32) as content_snippet
        FROM references_fts
        JOIN project_references pr ON references_fts.rowid = pr.id
        WHERE references_fts MATCH ? AND pr.project_id = ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, project_id, limit),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "project_id": r["project_id"],
            "type": r["type"],
            "title": r["title"],
            "file_path": r["file_path"],
            "file_type": r["file_type"],
            "file_size": r["file_size"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "title_snippet": r["title_snippet"],
            "content_snippet": r["content_snippet"],
        }
        for r in rows
    ]


# ==================== Conversation-Project Integration ====================

def move_conversation_to_project(conv_id: str, project_id: str | None) -> bool:
    """Move a conversation to a project (or remove from project if project_id is None)."""
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "UPDATE conversations SET project_id = ?, updated_at = ? WHERE id = ?",
        (project_id, now, conv_id),
    )
    conn.commit()
    return cur.rowcount > 0


def list_conversations_by_project(project_id: str | None, limit: int = 50, offset: int = 0) -> list[dict]:
    """List conversations filtered by project."""
    conn = _get_conn()
    if project_id:
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at, c.project_id,
                      (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
               FROM conversations c
               WHERE c.is_archived = 0 AND c.project_id = ?
               ORDER BY c.updated_at DESC
               LIMIT ? OFFSET ?""",
            (project_id, limit, offset),
        ).fetchall()
    else:
        # Conversations not in any project
        rows = conn.execute(
            """SELECT c.id, c.title, c.created_at, c.updated_at, c.project_id,
                      (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1) as last_message
               FROM conversations c
               WHERE c.is_archived = 0 AND c.project_id IS NULL
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
            "project_id": r["project_id"],
            "last_message": (r["last_message"] or "")[:100],
        }
        for r in rows
    ]


def get_project_context(project_id: str) -> str:
    """Get all reference content for AI context injection."""
    conn = _get_conn()
    project = get_project(project_id)
    if not project:
        return ""

    refs = conn.execute(
        """SELECT type, title, content FROM project_references
           WHERE project_id = ? ORDER BY type, created_at""",
        (project_id,),
    ).fetchall()

    if not refs:
        return f"Project: {project['name']}\nDescription: {project['description']}\n(No reference documents)"

    context_parts = [f"Project: {project['name']}"]
    if project["description"]:
        context_parts.append(f"Description: {project['description']}")
    context_parts.append("\nReference Documents:")

    for ref in refs:
        ref_type = "Note" if ref["type"] == "note" else "File"
        context_parts.append(f"\n[{ref_type}: {ref['title']}]")
        if ref["content"]:
            # Truncate very long content
            content = ref["content"][:4000]
            if len(ref["content"]) > 4000:
                content += "\n... (content truncated)"
            context_parts.append(content)

    return "\n".join(context_parts)
