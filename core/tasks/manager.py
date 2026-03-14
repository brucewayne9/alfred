"""Task manager — SQLite-backed daily task tracking for Mike's dashboard."""

import json
import sqlite3
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "tasks.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    original_date TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    sort_order INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tasks_date ON tasks(original_date);
CREATE INDEX IF NOT EXISTS idx_tasks_completed ON tasks(completed);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with _conn() as conn:
        conn.executescript(_CREATE_SQL)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["completed"] = bool(d["completed"])
    # Calculate days outstanding if not completed
    if not d["completed"]:
        orig = datetime.strptime(d["original_date"], "%Y-%m-%d").date()
        delta = (date.today() - orig).days
        d["days_outstanding"] = delta
    else:
        d["days_outstanding"] = 0
    return d


# ── CRUD ────────────────────────────────────────────────

def add_task(text: str, target_date: str | None = None, source: str = "manual") -> dict:
    """Add a new task. target_date defaults to today."""
    init_db()
    task_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    target = target_date or date.today().isoformat()
    # Get next sort order for the date
    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order FROM tasks WHERE original_date = ?",
            (target,),
        ).fetchone()
        order = row["next_order"] if row else 0
        conn.execute(
            "INSERT INTO tasks (id, text, completed, created_at, original_date, source, sort_order) VALUES (?, ?, 0, ?, ?, ?, ?)",
            (task_id, text, now, target, source, order),
        )
    return get_task(task_id)


def get_task(task_id: str) -> dict | None:
    init_db()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(row) if row else None


def toggle_task(task_id: str) -> dict | None:
    """Toggle completed status."""
    init_db()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return None
        new_status = 0 if row["completed"] else 1
        completed_at = datetime.now().isoformat() if new_status else None
        conn.execute(
            "UPDATE tasks SET completed = ?, completed_at = ? WHERE id = ?",
            (new_status, completed_at, task_id),
        )
    return get_task(task_id)


def update_task(task_id: str, text: str) -> dict | None:
    init_db()
    with _conn() as conn:
        conn.execute("UPDATE tasks SET text = ? WHERE id = ?", (text, task_id))
    return get_task(task_id)


def delete_task(task_id: str) -> bool:
    init_db()
    with _conn() as conn:
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    return cursor.rowcount > 0


def reorder_tasks(task_ids: list[str]) -> bool:
    """Set sort_order based on position in list."""
    init_db()
    with _conn() as conn:
        for i, tid in enumerate(task_ids):
            conn.execute("UPDATE tasks SET sort_order = ? WHERE id = ?", (i, tid))
    return True


# ── Queries ─────────────────────────────────────────────

def get_tasks_for_date(target_date: str | None = None) -> list[dict]:
    """Get tasks for a specific date (defaults to today)."""
    init_db()
    target = target_date or date.today().isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE original_date = ? ORDER BY completed ASC, sort_order ASC",
            (target,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_outstanding_tasks() -> list[dict]:
    """Get all incomplete tasks from before today."""
    init_db()
    today = date.today().isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE completed = 0 AND original_date < ? ORDER BY original_date ASC, sort_order ASC",
            (today,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_dashboard() -> dict:
    """Get full dashboard view: today's tasks, outstanding, and stats."""
    init_db()
    today = date.today().isoformat()
    today_tasks = get_tasks_for_date(today)
    outstanding = get_outstanding_tasks()

    # Stats
    with _conn() as conn:
        total_today = conn.execute(
            "SELECT COUNT(*) as c FROM tasks WHERE original_date = ?", (today,)
        ).fetchone()["c"]
        done_today = conn.execute(
            "SELECT COUNT(*) as c FROM tasks WHERE original_date = ? AND completed = 1", (today,)
        ).fetchone()["c"]
        # Completed in last 7 days
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        done_week = conn.execute(
            "SELECT COUNT(*) as c FROM tasks WHERE completed = 1 AND completed_at >= ?", (week_ago,)
        ).fetchone()["c"]

    return {
        "date": today,
        "today": today_tasks,
        "outstanding": outstanding,
        "stats": {
            "total_today": total_today,
            "done_today": done_today,
            "outstanding_count": len(outstanding),
            "done_this_week": done_week,
        },
    }


def rollover_tasks(from_date: str, to_date: str | None = None) -> int:
    """Copy incomplete tasks from a past date to a new date (today by default)."""
    init_db()
    target = to_date or date.today().isoformat()
    outstanding = []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks WHERE original_date = ? AND completed = 0",
            (from_date,),
        ).fetchall()
        outstanding = [dict(r) for r in rows]

    count = 0
    for task in outstanding:
        add_task(task["text"], target_date=target, source="rollover")
        count += 1
    return count


def bulk_add_tasks(texts: list[str], target_date: str | None = None, source: str = "morning_brief") -> list[dict]:
    """Add multiple tasks at once (used by morning brief / Claw)."""
    results = []
    for text in texts:
        if text.strip():
            results.append(add_task(text.strip(), target_date=target_date, source=source))
    return results
