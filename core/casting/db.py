# core/casting/db.py
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any
from config.settings import settings

def _db_path() -> Path:
    return Path(settings.casting_db_path)

def _conn() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c

def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS dj (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'host',
                status TEXT NOT NULL DEFAULT 'draft',
                persona_prompt TEXT NOT NULL DEFAULT '',
                archetype_tags TEXT NOT NULL DEFAULT '[]',
                expertise TEXT NOT NULL DEFAULT '',
                voice_source TEXT NOT NULL DEFAULT 'recorded',
                moods_present TEXT NOT NULL DEFAULT '[]',
                avatar TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS assignment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dj_id INTEGER NOT NULL,
                station_id INTEGER NOT NULL,
                slot TEXT NOT NULL,
                effective_at TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

def _row_to_dj(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["archetype_tags"] = json.loads(d.get("archetype_tags") or "[]")
    d["moods_present"] = json.loads(d.get("moods_present") or "[]")
    return d

def create_dj(*, name: str, role: str, persona_prompt: str, archetype_tags: list[str],
              expertise: str, voice_source: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO dj (name, role, persona_prompt, archetype_tags, expertise, voice_source) "
            "VALUES (?,?,?,?,?,?)",
            (name, role, persona_prompt, json.dumps(archetype_tags), expertise, voice_source),
        )
        return int(cur.lastrowid)

def get_dj(dj_id: int) -> dict[str, Any] | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM dj WHERE id=?", (dj_id,)).fetchone()
        return _row_to_dj(r) if r else None

def list_djs() -> list[dict[str, Any]]:
    with _conn() as c:
        return [_row_to_dj(r) for r in c.execute("SELECT * FROM dj ORDER BY id DESC")]

def update_dj(dj_id: int, *, persona_prompt: str | None = None, archetype_tags: list[str] | None = None,
              expertise: str | None = None, name: str | None = None) -> None:
    sets, vals = [], []
    if persona_prompt is not None: sets.append("persona_prompt=?"); vals.append(persona_prompt)
    if archetype_tags is not None: sets.append("archetype_tags=?"); vals.append(json.dumps(archetype_tags))
    if expertise is not None: sets.append("expertise=?"); vals.append(expertise)
    if name is not None: sets.append("name=?"); vals.append(name)
    if not sets: return
    vals.append(dj_id)
    with _conn() as c:
        c.execute(f"UPDATE dj SET {', '.join(sets)} WHERE id=?", vals)

def set_status(dj_id: int, status: str) -> None:
    with _conn() as c:
        c.execute("UPDATE dj SET status=? WHERE id=?", (status, dj_id))

def set_mood_present(dj_id: int, mood: str) -> None:
    with _conn() as c:
        r = c.execute("SELECT moods_present FROM dj WHERE id=?", (dj_id,)).fetchone()
        moods = json.loads(r["moods_present"]) if r else []
        if mood not in moods:
            moods.append(mood)
        c.execute("UPDATE dj SET moods_present=? WHERE id=?", (json.dumps(moods), dj_id))

def create_assignment(*, dj_id: int, station_id: int, slot: str, effective_at: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO assignment (dj_id, station_id, slot, effective_at) VALUES (?,?,?,?)",
            (dj_id, station_id, slot, effective_at),
        )
        return int(cur.lastrowid)

def list_assignments(station_id: int | None = None) -> list[dict[str, Any]]:
    q = ("SELECT a.*, d.name AS dj_name FROM assignment a JOIN dj d ON d.id=a.dj_id")
    args: tuple = ()
    if station_id is not None:
        q += " WHERE a.station_id=?"; args = (station_id,)
    q += " ORDER BY a.effective_at"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args)]

def due_assignments(now_iso: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT a.*, d.name AS dj_name FROM assignment a JOIN dj d ON d.id=a.dj_id "
            "WHERE a.applied=0 AND a.effective_at<=? ORDER BY a.effective_at",
            (now_iso,),
        ).fetchall()
        return [dict(r) for r in rows]

def mark_applied(assignment_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE assignment SET applied=1 WHERE id=?", (assignment_id,))
