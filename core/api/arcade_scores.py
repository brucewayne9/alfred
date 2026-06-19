"""W. Pharr Rd. Arcade — server-backed leaderboards."""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent.parent.parent / "data" / "arcade_scores.db"

VALID_GAMES = {
    "frogger", "snake", "pong", "breakout", "invaders", "pacman", "tetris", "asteroids",
    "galaga", "missile", "tron", "dk",
}

# light family-safe filter: a small blocklist of obvious slurs/curses we don't want
# kids leaving on the leaderboard. Not exhaustive — just covers the most common.
BAD_WORDS = {
    "fuck","shit","cunt","bitch","ass","dick","piss","fag","faggot","nigger","nigga","retard","whore","slut","tit","tits",
    "asshole","bastard","damn","hell","crap",
}
NAME_RE = re.compile(r"^[A-Za-z0-9 _'\-\.\!]+$")
SCORE_CAP = 99_999_999

_SALT = os.environ.get("ARCADE_HASH_SALT", "w_pharr_rd_arcade_2026")


def _ip_hash(ip: str) -> str:
    return hashlib.sha256((_SALT + (ip or "0")).encode()).hexdigest()[:16]


class _AutoCloseConn(sqlite3.Connection):
    """A Connection whose ``with`` block also CLOSES the connection on exit.

    Stock ``with sqlite3.connect(...) as c:`` only commits/rolls back the
    transaction — it never closes, leaking a file descriptor per call. Every
    helper here uses ``with _conn() as c:``, so closing on __exit__ plugs the
    leak for all call sites with no churn. (Mirrors core/forge/db.py.)
    """

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            super().__exit__(exc_type, exc_val, exc_tb)  # commit / rollback
        finally:
            self.close()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), factory=_AutoCloseConn)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def _init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS arcade_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game TEXT NOT NULL,
                name TEXT NOT NULL,
                aka  TEXT NOT NULL DEFAULT '',
                score INTEGER NOT NULL,
                ip_hash TEXT,
                ts INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_arcade_game_score ON arcade_scores(game, score DESC);
            CREATE INDEX IF NOT EXISTS idx_arcade_ts ON arcade_scores(ts DESC);
            """
        )


def _clean(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s[:max_len]
    return s


def _is_clean(s: str) -> bool:
    if not s:
        return True
    low = s.lower()
    for w in BAD_WORDS:
        if w in low:
            return False
    return True


class ScoreSubmit(BaseModel):
    game: str
    name: str = Field(..., min_length=1, max_length=15)
    aka: str = Field(default="", max_length=20)
    score: int = Field(..., ge=0, le=SCORE_CAP)


def register(app: FastAPI) -> None:
    """Register arcade leaderboard endpoints on the given FastAPI app."""
    _init_db()

    @app.get("/api/arcade/top/{game}")
    def top_scores(game: str, limit: int = 25):
        if game not in VALID_GAMES:
            raise HTTPException(400, "unknown game")
        limit = max(1, min(int(limit), 100))
        with _conn() as c:
            rows = c.execute(
                "SELECT name, aka, score, ts FROM arcade_scores WHERE game = ? ORDER BY score DESC, ts ASC LIMIT ?",
                (game, limit),
            ).fetchall()
        return {
            "game": game,
            "scores": [
                {"rank": i + 1, "name": r["name"], "aka": r["aka"], "score": r["score"], "ts": r["ts"]}
                for i, r in enumerate(rows)
            ],
        }

    @app.get("/api/arcade/topall")
    def top_per_game():
        """Top score per game — used by the lobby to show 'TOP: NAME · 1234'."""
        with _conn() as c:
            out = {}
            for game in sorted(VALID_GAMES):
                row = c.execute(
                    "SELECT name, aka, score, ts FROM arcade_scores WHERE game = ? ORDER BY score DESC, ts ASC LIMIT 1",
                    (game,),
                ).fetchone()
                out[game] = (
                    {"name": row["name"], "aka": row["aka"], "score": row["score"], "ts": row["ts"]}
                    if row
                    else None
                )
        return out

    @app.get("/api/arcade/recent")
    def recent_submissions(limit: int = 20):
        limit = max(1, min(int(limit), 100))
        with _conn() as c:
            rows = c.execute(
                "SELECT game, name, aka, score, ts FROM arcade_scores ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "recent": [
                {"game": r["game"], "name": r["name"], "aka": r["aka"], "score": r["score"], "ts": r["ts"]}
                for r in rows
            ]
        }

    @app.get("/api/arcade/stats")
    def stats():
        with _conn() as c:
            total = c.execute("SELECT COUNT(*) AS n FROM arcade_scores").fetchone()["n"]
            uniq = c.execute("SELECT COUNT(DISTINCT name) AS n FROM arcade_scores").fetchone()["n"]
            per = {}
            for game in sorted(VALID_GAMES):
                per[game] = c.execute(
                    "SELECT COUNT(*) AS n FROM arcade_scores WHERE game = ?", (game,)
                ).fetchone()["n"]
        return {"total_scores": total, "unique_players": uniq, "per_game": per}

    @app.post("/api/arcade/submit")
    def submit_score(payload: ScoreSubmit, request: Request):
        if payload.game not in VALID_GAMES:
            raise HTTPException(400, "unknown game")
        name = _clean(payload.name, 15)
        aka = _clean(payload.aka, 20)
        if not name:
            raise HTTPException(400, "name required")
        if not NAME_RE.match(name):
            raise HTTPException(400, "name contains invalid characters")
        if aka and not NAME_RE.match(aka):
            raise HTTPException(400, "aka contains invalid characters")
        if not _is_clean(name) or not _is_clean(aka):
            raise HTTPException(400, "keep it family friendly please")
        if payload.score < 0 or payload.score > SCORE_CAP:
            raise HTTPException(400, "invalid score")

        ip = request.client.host if request.client else ""
        iph = _ip_hash(ip)
        now = int(time.time())

        # very light rate limit: 1 submit per IP per 5 seconds
        with _conn() as c:
            recent = c.execute(
                "SELECT ts FROM arcade_scores WHERE ip_hash = ? ORDER BY ts DESC LIMIT 1",
                (iph,),
            ).fetchone()
            if recent and now - recent["ts"] < 5:
                raise HTTPException(429, "slow down — try again in a few seconds")

            c.execute(
                "INSERT INTO arcade_scores (game, name, aka, score, ip_hash, ts) VALUES (?, ?, ?, ?, ?, ?)",
                (payload.game, name, aka, int(payload.score), iph, now),
            )
            new_id = c.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

            # compute rank in this game
            rank = c.execute(
                "SELECT COUNT(*) AS n FROM arcade_scores WHERE game = ? AND (score > ? OR (score = ? AND ts < ?))",
                (payload.game, payload.score, payload.score, now),
            ).fetchone()["n"] + 1

            top10 = c.execute(
                "SELECT name, aka, score FROM arcade_scores WHERE game = ? ORDER BY score DESC, ts ASC LIMIT 10",
                (payload.game,),
            ).fetchall()

        return {
            "id": new_id,
            "rank": rank,
            "is_top10": rank <= 10,
            "is_first": rank == 1,
            "top10": [
                {"name": r["name"], "aka": r["aka"], "score": r["score"]}
                for r in top10
            ],
        }
