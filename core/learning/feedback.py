"""Feedback tracking system - captures corrections and ratings to improve Alfred.

This module tracks:
- Explicit feedback (thumbs up/down, corrections)
- Implicit feedback (task completion, retries)
- Correction patterns (what Alfred gets wrong)
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DB_PATH = Path("/home/aialfred/alfred/data/learning.db")


def _get_db():
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db():
    """Initialize the feedback database."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            message_id TEXT,
            feedback_type TEXT NOT NULL,  -- 'positive', 'negative', 'correction'
            original_response TEXT,
            correction TEXT,
            context TEXT,
            category TEXT,  -- 'tool_use', 'response_quality', 'accuracy', 'tone'
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            user_query TEXT NOT NULL,
            alfred_response TEXT,
            tools_used TEXT,  -- JSON array
            response_time_ms INTEGER,
            was_successful INTEGER DEFAULT 1,
            retry_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,  -- What Alfred got wrong
            correction TEXT NOT NULL,  -- How to fix it
            frequency INTEGER DEFAULT 1,
            last_seen TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
        CREATE INDEX IF NOT EXISTS idx_feedback_category ON feedback(category);
        CREATE INDEX IF NOT EXISTS idx_interactions_conv ON interactions(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_corrections_pattern ON corrections(pattern);
    """)
    conn.commit()
    conn.close()
    logger.info("Feedback database initialized")


# Initialize on import
init_feedback_db()


@dataclass
class Feedback:
    """A feedback entry."""
    feedback_type: str  # positive, negative, correction
    original_response: str = ""
    correction: str = ""
    context: str = ""
    category: str = ""
    conversation_id: str = ""
    message_id: str = ""


class FeedbackTracker:
    """Tracks and analyzes feedback to improve Alfred."""

    def __init__(self):
        self.db_path = DB_PATH

    def record(self, feedback: Feedback) -> int:
        """Record a feedback entry."""
        conn = _get_db()
        cursor = conn.execute("""
            INSERT INTO feedback
            (conversation_id, message_id, feedback_type, original_response,
             correction, context, category, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.conversation_id,
            feedback.message_id,
            feedback.feedback_type,
            feedback.original_response,
            feedback.correction,
            feedback.context,
            feedback.category,
            datetime.now(timezone.utc).isoformat(),
        ))
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # If it's a correction, also update the corrections table
        if feedback.feedback_type == "correction" and feedback.correction:
            self._record_correction(feedback.original_response, feedback.correction)

        logger.info(f"Recorded {feedback.feedback_type} feedback #{feedback_id}")
        return feedback_id

    def _record_correction(self, pattern: str, correction: str):
        """Record or update a correction pattern."""
        conn = _get_db()

        # Check if pattern exists
        existing = conn.execute(
            "SELECT id, frequency FROM corrections WHERE pattern = ?",
            (pattern[:500],)  # Truncate for indexing
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE corrections
                SET frequency = frequency + 1, last_seen = ?, correction = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), correction, existing["id"]))
        else:
            conn.execute("""
                INSERT INTO corrections (pattern, correction, last_seen, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                pattern[:500],
                correction,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ))

        conn.commit()
        conn.close()

    def record_interaction(
        self,
        user_query: str,
        alfred_response: str,
        tools_used: list[str] = None,
        response_time_ms: int = 0,
        was_successful: bool = True,
        conversation_id: str = "",
    ) -> int:
        """Record an interaction for analysis."""
        conn = _get_db()
        cursor = conn.execute("""
            INSERT INTO interactions
            (conversation_id, user_query, alfred_response, tools_used,
             response_time_ms, was_successful, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            conversation_id,
            user_query,
            alfred_response,
            json.dumps(tools_used or []),
            response_time_ms,
            1 if was_successful else 0,
            datetime.now(timezone.utc).isoformat(),
        ))
        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return interaction_id

    def get_stats(self) -> dict:
        """Get feedback statistics."""
        conn = _get_db()

        # Overall counts
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        positive = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'positive'"
        ).fetchone()[0]
        negative = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'negative'"
        ).fetchone()[0]
        corrections = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE feedback_type = 'correction'"
        ).fetchone()[0]

        # Category breakdown
        categories = conn.execute("""
            SELECT category, COUNT(*) as count
            FROM feedback
            WHERE category != ''
            GROUP BY category
        """).fetchall()

        # Recent corrections
        recent_corrections = conn.execute("""
            SELECT pattern, correction, frequency
            FROM corrections
            ORDER BY last_seen DESC
            LIMIT 10
        """).fetchall()

        # Interaction stats
        total_interactions = conn.execute(
            "SELECT COUNT(*) FROM interactions"
        ).fetchone()[0]
        successful = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE was_successful = 1"
        ).fetchone()[0]

        conn.close()

        return {
            "feedback": {
                "total": total,
                "positive": positive,
                "negative": negative,
                "corrections": corrections,
                "satisfaction_rate": positive / total if total > 0 else 0,
            },
            "categories": {row["category"]: row["count"] for row in categories},
            "recent_corrections": [
                {"pattern": r["pattern"][:100], "correction": r["correction"][:100], "frequency": r["frequency"]}
                for r in recent_corrections
            ],
            "interactions": {
                "total": total_interactions,
                "successful": successful,
                "success_rate": successful / total_interactions if total_interactions > 0 else 0,
            },
        }

    def get_corrections_for_context(self, context: str) -> list[dict]:
        """Get relevant corrections for a given context."""
        conn = _get_db()

        # Simple keyword matching - could be enhanced with embeddings
        words = context.lower().split()[:10]
        placeholders = " OR ".join(["pattern LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]

        if not params:
            return []

        corrections = conn.execute(f"""
            SELECT pattern, correction, frequency
            FROM corrections
            WHERE {placeholders}
            ORDER BY frequency DESC
            LIMIT 5
        """, params).fetchall()

        conn.close()
        return [dict(row) for row in corrections]


# Global instance
_tracker = FeedbackTracker()


def record_feedback(
    feedback_type: str,
    original_response: str = "",
    correction: str = "",
    context: str = "",
    category: str = "",
    conversation_id: str = "",
    message_id: str = "",
) -> int:
    """Record feedback (convenience function)."""
    return _tracker.record(Feedback(
        feedback_type=feedback_type,
        original_response=original_response,
        correction=correction,
        context=context,
        category=category,
        conversation_id=conversation_id,
        message_id=message_id,
    ))


def record_interaction(
    user_query: str,
    alfred_response: str,
    tools_used: list[str] = None,
    response_time_ms: int = 0,
    was_successful: bool = True,
    conversation_id: str = "",
) -> int:
    """Record an interaction (convenience function)."""
    return _tracker.record_interaction(
        user_query=user_query,
        alfred_response=alfred_response,
        tools_used=tools_used,
        response_time_ms=response_time_ms,
        was_successful=was_successful,
        conversation_id=conversation_id,
    )


def get_feedback_stats() -> dict:
    """Get feedback statistics (convenience function)."""
    return _tracker.get_stats()


def get_corrections_for_context(context: str) -> list[dict]:
    """Get relevant corrections (convenience function)."""
    return _tracker.get_corrections_for_context(context)
