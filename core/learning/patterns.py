"""Pattern detection - identifies recurring workflows and suggests automations.

This module:
- Tracks sequences of actions
- Identifies repeated patterns
- Suggests workflow automations
"""

import json
import logging
import sqlite3
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("/home/aialfred/alfred/data/learning.db")


def _get_db():
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_patterns_db():
    """Initialize the patterns tables."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS action_sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            action_type TEXT NOT NULL,  -- 'tool_call', 'query', 'command'
            action_name TEXT NOT NULL,
            action_data TEXT,  -- JSON
            sequence_order INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detected_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_name TEXT NOT NULL,
            pattern_sequence TEXT NOT NULL,  -- JSON array of actions
            frequency INTEGER DEFAULT 1,
            avg_time_between_actions INTEGER,  -- seconds
            suggested_automation TEXT,
            is_active INTEGER DEFAULT 1,
            last_seen TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workflow_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER,
            suggestion TEXT NOT NULL,
            status TEXT DEFAULT 'pending',  -- 'pending', 'accepted', 'dismissed'
            created_at TEXT NOT NULL,
            FOREIGN KEY (pattern_id) REFERENCES detected_patterns(id)
        );

        CREATE INDEX IF NOT EXISTS idx_seq_session ON action_sequences(session_id);
        CREATE INDEX IF NOT EXISTS idx_seq_created ON action_sequences(created_at);
        CREATE INDEX IF NOT EXISTS idx_patterns_freq ON detected_patterns(frequency DESC);
    """)
    conn.commit()
    conn.close()


# Initialize on import
init_patterns_db()


class PatternDetector:
    """Detects patterns in user behavior and suggests automations."""

    # Minimum occurrences to consider something a pattern
    MIN_PATTERN_FREQUENCY = 3

    # Time window for sequence detection (minutes)
    SEQUENCE_WINDOW = 30

    def __init__(self):
        self.db_path = DB_PATH
        self._current_session = None

    def start_session(self, session_id: str = None):
        """Start a new tracking session."""
        self._current_session = session_id or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return self._current_session

    def record_action(
        self,
        action_type: str,
        action_name: str,
        action_data: dict = None,
        session_id: str = None,
    ):
        """Record an action in the sequence."""
        conn = _get_db()

        session = session_id or self._current_session or "default"

        # Get current sequence order for this session
        result = conn.execute("""
            SELECT MAX(sequence_order) as max_order
            FROM action_sequences
            WHERE session_id = ?
            AND created_at > datetime('now', '-30 minutes')
        """, (session,)).fetchone()

        order = (result["max_order"] or 0) + 1

        conn.execute("""
            INSERT INTO action_sequences
            (session_id, action_type, action_name, action_data, sequence_order, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session,
            action_type,
            action_name,
            json.dumps(action_data or {}),
            order,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()

        # Check for patterns after recording
        if order >= 3:
            self._check_for_patterns(session)

    def _check_for_patterns(self, session_id: str):
        """Check if recent actions form a known pattern."""
        conn = _get_db()

        # Get recent actions
        actions = conn.execute("""
            SELECT action_type, action_name
            FROM action_sequences
            WHERE session_id = ?
            AND created_at > datetime('now', '-30 minutes')
            ORDER BY sequence_order
        """, (session_id,)).fetchall()

        if len(actions) < 3:
            conn.close()
            return

        # Create sequence signature
        sequence = [f"{a['action_type']}:{a['action_name']}" for a in actions[-5:]]
        sequence_str = json.dumps(sequence)

        # Check if this pattern exists
        existing = conn.execute("""
            SELECT id, frequency FROM detected_patterns
            WHERE pattern_sequence = ?
        """, (sequence_str,)).fetchone()

        now = datetime.now(timezone.utc).isoformat()

        if existing:
            conn.execute("""
                UPDATE detected_patterns
                SET frequency = frequency + 1, last_seen = ?
                WHERE id = ?
            """, (now, existing["id"]))

            # If pattern is frequent enough, create suggestion
            if existing["frequency"] + 1 >= self.MIN_PATTERN_FREQUENCY:
                self._create_suggestion(conn, existing["id"], sequence)
        else:
            # Create new pattern
            pattern_name = f"Pattern: {' -> '.join([a['action_name'] for a in actions[-3:]])}"
            conn.execute("""
                INSERT INTO detected_patterns
                (pattern_name, pattern_sequence, last_seen, created_at)
                VALUES (?, ?, ?, ?)
            """, (pattern_name, sequence_str, now, now))

        conn.commit()
        conn.close()

    def _create_suggestion(self, conn, pattern_id: int, sequence: list):
        """Create an automation suggestion for a pattern."""
        # Check if suggestion already exists
        existing = conn.execute("""
            SELECT id FROM workflow_suggestions
            WHERE pattern_id = ? AND status = 'pending'
        """, (pattern_id,)).fetchone()

        if existing:
            return

        # Generate suggestion text
        actions = [s.split(":")[-1] for s in sequence]
        suggestion = f"I noticed you frequently do these actions together: {' → '.join(actions)}. " \
                    f"Would you like me to create an automation for this workflow?"

        conn.execute("""
            INSERT INTO workflow_suggestions (pattern_id, suggestion, created_at)
            VALUES (?, ?, ?)
        """, (pattern_id, suggestion, datetime.now(timezone.utc).isoformat()))

    def get_pending_suggestions(self) -> list[dict]:
        """Get pending workflow suggestions."""
        conn = _get_db()
        rows = conn.execute("""
            SELECT ws.id, ws.suggestion, ws.created_at,
                   dp.pattern_name, dp.frequency
            FROM workflow_suggestions ws
            JOIN detected_patterns dp ON ws.pattern_id = dp.id
            WHERE ws.status = 'pending'
            ORDER BY dp.frequency DESC
            LIMIT 5
        """).fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def respond_to_suggestion(self, suggestion_id: int, accept: bool) -> bool:
        """Accept or dismiss a suggestion."""
        conn = _get_db()
        status = "accepted" if accept else "dismissed"
        cursor = conn.execute("""
            UPDATE workflow_suggestions
            SET status = ?
            WHERE id = ?
        """, (status, suggestion_id))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def get_frequent_patterns(self, limit: int = 10) -> list[dict]:
        """Get the most frequent patterns."""
        conn = _get_db()
        rows = conn.execute("""
            SELECT pattern_name, pattern_sequence, frequency, last_seen
            FROM detected_patterns
            WHERE is_active = 1
            ORDER BY frequency DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        patterns = []
        for row in rows:
            patterns.append({
                "name": row["pattern_name"],
                "sequence": json.loads(row["pattern_sequence"]),
                "frequency": row["frequency"],
                "last_seen": row["last_seen"],
            })
        return patterns

    def get_common_sequences(self, action_name: str) -> list[dict]:
        """Get common actions that follow a specific action."""
        conn = _get_db()

        # Find actions that commonly follow the given action
        rows = conn.execute("""
            SELECT a2.action_name, COUNT(*) as count
            FROM action_sequences a1
            JOIN action_sequences a2
                ON a1.session_id = a2.session_id
                AND a2.sequence_order = a1.sequence_order + 1
            WHERE a1.action_name = ?
            GROUP BY a2.action_name
            ORDER BY count DESC
            LIMIT 5
        """, (action_name,)).fetchall()
        conn.close()

        return [{"next_action": row["action_name"], "count": row["count"]} for row in rows]

    def get_stats(self) -> dict:
        """Get pattern detection statistics."""
        conn = _get_db()

        total_actions = conn.execute("SELECT COUNT(*) FROM action_sequences").fetchone()[0]
        total_patterns = conn.execute("SELECT COUNT(*) FROM detected_patterns").fetchone()[0]
        frequent_patterns = conn.execute(
            "SELECT COUNT(*) FROM detected_patterns WHERE frequency >= ?",
            (self.MIN_PATTERN_FREQUENCY,)
        ).fetchone()[0]
        pending_suggestions = conn.execute(
            "SELECT COUNT(*) FROM workflow_suggestions WHERE status = 'pending'"
        ).fetchone()[0]

        conn.close()

        return {
            "total_actions_tracked": total_actions,
            "patterns_detected": total_patterns,
            "frequent_patterns": frequent_patterns,
            "pending_suggestions": pending_suggestions,
        }


# Global instance
_detector = PatternDetector()


def record_action(
    action_type: str,
    action_name: str,
    action_data: dict = None,
    session_id: str = None,
):
    """Record an action (convenience function)."""
    _detector.record_action(action_type, action_name, action_data, session_id)


def detect_patterns() -> list[dict]:
    """Get detected patterns (convenience function)."""
    return _detector.get_frequent_patterns()


def get_workflow_suggestions() -> list[dict]:
    """Get workflow suggestions (convenience function)."""
    return _detector.get_pending_suggestions()


def respond_to_suggestion(suggestion_id: int, accept: bool) -> bool:
    """Respond to a suggestion (convenience function)."""
    return _detector.respond_to_suggestion(suggestion_id, accept)


def get_pattern_stats() -> dict:
    """Get pattern stats (convenience function)."""
    return _detector.get_stats()


def get_next_action_predictions(current_action: str) -> list[dict]:
    """Predict likely next actions (convenience function)."""
    return _detector.get_common_sequences(current_action)
