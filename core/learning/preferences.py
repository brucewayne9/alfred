"""Preference learning engine - tracks and learns user preferences over time.

This module:
- Tracks explicit preferences (user says "I prefer X")
- Learns implicit preferences (user always does X)
- Provides preference context for responses
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("/home/aialfred/alfred/data/learning.db")


def _get_db():
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_preferences_db():
    """Initialize the preferences tables."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,  -- 'communication', 'scheduling', 'tools', 'format', etc.
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,  -- 0-1, higher = more certain
            source TEXT DEFAULT 'explicit',  -- 'explicit', 'inferred', 'default'
            evidence_count INTEGER DEFAULT 1,
            last_updated TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(category, key)
        );

        CREATE TABLE IF NOT EXISTS preference_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preference_id INTEGER,
            old_value TEXT,
            new_value TEXT,
            reason TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (preference_id) REFERENCES preferences(id)
        );

        CREATE TABLE IF NOT EXISTS behavioral_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_type TEXT NOT NULL,  -- 'time_preference', 'tool_usage', 'response_style'
            signal_data TEXT NOT NULL,  -- JSON
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pref_category ON preferences(category);
        CREATE INDEX IF NOT EXISTS idx_pref_key ON preferences(key);
        CREATE INDEX IF NOT EXISTS idx_signals_type ON behavioral_signals(signal_type);
    """)
    conn.commit()
    conn.close()


# Initialize on import
init_preferences_db()


class PreferenceEngine:
    """Learns and manages user preferences."""

    # Default preferences
    DEFAULTS = {
        "communication": {
            "tone": "professional but friendly",
            "verbosity": "concise",
            "use_emojis": "no",
        },
        "scheduling": {
            "preferred_meeting_time": "morning",
            "timezone": "America/New_York",
            "buffer_between_meetings": "15 minutes",
        },
        "notifications": {
            "email_summary_frequency": "daily",
            "urgent_alert_channels": "all",
        },
        "tools": {
            "default_calendar": "google",
            "default_email": "gmail",
            "default_notes": "nextcloud",
        },
    }

    def __init__(self):
        self.db_path = DB_PATH
        self._ensure_defaults()

    def _ensure_defaults(self):
        """Ensure default preferences exist."""
        for category, prefs in self.DEFAULTS.items():
            for key, value in prefs.items():
                self.set(category, key, value, source="default", confidence=0.3)

    def set(
        self,
        category: str,
        key: str,
        value: Any,
        source: str = "explicit",
        confidence: float = 1.0,
        reason: str = "",
    ) -> bool:
        """Set or update a preference."""
        conn = _get_db()

        # Check existing
        existing = conn.execute(
            "SELECT id, value, confidence FROM preferences WHERE category = ? AND key = ?",
            (category, key)
        ).fetchone()

        value_str = json.dumps(value) if not isinstance(value, str) else value
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            # Only update if new confidence is higher or it's explicit
            if source == "explicit" or confidence > existing["confidence"]:
                # Record history
                conn.execute("""
                    INSERT INTO preference_history (preference_id, old_value, new_value, reason, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (existing["id"], existing["value"], value_str, reason, now))

                # Update preference
                conn.execute("""
                    UPDATE preferences
                    SET value = ?, confidence = ?, source = ?, evidence_count = evidence_count + 1, last_updated = ?
                    WHERE id = ?
                """, (value_str, confidence, source, now, existing["id"]))

                logger.info(f"Updated preference {category}.{key} = {value_str}")
        else:
            conn.execute("""
                INSERT INTO preferences (category, key, value, confidence, source, last_updated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (category, key, value_str, confidence, source, now, now))
            logger.info(f"Created preference {category}.{key} = {value_str}")

        conn.commit()
        conn.close()
        return True

    def get(self, category: str, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        conn = _get_db()
        row = conn.execute(
            "SELECT value FROM preferences WHERE category = ? AND key = ?",
            (category, key)
        ).fetchone()
        conn.close()

        if row:
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]
        return default

    def get_category(self, category: str) -> dict:
        """Get all preferences in a category."""
        conn = _get_db()
        rows = conn.execute(
            "SELECT key, value, confidence, source FROM preferences WHERE category = ?",
            (category,)
        ).fetchall()
        conn.close()

        result = {}
        for row in rows:
            try:
                result[row["key"]] = {
                    "value": json.loads(row["value"]) if row["value"].startswith(('[', '{', '"')) else row["value"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                }
            except json.JSONDecodeError:
                result[row["key"]] = {
                    "value": row["value"],
                    "confidence": row["confidence"],
                    "source": row["source"],
                }
        return result

    def get_all(self) -> dict:
        """Get all preferences organized by category."""
        conn = _get_db()
        rows = conn.execute(
            "SELECT category, key, value, confidence, source FROM preferences ORDER BY category, key"
        ).fetchall()
        conn.close()

        result = {}
        for row in rows:
            if row["category"] not in result:
                result[row["category"]] = {}
            try:
                value = json.loads(row["value"]) if row["value"].startswith(('[', '{', '"')) else row["value"]
            except json.JSONDecodeError:
                value = row["value"]
            result[row["category"]][row["key"]] = value

        return result

    def record_signal(self, signal_type: str, signal_data: dict):
        """Record a behavioral signal for later analysis."""
        conn = _get_db()
        conn.execute("""
            INSERT INTO behavioral_signals (signal_type, signal_data, created_at)
            VALUES (?, ?, ?)
        """, (signal_type, json.dumps(signal_data), datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()

    def analyze_signals(self, signal_type: str, limit: int = 100) -> dict:
        """Analyze signals to infer preferences."""
        conn = _get_db()
        rows = conn.execute("""
            SELECT signal_data FROM behavioral_signals
            WHERE signal_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (signal_type, limit)).fetchall()
        conn.close()

        if not rows:
            return {}

        signals = [json.loads(row["signal_data"]) for row in rows]

        # Simple frequency analysis
        if signal_type == "time_preference":
            hours = [s.get("hour", 12) for s in signals if "hour" in s]
            if hours:
                avg_hour = sum(hours) / len(hours)
                if avg_hour < 12:
                    return {"inferred": "morning", "confidence": 0.6}
                elif avg_hour < 17:
                    return {"inferred": "afternoon", "confidence": 0.6}
                else:
                    return {"inferred": "evening", "confidence": 0.6}

        elif signal_type == "tool_usage":
            tool_counts = {}
            for s in signals:
                tool = s.get("tool")
                if tool:
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
            if tool_counts:
                most_used = max(tool_counts, key=tool_counts.get)
                return {"most_used": most_used, "counts": tool_counts}

        return {"signals_analyzed": len(signals)}

    def get_context_prompt(self) -> str:
        """Generate a context string for the AI about user preferences."""
        prefs = self.get_all()

        lines = ["User preferences:"]
        for category, items in prefs.items():
            for key, value in items.items():
                lines.append(f"- {category}.{key}: {value}")

        return "\n".join(lines)

    def delete(self, category: str, key: str) -> bool:
        """Delete a preference."""
        conn = _get_db()
        cursor = conn.execute(
            "DELETE FROM preferences WHERE category = ? AND key = ?",
            (category, key)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted


# Global instance
_engine = PreferenceEngine()


def update_preference(
    category: str,
    key: str,
    value: Any,
    source: str = "explicit",
    confidence: float = 1.0,
) -> bool:
    """Update a preference (convenience function)."""
    return _engine.set(category, key, value, source, confidence)


def get_preference(category: str, key: str, default: Any = None) -> Any:
    """Get a preference (convenience function)."""
    return _engine.get(category, key, default)


def get_preferences(category: str = None) -> dict:
    """Get preferences (convenience function)."""
    if category:
        return _engine.get_category(category)
    return _engine.get_all()


def get_preference_context() -> str:
    """Get preference context for AI (convenience function)."""
    return _engine.get_context_prompt()


def record_behavioral_signal(signal_type: str, signal_data: dict):
    """Record a behavioral signal (convenience function)."""
    _engine.record_signal(signal_type, signal_data)
