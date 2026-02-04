"""Notification Manager - WebSocket push notifications for Alfred.

Handles broadcasting events (agent completion, system alerts, etc.) to connected clients.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Set
from weakref import WeakSet

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications."""
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_CANCELLED = "agent_cancelled"
    SYSTEM_ALERT = "system_alert"
    TASK_UPDATE = "task_update"


@dataclass
class Notification:
    """A notification to be broadcast to clients."""
    type: NotificationType
    data: dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        })


class NotificationManager:
    """Manages WebSocket connections and broadcasts notifications.

    Usage:
        manager = get_notification_manager()

        # In WebSocket endpoint:
        await manager.connect(websocket, user_id)

        # Broadcast notification:
        await manager.broadcast(Notification(
            type=NotificationType.AGENT_COMPLETED,
            data={"task_id": "abc123", "result": "..."}
        ))
    """

    def __init__(self):
        self._connections: dict[str, Set[WebSocket]] = {}  # user_id -> websockets
        self._all_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: str = "anonymous"):
        """Register a new WebSocket connection."""
        async with self._lock:
            if user_id not in self._connections:
                self._connections[user_id] = set()
            self._connections[user_id].add(websocket)
            self._all_connections.add(websocket)

        logger.info(f"Notification client connected: {user_id} (total: {len(self._all_connections)})")

    async def disconnect(self, websocket: WebSocket, user_id: str = "anonymous"):
        """Unregister a WebSocket connection."""
        async with self._lock:
            if user_id in self._connections:
                self._connections[user_id].discard(websocket)
                if not self._connections[user_id]:
                    del self._connections[user_id]
            self._all_connections.discard(websocket)

        logger.info(f"Notification client disconnected: {user_id} (total: {len(self._all_connections)})")

    async def broadcast(self, notification: Notification, user_id: str = None):
        """Broadcast a notification to connected clients.

        Args:
            notification: The notification to send
            user_id: If provided, only send to this user's connections
        """
        message = notification.to_json()

        if user_id:
            # Send to specific user
            connections = self._connections.get(user_id, set())
        else:
            # Broadcast to all
            connections = self._all_connections.copy()

        if not connections:
            logger.debug(f"No clients to receive notification: {notification.type.value}")
            return

        # Send to all connections, removing dead ones
        dead_connections = []
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            for uid, conns in self._connections.items():
                conns.discard(ws)
            self._all_connections.discard(ws)

        logger.info(f"Broadcast {notification.type.value} to {len(connections) - len(dead_connections)} clients")

    async def send_agent_started(self, task_id: str, agent_type: str, goal: str):
        """Convenience method for agent started notification."""
        await self.broadcast(Notification(
            type=NotificationType.AGENT_STARTED,
            data={
                "task_id": task_id,
                "agent_type": agent_type,
                "goal": goal[:200],
            }
        ))

    async def send_agent_completed(self, task_id: str, agent_type: str, result: Any, duration_seconds: float = None):
        """Convenience method for agent completed notification."""
        # Truncate result for notification (full result available via API)
        result_preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)

        await self.broadcast(Notification(
            type=NotificationType.AGENT_COMPLETED,
            data={
                "task_id": task_id,
                "agent_type": agent_type,
                "result_preview": result_preview,
                "duration_seconds": duration_seconds,
            }
        ))

    async def send_agent_failed(self, task_id: str, agent_type: str, error: str):
        """Convenience method for agent failed notification."""
        await self.broadcast(Notification(
            type=NotificationType.AGENT_FAILED,
            data={
                "task_id": task_id,
                "agent_type": agent_type,
                "error": error,
            }
        ))

    @property
    def connection_count(self) -> int:
        """Number of connected clients."""
        return len(self._all_connections)


# Global singleton
_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance."""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager
