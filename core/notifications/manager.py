"""Notification Manager - WebSocket push notifications for Alfred.

Handles broadcasting events (agent completion, system alerts, etc.) to connected clients.
Includes Web Push (VAPID) support for background notifications.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Set

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
    LONG_PROCESSING = "long_processing"


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
        # Push subscription storage: endpoint_url -> subscription_info dict
        self._push_subscriptions: dict[str, dict] = {}

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

    async def send_long_processing(self, message: str, query_preview: str, elapsed_seconds: float):
        """Convenience method for long processing notification (WebSocket toast)."""
        await self.broadcast(Notification(
            type=NotificationType.LONG_PROCESSING,
            data={
                "message": message,
                "query_preview": query_preview[:200],
                "elapsed_seconds": elapsed_seconds,
            }
        ))

    def add_push_subscription(self, subscription_info: dict):
        """Store a Web Push subscription."""
        endpoint = subscription_info.get("endpoint", "")
        if endpoint:
            self._push_subscriptions[endpoint] = subscription_info
            logger.info(f"Push subscription added (total: {len(self._push_subscriptions)})")

    def remove_push_subscription(self, endpoint: str):
        """Remove a Web Push subscription."""
        removed = self._push_subscriptions.pop(endpoint, None)
        if removed:
            logger.info(f"Push subscription removed (total: {len(self._push_subscriptions)})")

    async def send_push_notification(self, title: str, body: str, url: str = "/"):
        """Send Web Push notification to all subscribed clients."""
        from config.settings import settings

        if not settings.vapid_private_key or not self._push_subscriptions:
            logger.debug("Push notifications skipped: no VAPID key or no subscriptions")
            return

        payload = json.dumps({
            "title": title,
            "body": body,
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Run in executor to avoid blocking the event loop (pywebpush is sync)
        loop = asyncio.get_event_loop()
        dead_endpoints = []

        for endpoint, sub_info in self._push_subscriptions.items():
            try:
                await loop.run_in_executor(None, self._send_single_push, sub_info, payload)
            except Exception as e:
                logger.warning(f"Push notification failed for {endpoint[:50]}...: {e}")
                # 410 Gone means subscription expired
                if "410" in str(e) or "404" in str(e):
                    dead_endpoints.append(endpoint)

        for ep in dead_endpoints:
            self._push_subscriptions.pop(ep, None)

        sent = len(self._push_subscriptions) - len(dead_endpoints)
        if sent > 0:
            logger.info(f"Push notification sent to {sent} subscribers")

    @staticmethod
    def _send_single_push(subscription_info: dict, payload: str):
        """Send a single push notification (blocking, run in executor)."""
        from pywebpush import webpush
        from config.settings import settings

        vapid_claims = {"sub": f"mailto:{settings.vapid_contact_email}"}

        webpush(
            subscription_info=subscription_info,
            data=payload,
            vapid_private_key=settings.vapid_private_key,
            vapid_claims=vapid_claims,
        )

    @property
    def connection_count(self) -> int:
        """Number of connected clients."""
        return len(self._all_connections)

    @property
    def push_subscription_count(self) -> int:
        """Number of push subscriptions."""
        return len(self._push_subscriptions)


# Global singleton
_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance."""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager
