"""Notification system for Alfred - WebSocket push notifications."""

from .manager import (
    NotificationManager,
    NotificationType,
    Notification,
    get_notification_manager,
)
from .watcher import LongProcessingWatcher

__all__ = [
    "NotificationManager",
    "NotificationType",
    "Notification",
    "get_notification_manager",
    "LongProcessingWatcher",
]
