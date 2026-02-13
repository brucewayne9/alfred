"""Long Processing Watcher - Alerts users when Alfred is taking a while.

Spawns an asyncio background task that sleeps for a configurable threshold.
If the request completes before the threshold, the watcher is cancelled silently.
If not, it fires 3 notification channels: WebSocket toast, browser push, and email.
One alert per request maximum.
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class LongProcessingWatcher:
    """Watches for long-running requests and sends notifications.

    Usage:
        watcher = LongProcessingWatcher(query="create a spreadsheet...")
        watcher.start()
        try:
            result = await ask(query, ...)
        finally:
            watcher.cancel()
    """

    def __init__(self, query: str, threshold_seconds: int | None = None):
        from config.settings import settings

        self._query = query
        self._threshold = threshold_seconds or settings.long_processing_threshold_seconds
        self._task: asyncio.Task | None = None
        self._fired = False
        self._start_time = time.monotonic()

    def start(self):
        """Spawn the background watcher task."""
        self._start_time = time.monotonic()
        self._task = asyncio.create_task(self._watch())

    def cancel(self):
        """Cancel the watcher (call in finally block after ask() completes)."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.debug(f"Watcher cancelled after {time.monotonic() - self._start_time:.1f}s")

    @property
    def fired(self) -> bool:
        """Whether the watcher already sent notifications."""
        return self._fired

    async def _watch(self):
        """Sleep for threshold, then fire all notification channels."""
        try:
            await asyncio.sleep(self._threshold)
        except asyncio.CancelledError:
            return

        # If we get here, the threshold was exceeded
        self._fired = True
        elapsed = time.monotonic() - self._start_time
        query_preview = self._query[:100] + ("..." if len(self._query) > 100 else "")
        logger.info(f"Long processing detected ({elapsed:.0f}s): {query_preview}")

        # Fire all 3 channels concurrently, don't let one failure block others
        results = await asyncio.gather(
            self._send_websocket(query_preview, elapsed),
            self._send_push(query_preview),
            self._send_email(query_preview, elapsed),
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                channel = ["websocket", "push", "email"][i]
                logger.warning(f"Long processing {channel} notification failed: {result}")

    async def _send_websocket(self, query_preview: str, elapsed: float):
        """Send WebSocket toast notification via the notification manager."""
        from core.notifications import get_notification_manager
        manager = get_notification_manager()
        await manager.send_long_processing(
            message="Alfred is still working on your request. You'll be notified when it's done.",
            query_preview=query_preview,
            elapsed_seconds=elapsed,
        )

    async def _send_push(self, query_preview: str):
        """Send browser push notification via the notification manager."""
        from core.notifications import get_notification_manager
        manager = get_notification_manager()
        await manager.send_push_notification(
            title="Alfred is still working...",
            body=f"Your request is taking longer than usual: {query_preview}",
            url="/",
        )

    async def _send_email(self, query_preview: str, elapsed: float):
        """Send email alert via the existing EmailClient."""
        from config.settings import settings

        if not settings.long_processing_email_to:
            logger.debug("Email notification skipped: no recipient configured")
            return

        from integrations.email.client import email_client

        subject = "Alfred - Still processing your request"
        body = (
            f"Hi,\n\n"
            f"Alfred has been working on your request for over {int(elapsed)} seconds "
            f"and is still processing.\n\n"
            f"Request: {query_preview}\n\n"
            f"You'll receive a response in the chat when it's complete. "
            f"No action is needed on your part.\n\n"
            f"— Alfred"
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: email_client.send_email(
                account=settings.long_processing_email_from_account,
                to=settings.long_processing_email_to,
                subject=subject,
                body=body,
            ),
        )
        logger.info(f"Long processing email sent to {settings.long_processing_email_to}")
