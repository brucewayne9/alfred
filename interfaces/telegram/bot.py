"""Telegram bot for Alfred - chat with Alfred via Telegram.

This module provides a Telegram bot interface that connects to Alfred's brain,
allowing users to interact with Alfred through Telegram messages.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")  # Comma-separated user IDs


class TelegramBot:
    """Telegram bot interface for Alfred."""

    def __init__(self, token: str = None, allowed_users: list[str] = None):
        self.token = token or TELEGRAM_BOT_TOKEN
        self.allowed_users = allowed_users or [u.strip() for u in ALLOWED_USERS if u.strip()]
        self.application: Optional[Application] = None
        self._running = False

    def is_authorized(self, user_id: int) -> bool:
        """Check if a user is authorized to use the bot."""
        if not self.allowed_users or self.allowed_users == [""]:
            return True  # No restrictions if not configured
        return str(user_id) in self.allowed_users

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        if not self.is_authorized(user.id):
            await update.message.reply_text(
                "Sorry, you're not authorized to use this bot. "
                "Please contact the administrator."
            )
            return

        await update.message.reply_text(
            f"Hello {user.first_name}! I'm Alfred, your AI assistant.\n\n"
            "You can ask me anything - I can help with:\n"
            "- Checking your email and calendar\n"
            "- Managing your CRM contacts\n"
            "- Searching your knowledge base\n"
            "- Getting your daily briefing\n"
            "- And much more!\n\n"
            "Just send me a message to get started."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text(
            "**Alfred Commands:**\n\n"
            "/start - Start the bot\n"
            "/help - Show this help\n"
            "/briefing - Get your daily briefing\n"
            "/status - Check Alfred's status\n"
            "/email - Check your email\n"
            "/calendar - Check today's calendar\n\n"
            "Or just send me a message and I'll help you!",
            parse_mode="Markdown",
        )

    async def briefing_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /briefing command."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text("Generating your briefing...")

        try:
            from core.briefing.daily import generate_quick_briefing
            briefing = await generate_quick_briefing()
            await update.message.reply_text(briefing, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Briefing failed: {e}")
            await update.message.reply_text(f"Sorry, I couldn't generate your briefing: {e}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text(
            "Alfred Status: Online\n"
            "Interface: Telegram\n"
            "Ready to assist!"
        )

    async def email_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /email command."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text("Checking your email...")

        try:
            from core.tools.definitions import unread_email_count, check_email
            count = unread_email_count()
            emails = check_email()

            if count.get("count", 0) == 0:
                await update.message.reply_text("Your inbox is clear! No unread emails.")
            else:
                msg = f"You have {count.get('count', 0)} unread emails.\n\n"
                for email in emails.get("emails", [])[:5]:
                    msg += f"From: {email.get('from', 'Unknown')}\n"
                    msg += f"Subject: {email.get('subject', 'No subject')}\n\n"
                await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Email check failed: {e}")
            await update.message.reply_text(f"Sorry, I couldn't check your email: {e}")

    async def calendar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /calendar command."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text("Checking your calendar...")

        try:
            from core.tools.definitions import today_schedule
            schedule = today_schedule()
            events = schedule.get("events", [])

            if not events:
                await update.message.reply_text("You have no scheduled events today.")
            else:
                msg = f"Today's Schedule ({len(events)} events):\n\n"
                for event in events[:10]:
                    time_str = event.get("start", {}).get("dateTime", "")
                    if time_str:
                        time_str = time_str[11:16]  # Extract HH:MM
                    else:
                        time_str = "All day"
                    msg += f"{time_str} - {event.get('summary', 'Untitled')}\n"
                await update.message.reply_text(msg)
        except Exception as e:
            logger.error(f"Calendar check failed: {e}")
            await update.message.reply_text(f"Sorry, I couldn't check your calendar: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages."""
        user = update.effective_user
        if not self.is_authorized(user.id):
            await update.message.reply_text("Sorry, you're not authorized to use this bot.")
            return

        message_text = update.message.text
        logger.info(f"Telegram message from {user.id}: {message_text[:100]}")

        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            # Query Alfred's brain
            from core.brain.router import ask

            response = await ask(
                query=message_text,
                messages=None,  # New conversation
                smart_routing=True,
            )

            reply_text = response.get("response", "I'm sorry, I couldn't process that request.")

            # Telegram has a 4096 character limit
            if len(reply_text) > 4000:
                # Split into chunks
                chunks = [reply_text[i:i+4000] for i in range(0, len(reply_text), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(reply_text)

            # Record interaction for learning
            try:
                from core.learning.feedback import record_interaction
                record_interaction(
                    user_query=message_text,
                    alfred_response=reply_text,
                    conversation_id=f"telegram_{user.id}",
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(
                f"Sorry, I encountered an error processing your request. Please try again."
            )

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text(
            "I received your voice message, but voice processing via Telegram "
            "is not yet supported. Please send a text message instead."
        )

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages."""
        if not self.is_authorized(update.effective_user.id):
            return

        await update.message.reply_text(
            "I received your photo. Image analysis via Telegram is coming soon! "
            "For now, please describe what you need help with."
        )

    def setup(self):
        """Set up the bot with handlers."""
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not configured")

        self.application = Application.builder().token(self.token).build()

        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("briefing", self.briefing_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("email", self.email_command))
        self.application.add_handler(CommandHandler("calendar", self.calendar_command))

        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))

        logger.info("Telegram bot configured")

    async def start(self):
        """Start the bot."""
        if not self.application:
            self.setup()

        self._running = True
        logger.info("Starting Telegram bot...")

        # Initialize and start polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot is running")

    async def stop(self):
        """Stop the bot."""
        if self.application and self._running:
            self._running = False
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot stopped")


# Global bot instance
_bot: Optional[TelegramBot] = None


def get_telegram_bot() -> TelegramBot:
    """Get the global Telegram bot instance."""
    global _bot
    if _bot is None:
        _bot = TelegramBot()
    return _bot


async def start_telegram_bot():
    """Start the Telegram bot (convenience function)."""
    bot = get_telegram_bot()
    if bot.token:
        await bot.start()
        return True
    else:
        logger.warning("Telegram bot token not configured, skipping startup")
        return False


async def stop_telegram_bot():
    """Stop the Telegram bot (convenience function)."""
    global _bot
    if _bot:
        await _bot.stop()


# Standalone runner
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    async def main():
        bot = TelegramBot()
        bot.setup()
        await bot.start()

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await bot.stop()

    asyncio.run(main())
