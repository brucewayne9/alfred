"""Telegram bot for Alfred - chat with Alfred via Telegram.

This module provides a Telegram bot interface that connects to Alfred's brain,
allowing users to interact with Alfred through Telegram messages.
"""

import asyncio
import datetime
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import requests

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

# Voice transcription (local faster-whisper, OpenAI-compatible)
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8001/v1/audio/transcriptions")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small.en")
# Where voice brain-dumps land for the RuckTalk radio content engine
BRAIN_DUMP_DIR = Path(__file__).parent.parent.parent / "data" / "rucktalk" / "brain_dumps"


def _transcribe_voice(ogg_path: str) -> str:
    """Convert a Telegram voice note (.oga/opus) to 16k mono wav and transcribe
    via the local faster-whisper server. Blocking — call via asyncio.to_thread."""
    wav_path = ogg_path + ".wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
    )
    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                files={"file": (os.path.basename(wav_path), f, "audio/wav")},
                data={"model": WHISPER_MODEL},
                timeout=180,
            )
        resp.raise_for_status()
        try:
            return (resp.json().get("text") or "").strip()
        except Exception:
            return resp.text.strip()
    finally:
        for p in (ogg_path, wav_path):
            try:
                os.remove(p)
            except OSError:
                pass


def _save_brain_dump(text: str, source: str = "voice") -> Path:
    """Append a brain-dump to today's RuckTalk journal. Blocking.
    `source` is 'voice' (transcribed note) or 'text' (typed/dictated message)."""
    BRAIN_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    path = BRAIN_DUMP_DIR / f"{now:%Y-%m-%d}.md"
    with open(path, "a") as f:
        f.write(f"\n## {now:%H:%M} — {source} brain-dump\n{text}\n")
    return path


# Text messages starting with any of these (case-insensitive) are captured as
# RuckTalk brain-dumps instead of being routed to the chat brain. Lets Mike
# dictate with the keyboard mic (which sends TEXT, not a voice note) and still
# have it land in the content journal.
BRAIN_DUMP_PREFIXES = ("/dump", "brain dump", "braindump", "brain-dump")


def _extract_brain_dump(message_text: str) -> Optional[str]:
    """If the message is a brain-dump trigger, return the dump body (prefix
    stripped). Otherwise return None. Matches a leading trigger word only."""
    stripped = message_text.lstrip()
    low = stripped.lower()
    for prefix in BRAIN_DUMP_PREFIXES:
        if low.startswith(prefix):
            body = stripped[len(prefix):]
            # Drop a single leading separator (":", "-", "—") and surrounding space
            body = body.lstrip(" :;-—\n")
            return body.strip()
    return None


# Lanes recognized right after the dump trigger. radio -> show journal,
# social/instagram/ig -> content board card, anything else -> 'none' (triage).
_DUMP_LANES = (("radio", "radio"), ("social", "social"),
               ("instagram", "social"), ("ig", "social"))


def _extract_brain_dump_lane(message_text: str):
    """Return (lane, body) for a brain-dump message, else None.
    lane in {'radio','social','none'}. A leading lane word is stripped from body."""
    body = _extract_brain_dump(message_text)
    if body is None:
        return None
    low = body.lower()
    for word, lane in _DUMP_LANES:
        if low.startswith(word):
            rest = body[len(word):].lstrip(" :;-—\n")
            return (lane, rest.strip())
    return ("none", body)


class TelegramBot:
    """Telegram bot interface for Alfred."""

    def __init__(self, token: str = None, allowed_users: list[str] = None):
        # Re-read env at instantiation so dotenv loaded in __main__ takes effect
        # (module-level globals are captured at import, before load_dotenv runs).
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "") or TELEGRAM_BOT_TOKEN
        env_users = [u.strip() for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if u.strip()]
        self.allowed_users = allowed_users or env_users
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

        # RuckTalk brain-dump capture (lane-aware): radio -> show journal,
        # social/instagram -> content board card, untagged -> journal (Alfred triages).
        dump = _extract_brain_dump_lane(message_text)
        if dump is not None:
            lane, body = dump
            if not body:
                await update.message.reply_text(
                    "Ready when you are, sir — 'brain dump social <idea>' for a reel, "
                    "'brain dump radio <idea>' for the show."
                )
                return
            try:
                source = {"radio": "radio", "social": "social"}.get(lane, "text")
                path = await asyncio.to_thread(_save_brain_dump, body, source)
                wc = len(body.split())
                if lane == "social":
                    import requests as _rq
                    try:
                        await asyncio.to_thread(
                            lambda: _rq.post(
                                "http://localhost:8400/rt-board/api/card",
                                json={"raw": body}, timeout=10,
                            )
                        )
                    except Exception as _e:
                        logger.error(f"board card post failed: {_e}")
                    await update.message.reply_text(
                        f"🎬 Card's on the board ({wc} words) — I'll polish it.\n"
                        "aialfred.groundrushcloud.com/rt-board/"
                    )
                else:
                    await update.message.reply_text(
                        f"🎙️ Brain-dump captured for today's RuckTalk show ({wc} words). "
                        "Keep them coming, sir."
                    )
                logger.info(f"Brain-dump [{lane}] captured ({wc} words) -> {path}")
            except Exception as e:
                logger.error(f"Brain-dump save failed: {e}")
                await update.message.reply_text(
                    "Sorry, I hit an error saving that brain-dump. Give it another shot?"
                )
            return

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
        """Transcribe a voice note and capture it as a RuckTalk brain-dump."""
        if not self.is_authorized(update.effective_user.id):
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        tmp_path = None
        try:
            tg_file = await context.bot.get_file(update.message.voice.file_id)
            with tempfile.NamedTemporaryFile(suffix=".oga", delete=False) as tf:
                tmp_path = tf.name
            await tg_file.download_to_drive(tmp_path)

            text = await asyncio.to_thread(_transcribe_voice, tmp_path)
            tmp_path = None  # _transcribe_voice cleans up its inputs

            if not text:
                await update.message.reply_text(
                    "I couldn't make out that voice note — mind trying again?"
                )
                return

            path = await asyncio.to_thread(_save_brain_dump, text)
            word_count = len(text.split())
            logger.info(f"Voice brain-dump captured ({word_count} words) -> {path}")
            await update.message.reply_text(
                f"🎙️ Brain-dump captured for today's RuckTalk show ({word_count} words). "
                f"Here's what I heard — check I caught it right:\n\n{text}"
            )
        except Exception as e:
            logger.error(f"Voice handling failed: {e}")
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            await update.message.reply_text(
                "Sorry, I hit an error processing that voice note. Give it another shot?"
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
