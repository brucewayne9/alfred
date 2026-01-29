"""Text-to-Speech service. Supports multiple backends."""

import io
import logging
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

# Qwen3-TTS server URL (local or remote)
QWEN3_TTS_URL = "http://localhost:7860"


def _format_phone_for_speech(match) -> str:
    """Convert phone number to digit-by-digit format for TTS."""
    phone = match.group(0)
    # Extract just the digits
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        # Format as "X X X, X X X, X X X X" with pauses between groups
        return f"{' '.join(digits[0:3])}, {' '.join(digits[3:6])}, {' '.join(digits[6:10])}"
    elif len(digits) == 11 and digits[0] == '1':
        # Handle 1-XXX-XXX-XXXX format
        return f"1, {' '.join(digits[1:4])}, {' '.join(digits[4:7])}, {' '.join(digits[7:11])}"
    else:
        # Just space out all digits
        return ' '.join(digits)


def _clean_for_speech(text: str) -> str:
    """Strip markdown and formatting artifacts so TTS reads naturally."""
    # Format phone numbers to be read digit by digit
    # Matches formats like: (404) 555-1234, 404-555-1234, 404.555.1234, 4045551234
    phone_pattern = r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    text = re.sub(phone_pattern, _format_phone_for_speech, text)

    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\|[^\n]+\|', '', text)
    text = re.sub(r'^\s*[-:]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

_engine = None


class TTSEngine:
    """TTS engine abstraction. Currently wraps Kokoro or Piper."""

    def __init__(self, backend: str = "kokoro"):
        self.backend = backend
        self._model = None
        logger.info(f"TTS engine initialized with backend: {backend}")

    def synthesize(self, text: str, voice_id: str = "bm_daniel") -> bytes:
        """Convert text to speech audio bytes (WAV format)."""
        if self.backend == "kokoro":
            return self._synthesize_kokoro(text, voice_id)
        elif self.backend == "qwen3":
            return self._synthesize_qwen3(text, voice_id)
        elif self.backend == "piper":
            return self._synthesize_piper(text)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

    def _synthesize_qwen3(self, text: str, voice_id: str = "demo_speaker0") -> bytes:
        """Synthesize using Qwen3-TTS server."""
        try:
            url = f"{QWEN3_TTS_URL}/synthesize_speech/"
            params = {"text": text, "voice": voice_id}
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Qwen3-TTS error: {response.status_code} - {response.text}")
                # Fallback to Kokoro
                return self._synthesize_kokoro(text, "bm_daniel")
        except Exception as e:
            logger.error(f"Qwen3-TTS connection failed: {e}, falling back to Kokoro")
            return self._synthesize_kokoro(text, "bm_daniel")

    def _synthesize_kokoro(self, text: str, voice_id: str) -> bytes:
        """Synthesize using Kokoro TTS."""
        try:
            from kokoro import KPipeline
            if self._model is None:
                self._model = KPipeline(lang_code="b")
                logger.info("Kokoro TTS model loaded")

            audio_segments = []
            for _, _, audio in self._model(text, voice=voice_id):
                audio_segments.append(audio)

            if not audio_segments:
                return b""

            import numpy as np
            import soundfile as sf

            combined = np.concatenate(audio_segments)
            buf = io.BytesIO()
            sf.write(buf, combined, 24000, format="WAV")
            buf.seek(0)
            return buf.read()
        except ImportError:
            logger.warning("Kokoro not installed, falling back to espeak")
            return self._synthesize_espeak(text)

    def _synthesize_piper(self, text: str) -> bytes:
        """Synthesize using Piper TTS (lightweight, fast)."""
        try:
            result = subprocess.run(
                ["piper", "--model", "en_US-lessac-medium", "--output-raw"],
                input=text.encode(),
                capture_output=True,
                timeout=30,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Piper not available, falling back to espeak")
            return self._synthesize_espeak(text)

    def _synthesize_espeak(self, text: str) -> bytes:
        """Fallback espeak synthesis."""
        buf = io.BytesIO()
        result = subprocess.run(
            ["espeak-ng", "--stdout", text],
            capture_output=True,
            timeout=10,
        )
        return result.stdout


def get_engine(backend: str = "kokoro") -> TTSEngine:
    global _engine
    if _engine is None or _engine.backend != backend:
        _engine = TTSEngine(backend=backend)
    return _engine


def warmup():
    """Pre-load the TTS model so first request is fast."""
    engine = get_engine()
    engine.synthesize("warmup", "bm_daniel")
    logger.info("TTS engine warmed up")


def speak(text: str, voice_id: str = None) -> bytes:
    """Quick function to synthesize speech.

    Uses the TTS backend configured in settings (kokoro, qwen3, or piper).
    """
    from config.settings import settings
    backend = settings.tts_model

    # Set default voice based on backend
    if voice_id is None:
        voice_id = "demo_speaker0" if backend == "qwen3" else "bm_daniel"

    engine = get_engine(backend)
    clean = _clean_for_speech(text)
    return engine.synthesize(clean, voice_id)
