"""Text-to-Speech service. Supports multiple backends."""

import io
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_for_speech(text: str) -> str:
    """Strip markdown and formatting artifacts so TTS reads naturally."""
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
        elif self.backend == "piper":
            return self._synthesize_piper(text)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")

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


def speak(text: str, voice_id: str = "bm_daniel") -> bytes:
    """Quick function to synthesize speech."""
    engine = get_engine()
    clean = _clean_for_speech(text)
    return engine.synthesize(clean, voice_id)
