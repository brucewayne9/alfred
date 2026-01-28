"""Speech-to-Text service using faster-whisper."""

import io
import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_model = None


def get_model(model_size: str = "small", device: str = "cuda") -> WhisperModel:
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {model_size} on {device}")
        _model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8_float16" if device == "cuda" else "int8",
        )
        logger.info("Whisper model loaded")
    return _model


def warmup(model_size: str = "small"):
    """Pre-load the Whisper model so first request is fast."""
    get_model(model_size)
    logger.info("Whisper model warmed up")


def transcribe(audio_data: bytes, model_size: str = "small") -> str:
    """Transcribe audio bytes to text."""
    model = get_model(model_size)
    audio_file = io.BytesIO(audio_data)
    segments, info = model.transcribe(audio_file, beam_size=1, vad_filter=True)
    text = " ".join(segment.text for segment in segments).strip()
    logger.info(f"Transcribed ({info.language}, {info.duration:.1f}s): {text[:100]}")
    return text


async def transcribe_stream(audio_stream, model_size: str = "small"):
    """Transcribe streaming audio chunks."""
    model = get_model(model_size)
    buffer = io.BytesIO()
    async for chunk in audio_stream:
        buffer.write(chunk)
    buffer.seek(0)
    segments, info = model.transcribe(buffer, beam_size=1, vad_filter=True)
    return " ".join(segment.text for segment in segments).strip()
