"""Wake word detection using OpenWakeWord."""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Wake word model path (will be hey_alfred.onnx after training)
WAKE_WORD_MODEL = "hey_jarvis_v0.1"  # Use hey_jarvis until we train hey_alfred
CUSTOM_MODEL_PATH = Path("/home/aialfred/alfred/models/hey_alfred.onnx")

_model = None
_is_initialized = False


def get_model():
    """Get or initialize the wake word model."""
    global _model, _is_initialized

    if _is_initialized:
        return _model

    try:
        from openwakeword.model import Model

        # Check for custom trained model first
        if CUSTOM_MODEL_PATH.exists():
            logger.info(f"Loading custom wake word model: {CUSTOM_MODEL_PATH}")
            _model = Model(
                wakeword_models=[str(CUSTOM_MODEL_PATH)],
                inference_framework="onnx",
            )
        else:
            logger.info(f"Loading pre-trained wake word model: {WAKE_WORD_MODEL}")
            _model = Model(
                wakeword_models=[WAKE_WORD_MODEL],
                inference_framework="onnx",
            )

        _is_initialized = True
        logger.info("Wake word model loaded successfully")
        return _model

    except Exception as e:
        logger.error(f"Failed to load wake word model: {e}")
        return None


def detect_wake_word(audio_chunk: np.ndarray, threshold: float = 0.5) -> dict:
    """Check if audio chunk contains the wake word.

    Args:
        audio_chunk: Audio data as numpy array (16kHz, mono, int16)
        threshold: Detection threshold (0-1)

    Returns:
        dict with 'detected' (bool) and 'scores' (dict of model scores)
    """
    model = get_model()
    if model is None:
        return {"detected": False, "scores": {}, "error": "Model not loaded"}

    try:
        # Ensure audio is correct format
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)

        # Run prediction
        prediction = model.predict(audio_chunk)

        # Check all models for detection
        scores = {}
        detected = False

        for model_name, score in prediction.items():
            scores[model_name] = float(score)
            if score >= threshold:
                detected = True
                logger.info(f"Wake word detected! Model: {model_name}, Score: {score:.3f}")

        return {"detected": detected, "scores": scores}

    except Exception as e:
        logger.error(f"Wake word detection error: {e}")
        return {"detected": False, "scores": {}, "error": str(e)}


def reset_model():
    """Reset the model state (call after detection to avoid re-triggering)."""
    model = get_model()
    if model:
        model.reset()


class WakeWordListener:
    """Continuous wake word listener for audio streams."""

    def __init__(
        self,
        on_wake_word: Callable[[], None],
        threshold: float = 0.5,
        chunk_size: int = 1280,  # 80ms at 16kHz
    ):
        self.on_wake_word = on_wake_word
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.is_listening = False
        self._buffer = np.array([], dtype=np.int16)

    def process_audio(self, audio_data: bytes) -> bool:
        """Process incoming audio data.

        Args:
            audio_data: Raw audio bytes (16kHz, mono, 16-bit PCM)

        Returns:
            True if wake word was detected
        """
        if not self.is_listening:
            return False

        # Convert bytes to numpy
        chunk = np.frombuffer(audio_data, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, chunk])

        # Process in chunks
        detected = False
        while len(self._buffer) >= self.chunk_size:
            process_chunk = self._buffer[:self.chunk_size]
            self._buffer = self._buffer[self.chunk_size:]

            result = detect_wake_word(process_chunk, self.threshold)
            if result["detected"]:
                detected = True
                reset_model()
                self._buffer = np.array([], dtype=np.int16)  # Clear buffer
                if self.on_wake_word:
                    self.on_wake_word()
                break

        return detected

    def start(self):
        """Start listening for wake word."""
        self.is_listening = True
        self._buffer = np.array([], dtype=np.int16)
        logger.info("Wake word listener started")

    def stop(self):
        """Stop listening for wake word."""
        self.is_listening = False
        self._buffer = np.array([], dtype=np.int16)
        logger.info("Wake word listener stopped")
