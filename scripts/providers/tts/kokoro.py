"""Kokoro TTS provider. Wraps the local OpenAI-compatible service at :8880."""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Sequence

import requests

from .base import TtsProvider, TtsRequest, TtsResult

KOKORO_URL = "http://localhost:8880"


class KokoroTts(TtsProvider):
    name = "kokoro"

    def __init__(self, base_url: str = KOKORO_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_voices(self) -> Sequence[str]:
        r = requests.get(f"{self.base_url}/v1/audio/voices", timeout=5)
        r.raise_for_status()
        return r.json()["voices"]

    def synth(self, req: TtsRequest) -> TtsResult:
        out_path = req.output_path or Path(tempfile.mkstemp(suffix=".wav")[1])
        r = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": req.text,
                "voice": req.voice,
                "response_format": "wav",
                "speed": req.speed,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)

        # Calculate duration. Kokoro may generate wav with incorrect header sizes,
        # so we compute from file size and sample parameters.
        with wave.open(str(out_path), "rb") as w:
            framerate = w.getframerate()
            sample_width = w.getsampwidth()
            channels = w.getnchannels()

        file_size = out_path.stat().st_size
        bytes_per_frame = sample_width * channels
        # WAV header is typically 44 bytes
        data_size = file_size - 44
        frames = data_size // bytes_per_frame
        duration = frames / float(framerate)

        return TtsResult(audio_path=out_path, duration_s=duration, voice=req.voice)
