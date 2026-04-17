"""Qwen3 TTS provider. Wraps the FastAPI service at :7860.

Merges cloned voices (/cloned_voices) and designed voices (/voice_design/voices)
into a single voice pool. Synthesis goes through /synthesize_speech/ for cloned
voices or /voice_design/synthesize for designed voices.
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Sequence

import requests

from .base import TtsProvider, TtsRequest, TtsResult

QWEN3_URL = "http://localhost:7860"


class Qwen3Tts(TtsProvider):
    name = "qwen3"

    def __init__(self, base_url: str = QWEN3_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _fetch_voice_catalogs(self) -> tuple[list[str], list[str]]:
        """Return (cloned_names, designed_names)."""
        cloned = requests.get(f"{self.base_url}/cloned_voices", timeout=5).json()
        designed = requests.get(f"{self.base_url}/voice_design/voices", timeout=5).json()
        cloned_names = [v["name"] for v in cloned.get("cloned_voices", [])]
        designed_names = [v["name"] for v in designed.get("voices", [])]
        return cloned_names, designed_names

    def list_voices(self) -> Sequence[str]:
        cloned, designed = self._fetch_voice_catalogs()
        return cloned + designed

    def synth(self, req: TtsRequest) -> TtsResult:
        out_path = req.output_path or Path(tempfile.mkstemp(suffix=".wav")[1])
        cloned, designed = self._fetch_voice_catalogs()

        if req.voice in cloned:
            endpoint = "/synthesize_speech/"
        elif req.voice in designed:
            endpoint = "/voice_design/synthesize"
        else:
            raise ValueError(f"Unknown Qwen3 voice: {req.voice}")

        r = requests.get(
            f"{self.base_url}{endpoint}",
            params={"text": req.text, "voice": req.voice, "speed": req.speed},
            timeout=self.timeout,
        )
        r.raise_for_status()
        out_path.write_bytes(r.content)

        # Duration from wav — if Qwen3 emits malformed headers like Kokoro,
        # fall back to file-size computation.
        try:
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

            # Sanity check: if duration is absurd (e.g., from corrupted header),
            # recompute from file size using reasonable defaults.
            if duration < 0 or duration > 3600:
                duration = (out_path.stat().st_size - 44) / (24000 * 2)
        except Exception:
            # Fallback: estimate from file size assuming 24000 Hz, mono
            duration = (out_path.stat().st_size - 44) / (24000 * 2)

        return TtsResult(audio_path=out_path, duration_s=duration, voice=req.voice)
