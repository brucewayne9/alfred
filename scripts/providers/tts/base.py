"""TTS provider abstract interface.

Every TTS backend (Kokoro, Qwen3, future providers) implements this.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TtsRequest:
    text: str
    voice: str
    speed: float = 1.0
    output_path: Path | None = None  # None => provider picks a temp path


@dataclass(frozen=True)
class TtsResult:
    audio_path: Path
    duration_s: float
    voice: str


class TtsProvider(ABC):
    """Abstract TTS provider. Subclasses MUST implement name, list_voices, synth."""

    name: str  # short id: "kokoro", "qwen3"

    @abstractmethod
    def list_voices(self) -> Sequence[str]:
        """Return the voices this provider exposes."""

    @abstractmethod
    def synth(self, req: TtsRequest) -> TtsResult:
        """Synthesize speech. Return path to a .wav file on disk."""
