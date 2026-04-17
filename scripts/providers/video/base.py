"""Video provider abstract interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    duration_s: float
    width: int = 1080
    height: int = 1920
    seed: int | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class VideoResult:
    video_path: Path
    duration_s: float


class VideoProvider(ABC):
    name: str

    @abstractmethod
    def gen(self, req: VideoRequest) -> VideoResult: ...
