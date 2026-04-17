"""Image provider abstract interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    width: int = 1080
    height: int = 1920
    seed: int | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class ImageResult:
    image_path: Path
    width: int
    height: int


class ImageProvider(ABC):
    name: str

    @abstractmethod
    def gen(self, req: ImageRequest) -> ImageResult: ...
