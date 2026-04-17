"""Higgsfield image provider — stub until API access arrives."""
from __future__ import annotations

from .base import ImageProvider, ImageRequest, ImageResult


class HiggsfieldImage(ImageProvider):
    name = "higgsfield"

    def gen(self, req: ImageRequest) -> ImageResult:
        raise NotImplementedError(
            "Higgsfield image provider not implemented yet. "
            "Fill in when API credentials are available."
        )
