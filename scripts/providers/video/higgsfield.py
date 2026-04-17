"""Higgsfield video provider — stub until API access arrives."""
from __future__ import annotations

from .base import VideoProvider, VideoRequest, VideoResult


class HiggsfieldVideo(VideoProvider):
    name = "higgsfield"

    def gen(self, req: VideoRequest) -> VideoResult:
        raise NotImplementedError(
            "Higgsfield video provider not implemented yet. "
            "Fill in when API credentials are available."
        )
