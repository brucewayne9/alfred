"""LTX-2 Cloud video provider. Delegates to ComfyUI cloud LTX workflow."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .base import VideoProvider, VideoRequest, VideoResult

# LTX-2 is invoked via the same comfyui_gen.py with a --video flag.
COMFYUI_GEN_SCRIPT = Path("/home/aialfred/alfred/scripts/comfyui_gen.py")


class LtxCloud(VideoProvider):
    name = "ltx-cloud"

    def gen(self, req: VideoRequest) -> VideoResult:
        out = req.output_path or Path(tempfile.mkstemp(suffix=".mp4")[1])
        cmd = [
            "python3", str(COMFYUI_GEN_SCRIPT),
            "generate-video", req.prompt,
            "--cloud",
            "--duration", str(req.duration_s),
            "--output", str(out),
            "--width", str(req.width),
            "--height", str(req.height),
        ]
        subprocess.run(cmd, check=True, timeout=600)
        return VideoResult(video_path=out, duration_s=req.duration_s)
