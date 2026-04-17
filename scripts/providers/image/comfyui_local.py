"""ComfyUI local GPU provider. Delegates to existing comfyui_gen.py script."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .base import ImageProvider, ImageRequest, ImageResult

COMFYUI_GEN_SCRIPT = Path("/home/aialfred/alfred/scripts/comfyui_gen.py")


class ComfyUiLocal(ImageProvider):
    name = "comfyui-local"

    def gen(self, req: ImageRequest) -> ImageResult:
        out = req.output_path or Path(tempfile.mkstemp(suffix=".png")[1])
        cmd = [
            "python3", str(COMFYUI_GEN_SCRIPT),
            "generate", req.prompt,
            "--output", str(out),
            "--width", str(req.width),
            "--height", str(req.height),
        ]
        if req.seed is not None:
            cmd.extend(["--seed", str(req.seed)])
        subprocess.run(cmd, check=True, timeout=300)
        return ImageResult(image_path=out, width=req.width, height=req.height)
