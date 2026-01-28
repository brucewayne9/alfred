"""ComfyUI API client for image generation."""

import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/aialfred/ComfyUI/output")

# Simple SDXL Turbo workflow - generates in 4 steps
SDXL_TURBO_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 1.0,
            "denoise": 1.0,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "seed": None,  # Will be randomized
            "steps": 4
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 1024,
            "width": 1024
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": ""  # Will be filled with prompt
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["4", 1],
            "text": "blurry, low quality, distorted, ugly, bad anatomy"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "alfred",
            "images": ["8", 0]
        }
    }
}


async def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> dict:
    """Generate an image using ComfyUI.

    Args:
        prompt: Text description of the image to generate
        width: Image width (default 1024)
        height: Image height (default 1024)

    Returns:
        dict with keys: success, image_path, base64, error
    """
    import random

    # Build the workflow
    workflow = json.loads(json.dumps(SDXL_TURBO_WORKFLOW))
    workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height
    workflow["6"]["inputs"]["text"] = prompt

    client_id = str(uuid.uuid4())

    try:
        async with aiohttp.ClientSession() as session:
            # Queue the prompt
            async with session.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id}
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return {"success": False, "error": f"Failed to queue prompt: {error}"}
                result = await resp.json()
                prompt_id = result["prompt_id"]

            logger.info(f"ComfyUI prompt queued: {prompt_id}")

            # Poll for completion via history
            for _ in range(120):  # 60 second timeout
                await asyncio.sleep(0.5)
                async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
                    if resp.status != 200:
                        continue
                    history = await resp.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        if "9" in outputs and outputs["9"].get("images"):
                            image_info = outputs["9"]["images"][0]
                            filename = image_info["filename"]
                            subfolder = image_info.get("subfolder", "")

                            # Get the image
                            params = {"filename": filename, "subfolder": subfolder, "type": "output"}
                            async with session.get(f"{COMFYUI_URL}/view", params=params) as img_resp:
                                if img_resp.status == 200:
                                    image_data = await img_resp.read()
                                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                                    # Save to Alfred's generated folder
                                    from core.tools.files import GENERATED_DIR
                                    save_path = GENERATED_DIR / filename
                                    save_path.write_bytes(image_data)

                                    logger.info(f"Image generated: {filename}")
                                    return {
                                        "success": True,
                                        "filename": filename,
                                        "image_path": str(save_path),
                                        "base64": image_base64,
                                        "download_url": f"/download/{filename}",
                                    }

            return {"success": False, "error": "Generation timed out"}

    except aiohttp.ClientConnectorError:
        return {"success": False, "error": "ComfyUI is not running. Start it with: sudo systemctl start comfyui"}
    except Exception as e:
        logger.error(f"ComfyUI error: {e}")
        return {"success": False, "error": str(e)}


async def check_status() -> dict:
    """Check if ComfyUI is running and ready."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFYUI_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    stats = await resp.json()
                    return {"running": True, "stats": stats}
    except Exception:
        pass
    return {"running": False}
