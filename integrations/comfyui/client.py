"""ComfyUI API client for image generation and product video pipeline.

Supports quality SDXL generation with Juggernaut XL v9,
hi-res fix workflow for sharp details, optional 4x upscaling,
SVD-XT video generation from product photos,
and on-demand GPU service management.
"""

import asyncio
import base64
import json
import logging
import random
import shutil
import uuid
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path("/home/aialfred/ComfyUI/output")
MODELS_DIR = Path("/home/aialfred/ComfyUI/models")
COMFYUI_INPUT_DIR = Path("/home/aialfred/ComfyUI/input")

# Model files
QUALITY_CHECKPOINT = "Juggernaut-XL-v9.safetensors"
FAST_CHECKPOINT = "sd_xl_turbo_1.0_fp16.safetensors"
FLUX_CHECKPOINT = "flux1-dev-fp8.safetensors"
UPSCALE_MODEL = "4x-UltraSharp.pth"

# Face/hand detection models for FaceDetailer (Impact Pack)
# UltralyticsDetectorProvider expects subdir/filename format
FACE_DETECT_MODEL = "bbox/face_yolov8m.pt"
HAND_DETECT_MODEL = "bbox/hand_yolov8s.pt"

# SVD-XT model for video generation
SVD_CHECKPOINT = "svd_xt_1_1.safetensors"
SVD_CHECKPOINT_FALLBACK = "svd_xt.safetensors"

# Strong negative prompt tuned for Juggernaut XL — prevents phantom people,
# distorted anatomy, and extra limbs
NEGATIVE_PROMPT = (
    "multiple people, crowd, group, extra person, two people, couple, "
    "bad anatomy, bad hands, extra fingers, missing fingers, fused fingers, "
    "too many fingers, extra limbs, missing limbs, extra arms, extra legs, "
    "mutated hands, poorly drawn hands, malformed hands, mutated, "
    "deformed, disfigured, mutation, extra heads, double image, "
    "duplicate, clone, twin, split image, "
    "blurry, low quality, distorted, ugly, worst quality, low quality, "
    "jpeg artifacts, watermark, text, signature, logo, banner, "
    "cropped, out of frame, poorly drawn face, asymmetric face, "
    "cross-eyed, body horror, fused bodies, conjoined, "
    "floating limbs, disconnected limbs, extra digits, fewer digits"
)

# Quality booster prefix for prompts — Juggernaut XL responds well to these
QUALITY_PREFIX = (
    "masterpiece, best quality, highly detailed, sharp focus, "
    "professional photography, 8k uhd, high resolution, "
)

# Product photography prompts for img2img enhancement
PRODUCT_ENHANCE_PROMPT = (
    "professional product photography, studio lighting, clean white background, "
    "commercial quality, sharp focus, high detail, catalog photo, "
    "soft diffused lighting, no shadows, pristine product shot"
)

PRODUCT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, ugly, worst quality, "
    "jpeg artifacts, watermark, text, signature, logo, banner, "
    "cluttered background, messy, dirty, damaged, scratched, "
    "oversaturated, underexposed, overexposed, grainy, noisy"
)

# SDXL-native resolution buckets — these are the aspect ratios SDXL was
# actually trained on. Using non-native resolutions causes distortion.
SDXL_RESOLUTIONS = [
    (1024, 1024),  # 1:1
    (1152, 896),   # 9:7 (landscape)
    (896, 1152),   # 7:9 (portrait)
    (1216, 832),   # 3:2 (landscape)
    (832, 1216),   # 2:3 (portrait)
    (1344, 768),   # 7:4 (wide landscape)
    (768, 1344),   # 4:7 (tall portrait)
    (1536, 640),   # 12:5 (ultrawide)
    (640, 1536),   # 5:12 (ultra tall)
]


def _snap_to_sdxl_resolution(width: int, height: int) -> tuple[int, int]:
    """Snap requested dimensions to the nearest SDXL-native resolution bucket.

    SDXL produces much better results at its trained resolutions. Arbitrary
    sizes cause distortion, especially with anatomy.
    """
    target_ratio = width / height
    best = min(
        SDXL_RESOLUTIONS,
        key=lambda res: abs((res[0] / res[1]) - target_ratio),
    )
    return best


def _enhance_prompt(prompt: str) -> str:
    """Add quality boosters to the user's prompt.

    Juggernaut XL produces significantly better output when quality tags
    are prepended. We also avoid doubling them if the user already included
    similar terms.
    """
    lower = prompt.lower()
    # Don't double-add if user already included quality terms
    if any(term in lower for term in ["masterpiece", "best quality", "8k", "highly detailed"]):
        return prompt
    return QUALITY_PREFIX + prompt


def _has_model(subdir: str, filename: str) -> bool:
    return (MODELS_DIR / subdir / filename).exists()


def _add_face_detailer(
    workflow: dict,
    image_node: str,
    model_node: str,
    clip_node_idx: int,
    positive_node: str,
    negative_node: str | None,
    is_flux: bool = False,
) -> tuple[dict, str]:
    """Append FaceDetailer nodes to a workflow for face/hand fix-up.

    Detects faces and hands in the generated image, then re-inpaints those
    regions at higher detail to fix extra eyes, mouths, fingers, etc.

    Returns (updated_workflow, final_image_node) — the final image node ID
    to use for SaveImage instead of the original image_node.
    """
    if not _has_model("ultralytics", FACE_DETECT_MODEL):
        return workflow, image_node

    # FaceDetailer for faces
    workflow["20"] = {
        "class_type": "UltralyticsDetectorProvider",
        "inputs": {"model_name": FACE_DETECT_MODEL},
    }
    # Use lower denoise for FLUX (it's already high quality, just fixing artifacts)
    face_denoise = 0.35 if is_flux else 0.4
    workflow["21"] = {
        "class_type": "FaceDetailer",
        "inputs": {
            "image": [image_node, 0],
            "model": [model_node, 0],
            "clip": [model_node, clip_node_idx],
            "vae": [model_node, 2],
            "positive": [positive_node, 0],
            "negative": [negative_node or positive_node, 0],
            "bbox_detector": ["20", 0],
            "sam_model_opt": None,
            "segm_detector_opt": None,
            "seed": random.randint(0, 2**32 - 1),
            "steps": 15,
            "cfg": 1.0 if is_flux else 4.5,
            "sampler_name": "euler" if is_flux else "dpmpp_2m_sde",
            "scheduler": "simple" if is_flux else "karras",
            "denoise": face_denoise,
            "guide_size": 384,
            "guide_size_for": True,
            "max_size": 1024,
            "feather": 5,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3.0,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "wildcard": "",
            "cycle": 1,
        },
    }

    final_image = "21"

    # Add hand detailer if model available
    if _has_model("ultralytics", HAND_DETECT_MODEL):
        workflow["22"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": HAND_DETECT_MODEL},
        }
        workflow["23"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["21", 0],  # chain after face fix
                "model": [model_node, 0],
                "clip": [model_node, clip_node_idx],
                "vae": [model_node, 2],
                "positive": [positive_node, 0],
                "negative": [negative_node or positive_node, 0],
                "bbox_detector": ["22", 0],
                "sam_model_opt": None,
                "segm_detector_opt": None,
                "seed": random.randint(0, 2**32 - 1),
                "steps": 15,
                "cfg": 1.0 if is_flux else 4.5,
                "sampler_name": "euler" if is_flux else "dpmpp_2m_sde",
                "scheduler": "simple" if is_flux else "karras",
                "denoise": 0.35,
                "guide_size": 384,
                "guide_size_for": True,
                "max_size": 1024,
                "feather": 5,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
            },
        }
        final_image = "23"

    return workflow, final_image


def _select_checkpoint() -> tuple[str, dict]:
    """Select best available checkpoint and its settings.

    Juggernaut XL v9 optimal settings:
    - CFG 4.5 (lower than default — prevents hallucinated extra features)
    - DPM++ 2M SDE Karras (better anatomy than plain DPM++ 2M)
    - 35 steps (more coherent than 30)
    """
    if _has_model("checkpoints", QUALITY_CHECKPOINT):
        return QUALITY_CHECKPOINT, {
            "steps": 35,
            "cfg": 4.5,
            "sampler": "dpmpp_2m_sde",
            "scheduler": "karras",
        }
    # Fallback to Turbo
    return FAST_CHECKPOINT, {
        "steps": 6,
        "cfg": 1.5,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
    }


def _get_svd_checkpoint() -> str | None:
    """Find available SVD checkpoint, returns filename or None."""
    if _has_model("checkpoints", SVD_CHECKPOINT):
        return SVD_CHECKPOINT
    if _has_model("checkpoints", SVD_CHECKPOINT_FALLBACK):
        return SVD_CHECKPOINT_FALLBACK
    return None


def _build_quality_workflow(
    prompt: str,
    width: int,
    height: int,
    upscale: bool = False,
    mode: str = "quality",
) -> dict:
    """Build a ComfyUI workflow with quality settings.

    Modes:
      - quality: Juggernaut XL, 35 steps, CFG 4.5, DPM++ 2M SDE Karras,
                 with hi-res fix (generate at 768 base, latent upscale to
                 target, re-denoise for sharp detail)
      - fast: SDXL Turbo, 6 steps, CFG 1.5, euler_ancestral
    """
    if mode == "fast":
        ckpt = FAST_CHECKPOINT
        settings = {"steps": 6, "cfg": 1.5, "sampler": "euler_ancestral", "scheduler": "normal"}
    else:
        ckpt, settings = _select_checkpoint()

    # Snap to SDXL-native resolution
    width, height = _snap_to_sdxl_resolution(width, height)

    # Enhance the prompt with quality tags
    enhanced_prompt = _enhance_prompt(prompt)

    seed = random.randint(0, 2**32 - 1)

    # For quality mode with Juggernaut, use hi-res fix workflow:
    # 1. Generate at reduced resolution (768-based) for good composition
    # 2. Latent upscale to target resolution
    # 3. Re-denoise at low strength for sharp details
    use_hires_fix = (mode == "quality" and ckpt == QUALITY_CHECKPOINT)

    if use_hires_fix:
        # Calculate base resolution (roughly 0.75x of target, snapped to 64)
        base_width = (int(width * 0.75) // 64) * 64
        base_height = (int(height * 0.75) // 64) * 64

        workflow = {
            # Load checkpoint
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt},
            },
            # Base latent (smaller for initial composition)
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": base_height, "width": base_width},
            },
            # Positive prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": enhanced_prompt},
            },
            # Negative prompt
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": NEGATIVE_PROMPT},
            },
            # First pass KSampler — full denoise for composition
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": settings["cfg"],
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": settings["sampler"],
                    "scheduler": settings["scheduler"],
                    "seed": seed,
                    "steps": settings["steps"],
                },
            },
            # Latent upscale to target resolution
            "12": {
                "class_type": "LatentUpscale",
                "inputs": {
                    "samples": ["3", 0],
                    "upscale_method": "bislerp",
                    "width": width,
                    "height": height,
                    "crop": "disabled",
                },
            },
            # Second pass KSampler — low denoise to add detail without changing composition
            "13": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": settings["cfg"],
                    "denoise": 0.45,
                    "latent_image": ["12", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": settings["sampler"],
                    "scheduler": settings["scheduler"],
                    "seed": seed,
                    "steps": 20,
                },
            },
            # VAE Decode the hi-res result
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["13", 0], "vae": ["4", 2]},
            },
        }
    else:
        # Simple single-pass workflow (fast mode or fallback checkpoint)
        workflow = {
            # Load checkpoint
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt},
            },
            # Empty latent
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": height, "width": width},
            },
            # Positive prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": enhanced_prompt},
            },
            # Negative prompt
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": NEGATIVE_PROMPT},
            },
            # KSampler
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": settings["cfg"],
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": settings["sampler"],
                    "scheduler": settings["scheduler"],
                    "seed": seed,
                    "steps": settings["steps"],
                },
            },
            # VAE Decode
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
        }

    # Fix faces and hands with FaceDetailer (detects and re-inpaints problem areas)
    workflow, final_image = _add_face_detailer(
        workflow, image_node="8", model_node="4", clip_node_idx=1,
        positive_node="6", negative_node="7",
    )

    if upscale and _has_model("upscale_models", UPSCALE_MODEL):
        workflow["10"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": UPSCALE_MODEL},
        }
        workflow["11"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["10", 0], "image": [final_image, 0]},
        }
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alfred", "images": ["11", 0]},
        }
    else:
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alfred", "images": [final_image, 0]},
        }

    return workflow


def _build_flux_workflow(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    upscale: bool = False,
) -> dict:
    """Build a ComfyUI workflow for FLUX.1 dev FP8.

    FLUX key differences from SDXL:
    - No negative prompt (FLUX ignores it)
    - Uses 'guidance' parameter on the model (not CFG on sampler)
    - CFG on KSampler should be 1.0 (guidance is applied via ModelSamplingFlux)
    - 20 steps with euler/simple scheduler
    - Supports arbitrary resolutions (no need for SDXL bucket snapping)
    - Much better prompt following and text rendering
    """
    # Snap to multiples of 64
    width = (width // 64) * 64
    height = (height // 64) * 64

    seed = random.randint(0, 2**32 - 1)

    workflow = {
        # Load FLUX checkpoint (FP8 combined has model + clip + vae)
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": FLUX_CHECKPOINT},
        },
        # Empty latent image
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": height, "width": width},
        },
        # Positive prompt (FLUX uses CLIP from the checkpoint)
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": prompt},
        },
        # KSampler — CFG 1.0 (FLUX guidance is internal to the model)
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 1.0,
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["6", 0],  # FLUX ignores negative, but node needs a connection
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "simple",
                "seed": seed,
                "steps": 20,
            },
        },
        # VAE Decode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
    }

    # Fix faces and hands with FaceDetailer
    workflow, final_image = _add_face_detailer(
        workflow, image_node="8", model_node="4", clip_node_idx=1,
        positive_node="6", negative_node=None, is_flux=True,
    )

    if upscale and _has_model("upscale_models", UPSCALE_MODEL):
        workflow["10"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": UPSCALE_MODEL},
        }
        workflow["11"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["10", 0], "image": [final_image, 0]},
        }
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alfred_flux", "images": ["11", 0]},
        }
    else:
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alfred_flux", "images": [final_image, 0]},
        }

    return workflow


def _build_img2img_workflow(
    image_filename: str,
    prompt: str | None = None,
    denoise_strength: float = 0.45,
) -> dict:
    """Build a ComfyUI img2img workflow to enhance a product photo.

    Uses Juggernaut XL with partial denoise to improve lighting and quality
    while preserving the original product appearance.
    """
    ckpt, settings = _select_checkpoint()
    seed = random.randint(0, 2**32 - 1)
    final_prompt = _enhance_prompt(prompt or PRODUCT_ENHANCE_PROMPT)

    return {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        # Load the product image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # Encode image to latent via VAE
        "3": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["2", 0], "vae": ["1", 2]},
        },
        # Positive prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": final_prompt},
        },
        # Negative prompt
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": PRODUCT_NEGATIVE_PROMPT},
        },
        # KSampler — partial denoise to enhance without destroying
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": settings["cfg"],
                "denoise": denoise_strength,
                "latent_image": ["3", 0],
                "model": ["1", 0],
                "negative": ["5", 0],
                "positive": ["4", 0],
                "sampler_name": settings["sampler"],
                "scheduler": settings["scheduler"],
                "seed": seed,
                "steps": settings["steps"],
            },
        },
        # VAE Decode
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        # Save
        "8": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alfred_enhanced", "images": ["7", 0]},
        },
    }


def _build_svd_workflow(
    image_filename: str,
    motion_bucket_id: int = 127,
    fps: int = 6,
    video_frames: int = 25,
) -> dict | None:
    """Build SVD-XT video generation workflow from an image.

    Uses ImageOnlyCheckpointLoader -> SVD_img2vid_Conditioning ->
    VideoLinearCFGGuidance -> KSampler -> VAEDecode -> SaveImage.

    Returns None if no SVD model is available.
    """
    svd_ckpt = _get_svd_checkpoint()
    if not svd_ckpt:
        return None

    seed = random.randint(0, 2**32 - 1)

    return {
        # Load SVD checkpoint (returns MODEL, CLIP_VISION, VAE)
        "1": {
            "class_type": "ImageOnlyCheckpointLoader",
            "inputs": {"ckpt_name": svd_ckpt},
        },
        # Load the source image
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        # SVD conditioning (encodes image via CLIP vision + creates latent batch)
        "3": {
            "class_type": "SVD_img2vid_Conditioning",
            "inputs": {
                "init_image": ["2", 0],
                "vae": ["1", 2],
                "clip_vision": ["1", 1],
                "width": 1024,
                "height": 576,
                "video_frames": video_frames,
                "motion_bucket_id": motion_bucket_id,
                "fps": fps,
                "augmentation_level": 0.0,
            },
        },
        # Apply linear CFG guidance for temporal consistency
        "4": {
            "class_type": "VideoLinearCFGGuidance",
            "inputs": {
                "model": ["1", 0],
                "min_cfg": 1.0,
            },
        },
        # KSampler
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["3", 0],
                "negative": ["3", 1],
                "latent_image": ["3", 2],
                "seed": seed,
                "steps": 20,
                "cfg": 2.5,
                "sampler_name": "euler",
                "scheduler": "karras",
                "denoise": 1.0,
            },
        },
        # VAE Decode
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        # Save frames as image sequence
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"svd_clip_{motion_bucket_id}",
                "images": ["6", 0],
            },
        },
    }


async def _ensure_comfyui_running(heavy: bool = False):
    """Start ComfyUI via GPU manager if not already running.

    Args:
        heavy: If True, this is a heavy workload (upscaling) that may need
               to evict other GPU services for VRAM headroom.
    """
    try:
        from integrations.gpu_manager import ensure_running
        result = await ensure_running("comfyui", heavy=heavy)
        if not result["success"]:
            return False, result.get("error", "Failed to start ComfyUI")
        return True, None
    except ImportError:
        # GPU manager not available, check directly
        status = await check_status()
        if status["running"]:
            return True, None
        return False, "ComfyUI is not running"


def _copy_image_to_comfyui_input(source_path: str) -> str:
    """Copy an image to ComfyUI's input directory. Returns the filename."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Use a unique name to avoid collisions
    dest_name = f"product_{uuid.uuid4().hex[:8]}{src.suffix}"
    dest = COMFYUI_INPUT_DIR / dest_name
    shutil.copy2(src, dest)
    logger.info(f"Copied {src.name} -> {dest}")
    return dest_name


async def _frames_to_video_ffmpeg(frame_paths: list[str], output_path: str, fps: int = 6) -> bool:
    """Convert a sequence of frame images to an mp4 video using ffmpeg."""
    if not frame_paths:
        return False

    # Create a temporary file list for ffmpeg concat demuxer
    list_path = Path(output_path).parent / f"_frames_{uuid.uuid4().hex[:8]}.txt"
    try:
        with open(list_path, "w") as f:
            for frame in sorted(frame_paths):
                f.write(f"file '{frame}'\n")
                f.write(f"duration {1.0 / fps}\n")

        cmd = (
            f"ffmpeg -y -f concat -safe 0 -i '{list_path}' "
            f"-vf 'fps={fps}' -c:v libx264 -preset medium -crf 18 "
            f"-pix_fmt yuv420p '{output_path}'"
        )
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"ffmpeg frames->video failed: {stderr.decode()}")
            return False
        logger.info(f"Created video: {output_path}")
        return True
    finally:
        list_path.unlink(missing_ok=True)


async def _stitch_videos_ffmpeg(video_paths: list[str], output_path: str) -> bool:
    """Concatenate multiple mp4 clips into a single video using ffmpeg."""
    if len(video_paths) == 1:
        shutil.copy2(video_paths[0], output_path)
        return True

    list_path = Path(output_path).parent / f"_concat_{uuid.uuid4().hex[:8]}.txt"
    try:
        with open(list_path, "w") as f:
            for vp in video_paths:
                f.write(f"file '{vp}'\n")

        cmd = (
            f"ffmpeg -y -f concat -safe 0 -i '{list_path}' "
            f"-c copy '{output_path}'"
        )
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"ffmpeg stitch failed: {stderr.decode()}")
            return False
        logger.info(f"Stitched {len(video_paths)} clips -> {output_path}")
        return True
    finally:
        list_path.unlink(missing_ok=True)


async def _submit_and_wait(
    workflow: dict,
    save_node: str = "9",
    timeout_polls: int = 360,
) -> dict:
    """Submit a workflow to ComfyUI and poll until completion.

    Returns dict with: success, images (list of {filename, subfolder}), error
    """
    client_id = str(uuid.uuid4())

    try:
        async with aiohttp.ClientSession() as session:
            # Queue the prompt
            async with session.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return {"success": False, "error": f"Failed to queue prompt: {error_text}"}
                result = await resp.json()
                prompt_id = result["prompt_id"]

            logger.info(f"ComfyUI prompt queued: {prompt_id}")

            # Poll for completion
            for _ in range(timeout_polls):
                await asyncio.sleep(0.5)
                async with session.get(f"{COMFYUI_URL}/history/{prompt_id}") as resp:
                    if resp.status != 200:
                        continue
                    history = await resp.json()
                    if prompt_id in history:
                        status_info = history[prompt_id].get("status", {})
                        if status_info.get("status_str") == "error":
                            messages = status_info.get("messages", [])
                            return {"success": False, "error": f"Generation failed: {messages}"}

                        outputs = history[prompt_id].get("outputs", {})
                        if save_node in outputs and outputs[save_node].get("images"):
                            images = outputs[save_node]["images"]
                            return {"success": True, "images": images}

            return {"success": False, "error": "Generation timed out"}

    except aiohttp.ClientConnectorError:
        return {"success": False, "error": "ComfyUI is not reachable at port 8188"}
    except Exception as e:
        logger.error(f"ComfyUI error: {e}")
        return {"success": False, "error": str(e)}


async def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    mode: str = "quality",
    upscale: bool = False,
    model: str = "juggernaut",
) -> dict:
    """Generate an image using ComfyUI.

    Args:
        prompt: Text description of the image to generate
        width: Image width (default 1024)
        height: Image height (default 1024)
        mode: "quality" (hi-res fix) or "fast" (SDXL Turbo). Ignored when model="flux".
        upscale: Whether to apply 4x upscaling (slower, higher res output)
        model: "juggernaut" (default, SDXL) or "flux" (FLUX.1 dev FP8)

    Returns:
        dict with keys: success, image_path, base64, error
    """
    # Validate model choice
    if model == "flux" and not _has_model("checkpoints", FLUX_CHECKPOINT):
        return {"success": False, "error": "FLUX model not installed"}

    # Ensure ComfyUI is running (starts it on-demand if needed)
    # heavy=True for upscaling, quality mode, or FLUX (large model needs VRAM headroom)
    running, error = await _ensure_comfyui_running(heavy=upscale or mode == "quality" or model == "flux")
    if not running:
        return {"success": False, "error": error}

    # Build workflow based on model choice
    if model == "flux":
        workflow = _build_flux_workflow(prompt, width, height, upscale=upscale)
    else:
        workflow = _build_quality_workflow(prompt, width, height, upscale=upscale, mode=mode)

    # Figure out which checkpoint we're actually using for logging
    ckpt = workflow["4"]["inputs"]["ckpt_name"]
    steps = workflow["3"]["inputs"]["steps"]
    hires = "13" in workflow
    logger.info(
        f"Generating image: model={ckpt}, steps={steps}, "
        f"{width}x{height}, hires_fix={hires}, upscale={upscale}"
    )

    # FLUX is slower than SDXL — give it more time
    if model == "flux":
        timeout_polls = 600
    elif mode == "quality":
        timeout_polls = 360
    else:
        timeout_polls = 120
    if upscale:
        timeout_polls = 600

    result = await _submit_and_wait(workflow, save_node="9", timeout_polls=timeout_polls)

    if not result["success"]:
        return result

    # Process the first output image
    image_info = result["images"][0]
    filename = image_info["filename"]
    subfolder = image_info.get("subfolder", "")

    try:
        async with aiohttp.ClientSession() as session:
            params = {"filename": filename, "subfolder": subfolder, "type": "output"}
            async with session.get(f"{COMFYUI_URL}/view", params=params) as img_resp:
                if img_resp.status == 200:
                    image_data = await img_resp.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                    # Save to Alfred's generated folder
                    from core.tools.files import GENERATED_DIR
                    save_path = GENERATED_DIR / filename
                    save_path.write_bytes(image_data)

                    # Touch GPU manager to reset idle timer
                    try:
                        from integrations.gpu_manager import touch
                        await touch("comfyui")
                    except ImportError:
                        pass

                    logger.info(f"Image generated: {filename} ({len(image_data)} bytes)")
                    return {
                        "success": True,
                        "filename": filename,
                        "image_path": str(save_path),
                        "base64": image_base64,
                        "download_url": f"/download/{filename}",
                        "model": ckpt,
                        "upscaled": upscale,
                        "hires_fix": hires,
                    }

        return {"success": False, "error": "Failed to retrieve generated image"}

    except aiohttp.ClientConnectorError:
        return {"success": False, "error": "ComfyUI is not reachable at port 8188"}
    except Exception as e:
        logger.error(f"ComfyUI error: {e}")
        return {"success": False, "error": str(e)}


async def generate_product_video(
    image_path: str,
    prompt: str | None = None,
    clips: int = 3,
    enhance: bool = True,
    denoise_strength: float = 0.45,
) -> dict:
    """Generate a product showcase video from a product photo.

    Pipeline: product photo -> (optional) img2img enhance -> SVD-XT video clips -> ffmpeg stitch

    Args:
        image_path: Path to the source product image
        prompt: Optional prompt for image enhancement (defaults to product photography prompt)
        clips: Number of video clips to generate (1-3), each with different motion intensity
        enhance: Whether to enhance the product photo via img2img first
        denoise_strength: How much to alter during enhancement (0.3=subtle, 0.6=dramatic)

    Returns:
        dict with: success, video_path, download_url, clips_generated, enhanced, error
    """
    from core.tools.files import GENERATED_DIR

    clips = max(1, min(3, clips))
    denoise_strength = max(0.3, min(0.6, denoise_strength))

    # Check SVD model availability early
    svd_ckpt = _get_svd_checkpoint()
    if not svd_ckpt:
        return {
            "success": False,
            "error": (
                "SVD-XT model not found. Download it with:\n"
                "cd /home/aialfred/ComfyUI/models/checkpoints/ && "
                "wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1"
                "/resolve/main/svd_xt_1_1.safetensors"
            ),
        }

    # Ensure ComfyUI is running (SVD is heavy)
    running, error = await _ensure_comfyui_running(heavy=True)
    if not running:
        return {"success": False, "error": error}

    # Step 1: Copy source image to ComfyUI input
    try:
        input_filename = _copy_image_to_comfyui_input(image_path)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    # Step 2: (Optional) Enhance via img2img
    video_source_filename = input_filename
    enhanced_path = None

    if enhance:
        logger.info("Enhancing product photo via img2img...")
        enhance_workflow = _build_img2img_workflow(
            input_filename, prompt=prompt, denoise_strength=denoise_strength
        )
        enhance_result = await _submit_and_wait(
            enhance_workflow, save_node="8", timeout_polls=360
        )

        if enhance_result["success"] and enhance_result.get("images"):
            enhanced_info = enhance_result["images"][0]
            enhanced_filename = enhanced_info["filename"]
            enhanced_subfolder = enhanced_info.get("subfolder", "")

            # Download enhanced image and copy to input dir for SVD
            try:
                async with aiohttp.ClientSession() as session:
                    params = {"filename": enhanced_filename, "subfolder": enhanced_subfolder, "type": "output"}
                    async with session.get(f"{COMFYUI_URL}/view", params=params) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            # Save to generated dir
                            enhanced_path = GENERATED_DIR / enhanced_filename
                            enhanced_path.write_bytes(data)
                            # Copy to ComfyUI input for SVD to use
                            dest = COMFYUI_INPUT_DIR / enhanced_filename
                            dest.write_bytes(data)
                            video_source_filename = enhanced_filename
                            logger.info(f"Enhanced image saved: {enhanced_filename}")
            except Exception as e:
                logger.warning(f"Failed to retrieve enhanced image, using original: {e}")
        else:
            logger.warning(f"Enhancement failed ({enhance_result.get('error')}), using original image")

    # Step 3: Generate video clips with different motion intensities
    motion_levels = {
        1: [127],                   # Single moderate clip
        2: [60, 200],               # Subtle + dramatic
        3: [60, 140, 200],          # Subtle + moderate + dramatic
    }
    motion_buckets = motion_levels[clips]

    clip_frame_paths: list[list[str]] = []
    clip_video_paths: list[str] = []

    for i, motion_bucket_id in enumerate(motion_buckets):
        logger.info(f"Generating SVD clip {i+1}/{clips} (motion_bucket_id={motion_bucket_id})...")

        svd_workflow = _build_svd_workflow(
            video_source_filename,
            motion_bucket_id=motion_bucket_id,
            fps=6,
            video_frames=25,
        )
        if svd_workflow is None:
            return {"success": False, "error": "SVD model disappeared during generation"}

        # SVD takes longer — generous timeout (600 polls = 5 min)
        svd_result = await _submit_and_wait(svd_workflow, save_node="7", timeout_polls=600)

        if not svd_result["success"]:
            logger.error(f"SVD clip {i+1} failed: {svd_result.get('error')}")
            continue

        # Collect frame paths from ComfyUI output
        frames = []
        for img_info in svd_result["images"]:
            fname = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            frame_path = OUTPUT_DIR / subfolder / fname if subfolder else OUTPUT_DIR / fname
            if frame_path.exists():
                frames.append(str(frame_path))

        if not frames:
            logger.warning(f"SVD clip {i+1}: no frames found in output")
            continue

        clip_frame_paths.append(frames)

        # Touch GPU manager between clips
        try:
            from integrations.gpu_manager import touch
            await touch("comfyui")
        except ImportError:
            pass

    if not clip_frame_paths:
        return {"success": False, "error": "All video clip generations failed"}

    # Step 4: Convert frame sequences to mp4 clips
    for i, frames in enumerate(clip_frame_paths):
        clip_path = str(GENERATED_DIR / f"product_clip_{i}_{uuid.uuid4().hex[:6]}.mp4")
        success = await _frames_to_video_ffmpeg(frames, clip_path, fps=6)
        if success:
            clip_video_paths.append(clip_path)
        else:
            logger.warning(f"Failed to convert clip {i} frames to video")

    if not clip_video_paths:
        return {"success": False, "error": "Failed to convert any frame sequences to video"}

    # Step 5: Stitch all clips into final video
    final_filename = f"product_video_{uuid.uuid4().hex[:8]}.mp4"
    final_path = str(GENERATED_DIR / final_filename)

    if len(clip_video_paths) == 1:
        shutil.copy2(clip_video_paths[0], final_path)
    else:
        stitched = await _stitch_videos_ffmpeg(clip_video_paths, final_path)
        if not stitched:
            # Fall back to first clip
            shutil.copy2(clip_video_paths[0], final_path)
            final_filename = Path(clip_video_paths[0]).name

    # Clean up intermediate clip files
    for cp in clip_video_paths:
        Path(cp).unlink(missing_ok=True)

    logger.info(f"Product video complete: {final_filename} ({len(clip_video_paths)} clips)")

    return {
        "success": True,
        "video_path": final_path,
        "filename": final_filename,
        "download_url": f"/download/{final_filename}",
        "clips_generated": len(clip_video_paths),
        "enhanced": enhanced_path is not None,
        "enhanced_image": str(enhanced_path) if enhanced_path else None,
        "svd_model": svd_ckpt,
    }


async def generate_with_controlnet(
    prompt: str,
    image_path: str,
    control_type: str = "canny",
    strength: float = 0.8,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Generate an image guided by a ControlNet condition image.

    Takes a reference image (photo, sketch, depth map) and uses it to guide
    the composition of the generated image while applying the text prompt.

    Args:
        prompt: Text description of the desired output
        image_path: Path to the guide image (will be preprocessed based on control_type)
        control_type: "canny" (edges) or "depth" (depth map)
        strength: ControlNet influence 0.0-1.0 (default 0.8)
        width: Output width (snapped to SDXL bucket)
        height: Output height (snapped to SDXL bucket)

    Returns:
        dict with: success, filename, image_path, base64, download_url, error
    """
    controlnet_models = {
        "canny": "controlnet-canny-sdxl.safetensors",
        "depth": "controlnet-depth-sdxl.safetensors",
    }

    if control_type not in controlnet_models:
        return {"success": False, "error": f"Unknown control_type: {control_type}. Use 'canny' or 'depth'."}

    cn_model = controlnet_models[control_type]
    if not _has_model("controlnet", cn_model):
        return {"success": False, "error": f"ControlNet model not found: {cn_model}"}

    running, error = await _ensure_comfyui_running(heavy=True)
    if not running:
        return {"success": False, "error": error}

    # Copy guide image to ComfyUI input
    try:
        input_filename = _copy_image_to_comfyui_input(image_path)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    width, height = _snap_to_sdxl_resolution(width, height)
    enhanced_prompt = _enhance_prompt(prompt)
    ckpt, settings = _select_checkpoint()
    seed = random.randint(0, 2**32 - 1)
    strength = max(0.0, min(1.0, strength))

    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        # Load ControlNet model
        "2": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": cn_model},
        },
        # Load guide image
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": input_filename},
        },
        # Positive prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": enhanced_prompt},
        },
        # Negative prompt
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": NEGATIVE_PROMPT},
        },
        # Empty latent
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": height, "width": width},
        },
    }

    # Add Canny preprocessor if using canny mode (extracts edges from image)
    if control_type == "canny":
        workflow["7"] = {
            "class_type": "Canny",
            "inputs": {"image": ["3", 0], "low_threshold": 0.4, "high_threshold": 0.8},
        }
        cn_image_ref = ["7", 0]
    else:
        # For depth, use the image directly (user provides depth map or we use as-is)
        cn_image_ref = ["3", 0]

    # Apply ControlNet
    workflow["8"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": ["4", 0],
            "negative": ["5", 0],
            "control_net": ["2", 0],
            "image": cn_image_ref,
            "strength": strength,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }

    # KSampler
    workflow["9"] = {
        "class_type": "KSampler",
        "inputs": {
            "cfg": settings["cfg"],
            "denoise": 1.0,
            "latent_image": ["6", 0],
            "model": ["1", 0],
            "negative": ["8", 1],
            "positive": ["8", 0],
            "sampler_name": settings["sampler"],
            "scheduler": settings["scheduler"],
            "seed": seed,
            "steps": settings["steps"],
        },
    }

    # VAE Decode + Save
    workflow["10"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["1", 2]},
    }
    workflow["11"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "alfred_cn", "images": ["10", 0]},
    }

    logger.info(f"ControlNet generation: type={control_type}, strength={strength}, model={ckpt}")

    result = await _submit_and_wait(workflow, save_node="11", timeout_polls=360)
    if not result["success"]:
        return result

    # Fetch the output image
    image_info = result["images"][0]
    filename = image_info["filename"]
    subfolder = image_info.get("subfolder", "")

    try:
        async with aiohttp.ClientSession() as session:
            params = {"filename": filename, "subfolder": subfolder, "type": "output"}
            async with session.get(f"{COMFYUI_URL}/view", params=params) as img_resp:
                if img_resp.status == 200:
                    image_data = await img_resp.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                    from core.tools.files import GENERATED_DIR
                    save_path = GENERATED_DIR / filename
                    save_path.write_bytes(image_data)
                    try:
                        from integrations.gpu_manager import touch
                        await touch("comfyui")
                    except ImportError:
                        pass
                    return {
                        "success": True,
                        "filename": filename,
                        "image_path": str(save_path),
                        "base64": image_base64,
                        "download_url": f"/download/{filename}",
                        "control_type": control_type,
                        "model": ckpt,
                    }
        return {"success": False, "error": "Failed to retrieve generated image"}
    except Exception as e:
        logger.error(f"ControlNet error: {e}")
        return {"success": False, "error": str(e)}


async def generate_with_style_reference(
    prompt: str,
    reference_image_path: str,
    weight: float = 0.8,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Generate an image using a reference image for style/subject guidance (IP-Adapter).

    The reference image influences the style, colors, and subject appearance of
    the generated image while the text prompt controls the content.

    Args:
        prompt: Text description of the desired output
        reference_image_path: Path to the style reference image
        weight: How strongly the reference influences the output 0.0-1.0 (default 0.8)
        width: Output width (snapped to SDXL bucket)
        height: Output height (snapped to SDXL bucket)

    Returns:
        dict with: success, filename, image_path, base64, download_url, error
    """
    ipadapter_model = "ip-adapter-plus_sdxl_vit-h.safetensors"
    clip_vision_model = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

    if not _has_model("ipadapter", ipadapter_model):
        return {"success": False, "error": f"IP-Adapter model not found: {ipadapter_model}"}
    if not _has_model("clip_vision", clip_vision_model):
        return {"success": False, "error": f"CLIP Vision model not found: {clip_vision_model}"}

    running, error = await _ensure_comfyui_running(heavy=True)
    if not running:
        return {"success": False, "error": error}

    try:
        input_filename = _copy_image_to_comfyui_input(reference_image_path)
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    width, height = _snap_to_sdxl_resolution(width, height)
    enhanced_prompt = _enhance_prompt(prompt)
    ckpt, settings = _select_checkpoint()
    seed = random.randint(0, 2**32 - 1)
    weight = max(0.0, min(1.0, weight))

    workflow = {
        # Load checkpoint
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        # Load IP-Adapter model
        "2": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": ipadapter_model},
        },
        # Load CLIP Vision
        "3": {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": clip_vision_model},
        },
        # Load reference image
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": input_filename},
        },
        # Prep image for CLIP Vision
        "5": {
            "class_type": "PrepImageForClipVision",
            "inputs": {
                "image": ["4", 0],
                "interpolation": "LANCZOS",
                "crop_position": "center",
                "sharpening": 0.0,
            },
        },
        # Apply IP-Adapter
        "6": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["2", 0],
                "clip_vision": ["3", 0],
                "image": ["5", 0],
                "weight": weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "embeds_scaling": "V only",
                "start_at": 0.0,
                "end_at": 1.0,
                "unfold_batch": False,
            },
        },
        # Positive prompt
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": enhanced_prompt},
        },
        # Negative prompt
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": NEGATIVE_PROMPT},
        },
        # Empty latent
        "9": {
            "class_type": "EmptyLatentImage",
            "inputs": {"batch_size": 1, "height": height, "width": width},
        },
        # KSampler (uses IP-Adapter modified model from node 6)
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": settings["cfg"],
                "denoise": 1.0,
                "latent_image": ["9", 0],
                "model": ["6", 0],
                "negative": ["8", 0],
                "positive": ["7", 0],
                "sampler_name": settings["sampler"],
                "scheduler": settings["scheduler"],
                "seed": seed,
                "steps": settings["steps"],
            },
        },
        # VAE Decode + Save
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "alfred_ipa", "images": ["11", 0]},
        },
    }

    logger.info(f"IP-Adapter generation: weight={weight}, model={ckpt}")

    result = await _submit_and_wait(workflow, save_node="12", timeout_polls=360)
    if not result["success"]:
        return result

    image_info = result["images"][0]
    filename = image_info["filename"]
    subfolder = image_info.get("subfolder", "")

    try:
        async with aiohttp.ClientSession() as session:
            params = {"filename": filename, "subfolder": subfolder, "type": "output"}
            async with session.get(f"{COMFYUI_URL}/view", params=params) as img_resp:
                if img_resp.status == 200:
                    image_data = await img_resp.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                    from core.tools.files import GENERATED_DIR
                    save_path = GENERATED_DIR / filename
                    save_path.write_bytes(image_data)
                    try:
                        from integrations.gpu_manager import touch
                        await touch("comfyui")
                    except ImportError:
                        pass
                    return {
                        "success": True,
                        "filename": filename,
                        "image_path": str(save_path),
                        "base64": image_base64,
                        "download_url": f"/download/{filename}",
                        "model": ckpt,
                    }
        return {"success": False, "error": "Failed to retrieve generated image"}
    except Exception as e:
        logger.error(f"IP-Adapter error: {e}")
        return {"success": False, "error": str(e)}


async def check_status() -> dict:
    """Check if ComfyUI is running and ready."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{COMFYUI_URL}/system_stats",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    stats = await resp.json()
                    return {"running": True, "stats": stats}
    except Exception:
        pass
    return {"running": False}


async def list_models() -> dict:
    """List available models in ComfyUI directories."""
    result = {}
    for subdir in ["checkpoints", "upscale_models", "loras", "controlnet"]:
        model_dir = MODELS_DIR / subdir
        if model_dir.exists():
            files = [
                f.name for f in model_dir.iterdir()
                if f.is_file() and not f.name.startswith("put_")
            ]
            if files:
                result[subdir] = files
    return result
