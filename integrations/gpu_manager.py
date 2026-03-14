"""GPU Service Manager - On-demand start/stop of GPU-intensive services.

Manages VRAM budget by starting services only when needed and stopping
them when idle. Services managed: comfyui, qwen3-tts.

Strategy: "try coexist, evict only if necessary"
  - ComfyUI runs with --lowvram so it streams model chunks through VRAM
  - Both services CAN coexist for standard generation tasks
  - Only evict another service for heavy workloads (upscaling) or on OOM
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Service definitions: name -> config
GPU_SERVICES = {
    "comfyui": {
        "systemd_unit": "comfyui",
        "vram_mb": 5000,  # Juggernaut XL with lowvram streaming
        "vram_heavy_mb": 10000,  # FLUX FP8 / upscaling / SVD-XT video gen (needs most of VRAM)
        "health_url": "http://127.0.0.1:8188/system_stats",
        "startup_timeout": 30,
        "idle_timeout": 0,  # Disabled — ComfyUI uses ~154MB idle, not worth stopping
    },
    "qwen3-tts": {
        "systemd_unit": "qwen3-tts",
        "vram_mb": 3600,
        "vram_heavy_mb": 3600,
        "health_url": "http://127.0.0.1:7860/docs",
        "startup_timeout": 45,
        "idle_timeout": 300,
    },
}

TOTAL_VRAM_MB = 12282
# Reserve for Alfred API + system overhead
RESERVED_VRAM_MB = 1500

# Track last-used timestamps
_last_used: dict[str, float] = {}
_locks: dict[str, asyncio.Lock] = {}


def _get_lock(service: str) -> asyncio.Lock:
    if service not in _locks:
        _locks[service] = asyncio.Lock()
    return _locks[service]


async def _run_cmd(cmd: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    output = (stdout or b"").decode() + (stderr or b"").decode()
    return proc.returncode, output.strip()


async def _get_real_vram_free() -> int:
    """Get actual free VRAM from nvidia-smi (not estimates)."""
    code, output = await _run_cmd(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
    )
    if code == 0:
        try:
            return int(output.strip())
        except ValueError:
            pass
    return TOTAL_VRAM_MB - RESERVED_VRAM_MB


async def is_running(service: str) -> bool:
    """Check if a GPU service is currently active."""
    config = GPU_SERVICES.get(service)
    if not config:
        return False
    code, _ = await _run_cmd(f"systemctl is-active {config['systemd_unit']}")
    return code == 0


async def get_status() -> dict:
    """Get status of all GPU services and real VRAM usage."""
    services = {}
    for name, config in GPU_SERVICES.items():
        running = await is_running(name)
        services[name] = {
            "running": running,
            "last_used": _last_used.get(name),
        }

    real_free = await _get_real_vram_free()
    return {
        "services": services,
        "vram_total_mb": TOTAL_VRAM_MB,
        "vram_free_mb": real_free,
    }


async def _wait_for_health(service: str) -> bool:
    """Wait for service health endpoint to respond."""
    import aiohttp

    config = GPU_SERVICES[service]
    url = config["health_url"]
    deadline = time.time() + config["startup_timeout"]

    while time.time() < deadline:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def ensure_running(service: str, heavy: bool = False) -> dict:
    """Ensure a GPU service is running. Starts it if needed, managing VRAM.

    Args:
        service: Service name ("comfyui" or "qwen3-tts")
        heavy: If True, this is a heavy workload (e.g. upscaling) that needs
               more VRAM. Will evict other services if needed. If False,
               tries to coexist with whatever else is running.

    Returns dict with: success, already_running, started, stopped_services, error
    """
    if service not in GPU_SERVICES:
        return {"success": False, "error": f"Unknown service: {service}"}

    async with _get_lock(service):
        if await is_running(service):
            _last_used[service] = time.time()
            # If heavy mode requested, check if we need to evict others
            if heavy:
                real_free = await _get_real_vram_free()
                config = GPU_SERVICES[service]
                # Need enough headroom for heavy workload
                needed_extra = config["vram_heavy_mb"] - config["vram_mb"]
                if real_free < needed_extra:
                    stopped = await _evict_others(service)
                    return {"success": True, "already_running": True, "stopped_services": stopped}
            return {"success": True, "already_running": True}

        config = GPU_SERVICES[service]
        needed = config["vram_heavy_mb"] if heavy else config["vram_mb"]

        real_free = await _get_real_vram_free()
        stopped = []

        if real_free < needed:
            # Need to evict other services
            stopped = await _evict_others(service)
            # Re-check after eviction
            real_free = await _get_real_vram_free()
            if real_free < needed:
                return {
                    "success": False,
                    "error": f"Not enough VRAM: need ~{needed}MB, have {real_free}MB free",
                }

        # Start the service
        code, output = await _run_cmd(f"sudo systemctl start {config['systemd_unit']}")
        if code != 0:
            return {"success": False, "error": f"Failed to start: {output}"}

        # Wait for health
        healthy = await _wait_for_health(service)
        if not healthy:
            return {"success": False, "error": f"{service} started but health check timed out"}

        _last_used[service] = time.time()
        logger.info(f"GPU service {service} started (stopped: {stopped})")

        return {
            "success": True,
            "already_running": False,
            "started": True,
            "stopped_services": stopped,
        }


async def _evict_others(keep: str) -> list[str]:
    """Stop other GPU services to free VRAM. Returns list of stopped service names."""
    stopped = []
    for name in GPU_SERVICES:
        if name == keep:
            continue
        if await is_running(name):
            logger.info(f"Evicting {name} to free VRAM for {keep}")
            await stop(name)
            stopped.append(name)
            # Give GPU a moment to release memory
            await asyncio.sleep(2)
    return stopped


async def stop(service: str) -> dict:
    """Stop a GPU service to free VRAM."""
    if service not in GPU_SERVICES:
        return {"success": False, "error": f"Unknown service: {service}"}

    config = GPU_SERVICES[service]

    if not await is_running(service):
        return {"success": True, "already_stopped": True}

    code, output = await _run_cmd(f"sudo systemctl stop {config['systemd_unit']}")
    if code != 0:
        return {"success": False, "error": f"Failed to stop: {output}"}

    _last_used.pop(service, None)
    logger.info(f"GPU service {service} stopped")
    return {"success": True}


async def touch(service: str):
    """Mark a service as recently used (resets idle timer)."""
    _last_used[service] = time.time()


async def cleanup_idle():
    """Stop services that have been idle beyond their timeout. Call periodically."""
    now = time.time()
    stopped = []
    for name, config in GPU_SERVICES.items():
        if not await is_running(name):
            continue
        last = _last_used.get(name, 0)
        if config["idle_timeout"] <= 0:
            continue  # idle_timeout=0 means never auto-stop
        if last and (now - last) > config["idle_timeout"]:
            logger.info(f"Auto-stopping idle service: {name} (idle {now - last:.0f}s)")
            await stop(name)
            stopped.append(name)
    return stopped
