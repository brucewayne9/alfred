# core/casting/deploy.py
from __future__ import annotations
import subprocess
from pathlib import Path
from config.settings import settings

class DeployError(RuntimeError):
    pass

def _sh(cmd: list[str], timeout: int = 120) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise DeployError(f"cmd failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr}")
    return r.stdout

def _copy_wav(local: str, remote_name: str) -> None:
    host = settings.casting_ssh_host
    remote = f"{host}:{settings.casting_engine_voices_dir}/{remote_name}"
    # copy to a temp on 98 then sudo-move into the engine dir (engine dir is root-owned)
    tmp_remote = f"/tmp/{remote_name}"
    _sh(["timeout", "120", "scp", local, f"{host}:{tmp_remote}"])
    _sh(["timeout", "60", "ssh", host, "sudo", "docker", "cp", tmp_remote,
         f"azuracast:{settings.casting_engine_voices_dir}/{remote_name}"])

def _upsert_break(*, base_name: str, persona_prompt: str, station_id: int) -> None:
    host = settings.casting_ssh_host
    pw = settings.casting_az_db_pass
    safe_persona = persona_prompt.replace("\\", "\\\\").replace("'", "''")
    voice = f"{base_name}_neutral"
    sql = (
        "INSERT INTO station_ai_dj_breaks "
        "(station_id, name, engine, voice_id, content_template, is_enabled, trigger_value) "
        f"VALUES ({station_id}, '{base_name} Show', 'qwen', '{voice}', '{safe_persona}', 1, 13) "
        "ON DUPLICATE KEY UPDATE voice_id=VALUES(voice_id), "
        "content_template=VALUES(content_template), is_enabled=1;"
    )
    _sh(["timeout", "60", "ssh", host, "sudo", "docker", "exec", "azuracast",
         "mariadb", "-u", "azuracast", f"-p{pw}", "azuracast", "-e", sql])

def deploy_dj(*, dj_id: int, base_name: str, moods: list[str], persona_prompt: str,
              station_id: int) -> None:
    vdir = Path(settings.casting_voices_dir) / str(dj_id)
    for mood in moods:
        local = vdir / f"{mood}.wav"
        if not local.exists():
            raise DeployError(f"missing mood clip: {local}")
        _copy_wav(str(local), f"{base_name}_{mood}.wav")
    _upsert_break(base_name=base_name, persona_prompt=persona_prompt, station_id=station_id)
