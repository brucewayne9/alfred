# core/casting/deploy.py
"""Central Casting → AzuraCast deploy bridge.

Writes/updates rows in the live `station_ai_dj_breaks` table on server 98
(station 22) to match the REAL schema confirmed against a working qwen
solo-jock row (Sloan).

Key facts about the real schema:
- There is NO unique key on (station_id, name) — `id` is the only PRIMARY key.
  So we cannot use `ON DUPLICATE KEY UPDATE`. Instead we SELECT the row by
  (station_id, name); if found we UPDATE by id, else we INSERT.
- `tts_voice` is a NAME the 105:7860 Qwen server resolves (e.g. cc7_neutral),
  NOT a filesystem path. Voice clips are registered to 105's local Qwen
  resources dir via `core.casting.voice.register_to_engine` — a normal local
  file copy on the same box as Alfred Labs, NOT an ssh round-trip to 98.
- `tts_settings` is NOT NULL (longtext). We always write a safe default of
  '{}' — it may be enriched later (speed/etc).
- `is_enabled` defaults to 0 (False) for safe staged deploys: the row exists
  but does not air until an operator enables it.
"""
from __future__ import annotations
import re, subprocess
from config.settings import settings
from core.casting import voice


class DeployError(RuntimeError):
    pass


def _sh(cmd: list[str], timeout: int = 120) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise DeployError(f"cmd failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def _esc(s: str) -> str:
    """Escape a string for use inside a single-quoted SQL literal."""
    return s.replace("\\", "\\\\").replace("'", "''")


def slot_to_times(slot: str) -> tuple[str, str]:
    """'10a-2p' or '10:00-14:00' -> ('10:00','14:00'). Raise ValueError if unparseable."""
    slot = slot.strip()
    # Format 1: HH:MM-HH:MM
    m = re.fullmatch(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})", slot)
    if m:
        sh, sm, eh, em = m.groups()
        return f"{int(sh):02d}:{sm}", f"{int(eh):02d}:{em}"

    # Format 2: am/pm shorthand, e.g. '10a-2p', '6p-10p', '12a-12p'
    m = re.fullmatch(r"(\d{1,2})(a|p)-(\d{1,2})(a|p)", slot, re.IGNORECASE)
    if m:
        sh, sap, eh, eap = m.groups()
        return f"{_ampm_to_24(int(sh), sap):02d}:00", f"{_ampm_to_24(int(eh), eap):02d}:00"

    raise ValueError(f"unparseable slot: {slot!r}")


def _ampm_to_24(hour: int, ap: str) -> int:
    if not 1 <= hour <= 12:
        raise ValueError(f"hour out of range: {hour}")
    ap = ap.lower()
    if ap == "a":
        return 0 if hour == 12 else hour
    # pm
    return 12 if hour == 12 else hour + 12


def deploy_dj(*, dj_id: int, dj_name: str, moods: list[str], persona_prompt: str,
              station_id: int, schedule_start: str, schedule_end: str,
              rss_feeds: list[str] | None = None, trigger_value: str = "13",
              enabled: bool = False) -> None:
    """Register the DJ's voice on 105 and upsert its break row on 98.

    Voices are copied locally (105) into the Qwen resources dir; the break row
    references the deployed voice by NAME (cc<dj_id>_neutral). The row is
    written DISABLED by default (scratch-safe) — an operator enables it to air.
    """
    # 1. Register voice clips locally on 105 (no ssh — same box as Alfred Labs)
    voice.register_to_engine(dj_id, moods)
    tts_voice = voice.engine_voice_name(dj_id, "neutral")

    row_name = f"{dj_name} ({schedule_start}-{schedule_end})"

    # Escape every string literal that goes into SQL
    e_name = _esc(row_name)
    e_template = _esc(persona_prompt)
    e_voice = _esc(tts_voice)
    e_folder = _esc("beds")
    rss_text = "\n".join(rss_feeds or [])
    e_rss = _esc(rss_text)
    use_rss = 1 if rss_feeds else 0
    is_enabled = 1 if enabled else 0

    host = settings.casting_ssh_host
    pw = settings.casting_az_db_pass

    def _exec(sql: str) -> str:
        return _sh([
            "timeout", "60", "ssh", host,
            "sudo", "docker", "exec", "azuracast",
            "mariadb", "-u", "azuracast", f"-p{pw}", "azuracast",
            "-N", "-e", sql,
        ])

    # 2. SELECT existing row by (station_id, name)
    select_sql = (
        f"SELECT id FROM station_ai_dj_breaks "
        f"WHERE station_id={station_id} AND name='{e_name}' LIMIT 1;"
    )
    out = _exec(select_sql)
    existing_id = None
    for tok in out.split():
        if tok.isdigit():
            existing_id = int(tok)
            break

    if existing_id is not None:
        # 3a. UPDATE the existing row by id
        update_sql = (
            f"UPDATE station_ai_dj_breaks SET "
            f"tts_voice='{e_voice}', "
            f"content_template='{e_template}', "
            f"is_enabled={is_enabled}, "
            f"schedule_start_time='{_esc(schedule_start)}', "
            f"schedule_end_time='{_esc(schedule_end)}', "
            f"trigger_value='{_esc(trigger_value)}', "
            f"use_instrumental_bed=1, "
            f"instrumental_folder='{e_folder}', "
            f"tts_provider='qwen', "
            f"content_source='ai_generated', "
            f"trigger_type='time_based', "
            f"is_dual_host=0, "
            f"use_rss_feeds={use_rss}, "
            f"rss_feed_urls='{e_rss}', "
            f"tts_settings='{{}}' "
            f"WHERE id={existing_id};"
        )
        _exec(update_sql)
    else:
        # 3b. INSERT the full working column set
        insert_sql = (
            "INSERT INTO station_ai_dj_breaks "
            "(station_id, name, is_enabled, trigger_type, trigger_value, "
            "content_source, tts_provider, tts_voice, content_template, "
            "tts_settings, is_dual_host, use_instrumental_bed, instrumental_folder, "
            "use_rss_feeds, rss_feed_urls, schedule_start_time, schedule_end_time) "
            f"VALUES ({station_id}, '{e_name}', {is_enabled}, 'time_based', "
            f"'{_esc(trigger_value)}', 'ai_generated', 'qwen', '{e_voice}', "
            f"'{e_template}', '{{}}', 0, 1, '{e_folder}', {use_rss}, '{e_rss}', "
            f"'{_esc(schedule_start)}', '{_esc(schedule_end)}');"
        )
        _exec(insert_sql)
