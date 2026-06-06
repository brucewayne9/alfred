# Central Casting v1 (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private, login-gated studio inside Alfred Labs to create AI radio host personalities (voice + persona), preview them off-air, and schedule them onto AzuraCast station 22 — replacing the manual email-and-hand-place workflow.

**Architecture:** A new backend module `core/casting/` (DB + voice processing + persona drafting + preview + AzuraCast deploy bridge) exposed through a `core/api/central_casting.py` router registered in `core/api/main.py`, plus React screens under `frontend/src/`. Reuses the Qwen3-TTS server (105:7860) for voice cloning/synthesis and the proven `ssh server-98` + `docker exec ... mariadb` PDO path to write `station_ai_dj_breaks` and place mood wavs.

**Tech Stack:** Python 3.11 / FastAPI (port 8400), raw `sqlite3` (WAL) following `core/forge/db.py`, Pydantic v2 models, `ffmpeg`/`ffprobe` for audio normalization, Ollama Kimi K2.6 (`105:11434/api/chat`) for persona drafting, Qwen3-TTS (`105:7860/synthesize_speech/`), React + Zustand + Vite frontend.

**Scope:** v1 = **hosts only**. Guests/bookings (v1.5), stock voice shelf (v2), and generate-a-voice-from-description (v3) are separate follow-on plans per the design spec.

---

## File Structure

**Backend (new):**
- `core/casting/__init__.py` — package marker
- `core/casting/db.py` — sqlite schema + CRUD (personalities, voices, assignments)
- `core/casting/models.py` — Pydantic request/response models + the 8 mood enum
- `core/casting/mood_pack.py` — the canonical 8-read Mood Pack script (constant)
- `core/casting/voice.py` — validate + normalize (ffmpeg) + store mood wavs into the library
- `core/casting/persona.py` — brief/archetype → Kimi K2.6 → persona prompt
- `core/casting/archetypes.py` — seed archetype shelf (constant list)
- `core/casting/preview.py` — render an off-air sample break in a DJ's voice
- `core/casting/deploy.py` — push wavs to server 98 + write the `station_ai_dj_breaks` row
- `core/casting/scheduler.py` — apply due assignments (called by cron)
- `core/api/central_casting.py` — FastAPI routes (`register(app)` pattern)

**Backend (modify):**
- `core/api/main.py` — register the router
- `config/settings.py` — add casting settings

**Data (new):**
- `data/casting.db` — library DB
- `data/casting/voices/<dj_id>/<mood>.wav` — stored mood clips
- `data/casting/previews/` — rendered preview mp3s

**Frontend (new):**
- `frontend/src/api/casting.ts` — API client functions
- `frontend/src/stores/castingStore.ts` — Zustand store
- `frontend/src/components/casting/CastingApp.tsx` — module shell + tab nav
- `frontend/src/components/casting/Library.tsx` — DJ grid
- `frontend/src/components/casting/CreateDJ.tsx` — intake wizard (voice + persona)
- `frontend/src/components/casting/DeploySchedule.tsx` — schedule a DJ to a slot

**Frontend (modify):**
- `frontend/src/components/layout/AppLayout.tsx` — add a "Central Casting" route/nav entry

**Tests (new):**
- `tests/casting/test_db.py`
- `tests/casting/test_voice.py`
- `tests/casting/test_persona.py`
- `tests/casting/test_deploy.py`
- `tests/casting/test_scheduler.py`
- `tests/casting/test_api.py`

---

## Conventions for this plan

- Run all Python test commands from repo root `/home/aialfred/alfred` using `venv/bin/pytest`.
- Mood vocabulary (must match the live engine's tag set): `neutral, fired, serious, amused, thoughtful, reactions, wry, intimate`. The 8 reference clips are stored as `<mood>.wav`; the engine on 98 expects files named `<Base>_<mood>.wav`, so the deploy step renames on copy.
- Never read env at import; always `from config.settings import settings` inside functions/at module top only for the singleton object (the project pattern). Wrap every `ssh`/`docker exec` in `timeout`.

---

## Task 1: Settings + package skeleton

**Files:**
- Create: `core/casting/__init__.py`
- Modify: `config/settings.py`

- [ ] **Step 1: Add casting settings**

In `config/settings.py`, inside the `Settings` class (alongside the existing `azuracast_*` fields), add:

```python
    # --- Central Casting ---
    casting_db_path: str = "/home/aialfred/alfred/data/casting.db"
    casting_voices_dir: str = "/home/aialfred/alfred/data/casting/voices"
    casting_previews_dir: str = "/home/aialfred/alfred/data/casting/previews"
    qwen_tts_url: str = "http://75.43.156.105:7860"
    casting_ollama_url: str = "http://75.43.156.105:11434"
    casting_model: str = "kimi-k2.6:cloud"
    # AzuraCast deploy target (server 98)
    casting_ssh_host: str = "server-98"
    casting_az_db_pass: str = "Yc2tNakqcne2"
    casting_engine_voices_dir: str = "/var/azuracast"          # where <Base>_<mood>.wav live on 98
    casting_station_id: int = 22
```

- [ ] **Step 2: Create the package marker**

```python
# core/casting/__init__.py
"""Central Casting — AI radio personality studio."""
```

- [ ] **Step 3: Verify settings import**

Run: `cd /home/aialfred/alfred && venv/bin/python -c "from config.settings import settings; print(settings.casting_db_path, settings.casting_station_id)"`
Expected: prints `/home/aialfred/alfred/data/casting.db 22`

- [ ] **Step 4: Commit**

```bash
git add config/settings.py core/casting/__init__.py
git commit -m "feat(casting): add settings + package skeleton"
```

---

## Task 2: Models + mood pack + archetypes

**Files:**
- Create: `core/casting/models.py`
- Create: `core/casting/mood_pack.py`
- Create: `core/casting/archetypes.py`

- [ ] **Step 1: Define the mood pack constant**

```python
# core/casting/mood_pack.py
"""The canonical 8-read Mood Pack. Every recorded voice is captured on these
exact reads, same booth/mic/distance/session, ~20-35s each."""

MOODS = ["neutral", "fired", "serious", "amused", "thoughtful", "reactions", "wry", "intimate"]

MOOD_PACK = {
    "neutral": {
        "label": "Neutral (warm baseline / anchor)",
        "direction": "Read it warm and conversational, like talking to one person.",
        "script": (
            "Right now, somewhere out there, somebody just turned this on for the first time. "
            "Maybe you're driving, maybe you're up late, maybe you're just getting started. "
            "Wherever you are, you found us, and I'm glad you did. Settle in, get comfortable, "
            "and let's spend some time together. This is your station, and this is exactly where "
            "you're supposed to be."
        ),
    },
    "fired": {
        "label": "Fired up (high energy)",
        "direction": "Big energy, fast, excited. Like the best part of the day is RIGHT NOW.",
        "script": (
            "Okay, stop what you're doing, because you are not ready for this one. I've been waiting "
            "all day to get on this mic, and we are going in. Turn it up, roll the windows down, tell "
            "somebody to come listen. This is the part of the day you've been waiting for, and trust "
            "me, it is about to be electric."
        ),
    },
    "serious": {
        "label": "Serious (grounded, sincere)",
        "direction": "Slow down. Sincere, grounded, no performance. Mean it.",
        "script": (
            "Let me slow it down for a second, because this matters. We don't always say the real "
            "thing out loud, but I'm going to. Times are heavy for a lot of people right now, and if "
            "that's you, I want you to know you're not alone in it. Take a breath. We're going to get "
            "through the hard stuff together. I mean that."
        ),
    },
    "amused": {
        "label": "Amused (light, lands on a laugh)",
        "direction": "Light and playful, smiling through it, end on a genuine laugh.",
        "script": (
            "So I have to tell you what just happened, because I still can't believe it. You know that "
            "feeling when something's so ridiculous you can't even be mad, you just have to laugh? Yeah. "
            "That was my whole morning. Ha. You can't make this stuff up. I wish I could."
        ),
    },
    "thoughtful": {
        "label": "Thoughtful (reflective, slower, lower)",
        "direction": "Reflective and unhurried, slightly lower, like thinking out loud.",
        "script": (
            "You ever stop and really think about how you got here? Not the big stuff, just the little "
            "turns. A conversation you almost didn't have. A day you almost stayed home. Funny how the "
            "smallest moments end up shaping everything. I've been sitting with that lately, and maybe "
            "you should too."
        ),
    },
    "reactions": {
        "label": "Reactions (short ad-libs + the laugh)",
        "direction": "Read each as a separate beat with a pause between. Natural, varied.",
        "script": (
            "Mm. ... Right. ... Come on, now. ... No way. ... Ha, okay, okay. ... Whew. ... I hear you. "
            "... That's wild. ... Let's get into it."
        ),
    },
    "wry": {
        "label": "Wry / Sarcastic (dry, deadpan)",
        "direction": "Dry, deadpan, eyebrow up. Understated, not mean.",
        "script": (
            "Oh, no, this is great. This is exactly how I pictured my day going. You ever notice how the "
            "people with the most to say are usually the ones who've done the least? No? Just me? Look, "
            "I'm not saying I told you so. But I did tell you so. Anyway. Let's all act surprised."
        ),
    },
    "intimate": {
        "label": "Intimate / Late-night (hushed, close-mic)",
        "direction": "Hushed, close to the mic, slow and warm. Just you and one listener.",
        "script": (
            "Hey. It's just us now. The rest of the world's gone quiet, and that's kind of the best part, "
            "isn't it? No rush tonight. Pull the covers up, turn the lights down low, and let me keep you "
            "company for a while. Whatever today took out of you, set it down right here. I'm not going "
            "anywhere. Just stay with me."
        ),
    },
}
```

- [ ] **Step 2: Define the archetype shelf**

```python
# core/casting/archetypes.py
"""Seed persona archetypes the operator can start from / remix."""

ARCHETYPES = [
    {"id": "firebrand", "name": "Loose-cannon Firebrand",
     "summary": "High-octane opinion host, Tucker-meets-Rogan, says the quiet part out loud."},
    {"id": "wellness", "name": "Calm Wellness Host",
     "summary": "Soothing late-night wind-down, sleep/mindfulness/small habits, never combative."},
    {"id": "strategist", "name": "Art-of-War Strategist",
     "summary": "Warm-realist, tactical, sales-driven; substance over slogans; for go-getters."},
    {"id": "operator", "name": "48-Laws Operator",
     "summary": "Direct, power-and-leverage minded, no filler, respects the listener's time."},
    {"id": "big_sister", "name": "Big-Sister Confidante",
     "summary": "Warm, funny, real talk; equal parts hype and tough love."},
    {"id": "anchor", "name": "Trusted News Anchor",
     "summary": "Even-keeled, broad and curious, opinion through a measured lens."},
]
```

- [ ] **Step 3: Define Pydantic models**

```python
# core/casting/models.py
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Mood = Literal["neutral", "fired", "serious", "amused", "thoughtful", "reactions", "wry", "intimate"]
VoiceSource = Literal["recorded", "stock-shelf", "generated"]
Status = Literal["draft", "ready", "live"]
Role = Literal["host", "guest"]

class PersonaBrief(BaseModel):
    brief: str = Field(min_length=3, max_length=600)
    archetype_id: Optional[str] = None
    name: str = Field(min_length=1, max_length=80)

class PersonaDraft(BaseModel):
    persona_prompt: str
    archetype_tags: list[str] = []

class DJCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    role: Role = "host"
    persona_prompt: str = ""
    archetype_tags: list[str] = []
    expertise: str = ""
    voice_source: VoiceSource = "recorded"

class DJOut(BaseModel):
    id: int
    name: str
    role: Role
    status: Status
    persona_prompt: str
    archetype_tags: list[str]
    expertise: str
    voice_source: VoiceSource
    moods_present: list[str]
    avatar: Optional[str] = None

class AssignmentCreate(BaseModel):
    dj_id: int
    station_id: int
    slot: str = Field(min_length=1, max_length=40)   # e.g. "10a-2p"
    effective_at: str                                 # ISO8601, e.g. "2026-06-07T10:00:00"

class AssignmentOut(BaseModel):
    id: int
    dj_id: int
    dj_name: str
    station_id: int
    slot: str
    effective_at: str
    applied: bool
```

- [ ] **Step 4: Verify imports**

Run: `cd /home/aialfred/alfred && venv/bin/python -c "from core.casting.mood_pack import MOODS, MOOD_PACK; from core.casting.archetypes import ARCHETYPES; from core.casting.models import DJCreate; assert len(MOODS)==8 and len(MOOD_PACK)==8; print('ok', len(ARCHETYPES), 'archetypes')"`
Expected: `ok 6 archetypes`

- [ ] **Step 5: Commit**

```bash
git add core/casting/models.py core/casting/mood_pack.py core/casting/archetypes.py
git commit -m "feat(casting): models, 8-read mood pack, archetype shelf"
```

---

## Task 3: DB layer (TDD)

**Files:**
- Create: `core/casting/db.py`
- Test: `tests/casting/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_db.py
import os, tempfile, importlib
import pytest

@pytest.fixture()
def db(monkeypatch):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "casting.db")
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_db_path", path, raising=False)
    import core.casting.db as dbmod
    importlib.reload(dbmod)
    dbmod.init_db()
    return dbmod

def test_create_and_get_dj(db):
    dj_id = db.create_dj(name="Sloan", role="host", persona_prompt="warm realist",
                         archetype_tags=["strategist"], expertise="", voice_source="recorded")
    assert isinstance(dj_id, int)
    row = db.get_dj(dj_id)
    assert row["name"] == "Sloan"
    assert row["status"] == "draft"
    assert row["archetype_tags"] == ["strategist"]

def test_list_djs(db):
    db.create_dj(name="A", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    db.create_dj(name="B", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    assert len(db.list_djs()) == 2

def test_set_status_and_moods(db):
    dj_id = db.create_dj(name="A", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    db.set_mood_present(dj_id, "neutral")
    db.set_mood_present(dj_id, "fired")
    db.set_status(dj_id, "ready")
    row = db.get_dj(dj_id)
    assert row["status"] == "ready"
    assert set(row["moods_present"]) == {"neutral", "fired"}

def test_assignment_roundtrip(db):
    dj_id = db.create_dj(name="A", role="host", persona_prompt="", archetype_tags=[], expertise="", voice_source="recorded")
    aid = db.create_assignment(dj_id=dj_id, station_id=22, slot="10a-2p", effective_at="2026-06-07T10:00:00")
    due = db.due_assignments(now_iso="2026-06-07T10:05:00")
    assert any(a["id"] == aid for a in due)
    db.mark_applied(aid)
    assert db.due_assignments(now_iso="2026-06-07T10:05:00") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.db'`

- [ ] **Step 3: Implement the DB layer**

```python
# core/casting/db.py
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Any
from config.settings import settings

def _db_path() -> Path:
    return Path(settings.casting_db_path)

def _conn() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c

def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS dj (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'host',
                status TEXT NOT NULL DEFAULT 'draft',
                persona_prompt TEXT NOT NULL DEFAULT '',
                archetype_tags TEXT NOT NULL DEFAULT '[]',
                expertise TEXT NOT NULL DEFAULT '',
                voice_source TEXT NOT NULL DEFAULT 'recorded',
                moods_present TEXT NOT NULL DEFAULT '[]',
                avatar TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS assignment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dj_id INTEGER NOT NULL,
                station_id INTEGER NOT NULL,
                slot TEXT NOT NULL,
                effective_at TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

def _row_to_dj(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["archetype_tags"] = json.loads(d.get("archetype_tags") or "[]")
    d["moods_present"] = json.loads(d.get("moods_present") or "[]")
    return d

def create_dj(*, name: str, role: str, persona_prompt: str, archetype_tags: list[str],
              expertise: str, voice_source: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO dj (name, role, persona_prompt, archetype_tags, expertise, voice_source) "
            "VALUES (?,?,?,?,?,?)",
            (name, role, persona_prompt, json.dumps(archetype_tags), expertise, voice_source),
        )
        return int(cur.lastrowid)

def get_dj(dj_id: int) -> dict[str, Any] | None:
    with _conn() as c:
        r = c.execute("SELECT * FROM dj WHERE id=?", (dj_id,)).fetchone()
        return _row_to_dj(r) if r else None

def list_djs() -> list[dict[str, Any]]:
    with _conn() as c:
        return [_row_to_dj(r) for r in c.execute("SELECT * FROM dj ORDER BY id DESC")]

def update_dj(dj_id: int, *, persona_prompt: str | None = None, archetype_tags: list[str] | None = None,
              expertise: str | None = None, name: str | None = None) -> None:
    sets, vals = [], []
    if persona_prompt is not None: sets.append("persona_prompt=?"); vals.append(persona_prompt)
    if archetype_tags is not None: sets.append("archetype_tags=?"); vals.append(json.dumps(archetype_tags))
    if expertise is not None: sets.append("expertise=?"); vals.append(expertise)
    if name is not None: sets.append("name=?"); vals.append(name)
    if not sets: return
    vals.append(dj_id)
    with _conn() as c:
        c.execute(f"UPDATE dj SET {', '.join(sets)} WHERE id=?", vals)

def set_status(dj_id: int, status: str) -> None:
    with _conn() as c:
        c.execute("UPDATE dj SET status=? WHERE id=?", (status, dj_id))

def set_mood_present(dj_id: int, mood: str) -> None:
    with _conn() as c:
        r = c.execute("SELECT moods_present FROM dj WHERE id=?", (dj_id,)).fetchone()
        moods = json.loads(r["moods_present"]) if r else []
        if mood not in moods:
            moods.append(mood)
        c.execute("UPDATE dj SET moods_present=? WHERE id=?", (json.dumps(moods), dj_id))

def create_assignment(*, dj_id: int, station_id: int, slot: str, effective_at: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO assignment (dj_id, station_id, slot, effective_at) VALUES (?,?,?,?)",
            (dj_id, station_id, slot, effective_at),
        )
        return int(cur.lastrowid)

def list_assignments(station_id: int | None = None) -> list[dict[str, Any]]:
    q = ("SELECT a.*, d.name AS dj_name FROM assignment a JOIN dj d ON d.id=a.dj_id")
    args: tuple = ()
    if station_id is not None:
        q += " WHERE a.station_id=?"; args = (station_id,)
    q += " ORDER BY a.effective_at"
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args)]

def due_assignments(now_iso: str) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT a.*, d.name AS dj_name FROM assignment a JOIN dj d ON d.id=a.dj_id "
            "WHERE a.applied=0 AND a.effective_at<=? ORDER BY a.effective_at",
            (now_iso,),
        ).fetchall()
        return [dict(r) for r in rows]

def mark_applied(assignment_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE assignment SET applied=1 WHERE id=?", (assignment_id,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_db.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add core/casting/db.py tests/casting/test_db.py
git commit -m "feat(casting): sqlite library DB (dj + assignment) with tests"
```

---

## Task 4: Voice processing (TDD)

Qwen3-TTS is zero-shot — the reference wav **is** the model, so "processing" = validate the upload, normalize it (mono 24kHz, trim silence, loudnorm), and store it as `data/casting/voices/<dj_id>/<mood>.wav`. No training step.

**Files:**
- Create: `core/casting/voice.py`
- Test: `tests/casting/test_voice.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_voice.py
import os, tempfile, subprocess, importlib
import pytest

@pytest.fixture()
def voicemod(monkeypatch):
    tmp = tempfile.mkdtemp()
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_voices_dir", os.path.join(tmp, "voices"), raising=False)
    import core.casting.voice as v
    importlib.reload(v)
    return v

def _make_wav(path, seconds=2):
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
         "-ar", "44100", "-ac", "2", "-y", path],
        check=True, capture_output=True,
    )

def test_validate_rejects_too_short(voicemod, tmp_path):
    p = str(tmp_path / "tiny.wav"); _make_wav(p, seconds=1)
    ok, reason = voicemod.validate_clip(p)
    assert ok is False and "short" in reason.lower()

def test_validate_accepts_good_clip(voicemod, tmp_path):
    p = str(tmp_path / "good.wav"); _make_wav(p, seconds=20)
    ok, reason = voicemod.validate_clip(p)
    assert ok is True, reason

def test_store_mood_normalizes_and_places(voicemod, tmp_path):
    src = str(tmp_path / "src.wav"); _make_wav(src, seconds=20)
    out = voicemod.store_mood(dj_id=7, mood="neutral", src_path=src)
    assert out.endswith("/7/neutral.wav")
    assert os.path.exists(out)
    # normalized to mono 24k
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=channels,sample_rate", "-of", "csv=p=0", out],
                           capture_output=True, text=True)
    assert "1" in probe.stdout and "24000" in probe.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_voice.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.voice'`

- [ ] **Step 3: Implement voice processing**

```python
# core/casting/voice.py
from __future__ import annotations
import json, subprocess
from pathlib import Path
from config.settings import settings

MIN_SECONDS = 8.0      # below this is too short to be a usable reference
MAX_SECONDS = 90.0

def _voices_root() -> Path:
    return Path(settings.casting_voices_dir)

def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0

def validate_clip(path: str) -> tuple[bool, str]:
    if not Path(path).exists():
        return False, "file not found"
    dur = _probe_duration(path)
    if dur <= 0:
        return False, "could not read audio (corrupt or unsupported format)"
    if dur < MIN_SECONDS:
        return False, f"too short ({dur:.1f}s); need at least {MIN_SECONDS:.0f}s"
    if dur > MAX_SECONDS:
        return False, f"too long ({dur:.1f}s); keep it under {MAX_SECONDS:.0f}s"
    return True, "ok"

def store_mood(*, dj_id: int, mood: str, src_path: str) -> str:
    """Normalize to mono 24kHz, trim edge silence, loudnorm, write to the library."""
    dest_dir = _voices_root() / str(dj_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{mood}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path,
         "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
                "areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB,"
                "areverse,loudnorm=I=-18:TP=-1.5:LRA=11",
         "-ar", "24000", "-ac", "1", str(dest)],
        check=True, capture_output=True, timeout=120,
    )
    return str(dest)

def mood_path(dj_id: int, mood: str) -> str:
    return str(_voices_root() / str(dj_id) / f"{mood}.wav")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_voice.py -v`
Expected: PASS (3 passed). (Requires `ffmpeg`/`ffprobe` on PATH — already present on 105.)

- [ ] **Step 5: Commit**

```bash
git add core/casting/voice.py tests/casting/test_voice.py
git commit -m "feat(casting): voice validate + normalize + store mood clips with tests"
```

---

## Task 5: Persona drafting (TDD)

**Files:**
- Create: `core/casting/persona.py`
- Test: `tests/casting/test_persona.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_persona.py
import importlib
import core.casting.persona as p

def test_build_prompt_includes_brief_and_archetype():
    msg = p._build_prompt(name="Sloan", brief="warm realist, sales-driven", archetype_summary="Art-of-War strategist")
    assert "Sloan" in msg and "sales-driven" in msg and "Art-of-War" in msg

def test_draft_persona_uses_llm(monkeypatch):
    importlib.reload(p)
    captured = {}
    def fake_chat(url, json=None, timeout=None):
        captured["url"] = url; captured["body"] = json
        class R:
            status_code = 200
            def json(self): return {"message": {"content": "PERSONA: Sloan is a warm-realist host."}}
        return R()
    monkeypatch.setattr(p.requests, "post", fake_chat)
    out = p.draft_persona(name="Sloan", brief="warm realist", archetype_id="strategist")
    assert "warm-realist" in out.persona_prompt
    assert captured["body"]["model"]  # model set
    assert captured["body"]["think"] is False
    assert captured["body"]["stream"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_persona.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.persona'`

- [ ] **Step 3: Implement persona drafting**

```python
# core/casting/persona.py
from __future__ import annotations
import requests
from config.settings import settings
from core.casting.archetypes import ARCHETYPES
from core.casting.models import PersonaDraft

_SYSTEM = (
    "You are a radio program director. Write a tight first-person PERSONA BRIEF for an AI radio host. "
    "Cover: who they are, their voice/tone, their lanes (topics), how they open and close, and their "
    "delivery quirks. Keep it under 320 words. Output ONLY the persona text, no preamble. "
    "The host emits delivery tags ([neutral],[fired],[serious],[amused],[thoughtful],[wry],[intimate]) "
    "and {laugh} in their scripts, so describe WHEN each mood shows up."
)

def _archetype_summary(archetype_id: str | None) -> str:
    for a in ARCHETYPES:
        if a["id"] == archetype_id:
            return a["summary"]
    return ""

def _build_prompt(*, name: str, brief: str, archetype_summary: str) -> str:
    parts = [f"Host name: {name}.", f"Operator brief: {brief}."]
    if archetype_summary:
        parts.append(f"Start from this archetype and remix it: {archetype_summary}.")
    return " ".join(parts)

def draft_persona(*, name: str, brief: str, archetype_id: str | None = None) -> PersonaDraft:
    user_msg = _build_prompt(name=name, brief=brief, archetype_summary=_archetype_summary(archetype_id))
    body = {
        "model": settings.casting_model,
        "think": False,
        "stream": False,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "options": {"num_predict": 700, "temperature": 0.8},
    }
    resp = requests.post(f"{settings.casting_ollama_url}/api/chat", json=body, timeout=120)
    resp.raise_for_status()
    content = (resp.json().get("message", {}) or {}).get("content", "").strip()
    tags = [archetype_id] if archetype_id else []
    return PersonaDraft(persona_prompt=content, archetype_tags=tags)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_persona.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add core/casting/persona.py tests/casting/test_persona.py
git commit -m "feat(casting): persona drafting via Kimi K2.6 with tests"
```

---

## Task 6: Off-air preview render

**Files:**
- Create: `core/casting/preview.py`
- Test: `tests/casting/test_preview.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_preview.py
import importlib
import core.casting.preview as pv

def test_preview_calls_qwen_with_neutral_voice(monkeypatch, tmp_path):
    importlib.reload(pv)
    calls = {}
    def fake_get(url, params=None, timeout=None):
        calls["url"] = url; calls["params"] = params
        class R:
            status_code = 200
            content = b"RIFFfakewav"
        return R()
    monkeypatch.setattr(pv.requests, "get", fake_get)
    out_path = str(tmp_path / "preview.wav")
    result = pv.render_preview(voice_wav="/voices/7/neutral.wav",
                               line="Good morning, this is a test.", out_path=out_path)
    assert result == out_path
    assert "synthesize_speech" in calls["url"]
    assert calls["params"]["text"] == "Good morning, this is a test."
    with open(out_path, "rb") as fh:
        assert fh.read().startswith(b"RIFF")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_preview.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.preview'`

- [ ] **Step 3: Implement preview**

```python
# core/casting/preview.py
from __future__ import annotations
import requests
from pathlib import Path
from config.settings import settings

DEFAULT_LINE = (
    "You're listening to News Muse. I'm glad you're here. Let's get into it."
)

def render_preview(*, voice_wav: str, line: str = DEFAULT_LINE, out_path: str) -> str:
    """Render a short off-air sample using the DJ's neutral reference clip.
    voice_wav is the absolute path to the reference clip as the Qwen server sees it
    (the Qwen server shares the 105 filesystem with Alfred Labs)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        f"{settings.qwen_tts_url}/synthesize_speech/",
        params={"text": line, "voice": voice_wav, "speed": "1.0"},
        timeout=120,
    )
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path
```

> **NOTE for executor:** Confirm the live 105:7860 `voice` param convention. The existing
> `interfaces/voice/tts.py` passes a voice *id/name*; the radio engine references clones by file
> basename. During execution, verify whether `voice=` expects a registered name or an absolute
> path, and adjust `render_preview`/`deploy` accordingly (one-line change). Test uses a mock so it
> passes regardless; this note flags the one live-integration check.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_preview.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add core/casting/preview.py tests/casting/test_preview.py
git commit -m "feat(casting): off-air preview render via Qwen with test"
```

---

## Task 7: AzuraCast deploy bridge (TDD)

Deploy = (1) copy the DJ's 8 mood wavs to server 98 renamed `<Base>_<mood>.wav`, and (2) upsert the `station_ai_dj_breaks` row (persona as `content_template`, base voice name, qwen engine) via `docker exec ... mariadb`. All shell calls wrapped in `timeout`.

**Files:**
- Create: `core/casting/deploy.py`
- Test: `tests/casting/test_deploy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_deploy.py
import importlib
import core.casting.deploy as dep

def test_deploy_builds_scp_and_sql(monkeypatch, tmp_path):
    importlib.reload(dep)
    # create fake mood wavs
    vdir = tmp_path / "7"; vdir.mkdir()
    for m in ["neutral", "fired"]:
        (vdir / f"{m}.wav").write_bytes(b"RIFF")
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_voices_dir", str(tmp_path), raising=False)

    ran = []
    def fake_run(cmd, **kw):
        ran.append(cmd)
        class R:
            returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr(dep.subprocess, "run", fake_run)

    dep.deploy_dj(dj_id=7, base_name="Sloan", moods=["neutral", "fired"],
                  persona_prompt="warm realist", station_id=22)

    # an scp/rsync push happened with timeout wrapping
    pushed = " ".join(" ".join(c) for c in ran)
    assert "Sloan_neutral.wav" in pushed and "Sloan_fired.wav" in pushed
    # a mariadb upsert happened referencing the breaks table + station 22
    assert "station_ai_dj_breaks" in pushed and "22" in pushed
    assert pushed.count("timeout") >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_deploy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.deploy'`

- [ ] **Step 3: Implement the deploy bridge**

```python
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
```

> **NOTE for executor:** The exact `station_ai_dj_breaks` column set and the unique key for the
> `ON DUPLICATE KEY` must be confirmed against the live schema on 98
> (`ssh server-98 'sudo docker exec azuracast mariadb -u azuracast -pYc2tNakqcne2 azuracast -e "DESCRIBE station_ai_dj_breaks;"'`).
> Adjust the column list/trigger fields to match. The existing live rows (breaks 13–16) are the
> reference template. Keep the persona/time/cross-promo blocks consistent with those rows.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_deploy.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add core/casting/deploy.py tests/casting/test_deploy.py
git commit -m "feat(casting): AzuraCast deploy bridge (wav push + break upsert) with test"
```

---

## Task 8: Scheduler (TDD)

**Files:**
- Create: `core/casting/scheduler.py`
- Test: `tests/casting/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_scheduler.py
import os, tempfile, importlib
import pytest

@pytest.fixture()
def env(monkeypatch):
    tmp = tempfile.mkdtemp()
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_db_path", os.path.join(tmp, "casting.db"), raising=False)
    import core.casting.db as dbmod; importlib.reload(dbmod); dbmod.init_db()
    import core.casting.scheduler as sch; importlib.reload(sch)
    return dbmod, sch

def test_apply_due_calls_deploy_and_marks(env, monkeypatch):
    dbmod, sch = env
    dj_id = dbmod.create_dj(name="Sloan", role="host", persona_prompt="warm",
                            archetype_tags=[], expertise="", voice_source="recorded")
    dbmod.set_mood_present(dj_id, "neutral")
    dbmod.set_status(dj_id, "ready")
    aid = dbmod.create_assignment(dj_id=dj_id, station_id=22, slot="10a-2p",
                                  effective_at="2026-06-07T10:00:00")
    deployed = {}
    monkeypatch.setattr(sch, "deploy_dj", lambda **kw: deployed.update(kw))
    applied = sch.apply_due(now_iso="2026-06-07T10:01:00")
    assert applied == 1
    assert deployed["base_name"] == "Sloan"
    assert dbmod.due_assignments(now_iso="2026-06-07T10:01:00") == []
    assert dbmod.get_dj(dj_id)["status"] == "live"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.scheduler'`

- [ ] **Step 3: Implement the scheduler**

```python
# core/casting/scheduler.py
from __future__ import annotations
from datetime import datetime
from core.casting import db
from core.casting.deploy import deploy_dj, DeployError

def apply_due(now_iso: str | None = None) -> int:
    now_iso = now_iso or datetime.now().isoformat(timespec="seconds")
    applied = 0
    for a in db.due_assignments(now_iso=now_iso):
        dj = db.get_dj(a["dj_id"])
        if not dj or dj["status"] == "draft" or not dj["moods_present"]:
            continue
        try:
            deploy_dj(dj_id=dj["id"], base_name=dj["name"].replace(" ", "_"),
                      moods=dj["moods_present"], persona_prompt=dj["persona_prompt"],
                      station_id=a["station_id"])
        except DeployError:
            # leave unapplied so the next run retries; alerting handled by caller
            continue
        db.mark_applied(a["id"])
        db.set_status(dj["id"], "live")
        applied += 1
    return applied

if __name__ == "__main__":
    print(f"applied {apply_due()} assignment(s)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_scheduler.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Add the cron entry**

Run (one-time, on 105):
```bash
( crontab -l 2>/dev/null; echo "*/5 * * * * cd /home/aialfred/alfred && venv/bin/python -m core.casting.scheduler >> data/casting/scheduler.log 2>&1" ) | crontab -
```
Expected: `crontab -l` shows the new line.

- [ ] **Step 6: Commit**

```bash
git add core/casting/scheduler.py tests/casting/test_scheduler.py
git commit -m "feat(casting): assignment scheduler (apply-due + cron) with test"
```

---

## Task 9: API router (TDD with FastAPI TestClient)

**Files:**
- Create: `core/casting/api_router.py` *(the route definitions)*
- Create: `core/api/central_casting.py` *(thin `register(app)` wrapper)*
- Modify: `core/api/main.py`
- Test: `tests/casting/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/casting/test_api.py
import os, tempfile, importlib
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.mkdtemp()
    from config.settings import settings
    monkeypatch.setattr(settings, "casting_db_path", os.path.join(tmp, "casting.db"), raising=False)
    monkeypatch.setattr(settings, "casting_voices_dir", os.path.join(tmp, "voices"), raising=False)
    monkeypatch.setattr(settings, "casting_previews_dir", os.path.join(tmp, "prev"), raising=False)
    import core.casting.db as dbmod; importlib.reload(dbmod); dbmod.init_db()
    import core.casting.api_router as r; importlib.reload(r)
    # bypass auth in tests
    app = FastAPI()
    app.dependency_overrides = {}
    r.register(app, auth_dep=lambda: {"username": "test"})
    return TestClient(app)

def test_moodpack_endpoint(client):
    res = client.get("/api/casting/moodpack")
    assert res.status_code == 200
    assert len(res.json()["moods"]) == 8

def test_create_and_list_dj(client):
    res = client.post("/api/casting/djs", json={"name": "Sloan", "role": "host",
        "persona_prompt": "warm realist", "archetype_tags": ["strategist"],
        "expertise": "", "voice_source": "recorded"})
    assert res.status_code == 200, res.text
    dj_id = res.json()["id"]
    lst = client.get("/api/casting/djs").json()
    assert any(d["id"] == dj_id for d in lst)

def test_archetypes_endpoint(client):
    res = client.get("/api/casting/archetypes")
    assert res.status_code == 200 and len(res.json()) >= 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.casting.api_router'`

- [ ] **Step 3: Implement the route module**

```python
# core/casting/api_router.py
from __future__ import annotations
import shutil, tempfile
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from config.settings import settings
from core.casting import db, voice, persona as persona_mod, preview as preview_mod
from core.casting.mood_pack import MOOD_PACK, MOODS
from core.casting.archetypes import ARCHETYPES
from core.casting.models import (DJCreate, DJOut, PersonaBrief, PersonaDraft,
                                 AssignmentCreate, AssignmentOut)

def _dj_out(row: dict) -> dict:
    return DJOut(
        id=row["id"], name=row["name"], role=row["role"], status=row["status"],
        persona_prompt=row["persona_prompt"], archetype_tags=row["archetype_tags"],
        expertise=row["expertise"], voice_source=row["voice_source"],
        moods_present=row["moods_present"], avatar=row.get("avatar"),
    ).model_dump()

def register(app: FastAPI, auth_dep=None) -> None:
    from core.security.auth import require_auth
    guard = auth_dep or require_auth

    @app.get("/api/casting/moodpack")
    async def moodpack(_user=Depends(guard)):
        return {"moods": [{"mood": m, **MOOD_PACK[m]} for m in MOODS]}

    @app.get("/api/casting/archetypes")
    async def archetypes(_user=Depends(guard)):
        return ARCHETYPES

    @app.get("/api/casting/djs")
    async def list_djs(_user=Depends(guard)):
        return [_dj_out(r) for r in db.list_djs()]

    @app.post("/api/casting/djs")
    async def create_dj(body: DJCreate, _user=Depends(guard)):
        dj_id = db.create_dj(name=body.name, role=body.role, persona_prompt=body.persona_prompt,
                             archetype_tags=body.archetype_tags, expertise=body.expertise,
                             voice_source=body.voice_source)
        return _dj_out(db.get_dj(dj_id))

    @app.get("/api/casting/djs/{dj_id}")
    async def get_dj(dj_id: int, _user=Depends(guard)):
        row = db.get_dj(dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        return _dj_out(row)

    @app.post("/api/casting/persona/draft", response_model=PersonaDraft)
    async def draft(body: PersonaBrief, _user=Depends(guard)):
        return persona_mod.draft_persona(name=body.name, brief=body.brief,
                                         archetype_id=body.archetype_id)

    @app.post("/api/casting/djs/{dj_id}/voice/{mood}")
    async def upload_mood(dj_id: int, mood: str, file: UploadFile = File(...), _user=Depends(guard)):
        if mood not in MOODS:
            raise HTTPException(400, f"unknown mood '{mood}'")
        if not db.get_dj(dj_id):
            raise HTTPException(404, "DJ not found")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        ok, reason = voice.validate_clip(tmp_path)
        if not ok:
            Path(tmp_path).unlink(missing_ok=True)
            raise HTTPException(422, reason)
        voice.store_mood(dj_id=dj_id, mood=mood, src_path=tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        db.set_mood_present(dj_id, mood)
        # neutral present => at least a working DJ
        row = db.get_dj(dj_id)
        if "neutral" in row["moods_present"] and row["status"] == "draft":
            db.set_status(dj_id, "ready")
        return _dj_out(db.get_dj(dj_id))

    @app.post("/api/casting/djs/{dj_id}/preview")
    async def preview(dj_id: int, _user=Depends(guard)):
        row = db.get_dj(dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        if "neutral" not in row["moods_present"]:
            raise HTTPException(422, "need a neutral clip before preview")
        out = str(Path(settings.casting_previews_dir) / f"{dj_id}.wav")
        preview_mod.render_preview(voice_wav=voice.mood_path(dj_id, "neutral"), out_path=out)
        return FileResponse(out, media_type="audio/wav", filename=f"preview_{dj_id}.wav")

    @app.get("/api/casting/assignments")
    async def list_assignments(station_id: int | None = None, _user=Depends(guard)):
        return [AssignmentOut(id=a["id"], dj_id=a["dj_id"], dj_name=a["dj_name"],
                              station_id=a["station_id"], slot=a["slot"],
                              effective_at=a["effective_at"], applied=bool(a["applied"])).model_dump()
                for a in db.list_assignments(station_id)]

    @app.post("/api/casting/assignments")
    async def create_assignment(body: AssignmentCreate, _user=Depends(guard)):
        row = db.get_dj(body.dj_id)
        if not row:
            raise HTTPException(404, "DJ not found")
        if row["status"] == "draft":
            raise HTTPException(422, "DJ is still a draft; add a neutral clip first")
        aid = db.create_assignment(dj_id=body.dj_id, station_id=body.station_id,
                                   slot=body.slot, effective_at=body.effective_at)
        return {"id": aid}
```

- [ ] **Step 4: Create the thin register wrapper**

```python
# core/api/central_casting.py
from fastapi import FastAPI
from core.casting.db import init_db
from core.casting.api_router import register as _register

def register(app: FastAPI) -> None:
    init_db()
    _register(app)
```

- [ ] **Step 5: Register in main.py**

In `core/api/main.py`, after the last existing module registration block (follow the exact `try/except` pattern already used for `roen_admin`), add:

```python
    try:
        from core.api.central_casting import register as _register_central_casting
        _register_central_casting(app)
    except Exception as _e:
        logger.exception("central_casting register failed: %s", _e)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting/test_api.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Smoke-test the live server boots with the router**

Run: `cd /home/aialfred/alfred && venv/bin/python -c "from core.api.main import app; print([r.path for r in app.routes if '/casting/' in r.path])"`
Expected: prints the list of `/api/casting/...` paths.

- [ ] **Step 8: Commit**

```bash
git add core/casting/api_router.py core/api/central_casting.py core/api/main.py tests/casting/test_api.py
git commit -m "feat(casting): FastAPI router (djs, persona, voice upload, preview, assignments)"
```

---

## Task 10: Frontend — API client + store

**Files:**
- Create: `frontend/src/api/casting.ts`
- Create: `frontend/src/stores/castingStore.ts`

- [ ] **Step 1: Add the API client**

```typescript
// frontend/src/api/casting.ts
import { apiFetch, apiUpload } from './client'

export interface DJ {
  id: number; name: string; role: 'host' | 'guest'; status: 'draft' | 'ready' | 'live'
  persona_prompt: string; archetype_tags: string[]; expertise: string
  voice_source: string; moods_present: string[]; avatar?: string | null
}
export interface MoodRead { mood: string; label: string; direction: string; script: string }
export interface Archetype { id: string; name: string; summary: string }
export interface Assignment { id: number; dj_id: number; dj_name: string; station_id: number; slot: string; effective_at: string; applied: boolean }

export const castingApi = {
  moodpack: () => apiFetch('/api/casting/moodpack').then(r => r.json()).then(d => d.moods as MoodRead[]),
  archetypes: () => apiFetch('/api/casting/archetypes').then(r => r.json() as Promise<Archetype[]>),
  listDJs: () => apiFetch('/api/casting/djs').then(r => r.json() as Promise<DJ[]>),
  createDJ: (body: Partial<DJ>) => apiFetch('/api/casting/djs', { method: 'POST', body: JSON.stringify(body) }).then(r => r.json() as Promise<DJ>),
  draftPersona: (name: string, brief: string, archetype_id?: string) =>
    apiFetch('/api/casting/persona/draft', { method: 'POST', body: JSON.stringify({ name, brief, archetype_id }) }).then(r => r.json()),
  uploadMood: (djId: number, mood: string, file: File) => {
    const fd = new FormData(); fd.append('file', file)
    return apiUpload(`/api/casting/djs/${djId}/voice/${mood}`, fd).then(r => r.json() as Promise<DJ>)
  },
  previewUrl: (djId: number) => `/api/casting/djs/${djId}/preview`,
  listAssignments: (stationId?: number) =>
    apiFetch(`/api/casting/assignments${stationId ? `?station_id=${stationId}` : ''}`).then(r => r.json() as Promise<Assignment[]>),
  createAssignment: (dj_id: number, station_id: number, slot: string, effective_at: string) =>
    apiFetch('/api/casting/assignments', { method: 'POST', body: JSON.stringify({ dj_id, station_id, slot, effective_at }) }).then(r => r.json()),
}
```

> **NOTE for executor:** Confirm `apiFetch`/`apiUpload` signatures in `frontend/src/api/client.ts`
> (return type, whether they set `Content-Type`/credentials). Adjust the calls above to match the
> exact wrapper (e.g. if `apiFetch` already parses JSON, drop the `.then(r => r.json())`).

- [ ] **Step 2: Add the Zustand store**

```typescript
// frontend/src/stores/castingStore.ts
import { create } from 'zustand'
import { castingApi, DJ, Archetype, MoodRead } from '../api/casting'

interface CastingState {
  djs: DJ[]; archetypes: Archetype[]; moodPack: MoodRead[]; loading: boolean
  refresh: () => Promise<void>
  loadStatic: () => Promise<void>
}

export const useCastingStore = create<CastingState>((set) => ({
  djs: [], archetypes: [], moodPack: [], loading: false,
  refresh: async () => {
    set({ loading: true })
    try { set({ djs: await castingApi.listDJs() }) } finally { set({ loading: false }) }
  },
  loadStatic: async () => {
    const [archetypes, moodPack] = await Promise.all([castingApi.archetypes(), castingApi.moodpack()])
    set({ archetypes, moodPack })
  },
}))
```

- [ ] **Step 3: Type-check**

Run: `cd /home/aialfred/alfred/frontend && npx tsc --noEmit`
Expected: no errors in `src/api/casting.ts` / `src/stores/castingStore.ts` (fix any signature mismatches flagged by the NOTE above).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/casting.ts frontend/src/stores/castingStore.ts
git commit -m "feat(casting): frontend api client + zustand store"
```

---

## Task 11: Frontend — screens + nav

Build with the **frontend-design** quality bar (this is Mike's flagship tool). Three screens behind one module shell. Use existing component/styling conventions from neighboring components in `frontend/src/components/`.

**Files:**
- Create: `frontend/src/components/casting/CastingApp.tsx`
- Create: `frontend/src/components/casting/Library.tsx`
- Create: `frontend/src/components/casting/CreateDJ.tsx`
- Create: `frontend/src/components/casting/DeploySchedule.tsx`
- Modify: `frontend/src/components/layout/AppLayout.tsx`

- [ ] **Step 1: Module shell with tabs**

```tsx
// frontend/src/components/casting/CastingApp.tsx
import { useEffect, useState } from 'react'
import { useCastingStore } from '../../stores/castingStore'
import { Library } from './Library'
import { CreateDJ } from './CreateDJ'
import { DeploySchedule } from './DeploySchedule'

type Tab = 'library' | 'create' | 'deploy'

export function CastingApp() {
  const [tab, setTab] = useState<Tab>('library')
  const { loadStatic, refresh } = useCastingStore()
  useEffect(() => { loadStatic(); refresh() }, [loadStatic, refresh])
  return (
    <div className="casting-app">
      <header className="casting-header">
        <h1>Central Casting</h1>
        <nav>
          <button className={tab === 'library' ? 'active' : ''} onClick={() => setTab('library')}>Library</button>
          <button className={tab === 'create' ? 'active' : ''} onClick={() => setTab('create')}>Create a DJ</button>
          <button className={tab === 'deploy' ? 'active' : ''} onClick={() => setTab('deploy')}>Deploy</button>
        </nav>
      </header>
      {tab === 'library' && <Library />}
      {tab === 'create' && <CreateDJ onDone={() => setTab('library')} />}
      {tab === 'deploy' && <DeploySchedule />}
    </div>
  )
}
```

- [ ] **Step 2: Library grid (with off-air preview)**

```tsx
// frontend/src/components/casting/Library.tsx
import { useCastingStore } from '../../stores/castingStore'
import { castingApi } from '../../api/casting'

export function Library() {
  const { djs, loading } = useCastingStore()
  if (loading) return <p>Loading…</p>
  if (!djs.length) return <p>No DJs yet. Create one to get started.</p>
  return (
    <div className="casting-grid">
      {djs.map(dj => (
        <div className="dj-card" key={dj.id}>
          <div className="dj-card-top">
            <span className="dj-name">{dj.name}</span>
            <span className={`dj-status dj-status-${dj.status}`}>{dj.status}</span>
          </div>
          <div className="dj-tags">{dj.archetype_tags.join(' · ')}</div>
          <div className="dj-moods">{dj.moods_present.length}/8 moods</div>
          <audio controls preload="none" src={castingApi.previewUrl(dj.id)} />
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Create-a-DJ wizard (persona draft + per-mood upload)**

```tsx
// frontend/src/components/casting/CreateDJ.tsx
import { useState } from 'react'
import { useCastingStore } from '../../stores/castingStore'
import { castingApi, DJ } from '../../api/casting'

export function CreateDJ({ onDone }: { onDone: () => void }) {
  const { archetypes, moodPack, refresh } = useCastingStore()
  const [name, setName] = useState('')
  const [brief, setBrief] = useState('')
  const [archetypeId, setArchetypeId] = useState('')
  const [persona, setPersona] = useState('')
  const [dj, setDj] = useState<DJ | null>(null)
  const [busy, setBusy] = useState(false)

  const draft = async () => {
    setBusy(true)
    try {
      const d = await castingApi.draftPersona(name, brief, archetypeId || undefined)
      setPersona(d.persona_prompt)
    } finally { setBusy(false) }
  }
  const save = async () => {
    const created = await castingApi.createDJ({
      name, role: 'host', persona_prompt: persona,
      archetype_tags: archetypeId ? [archetypeId] : [], voice_source: 'recorded',
    })
    setDj(created); await refresh()
  }
  const upload = async (mood: string, file: File) => {
    if (!dj) return
    const updated = await castingApi.uploadMood(dj.id, mood, file)
    setDj(updated); await refresh()
  }

  return (
    <div className="create-dj">
      <label>Name<input value={name} onChange={e => setName(e.target.value)} /></label>
      <label>Archetype
        <select value={archetypeId} onChange={e => setArchetypeId(e.target.value)}>
          <option value="">(none)</option>
          {archetypes.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
      </label>
      <label>Brief
        <textarea value={brief} onChange={e => setBrief(e.target.value)}
          placeholder="warm Latina, mid-30s, sales-driven, sports-obsessed, big-sister energy" />
      </label>
      <button disabled={busy || !name || !brief} onClick={draft}>Draft persona</button>
      <textarea className="persona-out" value={persona} onChange={e => setPersona(e.target.value)} rows={12} />
      <button disabled={!name || !persona} onClick={save}>{dj ? 'Saved ✓' : 'Save DJ'}</button>

      {dj && (
        <section className="mood-capture">
          <h3>Record the Mood Pack — same mic, same distance, one session</h3>
          {moodPack.map(m => (
            <div className="mood-row" key={m.mood}>
              <div className="mood-meta">
                <strong>{m.label}</strong>
                <em>{m.direction}</em>
                <p>{m.script}</p>
              </div>
              <div className="mood-upload">
                {dj.moods_present.includes(m.mood) ? <span>✓ captured</span> : null}
                <input type="file" accept="audio/*"
                  onChange={e => e.target.files?.[0] && upload(m.mood, e.target.files[0])} />
              </div>
            </div>
          ))}
          <button onClick={onDone}>Done</button>
        </section>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Deploy / schedule screen**

```tsx
// frontend/src/components/casting/DeploySchedule.tsx
import { useEffect, useState } from 'react'
import { useCastingStore } from '../../stores/castingStore'
import { castingApi, Assignment } from '../../api/casting'

const STATION_ID = 22

export function DeploySchedule() {
  const { djs } = useCastingStore()
  const [assignments, setAssignments] = useState<Assignment[]>([])
  const [djId, setDjId] = useState<number | ''>('')
  const [slot, setSlot] = useState('10a-2p')
  const [effectiveAt, setEffectiveAt] = useState('')
  const reload = () => castingApi.listAssignments(STATION_ID).then(setAssignments)
  useEffect(() => { reload() }, [])

  const schedule = async () => {
    if (!djId || !effectiveAt) return
    await castingApi.createAssignment(Number(djId), STATION_ID, slot, effectiveAt)
    await reload()
  }
  const deployable = djs.filter(d => d.status !== 'draft')

  return (
    <div className="deploy-schedule">
      <h3>Schedule a host onto News Muse (station {STATION_ID})</h3>
      <select value={djId} onChange={e => setDjId(e.target.value ? Number(e.target.value) : '')}>
        <option value="">Pick a DJ…</option>
        {deployable.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select>
      <input value={slot} onChange={e => setSlot(e.target.value)} placeholder="slot e.g. 10a-2p" />
      <input type="datetime-local" value={effectiveAt} onChange={e => setEffectiveAt(e.target.value)} />
      <button disabled={!djId || !effectiveAt} onClick={schedule}>Schedule swap</button>

      <h4>Upcoming lineup</h4>
      <ul className="lineup">
        {assignments.map(a => (
          <li key={a.id}>
            <strong>{a.slot}</strong> → {a.dj_name} @ {a.effective_at} {a.applied ? '(live)' : '(queued)'}
          </li>
        ))}
      </ul>
    </div>
  )
}
```

- [ ] **Step 5: Wire into AppLayout nav**

In `frontend/src/components/layout/AppLayout.tsx`, follow the existing pattern for how other modules/pages are added to the nav and the rendered view switch. Add a nav entry "Central Casting" that renders `<CastingApp />`:

```tsx
import { CastingApp } from '../casting/CastingApp'
// ...add to the nav items list and the view switch the same way existing entries are added...
// nav item key: 'central-casting'  →  renders <CastingApp />
```

- [ ] **Step 6: Add minimal styles**

Add a `frontend/src/components/casting/casting.css` (imported by `CastingApp.tsx`) with grid/card styles consistent with the app's existing visual language. (Use the frontend-design skill for polish.)

- [ ] **Step 7: Type-check + build**

Run: `cd /home/aialfred/alfred/frontend && npx tsc --noEmit && npm run build`
Expected: build succeeds, `dist/` updated.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/casting/ frontend/src/components/layout/AppLayout.tsx
git commit -m "feat(casting): library, create-DJ wizard, deploy/schedule screens + nav"
```

---

## Task 12: End-to-end verification (manual, off-air)

- [ ] **Step 1: Restart the API** so the new router loads.

Run: `sudo systemctl restart alfred.service && sleep 3 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8400/api/casting/moodpack`
Expected: `401` (auth required) — proves the route exists and is guarded. (A `200` after login proves the full path.)

- [ ] **Step 2: Create a test DJ end-to-end via the UI** (login → Central Casting → Create a DJ): draft a persona, upload at least a **neutral** clip (use one of MJ's existing clips from `data` or a fresh read), confirm it flips to **ready**.

- [ ] **Step 3: Preview off-air.** Hit the preview control on the Library card; confirm audio plays and **nothing changed on station 22** (check `nowplaying` / liquidsoap log is untouched).

- [ ] **Step 4: Schedule + apply (against a scratch slot).** Create an assignment with `effective_at` a minute in the future for a **non-live test slot name**, run `venv/bin/python -m core.casting.scheduler`, and confirm: wavs landed on 98 (`ssh server-98 'sudo docker exec azuracast ls /var/azuracast | grep <Name>_'`) and a `station_ai_dj_breaks` row exists. **Do not point it at a real live daypart until you've previewed.**

- [ ] **Step 5: Full test suite green.**

Run: `cd /home/aialfred/alfred && venv/bin/pytest tests/casting -v`
Expected: all pass.

- [ ] **Step 6: Commit any fixes from verification, then open a PR** from `feat/central-casting` → `main`.

---

## Self-Review notes (coverage map)

- **Login-gated, single user, agent-drivable API** → Tasks 9 (auth guard via `require_auth`) + 12.
- **DJ = voice + personality** → Tasks 2 (models), 3 (DB), 5 (persona), 4 (voice).
- **8-mood Pack standardized capture** → Tasks 2 (`mood_pack.py`), 4 (store), 11 (capture UI shows scripts + directions).
- **Persona from brief + archetype shelf** → Tasks 2 (`archetypes.py`), 5, 11.
- **Library outside AzuraCast; deploy = push** → Tasks 3 (own DB), 7 (deploy bridge).
- **Scheduled self-flipping deploy** → Task 8 (scheduler + cron).
- **Preview before air; never test live** → Tasks 6, 9 (preview route), 12 (explicit off-air checks).
- **Error handling** → voice validation (4/9 returns 422 with reason), deploy retries on failure (8), DeployError surfaced.
- **Build order v1 hosts only** → guests/stock-shelf/generate deferred (noted in scope; `role` + `voice_source` fields already present so v1.5/v2 slot in without migration).

**Deferred to follow-on plans (intentional, per spec):** guest bookings + duo-engine wiring (v1.5), stock voice shelf (v2), generate-a-voice-from-description (v3), avatar image upload, Telegram deploy-failure alerts (wire into the scheduler's `except` once base path is proven).
