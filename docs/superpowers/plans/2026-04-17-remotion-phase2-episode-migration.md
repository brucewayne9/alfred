# Remotion Phase 2 — Episode Pipeline Migration (RuckTalkClip → MagazineRig)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `scripts/rucktalk_episode_pipeline.py` from rendering episode clips via the old `RuckTalkClip` Remotion composition to the new `MagazineRig`, with a feature-flag rollout, a side-by-side comparison gate, and Mike's explicit approval before the flip.

**Architecture:** Extract the rig-specific prop builder into a new module `scripts/rucktalk_rig_props.py` with two functions (`build_rucktalkclip_props`, `build_magazinerig_props`). Introduce a `EPISODE_RIG` constant in `rucktalk_episode_pipeline.py` (default: `"RuckTalkClip"` during the cutover, flipping to `"MagazineRig"` at Task 7). `_render_branded_clip` picks the prop builder + composition id by looking at `EPISODE_RIG`. Add a `--compare` CLI flag that renders BOTH rigs side-by-side on a chosen episode so Mike can review before we flip.

**Tech Stack:** Python 3.11, pytest, Remotion 4.0.438 (already installed from Phase 1), subprocess invocation of `npx remotion render`.

**Source spec:** `/home/aialfred/alfred/docs/superpowers/specs/2026-04-17-remotion-template-library-design.md` Section 6.

**Scope boundary:** This plan only touches the **episode pipeline**. `scripts/rucktalk_daily_social.py` is Phase 3, separate. Do not modify the Remotion repo (the new `MagazineRig` composition already exists there from Phase 1).

**Repos:** All changes happen in `/home/aialfred/alfred/`. The Remotion library is already merged on master of `/home/aialfred/remotion/` and needs no further changes for Phase 2.

---

## File Structure

**Create:**
- `scripts/rucktalk_rig_props.py` — two prop-builder functions, one per rig shape. Pure functions that take caption phrases + episode metadata and return a validated props dict.
- `tests/scripts/__init__.py` — new test package
- `tests/scripts/test_rucktalk_rig_props.py` — unit tests for both prop builders

**Modify:**
- `scripts/rucktalk_episode_pipeline.py` — replace inline prop-building inside `_render_branded_clip` (lines ~914-946) with a call to the new prop-builder module; add `EPISODE_RIG` constant at the top of the module; add a `--compare` CLI flag and helper `_compare_rigs_render()`; change the composition id passed to `npx remotion render` based on `EPISODE_RIG`.

**Do not touch:**
- `/home/aialfred/remotion/` — new library is already there on master.
- `scripts/rucktalk_daily_social.py` — that's Phase 3.
- Anything under `scripts/providers/` — that's from Phase 1, stable.

---

## Prerequisites Check

Before starting Task 1, verify the following are true:

- [ ] **P1:** `MagazineRig` composition exists in `/home/aialfred/remotion/src/Root.tsx` on branch `master`. Verify: `cd /home/aialfred/remotion && grep -c 'id="MagazineRig"' src/Root.tsx` → returns 1.
- [ ] **P2:** Remotion library v1.0 tag exists. Verify: `cd /home/aialfred/remotion && git tag -l remotion-library-v1.0` → returns `remotion-library-v1.0`.
- [ ] **P3:** `npm run hero briefs/sample-magazine.json -- --low` renders a valid mp4. (Sanity check, optional — only if the team wants to verify the library independently first.)

If any prerequisite fails, stop and escalate — do not start Phase 2 against a broken foundation.

---

## Task 1: Extract rig props into a new module

**Files:**
- Create: `scripts/rucktalk_rig_props.py`
- Create: `tests/scripts/__init__.py`
- Create: `tests/scripts/test_rucktalk_rig_props.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_rucktalk_rig_props.py`:

```python
"""Tests for rig prop builders — Phase 2 migration."""
from scripts.rucktalk_rig_props import (
    build_rucktalkclip_props,
    build_magazinerig_props,
    CaptionPhrase,
)


def _sample_phrases() -> list[CaptionPhrase]:
    return [
        {"text": "HELLO WORLD", "startFrame": 0, "endFrame": 60},
        {"text": "TESTING", "startFrame": 70, "endFrame": 120},
    ]


def test_rucktalkclip_shape():
    p = build_rucktalkclip_props(
        clip_filename="clip_test.mp4",
        episode_number=42,
        episode_title="The Answer",
        context_line="Comfort is the enemy.",
        host_name="MIKE JOHNSON",
        guest_name=None,
        caption_phrases=_sample_phrases(),
    )
    assert p["videoSrc"] == "clip_test.mp4"
    assert p["episodeNumber"] == 42
    assert p["episodeTitle"] == "The Answer"
    assert p["contextLine"] == "Comfort is the enemy."
    assert p["hostName"] == "MIKE JOHNSON"
    assert p["guestName"] == ""
    assert p["captionPhrases"] == _sample_phrases()


def test_rucktalkclip_with_guest():
    p = build_rucktalkclip_props(
        clip_filename="c.mp4",
        episode_number=1,
        episode_title="T",
        context_line="",
        host_name="MIKE",
        guest_name="DR SMITH",
        caption_phrases=_sample_phrases(),
    )
    assert p["guestName"] == "DR SMITH"


def test_magazinerig_shape():
    p = build_magazinerig_props(
        clip_filename="clip_test.mp4",
        episode_number=42,
        episode_title="The Answer",
        host_name="MIKE JOHNSON",
        guest_name=None,
        caption_phrases=_sample_phrases(),
    )
    assert p["brand"] == "rucktalk"
    assert p["clipSrc"] == "clip_test.mp4"
    assert p["episodeNumber"] == 42
    assert p["episodeTitle"] == "The Answer"
    assert p["hostName"] == "MIKE JOHNSON"
    assert p["captionPhrases"] == _sample_phrases()
    # contextLine is dropped — must NOT be in output
    assert "contextLine" not in p
    # videoSrc is renamed — must NOT be in output
    assert "videoSrc" not in p


def test_magazinerig_with_guest():
    p = build_magazinerig_props(
        clip_filename="c.mp4",
        episode_number=1,
        episode_title="T",
        host_name="MIKE",
        guest_name="DR SMITH",
        caption_phrases=_sample_phrases(),
    )
    assert p["guestName"] == "DR SMITH"


def test_magazinerig_omits_none_guest():
    """When no guest, the key should either be absent or empty string — never None."""
    p = build_magazinerig_props(
        clip_filename="c.mp4",
        episode_number=1,
        episode_title="T",
        host_name="MIKE",
        guest_name=None,
        caption_phrases=_sample_phrases(),
    )
    assert p.get("guestName", "") == ""
```

- [ ] **Step 2: Create tests/scripts package marker and verify fail**

```bash
cd /home/aialfred/alfred
mkdir -p tests/scripts
touch tests/scripts/__init__.py
python3 -m pytest tests/scripts/test_rucktalk_rig_props.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError: No module named 'scripts.rucktalk_rig_props'`.

- [ ] **Step 3: Implement the module**

Create `scripts/rucktalk_rig_props.py`:

```python
"""Prop builders for the RuckTalk episode pipeline Remotion rigs.

Two builders — one per rig — isolate the prop-shaping concern from the
render-orchestration concern in rucktalk_episode_pipeline.py. During the
Phase 2 migration both exist side-by-side so we can render BOTH for a
given episode and compare output. Phase 4 will delete the old builder
(and the deprecated RuckTalkClip composition) once the cutover is stable.
"""
from __future__ import annotations

from typing import TypedDict


class CaptionPhrase(TypedDict):
    text: str
    startFrame: int
    endFrame: int


class RuckTalkClipProps(TypedDict):
    videoSrc: str
    episodeNumber: int
    episodeTitle: str
    contextLine: str
    hostName: str
    guestName: str
    captionPhrases: list[CaptionPhrase]


class MagazineRigProps(TypedDict, total=False):
    # total=False so guestName can be omitted — MagazineRig treats it optional.
    brand: str
    clipSrc: str
    episodeNumber: int
    episodeTitle: str
    captionPhrases: list[CaptionPhrase]
    hostName: str
    guestName: str


def build_rucktalkclip_props(
    *,
    clip_filename: str,
    episode_number: int,
    episode_title: str,
    context_line: str,
    host_name: str,
    guest_name: str | None,
    caption_phrases: list[CaptionPhrase],
) -> RuckTalkClipProps:
    """Props for the deprecated RuckTalkClip composition (Phase 1)."""
    return {
        "videoSrc": clip_filename,
        "episodeNumber": episode_number,
        "episodeTitle": episode_title,
        "contextLine": context_line,
        "hostName": host_name,
        "guestName": guest_name or "",
        "captionPhrases": caption_phrases,
    }


def build_magazinerig_props(
    *,
    clip_filename: str,
    episode_number: int,
    episode_title: str,
    host_name: str,
    guest_name: str | None,
    caption_phrases: list[CaptionPhrase],
) -> MagazineRigProps:
    """Props for the new MagazineRig composition (Phase 2 target).

    Differences from RuckTalkClip shape:
      - adds: brand (always "rucktalk")
      - renames: videoSrc -> clipSrc
      - drops:   contextLine (MagazineRig does not render it)
    """
    props: MagazineRigProps = {
        "brand": "rucktalk",
        "clipSrc": clip_filename,
        "episodeNumber": episode_number,
        "episodeTitle": episode_title,
        "hostName": host_name,
        "captionPhrases": caption_phrases,
    }
    if guest_name:
        props["guestName"] = guest_name
    else:
        props["guestName"] = ""
    return props
```

- [ ] **Step 4: Run tests — all 5 PASS**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/test_rucktalk_rig_props.py -v 2>&1 | tail -10
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_rig_props.py tests/scripts/__init__.py tests/scripts/test_rucktalk_rig_props.py
git commit -m "feat(rucktalk): extract rig prop builders for Phase 2 migration

New module scripts/rucktalk_rig_props.py exposes:
  - build_rucktalkclip_props (matches current RuckTalkClip shape)
  - build_magazinerig_props  (new MagazineRig shape, brand + clipSrc)

Pure functions, 5 unit tests covering shape + guest/no-guest paths.
No changes to rucktalk_episode_pipeline.py yet — wiring happens in Task 2."
```

---

## Task 2: Add EPISODE_RIG flag + wire the prop builders into `_render_branded_clip`

**Files:**
- Modify: `scripts/rucktalk_episode_pipeline.py`

- [ ] **Step 1: Read the current `_render_branded_clip` function**

Open `scripts/rucktalk_episode_pipeline.py` and locate `_render_branded_clip` (starts at line ~853, ends at line ~966). The portion to change is the props-build + subprocess call (roughly lines 914-963). Keep the caption-phrase generation (lines 871-912) unchanged — that logic is rig-independent.

- [ ] **Step 2: Add the EPISODE_RIG constant and the import**

Near the top of `scripts/rucktalk_episode_pipeline.py` (after the existing imports), add:

```python
import os
from scripts.rucktalk_rig_props import (
    build_rucktalkclip_props,
    build_magazinerig_props,
)

# Phase 2 migration flag. Default is the deprecated rig during cutover;
# Task 7 flips it to "MagazineRig". Set EPISODE_RIG env var to override at runtime.
EPISODE_RIG = os.environ.get("EPISODE_RIG", "RuckTalkClip")
```

(`os` may already be imported — if so, do not duplicate the import.)

- [ ] **Step 3: Replace the inline props-building block**

Inside `_render_branded_clip`, find this block (around lines 924-946):

```python
    # Build Remotion props — videoSrc is the filename in public/, staticFile() resolves it
    props = {
        "videoSrc": clip_filename,
        "episodeNumber": episode_number,
        "episodeTitle": episode_title,
        "contextLine": context_line,
        "hostName": host_name,
        "guestName": guest_name or "",
        "captionPhrases": phrases,
    }

    # Write props to a temp file to avoid shell escaping issues with JSON
    props_file = Path(remotion_dir) / f"props_{output_path.stem}.json"
    props_file.write_text(json.dumps(props))

    npx = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/npx"
    cmd = [
        npx, "remotion", "render",
        "src/index.ts", "RuckTalkClip",
        f"--props={str(props_file)}",
        f"--frames=0-{min(duration_frames, 1800)}",
        str(output_path),
    ]
```

Replace it with:

```python
    # Build props + pick composition id based on EPISODE_RIG flag
    if EPISODE_RIG == "MagazineRig":
        props = build_magazinerig_props(
            clip_filename=clip_filename,
            episode_number=episode_number,
            episode_title=episode_title,
            host_name=host_name,
            guest_name=guest_name,
            caption_phrases=phrases,
        )
        composition_id = "MagazineRig"
    else:
        props = build_rucktalkclip_props(
            clip_filename=clip_filename,
            episode_number=episode_number,
            episode_title=episode_title,
            context_line=context_line,
            host_name=host_name,
            guest_name=guest_name,
            caption_phrases=phrases,
        )
        composition_id = "RuckTalkClip"

    # Write props to a temp file to avoid shell escaping issues with JSON
    props_file = Path(remotion_dir) / f"props_{output_path.stem}.json"
    props_file.write_text(json.dumps(props))

    logger.info("Rendering clip via %s (EPISODE_RIG=%s)", composition_id, EPISODE_RIG)

    npx = "/home/aialfred/.nvm/versions/node/v22.22.0/bin/npx"
    cmd = [
        npx, "remotion", "render",
        "src/index.ts", composition_id,
        f"--props={str(props_file)}",
        f"--frames=0-{min(duration_frames, 1800)}",
        str(output_path),
    ]
```

- [ ] **Step 4: Syntax check**

```bash
cd /home/aialfred/alfred
python3 -c "import scripts.rucktalk_episode_pipeline" 2>&1 | tail -5
```

Expected: no output (clean import). If `ImportError` appears, fix the `scripts.rucktalk_rig_props` import path.

- [ ] **Step 5: Run the existing unit tests plus the new ones**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/test_rucktalk_rig_props.py tests/providers/ -v 2>&1 | tail -15
```

Expected: 5 (new) + ~10 (providers) tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_episode_pipeline.py
git commit -m "feat(rucktalk): wire EPISODE_RIG flag into _render_branded_clip

New EPISODE_RIG env var (default RuckTalkClip) selects which rig the
episode pipeline renders. Delegates prop-building to the new
scripts/rucktalk_rig_props module. No behavior change at default —
renders the same deprecated rig as before unless EPISODE_RIG=MagazineRig
is set in the environment."
```

---

## Task 3: Add `--compare` CLI for side-by-side render

**Files:**
- Modify: `scripts/rucktalk_episode_pipeline.py`

- [ ] **Step 1: Locate the CLI arg-parser**

Find where `argparse.ArgumentParser` is instantiated in `rucktalk_episode_pipeline.py`. Grep for it:

```bash
cd /home/aialfred/alfred
grep -n "ArgumentParser\|add_argument" scripts/rucktalk_episode_pipeline.py | head -20
```

If no CLI exists (the file is import-only), skip adding CLI flags and instead expose a new top-level function `compare_rigs_for_clip()` that can be invoked from `python3 -c` or an ad-hoc script.

- [ ] **Step 2: Implement `compare_rigs_for_clip()`**

Add this new function at the end of `scripts/rucktalk_episode_pipeline.py` (after `_render_branded_clip`):

```python
def compare_rigs_for_clip(
    raw_clip_path: Path,
    output_dir: Path,
    episode_number: int,
    episode_title: str,
    context_line: str,
    host_name: str,
    guest_name: str | None,
    transcript: dict,
    clip_start: float,
    clip_end: float,
    duration_frames: int,
) -> tuple[Path, Path]:
    """Render the same clip through BOTH rigs and return paths to both outputs.

    Used once, at the Phase 2 cutover gate, to produce a side-by-side
    comparison for human review before flipping EPISODE_RIG default.
    """
    global EPISODE_RIG
    output_dir.mkdir(parents=True, exist_ok=True)

    old_path = output_dir / f"ep{episode_number}_rucktalkclip.mp4"
    new_path = output_dir / f"ep{episode_number}_magazinerig.mp4"

    # Save and restore EPISODE_RIG so we don't leak state to the caller.
    saved = EPISODE_RIG
    try:
        EPISODE_RIG = "RuckTalkClip"
        _render_branded_clip(
            raw_clip_path, old_path, episode_number, episode_title,
            context_line, host_name, guest_name, transcript,
            clip_start, clip_end, duration_frames,
        )
        EPISODE_RIG = "MagazineRig"
        _render_branded_clip(
            raw_clip_path, new_path, episode_number, episode_title,
            context_line, host_name, guest_name, transcript,
            clip_start, clip_end, duration_frames,
        )
    finally:
        EPISODE_RIG = saved

    return old_path, new_path
```

Note: the `global EPISODE_RIG` trick works because `_render_branded_clip` reads the module-level constant at call time; reassigning the module attribute flips which branch runs. Tests in Task 4 verify this works.

- [ ] **Step 3: Write the test**

Create `tests/scripts/test_episode_pipeline_compare.py`:

```python
"""Smoke test for compare_rigs_for_clip — verifies it calls the render
function twice with the two composition ids. Uses monkeypatch, no real render."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import scripts.rucktalk_episode_pipeline as pipeline


def test_compare_rigs_calls_both(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_render(raw_clip_path, output_path, *args, **kwargs):
        # Infer which rig was requested from the module-level EPISODE_RIG at call time
        calls.append(pipeline.EPISODE_RIG)
        output_path.write_bytes(b"fake mp4")
        return True

    monkeypatch.setattr(pipeline, "_render_branded_clip", fake_render)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"x")
    out_dir = tmp_path / "out"

    old_path, new_path = pipeline.compare_rigs_for_clip(
        raw_clip_path=raw,
        output_dir=out_dir,
        episode_number=42,
        episode_title="Test",
        context_line="ctx",
        host_name="MIKE",
        guest_name=None,
        transcript={"segments": []},
        clip_start=0.0,
        clip_end=5.0,
        duration_frames=150,
    )

    assert calls == ["RuckTalkClip", "MagazineRig"]
    assert old_path.name == "ep42_rucktalkclip.mp4"
    assert new_path.name == "ep42_magazinerig.mp4"
    assert old_path.exists() and new_path.exists()


def test_compare_rigs_restores_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pipeline, "_render_branded_clip",
                        lambda *a, **k: a[1].write_bytes(b"x") or True)
    pipeline.EPISODE_RIG = "RuckTalkClip"

    raw = tmp_path / "raw.mp4"; raw.write_bytes(b"x")
    pipeline.compare_rigs_for_clip(
        raw_clip_path=raw, output_dir=tmp_path / "out",
        episode_number=1, episode_title="T", context_line="c",
        host_name="M", guest_name=None, transcript={"segments": []},
        clip_start=0.0, clip_end=1.0, duration_frames=30,
    )

    assert pipeline.EPISODE_RIG == "RuckTalkClip"  # restored
```

- [ ] **Step 4: Run — both tests PASS**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/test_episode_pipeline_compare.py -v 2>&1 | tail -10
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_episode_pipeline.py tests/scripts/test_episode_pipeline_compare.py
git commit -m "feat(rucktalk): compare_rigs_for_clip — render both rigs side-by-side

Helper used at the Phase 2 cutover gate. Renders the same raw clip twice
— once through RuckTalkClip, once through MagazineRig — and returns both
paths. Safely saves/restores EPISODE_RIG. Tests verify the two render
calls hit the right composition ids and that the flag is restored."
```

---

## Task 4: Render a real side-by-side and email Mike for approval

**Files:** (no new files)

- [ ] **Step 1: Pick a recent real episode clip**

Find the most recent rendered episode clip from the rucktalk pipeline:

```bash
ls -t /home/aialfred/alfred/data/rucktalk/clips/*.mp4 2>/dev/null | head -3
ls -t /home/aialfred/alfred/data/rucktalk/episodes/ 2>/dev/null | head -3
```

Identify one episode we already have the transcript JSON and raw clip for. If none exist, STOP and ask the user for a sample episode to use.

- [ ] **Step 2: Run compare_rigs_for_clip on it**

From a Python one-liner. Use the episode metadata from its transcript/metadata JSON. Example — replace the placeholders with real values from the chosen episode:

```bash
cd /home/aialfred/alfred
mkdir -p data/rucktalk/phase2_compare
python3 <<'PY'
import json
from pathlib import Path
from scripts.rucktalk_episode_pipeline import compare_rigs_for_clip

# EDIT THESE with the real episode's data:
RAW_CLIP = Path("/home/aialfred/alfred/data/rucktalk/clips/clip_REAL_EPISODE.mp4")
TRANSCRIPT = Path("/home/aialfred/alfred/data/rucktalk/episodes/EP100_transcript.json")
EPISODE_NUMBER = 100
EPISODE_TITLE = "Reinvent Or Get Left Behind"
CONTEXT_LINE = "Ford got comfortable. The competition reinvented."
HOST_NAME = "MIKE JOHNSON"
GUEST_NAME = None
CLIP_START = 0.0
CLIP_END = 60.0
DURATION_FRAMES = 1800

transcript = json.loads(TRANSCRIPT.read_text())
out_dir = Path("/home/aialfred/alfred/data/rucktalk/phase2_compare")

old_path, new_path = compare_rigs_for_clip(
    raw_clip_path=RAW_CLIP,
    output_dir=out_dir,
    episode_number=EPISODE_NUMBER,
    episode_title=EPISODE_TITLE,
    context_line=CONTEXT_LINE,
    host_name=HOST_NAME,
    guest_name=GUEST_NAME,
    transcript=transcript,
    clip_start=CLIP_START,
    clip_end=CLIP_END,
    duration_frames=DURATION_FRAMES,
)
print(f"OLD (RuckTalkClip): {old_path} ({old_path.stat().st_size/1024/1024:.1f} MB)")
print(f"NEW (MagazineRig):  {new_path} ({new_path.stat().st_size/1024/1024:.1f} MB)")
PY
```

Expected: both files rendered, non-zero size. Renders take 1–3 minutes each.

- [ ] **Step 3: Email Mike with both attachments**

```bash
cd /home/aialfred/alfred
python3 <<'PY'
import os, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("/home/aialfred/alfred/config/.env")

pw = os.environ["EMAIL_PASS_ALFRED_GW"]
compare_dir = Path("/home/aialfred/alfred/data/rucktalk/phase2_compare")
old_mp4 = next(compare_dir.glob("*_rucktalkclip.mp4"))
new_mp4 = next(compare_dir.glob("*_magazinerig.mp4"))

msg = MIMEMultipart()
msg["From"] = "Alfred <alfred@groundrushinc.com>"
msg["To"] = "mjohnson@groundrushinc.com"
msg["Subject"] = "Remotion Phase 2 — episode rig cutover gate (please approve)"

body = MIMEText("""<html><body style="font-family:sans-serif;line-height:1.55;max-width:640px;padding:20px">
<p>Sir,</p>
<p>Phase 2 cutover gate. Both attachments are <strong>the same source clip</strong> rendered through two different rigs:</p>
<ul>
  <li><strong>*_rucktalkclip.mp4</strong> — current production rig (deprecated).</li>
  <li><strong>*_magazinerig.mp4</strong> — new MagazineRig from Phase 1 library.</li>
</ul>
<p>After your approval I flip the EPISODE_RIG default in <code>rucktalk_episode_pipeline.py</code>
from <code>RuckTalkClip</code> → <code>MagazineRig</code>. The old rig stays available
via <code>EPISODE_RIG=RuckTalkClip</code> env var for one-command rollback if anything
goes wrong.</p>
<p>Reply with one of:</p>
<ul>
  <li><strong>Approve</strong> — flip the default, ship.</li>
  <li><strong>Revise</strong> — tell me what to change in MagazineRig (colors, layout, caption style, etc.).</li>
  <li><strong>Keep old</strong> — abort Phase 2, stay on RuckTalkClip.</li>
</ul>
<p>— Alfred</p>
</body></html>""", "html")
msg.attach(body)

for f in (old_mp4, new_mp4):
    with open(f, "rb") as fh:
        part = MIMEBase("video", "mp4")
        part.set_payload(fh.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment", filename=f.name)
    msg.attach(part)

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
    s.login("alfred@groundrushinc.com", pw)
    s.send_message(msg)
print(f"sent — {old_mp4.name}, {new_mp4.name}")
PY
```

- [ ] **Step 4: STOP — wait for approval**

This is a **human approval gate**. Do NOT proceed to Task 5 until Mike replies with one of: `Approve`, `Revise`, `Keep old`.

- If **Approve** → continue to Task 5.
- If **Revise** → gather his notes, loop back to the Remotion library (not this plan) and amend `MagazineRig`, re-render, re-send. This plan is paused until the new render is approved.
- If **Keep old** → abandon Phase 2. Close the plan. No changes needed beyond what Tasks 1-3 already shipped (the flag stays at `RuckTalkClip` default forever).

There is no commit for Task 4 — it produces artifacts, not code changes.

---

## Task 5: Flip the default to MagazineRig

**Files:**
- Modify: `scripts/rucktalk_episode_pipeline.py` (change one constant)

Execute this task ONLY after Mike approves in Task 4.

- [ ] **Step 1: Change the default**

Edit `scripts/rucktalk_episode_pipeline.py`. Locate the line:

```python
EPISODE_RIG = os.environ.get("EPISODE_RIG", "RuckTalkClip")
```

Replace with:

```python
EPISODE_RIG = os.environ.get("EPISODE_RIG", "MagazineRig")
```

- [ ] **Step 2: Verify the module still imports cleanly**

```bash
cd /home/aialfred/alfred
python3 -c "from scripts.rucktalk_episode_pipeline import EPISODE_RIG; print(EPISODE_RIG)"
```

Expected: prints `MagazineRig`.

- [ ] **Step 3: Re-run all tests**

```bash
cd /home/aialfred/alfred
python3 -m pytest tests/scripts/ tests/providers/ -v 2>&1 | tail -15
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/aialfred/alfred
git add scripts/rucktalk_episode_pipeline.py
git commit -m "feat(rucktalk): flip EPISODE_RIG default from RuckTalkClip to MagazineRig

Phase 2 cutover approved by Mike via email gate. The episode pipeline
now renders clips via the new MagazineRig composition by default.

Rollback: set env var EPISODE_RIG=RuckTalkClip on the cron shell to
revert without a code change."
```

---

## Task 6: Monitor the next real episode render

**Files:** (no new files)

- [ ] **Step 1: Identify next episode trigger**

The RuckTalk episode pipeline runs from cron (see `crontab -l`). The next new episode drop into NextCloud will trigger a full render cycle. Alternatively, force a re-render of the most recent episode to verify:

```bash
cd /home/aialfred/alfred
# Remove cached branded output to force re-render
# (adjust path to the real episode's branded clip dir as needed)
ls data/rucktalk/branded/ 2>/dev/null
```

If you need to test immediately without waiting for a new episode, delete the cached branded mp4s for ONE episode and re-invoke the pipeline for that episode:

```bash
python3 -c "from scripts.rucktalk_episode_pipeline import process_pending_episodes; process_pending_episodes()"
```

- [ ] **Step 2: Verify the render succeeded**

Watch the log:

```bash
tail -f /home/aialfred/alfred/data/logs/rucktalk_episode.log 2>/dev/null
```

Look for the line `Rendering clip via MagazineRig (EPISODE_RIG=MagazineRig)` followed by `Branded clip rendered:`.

- [ ] **Step 3: Verify the uploaded clip on WordPress / YouTube**

Check that the branded clip landed on:
- WordPress (rucktalk.com admin — the episode post)
- YouTube (scheduled clip uploads, if applicable)

Manually open one clip and confirm it's playable and visually correct.

- [ ] **Step 4: Notify Mike**

```bash
cd /home/aialfred/alfred
python3 <<'PY'
import sys; sys.path.insert(0, ".")
from integrations.email.client import EmailClient
EmailClient().send_email(
    account="alfred-gw",
    to="mjohnson@groundrushinc.com",
    subject="Phase 2 cutover live — next episode rendering via MagazineRig",
    body="""Sir,

EPISODE_RIG default is now MagazineRig. The next real episode (or a forced
re-render) produced a working clip using the new rig. WordPress + clip queue
checked OK.

Rollback remains one env var away: EPISODE_RIG=RuckTalkClip on the cron.

I'll watch the next two episode cycles for any regressions.

— Alfred""",
)
print("sent")
PY
```

No commit for Task 6 — it's verification.

---

## Task 7: Two-cycle stability watch

**Files:** (no new files)

- [ ] **Step 1: Monitor the next TWO episodes**

When the next two episode drops complete, confirm:
- Both rendered via MagazineRig (log line `Rendering clip via MagazineRig`)
- Both uploaded successfully to WordPress + clip queue
- Neither triggered a Telegram error alert

- [ ] **Step 2: Summary email to Mike**

After TWO clean cycles:

```bash
cd /home/aialfred/alfred
python3 <<'PY'
import sys; sys.path.insert(0, ".")
from integrations.email.client import EmailClient
EmailClient().send_email(
    account="alfred-gw",
    to="mjohnson@groundrushinc.com",
    subject="Phase 2 stable — 2 episode cycles clean on MagazineRig",
    body="""Sir,

Two consecutive episode cycles have rendered cleanly via MagazineRig with no
errors, no rollbacks, no manual intervention. Phase 2 is considered stable.

Next: Phase 3 (cutover rucktalk_daily_social.py) on your word.

— Alfred""",
)
print("sent")
PY
```

- [ ] **Step 3: Tag the migration**

```bash
cd /home/aialfred/alfred
git tag -a rucktalk-phase2-stable -m "Phase 2 stable — episode pipeline on MagazineRig, 2 clean cycles"
```

---

## Self-Review

**Spec coverage (Section 6 Phase 2):**
- "Cut `rucktalk_episode_pipeline.py` over from `RuckTalkClip` → `MagazineRig`" → Tasks 1, 2, 5 ✓
- "Render side-by-side and compare before shipping" → Tasks 3, 4 ✓
- Rollback plan (one env var) → Task 5 step 4 commit message + EPISODE_RIG flag ✓
- Side-by-side + human approval gate → Task 4 ✓
- Two-cycle stability before declaring done → Task 7 ✓

**Placeholder scan:** none. Every step has exact code or exact commands. Task 4 step 2 contains placeholder variables (`REAL_EPISODE`, etc.) the implementer MUST fill in from real episode data — that's flagged clearly and is intentional (the plan can't know which episode the team picks).

**Type consistency:**
- `CaptionPhrase` TypedDict defined in `rucktalk_rig_props.py` Task 1, referenced by both builders consistently.
- `build_magazinerig_props` always emits `brand="rucktalk"` and `clipSrc` (not `videoSrc`); tests assert this.
- `_render_branded_clip` signature is unchanged — only the body rebind's `props` and `composition_id`.
- `EPISODE_RIG` is the single source of truth for rig selection, read at call time so `compare_rigs_for_clip` can flip it safely.

**Out of scope (deferred):**
- Phase 3 (`rucktalk_daily_social.py` migration) — separate plan.
- Phase 4 (delete `_deprecated/` templates, drop `build_rucktalkclip_props`) — separate plan.
- Any changes to Remotion library compositions — Phase 1 shipped those, locked.

Plan verified. Ready for execution.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-17-remotion-phase2-episode-migration.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good fit because Task 4 is a manual approval gate and the rest are small.
2. **Inline Execution** — I execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach, sir?
