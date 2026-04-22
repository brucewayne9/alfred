"""AutoBrief builders for the daily social engine (Phase 3 migration).

Converts the Python-side daily content state (LLM-generated script, Kokoro
audio duration, ComfyUI Cloud video) into the AutoBrief shape consumed by
the Remotion engine/autoProps.ts.

One builder per daily social mode:
  - build_monologue_brief   -> resolves to KineticTypeRig via autoProps rotation
  - build_conversation_brief -> resolves to GritDocRig   via autoProps rotation

Rotation is controlled by the caller so autoProps picks the expected rig.
autoProps cycles through ["MagazineRig", "GritDocRig", "KineticTypeRig"];
the caller must pass rotation values that land on the intended rig.
"""
from __future__ import annotations

import re
from typing import TypedDict


class CaptionPhrase(TypedDict):
    word: str
    startFrame: int
    endFrame: int
    variant: str  # "single" | "stacked" | "scaleOnBeat"


class MonologueAutoBrief(TypedDict):
    brand: str
    date: str
    rotation: int
    bgClip: str
    wordBeats: list[CaptionPhrase]


class ConversationClip(TypedDict):
    src: str
    durationFrames: int


class ConversationAutoBrief(TypedDict):
    brand: str
    date: str
    rotation: int
    clips: list[ConversationClip]


# autoProps.ts cycles through these three at index `rotation % 3`.
# MagazineRig is skipped for daily content (no episode data available),
# so valid rotation values are those that land on GritDocRig (1) or
# KineticTypeRig (2).
_ROTATION_FOR_KINETIC_TYPE = 2
_ROTATION_FOR_GRIT_DOC = 1


_MAX_CHARS_PER_BEAT = 12  # KineticTypeRig renders at ~42% of 1920px height —
                          # anything longer overflows horizontally.


def derive_word_beats_from_script(
    script: str,
    audio_duration_s: float,
    fps: int = 30,
) -> list[CaptionPhrase]:
    """Break a TTS script into timed caption beats that fit KineticTypeRig.

    Strategy: split on BOTH sentence-enders (.!?) AND commas, then further
    chunk any remaining beat > _MAX_CHARS_PER_BEAT into word-level groups of
    1-2 words. This keeps each on-screen phrase short enough that the rig's
    large hero typography doesn't overflow.
    """
    # Split on sentence-enders and commas
    raw = re.split(r"(?<=[.!?,])\s+|\s*,\s*", script.strip())
    phrases = [p.strip() for p in raw if p and p.strip()]
    if not phrases:
        return []

    # Second pass — break over-long phrases into 1-2 word chunks
    chunks: list[str] = []
    for p in phrases:
        if len(p) <= _MAX_CHARS_PER_BEAT:
            chunks.append(p)
            continue
        words = p.split()
        i = 0
        while i < len(words):
            # Try 2 words; back off to 1 if still too long
            candidate = " ".join(words[i:i + 2])
            if len(candidate) <= _MAX_CHARS_PER_BEAT:
                chunks.append(candidate)
                i += 2
            else:
                chunks.append(words[i])
                i += 1

    # Allocate frames proportionally to chunk char count.
    # Each beat needs enough frames to be both legible AND cover the pop-in/
    # pop-out fade window on the rig side (popIn=3 + popOut=4 = 7 → 12 is safer).
    MIN_BEAT_FRAMES = 12
    total_chars = sum(len(c) for c in chunks) or 1
    total_frames = int(audio_duration_s * fps)

    beats: list[CaptionPhrase] = []
    cursor = 0
    for i, c in enumerate(chunks):
        share = len(c) / total_chars
        span = max(MIN_BEAT_FRAMES, int(total_frames * share))
        end = cursor + span if i < len(chunks) - 1 else max(total_frames, cursor + MIN_BEAT_FRAMES)
        clean = re.sub(r"[^A-Za-z0-9\s'-]", "", c).upper().strip()
        if not clean:
            continue
        beats.append({
            "word": clean,
            "startFrame": cursor,
            "endFrame": end,
            "variant": "scaleOnBeat",
        })
        cursor = end

    return beats


def build_monologue_brief(
    *,
    date: str,
    rotation: int,
    script: str,
    bg_clip_public_name: str,
    audio_duration_s: float,
    fps: int = 30,
) -> MonologueAutoBrief:
    """Build an AutoBrief that resolves to KineticTypeRig.

    Caller is responsible for choosing a `rotation` that lands on
    KineticTypeRig (rotation % 3 == 2). `bg_clip_public_name` is the
    filename inside Remotion's public/ dir (the mp4 must be copied there
    before render).
    """
    beats = derive_word_beats_from_script(script, audio_duration_s, fps)
    if not beats:
        raise ValueError("script produced zero word beats")
    return {
        "brand": "rucktalk",
        "date": date,
        "rotation": rotation,
        "bgClip": bg_clip_public_name,
        "wordBeats": beats,
    }


def build_conversation_brief(
    *,
    date: str,
    rotation: int,
    bg_clips: list[ConversationClip],
) -> ConversationAutoBrief:
    """Build an AutoBrief that resolves to GritDocRig (no captions, just B-roll montage).

    Caller must pass rotation such that rotation % 3 == 1 (GritDocRig).
    """
    if not bg_clips:
        raise ValueError("bg_clips must not be empty")
    return {
        "brand": "rucktalk",
        "date": date,
        "rotation": rotation,
        "clips": bg_clips,
    }


def pick_rotation_for_kinetic_type() -> int:
    """Return a rotation index that makes autoProps pick KineticTypeRig."""
    return _ROTATION_FOR_KINETIC_TYPE


def pick_rotation_for_grit_doc() -> int:
    """Return a rotation index that makes autoProps pick GritDocRig."""
    return _ROTATION_FOR_GRIT_DOC
