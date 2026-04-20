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


def derive_word_beats_from_script(
    script: str,
    audio_duration_s: float,
    fps: int = 30,
) -> list[CaptionPhrase]:
    """Break a TTS script into timed caption phrases.

    Sentences are the unit of typographic emphasis. Time is allocated
    proportionally to each sentence's character count (approximation of
    speaking time). Result is a list of phrases with non-overlapping
    frame ranges, guaranteed to fit within [0, audio_duration_s*fps].
    """
    # Split on sentence-ending punctuation, drop empties
    raw = re.split(r"(?<=[.!?])\s+", script.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        return []

    total_chars = sum(len(s) for s in sentences) or 1
    total_frames = int(audio_duration_s * fps)

    beats: list[CaptionPhrase] = []
    cursor = 0
    for i, s in enumerate(sentences):
        share = len(s) / total_chars
        span = max(1, int(total_frames * share))
        end = cursor + span if i < len(sentences) - 1 else total_frames
        # Upper-case short punchy phrase for hero caption display
        clean = re.sub(r"[^A-Za-z0-9\s'-]", "", s).upper().strip()
        variant = "scaleOnBeat" if len(clean) <= 20 else "stacked"
        beats.append({
            "word": clean,
            "startFrame": cursor,
            "endFrame": end,
            "variant": variant,
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
