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
