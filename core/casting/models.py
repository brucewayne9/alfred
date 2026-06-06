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
