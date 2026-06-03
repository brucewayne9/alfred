"""Forge caption-style catalog — the single source of truth for the Film Montage
caption gallery.

Python owns the presets; the resolved token spec is passed straight into Remotion
props (CaptionStudioRig -> CaptionStyleEngine), so adding a style is pure data
here — no Remotion edits. The Forge UI fetches `list_styles()` to build the
visual gallery; the renderer calls `resolve(style_id)` to get the token spec.

Token keys are camelCase because they pass through verbatim as Remotion props
(see src/primitives/CaptionStyleEngine.tsx :: CaptionStyleSpec).

Engines (all implemented in CaptionStyleEngine):
  karaoke    — whole line, word fills base->active L->R (classic TikTok)
  highlight  — marker swipe behind the active word (Hormozi marker)
  wordpop    — words spring in one by one, active word tinted
  box        — wordpop + active word sits in a filled rounded box (Hormozi box)
  kinetic    — words slide up + skew in, energetic
  typewriter — characters type across the line with a blinking cursor
  minimal    — clean single line, subtle fade (IG-clean)
"""
from __future__ import annotations

SANS = "Montserrat, Inter, sans-serif"
MONO = "JetBrains Mono, ui-monospace, monospace"
SERIF = "Georgia, 'Times New Roman', serif"

# Palette (Mainstay / Rod Wave moody)
GOLD = "#E8B84B"
WHITE = "#ffffff"
GREY = "#b9b4a8"
INK = "#11100c"
PINK = "#FF4D8D"
ICE = "#8FD3FF"
MINT = "#6FE3B0"
RED = "#FF3B3B"
PURPLE = "#B57BFF"
GREEN = "#3DF07A"

# Vertical anchors (px from bottom in a 1920-tall frame)
LOWER = 470
MID = 900
TOP = 1380


def _s(engine, **kw) -> dict:
    """Build a token spec with sane defaults; kw overrides."""
    base = {
        "engine": engine,
        "font": SANS,
        "weight": 800,
        "size": 72,
        "upper": True,
        "italic": False,
        "color": WHITE,
        "active": GOLD,
        "boxFill": None,
        "boxRadius": 12,
        "boxPadX": 18,
        "boxPadY": 8,
        "outline": 0,
        "outlineColor": INK,
        "shadow": True,
        "bottomPx": LOWER,
        "letterSpacing": 1,
        "maxWords": 4,
        "intensity": 1.0,
    }
    base.update(kw)
    return base


# (id, name, family, spec)
CATALOG: list[dict] = [
    # ── Karaoke ────────────────────────────────────────────────────────────
    {"id": "karaoke_gold",  "name": "Gold Sweep",     "family": "Karaoke", "spec": _s("karaoke", color=WHITE, active=GOLD)},
    {"id": "karaoke_white", "name": "Clean Fill",     "family": "Karaoke", "spec": _s("karaoke", color=GREY, active=WHITE)},
    {"id": "karaoke_pink",  "name": "Neon Pink",      "family": "Karaoke", "spec": _s("karaoke", color=WHITE, active=PINK)},
    {"id": "karaoke_ice",   "name": "Ice Sweep",      "family": "Karaoke", "spec": _s("karaoke", color=WHITE, active=ICE)},
    {"id": "karaoke_mint",  "name": "Mint Sweep",     "family": "Karaoke", "spec": _s("karaoke", color=WHITE, active=MINT)},
    {"id": "karaoke_caps",  "name": "Big Caps Gold",  "family": "Karaoke", "spec": _s("karaoke", color=WHITE, active=GOLD, size=86, letterSpacing=2)},

    # ── Highlight (marker) ─────────────────────────────────────────────────
    {"id": "hl_gold",   "name": "Gold Marker",   "family": "Highlight", "spec": _s("highlight", active=GOLD)},
    {"id": "hl_pink",   "name": "Pink Marker",   "family": "Highlight", "spec": _s("highlight", active=PINK)},
    {"id": "hl_white",  "name": "White Marker",  "family": "Highlight", "spec": _s("highlight", active=WHITE)},
    {"id": "hl_purple", "name": "Purple Marker", "family": "Highlight", "spec": _s("highlight", active=PURPLE)},
    {"id": "hl_red",    "name": "Red Marker",    "family": "Highlight", "spec": _s("highlight", active=RED)},

    # ── Word-Pop ───────────────────────────────────────────────────────────
    {"id": "pop_gold",  "name": "Pop Gold",   "family": "Word-Pop", "spec": _s("wordpop", active=GOLD)},
    {"id": "pop_white", "name": "Pop White",  "family": "Word-Pop", "spec": _s("wordpop", active=WHITE, color=GREY)},
    {"id": "pop_pink",  "name": "Pop Pink",   "family": "Word-Pop", "spec": _s("wordpop", active=PINK)},
    {"id": "pop_big",   "name": "Pop Hero",   "family": "Word-Pop", "spec": _s("wordpop", active=WHITE, size=92, letterSpacing=1.5)},
    {"id": "pop_punch", "name": "Punch Pink", "family": "Word-Pop", "spec": _s("wordpop", active=PINK, size=88, intensity=1.4)},

    # ── Box (Hormozi) ──────────────────────────────────────────────────────
    {"id": "box_gold",   "name": "Gold Box",    "family": "Box", "spec": _s("box", color=WHITE, boxFill=GOLD, active=INK)},
    {"id": "box_black",  "name": "Black Box",   "family": "Box", "spec": _s("box", color=WHITE, boxFill=INK, active=GOLD)},
    {"id": "box_white",  "name": "White Box",   "family": "Box", "spec": _s("box", color=WHITE, boxFill=WHITE, active=INK)},
    {"id": "box_pink",   "name": "Pink Box",    "family": "Box", "spec": _s("box", color=WHITE, boxFill=PINK, active=WHITE)},
    {"id": "box_red",    "name": "Red Box",     "family": "Box", "spec": _s("box", color=WHITE, boxFill=RED, active=WHITE)},
    {"id": "box_purple", "name": "Purple Box",  "family": "Box", "spec": _s("box", color=WHITE, boxFill=PURPLE, active=WHITE)},
    {"id": "box_gold_top", "name": "Gold Box · Top", "family": "Box", "spec": _s("box", color=WHITE, boxFill=GOLD, active=INK, bottomPx=TOP)},

    # ── Kinetic ────────────────────────────────────────────────────────────
    {"id": "kin_gold",  "name": "Kinetic Gold",  "family": "Kinetic", "spec": _s("kinetic", active=GOLD)},
    {"id": "kin_white", "name": "Kinetic White", "family": "Kinetic", "spec": _s("kinetic", active=WHITE, color=GREY)},
    {"id": "kin_pink",  "name": "Kinetic Pink",  "family": "Kinetic", "spec": _s("kinetic", active=PINK)},
    {"id": "kin_hard",  "name": "Slam Cut",      "family": "Kinetic", "spec": _s("kinetic", active=WHITE, size=84, intensity=1.5)},
    {"id": "kin_mint",  "name": "Kinetic Mint",  "family": "Kinetic", "spec": _s("kinetic", active=MINT)},

    # ── Typewriter ─────────────────────────────────────────────────────────
    {"id": "type_gold",  "name": "Type Gold",   "family": "Typewriter", "spec": _s("typewriter", font=MONO, upper=False, color=WHITE, active=GOLD, size=58, weight=600)},
    {"id": "type_green", "name": "Terminal",    "family": "Typewriter", "spec": _s("typewriter", font=MONO, upper=False, color=GREEN, active=GREEN, size=56, weight=600)},
    {"id": "type_white", "name": "Type Clean",  "family": "Typewriter", "spec": _s("typewriter", font=MONO, upper=False, color=WHITE, active=WHITE, size=58, weight=600)},

    # ── Minimal (IG-clean) ─────────────────────────────────────────────────
    {"id": "min_white",  "name": "Clean White",  "family": "Minimal", "spec": _s("minimal", upper=False, color=WHITE, size=60, weight=600, shadow=True)},
    {"id": "min_gold",   "name": "Clean Gold",   "family": "Minimal", "spec": _s("minimal", upper=False, color=GOLD, size=60, weight=600)},
    {"id": "min_caps",   "name": "Wide Caps",    "family": "Minimal", "spec": _s("minimal", upper=True, color=WHITE, size=52, weight=700, letterSpacing=6)},
    {"id": "min_italic", "name": "Editorial",    "family": "Minimal", "spec": _s("minimal", font=SERIF, italic=True, upper=False, color=WHITE, size=62, weight=600)},
    {"id": "min_center", "name": "Center Quote", "family": "Minimal", "spec": _s("minimal", font=SERIF, upper=False, color=WHITE, size=64, weight=600, bottomPx=MID)},
    {"id": "min_small",  "name": "Subtle",       "family": "Minimal", "spec": _s("minimal", upper=False, color=WHITE, size=46, weight=500, letterSpacing=1)},

    # ── Big Word (one giant word at a time — Hormozi/Opus) ──────────────────
    {"id": "big_gold",  "name": "Big Gold",   "family": "Big Word", "spec": _s("bigword", color=GOLD, active=GOLD, size=124, weight=900, bottomPx=MID)},
    {"id": "big_white", "name": "Big White",  "family": "Big Word", "spec": _s("bigword", color=WHITE, active=WHITE, size=124, weight=900, bottomPx=MID)},
    {"id": "big_pink",  "name": "Big Pink",   "family": "Big Word", "spec": _s("bigword", color=PINK, active=PINK, size=124, weight=900, bottomPx=MID)},
    {"id": "big_punch", "name": "Slam Word",  "family": "Big Word", "spec": _s("bigword", color=WHITE, active=WHITE, size=132, weight=900, intensity=1.5, outline=3, bottomPx=MID)},
    {"id": "big_ice",   "name": "Big Ice",    "family": "Big Word", "spec": _s("bigword", color=ICE, active=ICE, size=120, weight=900, bottomPx=MID)},

    # ── Caption Bar (text on a solid background bar) ────────────────────────
    {"id": "bar_dark",  "name": "Dark Bar",   "family": "Caption Bar", "spec": _s("bar", color=WHITE, active=GOLD, boxFill="rgba(8,7,10,0.82)", boxRadius=14)},
    {"id": "bar_black", "name": "Black Bar",  "family": "Caption Bar", "spec": _s("bar", color=WHITE, active=GOLD, boxFill=INK, boxRadius=10)},
    {"id": "bar_gold",  "name": "Gold Bar",   "family": "Caption Bar", "spec": _s("bar", color=INK, active="#3a2c05", boxFill=GOLD, boxRadius=14)},
    {"id": "bar_white", "name": "White Bar",  "family": "Caption Bar", "spec": _s("bar", color=INK, active=PINK, boxFill=WHITE, boxRadius=14)},
    {"id": "bar_pink",  "name": "Pink Bar",   "family": "Caption Bar", "spec": _s("bar", color=WHITE, active=INK, boxFill=PINK, boxRadius=18)},

    # ── Stack (words build vertically) ─────────────────────────────────────
    {"id": "stack_gold",  "name": "Stack Gold",  "family": "Stack", "spec": _s("stack", color=WHITE, active=GOLD, size=78, weight=900, bottomPx=MID)},
    {"id": "stack_white", "name": "Stack White", "family": "Stack", "spec": _s("stack", color=GREY, active=WHITE, size=78, weight=900, bottomPx=MID)},
    {"id": "stack_pink",  "name": "Stack Pink",  "family": "Stack", "spec": _s("stack", color=WHITE, active=PINK, size=78, weight=900, bottomPx=MID)},

    # ── Wave (words bob in motion) ─────────────────────────────────────────
    {"id": "wave_gold",  "name": "Wave Gold",  "family": "Wave", "spec": _s("wave", color=WHITE, active=GOLD)},
    {"id": "wave_white", "name": "Wave White", "family": "Wave", "spec": _s("wave", color=GREY, active=WHITE)},
    {"id": "wave_mint",  "name": "Wave Mint",  "family": "Wave", "spec": _s("wave", color=WHITE, active=MINT)},
]

DEFAULT_STYLE = "box_gold"

_BY_ID = {c["id"]: c for c in CATALOG}


def list_styles() -> list[dict]:
    """Gallery payload for the UI: id, name, family + preview hints (no internals)."""
    out = []
    for c in CATALOG:
        sp = c["spec"]
        out.append({
            "id": c["id"],
            "name": c["name"],
            "family": c["family"],
            "engine": sp["engine"],
            "color": sp["color"],
            "active": sp["active"],
            "boxFill": sp["boxFill"],
            "upper": sp["upper"],
        })
    return out


def families() -> list[str]:
    seen, out = set(), []
    for c in CATALOG:
        if c["family"] not in seen:
            seen.add(c["family"]); out.append(c["family"])
    return out


def resolve(style_id: str | None) -> dict:
    """Return the token spec for a style id, falling back to DEFAULT_STYLE."""
    c = _BY_ID.get(style_id or "") or _BY_ID[DEFAULT_STYLE]
    return dict(c["spec"])


def is_valid(style_id: str | None) -> bool:
    return (style_id or "") in _BY_ID
