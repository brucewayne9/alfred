from core.forge.renderers.leak_graphic import build_prompt


def test_build_prompt_uses_override_when_given():
    p = build_prompt("sad caption", override="neon city street")
    assert "neon city street" in p


def test_build_prompt_falls_back_to_caption():
    p = build_prompt("sad caption")
    assert "sad caption" in p
