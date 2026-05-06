"""Note generator — prompt construction and constraint validation tests."""
from __future__ import annotations
import pytest
from unittest.mock import patch

from core.jewelry.bracelet_box import note_writer


def make_picks():
    return [
        {'id': 1, 'name': 'Amber drop',   'short': 'warm beaded',  'color_family': 'warm',    'material_class': 'beaded'},
        {'id': 2, 'name': 'Sage chain',   'short': 'cool metal',   'color_family': 'cool',    'material_class': 'metal-chain'},
        {'id': 3, 'name': 'River pearl',  'short': 'neutral',      'color_family': 'neutral', 'material_class': 'gemstone'},
        {'id': 4, 'name': 'Knot leather', 'short': 'leather',      'color_family': 'neutral', 'material_class': 'leather'},
        {'id': 5, 'name': 'Spark thread', 'short': 'mixed',        'color_family': 'mixed',   'material_class': 'mixed-media'},
    ]


def test_prompt_includes_picks_and_first_name():
    prompt = note_writer.build_prompt(
        picks=make_picks(), first_name="Maria",
        past_notes=[], order_count=1,
    )
    assert "Maria" in prompt
    assert "Amber drop" in prompt
    assert "Sage chain" in prompt
    assert "first order" in prompt.lower()


def test_prompt_anonymous_when_no_first_name():
    prompt = note_writer.build_prompt(
        picks=make_picks(), first_name=None,
        past_notes=[], order_count=1,
    )
    assert "anonymous" in prompt.lower()


def test_prompt_avoids_past_themes():
    past = [
        "Roen chose this set for the warm autumn light, leaning into earth-tones throughout.",
        "These five lean into deep moody contrasts, with the leather grounding the metallic accents.",
    ]
    prompt = note_writer.build_prompt(
        picks=make_picks(), first_name="Lee",
        past_notes=past, order_count=2,
    )
    assert "avoid" in prompt.lower()
    assert "warm autumn" in prompt or past[0][:80] in prompt


def test_prompt_order_count_three_is_warm_not_gushy():
    prompt = note_writer.build_prompt(
        picks=make_picks(), first_name="X",
        past_notes=[], order_count=3,
    )
    assert "order #3" in prompt or "warm but not gushy" in prompt


def test_word_count_constraint():
    just_right = " ".join(["word"] * 75)
    assert note_writer.is_valid_length(just_right)
    assert not note_writer.is_valid_length(" ".join(["word"] * 50))
    assert not note_writer.is_valid_length(" ".join(["word"] * 200))


def test_no_exclamations():
    assert note_writer.has_no_exclamations("Calm voice. Quiet ending.")
    assert not note_writer.has_no_exclamations("This is great!")


def test_has_signoff():
    assert note_writer.has_signoff("...end of note. with care, roen")
    assert note_writer.has_signoff("...goodbye. yours, roen")
    assert note_writer.has_signoff("...from the studio")
    assert not note_writer.has_signoff("Just a paragraph with no closer")


@patch('core.jewelry.bracelet_box.note_writer._call_kimi')
def test_generate_returns_passing_note(mock_kimi):
    """Model returns a valid note on first try; we accept it."""
    valid = (
        "Roen chose this set with quiet contrast in mind. The Amber drop warms the wrist "
        "while the Sage chain cools it; the River pearl and Knot leather sit neutral between "
        "them, and the Spark thread ties the whole set together with one bright accent that "
        "carries the eye across all five pieces in a single coherent gesture. The throughline "
        "is a calm conversation between warm and cool. with care, roen"
    )
    mock_kimi.return_value = valid
    out = note_writer.generate(picks=make_picks(), first_name="Maria",
                                past_notes=[], order_count=1)
    assert out == valid
    mock_kimi.assert_called_once()


@patch('core.jewelry.bracelet_box.note_writer._call_kimi')
def test_generate_retries_on_invalid(mock_kimi):
    """Model returns an invalid note (has '!') first; valid second attempt."""
    invalid = "Roen chose this set! " + " ".join(["word"] * 75) + " with care, roen"
    valid = "Roen chose this set " + " ".join(["word"] * 73) + " with care, roen"
    mock_kimi.side_effect = [invalid, valid]
    out = note_writer.generate(picks=make_picks(), first_name="X",
                                past_notes=[], order_count=1)
    assert out == valid
    assert mock_kimi.call_count == 2


@patch('core.jewelry.bracelet_box.note_writer._call_kimi')
def test_generate_returns_last_attempt_even_if_invalid(mock_kimi):
    """If both attempts fail validation, return the last attempt (Sarah will edit)."""
    bad = "this is too short. with care, roen"
    mock_kimi.return_value = bad
    out = note_writer.generate(picks=make_picks(), first_name="X",
                                past_notes=[], order_count=1)
    assert out == bad
    assert mock_kimi.call_count == 2  # tried both


@patch('core.jewelry.bracelet_box.note_writer._call_kimi')
def test_generate_handles_request_exception(mock_kimi):
    """If kimi raises, retry; if both raise, return empty string."""
    import requests as r
    mock_kimi.side_effect = r.exceptions.Timeout("simulated")
    out = note_writer.generate(picks=make_picks(), first_name="X",
                                past_notes=[], order_count=1)
    assert out == ""
    assert mock_kimi.call_count == 2
