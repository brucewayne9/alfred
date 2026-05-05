"""Vision module returns structured tags alongside the description."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from core.jewelry import vision

# Patch _encode_image globally so tests never need real image files on disk
_FAKE_B64 = "ZmFrZQ=="  # base64 for "fake"


@patch('core.jewelry.vision._encode_image', return_value=_FAKE_B64)
@patch('core.jewelry.vision._call_ollama')
def test_vision_returns_tags(mock_call, mock_enc):
    mock_call.return_value = {
        "message": {
            "content": (
                '{"description": "A delicate beaded bracelet in warm earth tones.",'
                ' "color_family": "warm",'
                ' "dominant_hex": "#C8794E",'
                ' "material_class": "beaded",'
                ' "style_class": "minimal"}'
            )
        }
    }
    result = vision.describe_piece(['/tmp/fake.jpg'])
    assert isinstance(result, dict)
    assert result['description']
    assert result['color_family'] == "warm"
    assert result['dominant_hex'] == "#C8794E"
    assert result['material_class'] == "beaded"
    assert result['style_class'] == "minimal"


@patch('core.jewelry.vision._encode_image', return_value=_FAKE_B64)
@patch('core.jewelry.vision._call_ollama')
def test_vision_handles_missing_tags_gracefully(mock_call, mock_enc):
    """If the model omits tags, default sentinels are filled in."""
    mock_call.return_value = {
        "message": {"content": '{"description": "Just a bracelet."}'}
    }
    result = vision.describe_piece(['/tmp/fake.jpg'])
    assert result['description'] == "Just a bracelet."
    assert result['color_family'] == 'mixed'
    assert result['material_class'] == 'other'
    assert result['style_class'] == 'classic'
    assert result['dominant_hex'] == '#888888'


@patch('core.jewelry.vision._encode_image', return_value=_FAKE_B64)
@patch('core.jewelry.vision._call_ollama')
def test_vision_handles_invalid_color_family(mock_call, mock_enc):
    """If the model returns a value not in the closed vocab, default in."""
    mock_call.return_value = {
        "message": {"content": '{"description": "x", "color_family": "rainbow", "material_class": "beaded", "style_class": "minimal"}'}
    }
    result = vision.describe_piece(['/tmp/fake.jpg'])
    assert result['color_family'] == 'mixed'  # rainbow → default
    assert result['material_class'] == 'beaded'  # valid, kept


@patch('core.jewelry.vision._encode_image', return_value=_FAKE_B64)
@patch('core.jewelry.vision._call_ollama')
def test_vision_handles_unparseable_json(mock_call, mock_enc):
    """If the model emits non-JSON, fall back to defaulted dict with raw text."""
    mock_call.return_value = {
        "message": {"content": "this is not json at all, just a paragraph"}
    }
    result = vision.describe_piece(['/tmp/fake.jpg'])
    assert isinstance(result, dict)
    assert result['description'] == "this is not json at all, just a paragraph"
    assert result['color_family'] == 'mixed'
    assert result['style_class'] == 'classic'


def test_vision_module_constants():
    """Confirm the closed-vocab constants are accessible from the module."""
    assert "warm" in vision.COLOR_FAMILIES
    assert "beaded" in vision.MATERIAL_CLASSES
    assert "minimal" in vision.STYLE_CLASSES
