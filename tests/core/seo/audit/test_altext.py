# tests/core/seo/audit/test_altext.py
"""Smoke tests for the alt-text generator.

We mock both httpx layers so the test never actually contacts the
network or Ollama. The point is just to confirm the plumbing handles a
realistic Ollama response and a fetch failure cleanly.
"""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from core.seo.audit.altext import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    generate_alt_text,
)


# A 1x1 transparent PNG — enough to be "real image bytes" for the test.
ONE_PX_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII="
)


def _ollama_resp(text: str) -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json = MagicMock(return_value={"response": text})
    return m


def _ollama_resp_reasoning_field(text: str) -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.raise_for_status = MagicMock()
    m.json = MagicMock(return_value={"response": "", "reasoning": text})
    return m


def _image_resp(content: bytes) -> MagicMock:
    m = MagicMock()
    m.status_code = 200
    m.content = content
    m.raise_for_status = MagicMock()
    return m


def test_generate_alt_text_from_bytes_happy_path():
    with patch(
        "core.seo.audit.altext.httpx.post",
        return_value=_ollama_resp("A delicate gold chain necklace with evil-eye pendant"),
    ) as mock_post:
        alt = generate_alt_text(ONE_PX_PNG)
    assert alt == "A delicate gold chain necklace with evil-eye pendant"
    # Confirm we hit the right Ollama endpoint with the right model.
    args, kwargs = mock_post.call_args
    assert args[0] == OLLAMA_URL
    assert kwargs["json"]["model"] == DEFAULT_MODEL
    assert "images" in kwargs["json"] and len(kwargs["json"]["images"]) == 1


def test_generate_alt_text_from_url_fetches_image():
    with patch(
        "core.seo.audit.altext.httpx.get", return_value=_image_resp(ONE_PX_PNG)
    ) as mock_get, patch(
        "core.seo.audit.altext.httpx.post",
        return_value=_ollama_resp("Silver charm bracelet on white background"),
    ):
        alt = generate_alt_text("https://cdn.example.com/bracelet.jpg")
    assert alt == "Silver charm bracelet on white background"
    assert mock_get.call_args[0][0] == "https://cdn.example.com/bracelet.jpg"


def test_generate_alt_text_strips_quotes_from_model_output():
    with patch(
        "core.seo.audit.altext.httpx.post",
        return_value=_ollama_resp('"Gold hoop earrings on marble"'),
    ):
        alt = generate_alt_text(ONE_PX_PNG)
    assert alt == "Gold hoop earrings on marble"


def test_generate_alt_text_reads_reasoning_field_fallback():
    """Some models on 105 (e.g. kimi-k2.6) put output in `reasoning`."""
    with patch(
        "core.seo.audit.altext.httpx.post",
        return_value=_ollama_resp_reasoning_field("A pearl pendant on chain"),
    ):
        alt = generate_alt_text(ONE_PX_PNG)
    assert alt == "A pearl pendant on chain"


def test_generate_alt_text_returns_none_on_fetch_failure():
    with patch(
        "core.seo.audit.altext.httpx.get",
        side_effect=Exception("dns boom"),
    ):
        alt = generate_alt_text("https://broken.invalid/x.jpg")
    assert alt is None


def test_generate_alt_text_returns_none_on_ollama_failure():
    with patch(
        "core.seo.audit.altext.httpx.post",
        side_effect=Exception("connection refused"),
    ):
        alt = generate_alt_text(ONE_PX_PNG)
    assert alt is None


def test_generate_alt_text_returns_none_on_empty_response():
    with patch(
        "core.seo.audit.altext.httpx.post",
        return_value=_ollama_resp(""),
    ):
        alt = generate_alt_text(ONE_PX_PNG)
    assert alt is None
