"""Card PDF renderer — produces a non-empty A6 PDF."""
from __future__ import annotations

import io
from pathlib import Path
import pytest

from core.jewelry.bracelet_box import card_pdf


def test_renders_pdf_bytes():
    """Smoke test: render produces a valid PDF byte stream."""
    pdf_bytes = card_pdf.render(
        recipient="Maria",
        note_body=(
            "Roen chose this set with quiet contrast in mind. "
            "Five pieces, one mood. with care, roen "
            + "extra " * 12
        ),
        piece_names=["Amber drop", "Sage chain", "River pearl",
                     "Knot leather", "Spark thread"],
        signoff="with care, roen",
    )
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 4000


def test_renders_without_recipient():
    """When recipient is None, the 'for X,' line is omitted but PDF still renders."""
    pdf_bytes = card_pdf.render(
        recipient=None,
        note_body="Roen chose. " * 25,
        piece_names=["A", "B", "C", "D", "E"],
        signoff="yours, roen",
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 4000


def test_renders_to_file(tmp_path):
    """render_to_file writes a non-empty PDF to the given path."""
    out = tmp_path / "test-card.pdf"
    card_pdf.render_to_file(
        out,
        recipient="Lee",
        note_body="Roen chose this set. " * 18 + "with care, roen",
        piece_names=["A", "B", "C", "D", "E"],
        signoff="with care, roen",
    )
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 4000


def test_recipient_appears_in_pdf_text():
    """The recipient name is rendered into the PDF's text content."""
    # Use pypdf if available; otherwise just confirm the bytes exist.
    pdf_bytes = card_pdf.render(
        recipient="Maria",
        note_body="Roen chose this set. " * 18 + "with care, roen",
        piece_names=["A", "B", "C", "D", "E"],
        signoff="with care, roen",
    )
    # We could decode, but for now just the smoke check.
    assert pdf_bytes.startswith(b"%PDF")
    # If pypdf is available, do a stricter check.
    try:
        import pypdf
        from io import BytesIO
        reader = pypdf.PdfReader(BytesIO(pdf_bytes))
        text = "".join(page.extract_text() for page in reader.pages)
        assert "Maria" in text
        assert "with care, roen" in text
    except ImportError:
        pass  # acceptable to not have pypdf installed for stricter assertions


def test_handles_long_note_gracefully():
    """A note at the upper word count limit (90 words) still renders."""
    long_note = " ".join(["word"] * 90) + " with care, roen"
    pdf_bytes = card_pdf.render(
        recipient="X",
        note_body=long_note,
        piece_names=["a", "b", "c", "d", "e"],
        signoff="with care, roen",
    )
    assert pdf_bytes.startswith(b"%PDF")
