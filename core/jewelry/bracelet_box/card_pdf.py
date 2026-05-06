"""Render the thank-you card as an A6 PDF using WeasyPrint.

The Jinja2 template lives at services/roen-minimal/templates/card.html
and the rowan-mark SVG at services/roen-minimal/assets/svg/rowan-mark.svg.
The SVG uses stroke="currentColor"; the template's .mark CSS sets the
color to terracotta (#B85C3D).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

TEMPLATE_DIR = Path("/home/aialfred/alfred/services/roen-minimal/templates")
SVG_PATH = Path("/home/aialfred/alfred/services/roen-minimal/assets/svg/rowan-mark.svg")

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(['html']),
)


def render(
    recipient: Optional[str],
    note_body: str,
    piece_names: List[str],
    signoff: str,
) -> bytes:
    """Render a single-page A6 PDF and return the bytes."""
    template = _env.get_template("card.html")
    rowan_mark_svg = SVG_PATH.read_text(encoding="utf-8")
    html = template.render(
        recipient=recipient,
        note_body=note_body,
        piece_names=piece_names,
        signoff=signoff,
        rowan_mark=rowan_mark_svg,
    )
    return HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf()


def render_to_file(out_path: Path, **kwargs) -> Path:
    """Convenience wrapper: render and write to disk. Returns out_path."""
    out_path = Path(out_path)
    out_path.write_bytes(render(**kwargs))
    return out_path
