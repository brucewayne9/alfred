"""Closed-vocab tag values used by the picker, the vision step, and other consumers."""
from __future__ import annotations

# Re-export the vocab from vision.py to keep a single source of truth.
# The picker can also rely on these constants for validation.
from core.jewelry.vision import (
    COLOR_FAMILIES,
    MATERIAL_CLASSES,
    STYLE_CLASSES,
)

BUNDLE_SIZE = 5


class InsufficientStock(Exception):
    """Raised when fewer than BUNDLE_SIZE eligible bracelets exist."""
