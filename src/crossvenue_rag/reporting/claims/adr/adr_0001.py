"""What docs/adr/0001-no-framework.md claims.

The record states no number of its own; the size of `src/` it links is checked
in design.md.
"""

from __future__ import annotations

from ...check import Doc
from ...sources import Sources


def no_framework_record(doc: Doc, src: Sources) -> None:
    """docs/adr/0001-no-framework.md: nothing beyond the record's number."""
