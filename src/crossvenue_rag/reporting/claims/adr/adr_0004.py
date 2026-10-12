"""What docs/adr/0004-index-version.md claims: one definition, no measured number."""

from __future__ import annotations

from ...check import Doc
from ...sources import Sources
from ..reasons import LIST_IN_SENTENCE


def index_version_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0004-index-version.md."""
    doc.definition("Two checks cover that gap", reason=LIST_IN_SENTENCE)
