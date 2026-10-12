"""What docs/adr/0008-package-structure.md claims.

The package in the repository, and the flat package before the restructuring.
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import repository
from ...sources import Sources
from ..reasons import ROOT_DEFINITION

BEFORE = (
    "the flat package before the restructuring of 2026-09-24, which the repository does not hold"
)


def package_structure_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0008-package-structure.md."""
    _context(doc)
    _decision(doc, src)


def _context(doc: Doc) -> None:
    doc.given("The package was {} flat modules.", "35", reason=BEFORE)
    doc.given(
        "{} held several responsibilities each (the labeling tool at {} lines, the CLI at {})",
        "2",
        "873",
        "844",
        reason=BEFORE,
    )


def _decision(doc: Doc, src: Sources) -> None:
    doc.check("- {} subpackages", formatting.count(repository.subpackage_count(src)))
    doc.definition("what at least two layers share", reason=ROOT_DEFINITION)
    doc.check(
        "{} `lint-imports` contracts", formatting.count(repository.import_contract_count(src))
    )
    doc.given(
        "No module above about {} lines",
        formatting.count(repository.MODULE_CODE_LINES_BOUND),
        reason="the size rule this record decides",
    )
