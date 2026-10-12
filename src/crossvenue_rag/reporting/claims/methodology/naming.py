"""evals/methodology.md: the numbers that name rather than count.

Section and list numbers, number words used as words, and the reference lists.
"""

from __future__ import annotations

from ...check import Doc
from ...compute import references as reference_list
from ...sources import Sources
from ..reasons import TEST_NAME
from .texts import CONVENTION, LITERATURE, WORDING, WORDING_STUDIES

# Section and list numbers, with up to four words after them.
NUMBERED = r"(?m)^(?:#{2,3} \d+[a-z]?\. |\s*\d+\. )(?:[^\s\d]+(?= |$)(?: [^\s\d]+(?= |$)){0,3})?"


def wording(doc: Doc) -> None:
    """Section and list numbers, number words used as words, and names."""
    doc.definitions_matching(NUMBERED, reason=CONVENTION)
    doc.definitions(WORDING, reason=CONVENTION)
    doc.definitions(WORDING_STUDIES, reason=LITERATURE)


def metric_names(doc: Doc) -> None:
    """Number words in the names of tests and metric families."""
    doc.definition("two-sided", reason=TEST_NAME)
    doc.definition("The two form the primary family", reason="the two primary metrics")
    doc.definition("Holm-adjusted within the two.", reason="the two secondary metrics")


def reference_claims(doc: Doc, src: Sources) -> None:
    """The references: each section's list exactly as docs/references.yaml records it.

    Raises:
        ValueError: `doc`'s text has no `REFERENCES_HEADING` line.
    """
    recorded = reference_list.references(src)
    for section in dict.fromkeys([*reference_list.written_sections(doc.text), *recorded]):
        doc.items(f"### {section}", recorded.get(section, []), reason=LITERATURE)
    doc.definition(
        "disagreement grows as topics fall toward 10, with no threshold",
        reason=LITERATURE + " (the abstract of the 2009 poster)",
    )
    doc.definition("The Smucker, Allan & Carterette 2009 poster", reason=LITERATURE)
    doc.definition(
        "The two caveats on \u03ba in \u00a75", reason="the two caveats the section states"
    )
