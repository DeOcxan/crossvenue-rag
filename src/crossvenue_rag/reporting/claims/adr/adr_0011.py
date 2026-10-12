"""What docs/adr/0011-preregistration.md claims.

The family of answer comparisons and the primary metrics.
"""

from __future__ import annotations

from .... import formatting
from ....generation.answer_rows import SCOPES
from ....generation.metrics import METRICS
from ...check import Doc
from ...compute import answers
from ...sources import Sources
from ..comparisons import comparison_questions, significance


def preregistration_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0011-preregistration.md."""
    _context(doc, src)
    _decision(doc, src)


def _context(doc: Doc, src: Sources) -> None:
    comparisons = formatting.count(answers.paired_comparison_count(src))
    doc.check(
        "{} pairs of the {} configurations \u00d7 {} metrics \u00d7 {} scopes, "
        "{} paired comparisons",
        formatting.count(answers.configuration_pairs(src)),
        formatting.count(len(answers.configurations(src))),
        formatting.count(len(METRICS)),
        formatting.count(len(SCOPES)),
        comparisons,
    )
    doc.given(
        "at \u03b1 = {}",
        significance(),
        reason="the significance level of methodology section 9",
    )
    doc.check(
        "about {} of {} null comparisons",
        formatting.count(answers.expected_false_positives(src)),
        comparisons,
    )


def _decision(doc: Doc, src: Sources) -> None:
    doc.check("over the {} comparison questions", comparison_questions(src))
    doc.check(
        "the {} form one family",
        formatting.count(len(answers.primary_metrics(src))),
    )
