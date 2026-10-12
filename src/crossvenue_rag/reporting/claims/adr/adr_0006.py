"""What docs/adr/0006-single-annotator.md claims.

The sample, the gold set, the agreement target and the gap before re-judging.
"""

from __future__ import annotations

from .... import formatting
from ....dataset.agreement import KAPPA_TARGET, MIN_RETEST_DAYS
from ...check import Doc
from ...compute import datasets
from ...sources import Sources
from ..comparisons import question_count_text, round2_sample_text
from ..methodology import protocol
from ..reasons import ARXIV_CITATION, LIST_IN_SENTENCE


def single_annotator_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0006-single-annotator.md."""
    doc.definition("Four ways to produce it were available", reason=LIST_IN_SENTENCE)
    _decision(doc, src)
    doc.given(
        "{} candidates per question",
        protocol.pool_display_text(src),
        reason="eval.pool_display",
    )


def _decision(doc: Doc, src: Sources) -> None:
    doc.check("a {}-question subsample", round2_sample_text(src))
    doc.given(
        "re-judged blind at least {} days later",
        formatting.count(MIN_RETEST_DAYS),
        reason="the gap registered in evals/methodology.md section 5",
    )
    doc.given(
        "Target \u03ba >= {}.",
        formatting.own_form(KAPPA_TARGET),
        reason="the agreement target of methodology section 5",
    )
    doc.given(
        "Below {}, \u03ba is reported",
        formatting.own_form(KAPPA_TARGET),
        reason="the agreement target",
    )
    doc.check(
        "Re-labeling {} questions ({} decisions)",
        question_count_text(src),
        formatting.count(datasets.relabel_decisions(src)),
    )
    doc.definition("van Elburg et al. 2025, [arXiv:2508.11758]", reason=ARXIV_CITATION)
    doc.definition("Rahmani et al. 2025, [arXiv:2506.10301]", reason=ARXIV_CITATION)
