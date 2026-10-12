"""What docs/results/findings.md claims.

The functions, each for a section or a run of sections, run in check order, which
differs from document order. The hypothesis paragraph, off-venue retrieval and the numbered
headings are checked after the pooling section, the table captions and number
words last.
"""

from __future__ import annotations

from ....evaluation.comparisons import OFF_VENUE_SCOPES
from ...check import Doc
from ...sources import Sources
from ..hypothesis import hypothesis
from . import (
    answers,
    calibration,
    caveats,
    corpus,
    headings,
    judged,
    latency,
    number_words,
    off_venue,
    pooling,
    references,
    retrieval,
    rounds,
)


def findings(doc: Doc, src: Sources) -> None:
    """States the claims of docs/results/findings.md.

    Raises:
        ValueError: `doc`'s text has no `number_words.KIND_TABLE` line, or that table has no
            `kind` column.
    """
    _retrieval(doc, src)
    hypothesis(doc, src)
    _off_venue(doc, src)
    rounds.agreement_finding(doc, src)
    rounds.sensitivity_finding(doc, src)
    _corpus(doc, src)
    _answers(doc, src)
    references.runs(doc, src)
    references.captions(doc, src)
    number_words.number_words(doc, src)
    number_words.kinds(doc, src)


def _retrieval(doc: Doc, src: Sources) -> None:
    """The header and the retrieval results, through the pooling bias."""
    retrieval.header(doc, src)
    retrieval.parameters(doc, src)
    retrieval.comparison_table(doc, src)
    retrieval.venue_split(doc, src)
    retrieval.split_recall(doc, src)
    retrieval.split_cost(doc, src)
    retrieval.reranker(doc, src)
    retrieval.first_stage(doc, src)
    retrieval.restructuring(doc, src)
    pooling.detection(doc, src)
    pooling.augmentation_claims(doc, src)
    pooling.rescored_gap(doc, src)
    pooling.condensed(doc, src)


def _off_venue(doc: Doc, src: Sources) -> None:
    """Off-venue retrieval and the numbered headings."""
    off_venue.definitions(doc, src)
    for scope in OFF_VENUE_SCOPES:
        off_venue.measures(doc, src, scope)
    off_venue.baseline(doc, src)
    off_venue.comparison_claims(doc, src)
    off_venue.bm25(doc, src)
    headings.headings(doc, src)


def _corpus(doc: Doc, src: Sources) -> None:
    """Latency, the bench, the caveats and the corpus."""
    latency.latency(doc, src)
    latency.reranker_bench(doc, src)
    latency.bench_saving(doc, src)
    latency.bench_machine(doc, src)
    caveats.caveats(doc, src)
    corpus.pancakeswap(doc, src)
    corpus.duplicate_claims(doc, src)


def _answers(doc: Doc, src: Sources) -> None:
    """The answers, the judge calibration and the judged answers."""
    answers.matrix(doc, src)
    answers.refusals(doc, src)
    answers.prompt_v2(doc, src)
    answers.clarification_claims(doc, src)
    answers.wrong_refusal_claims(doc, src)
    answers.mercury(doc, src)
    answers.cost_claims(doc, src)
    answers.judge_cost(doc, src)
    calibration.table(doc, src)
    calibration.judge_choice(doc, src)
    calibration.rejection_claims(doc, src)
    calibration.repeat(doc, src)
    judged.primary(doc, src)
    judged.refusal_coverage(doc, src)
    judged.generator_comparison_claims(doc, src)
    judged.h_v2(doc, src)
    judged.attribution_claims(doc, src)
    judged.prompt_v2_primary(doc, src)
