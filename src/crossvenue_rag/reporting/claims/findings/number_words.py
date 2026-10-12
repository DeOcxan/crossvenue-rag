"""docs/results/findings.md: the counts it writes as words or in passing."""

from __future__ import annotations

from .... import formatting
from ....models import RETRIEVAL_STEPS
from ...check import Doc
from ...compute import agreement, answers, calibration, datasets, documents, retrieval
from ...run_ids import ANSWERS_V1
from ...sources import Sources
from ..comparisons import level
from ..reasons import LIST_FOLLOWS, TEST_NAME

# The table of the claims Opus accepted and the human rejected, and its kinds.
KIND_TABLE = "| claim | answer | what the chunk does not state | kind |"

QUALIFIER = "qualifier not in the chunk"

EXAMPLE = "example turned into a rule"

DATE = "date from a heading"


def number_words(doc: Doc, src: Sources) -> None:
    """Counts the text writes as words or in passing."""
    judge = src.judged_v1.config.generation.chosen_judge()
    doc.check("On the {} ambiguous", formatting.count(datasets.category_count(src, "ambiguous")))
    doc.check(
        "For the {} round-1 pool methods, whose", formatting.count(len(retrieval.ROUND1_POOL))
    )
    doc.check(
        "The steps sum is the sum of the {} measured steps",
        formatting.count(len(RETRIEVAL_STEPS)),
    )
    doc.claim(
        "meets {} of {} thresholds on the point estimate and one on the interval",
        formatting.count(calibration.passes(src, judge)),
        formatting.count(calibration.dimension_count(src)),
        holds=len(calibration.passes_on_interval(src, judge)) == 1,
        what="Opus passes exactly one dimension on an informative interval",
    )
    doc.check(
        '`cv-0121 "which bingx endpoint returns the usdt-m"`, whose {} claims cite',
        formatting.count(answers.claim_count(src, "qwen@venue_split:cv-0121")),
    )
    doc.caption(
        "The {} claims Opus accepted and the human rejected, run `results/{run}`",
        (ANSWERS_V1,),
        formatting.count(calibration.verdict_pairs(src, judge, "supported")[(True, False)]),
    )
    doc.check(
        "The {} claims Opus rejected and the human accepted",
        formatting.count(calibration.verdict_pairs(src, judge, "supported")[(False, True)]),
    )
    doc.check("the same {} configurations and", formatting.count(len(answers.configurations(src))))
    doc.check(
        "Holm-adjusted across the {} configurations, {}% paired bootstrap intervals",
        formatting.count(len(answers.configurations(src))),
        level(),
    )
    doc.check("Holm within the {} metrics", formatting.count(len(answers.primary_metrics(src))))
    doc.check("stayed invalid after {} retries", formatting.count(answers.max_invalid_retries(src)))


def kinds(doc: Doc, src: Sources) -> None:
    """The kinds of the claims Opus accepted and the human rejected, and the definitions.

    Raises:
        ValueError: `doc`'s text has no `KIND_TABLE` line, or that table has no `kind` column.
    """
    counts = documents.table_column_counts(doc.text, KIND_TABLE, "kind")
    doc.claim(
        "By kind, {} add a qualifier the chunk does not carry, {} turn an example or a field list "
        "into a rule, and one takes a date from a changelog heading",
        formatting.count(counts.get(QUALIFIER, 0)),
        formatting.count(counts.get(EXAMPLE, 0)),
        holds=calibration.rejection_kinds_hold(src, counts, (QUALIFIER, EXAMPLE, DATE), DATE),
        what="the kind column has one date row, only those kinds, and one row per claim Opus "
        "accepted and the human rejected",
    )
    doc.check(
        "{} of those have answers in the corpus",
        formatting.count(len(agreement.recategorized(src))),
    )
    doc.definition("two-sided paired randomization", reason=TEST_NAME)
    doc.definition("two-sided sign-flip", reason=TEST_NAME)
    doc.definition(
        "as id plus their first six words", reason="the citation convention of docs/how-to/label.md"
    )
    doc.definition("was timed between two checks", reason="the bench's procedure")
    doc.definition("The two share concepts", reason="Uniswap and PancakeSwap")
    doc.definition("Two checks not registered", reason=LIST_FOLLOWS)
