"""What evals/datasheet.md claims under Known limitations."""

from __future__ import annotations

from ... import formatting
from ...dataset.validate import LEAKAGE_NGRAM
from ...evaluation.labels import fmt
from ..check import Doc
from ..compute import agreement, datasets, datasets_rounds, gold_sheet, retrieval
from ..outcomes import Param
from ..run_ids import RETRIEVAL_RUN
from ..sources import Sources
from .comparisons import (
    comparison_questions,
    kappa_bounds,
    level,
    round2_sample_text,
    top_k_text,
)
from .methodology import protocol
from .reasons import INTERVAL_LEVEL, REGISTERED


def limitations(doc: Doc, src: Sources) -> None:
    """The known limitations: pooling, slices, agreement, the source marker, D23 and ccxt."""
    doc.check("round 1 judged the fused top {} of", protocol.pool_display_text(src))
    doc.check("of the {} round-1 pool methods", formatting.count(len(retrieval.ROUND1_POOL)))
    doc.check(
        "own top {} on the {} comparison questions", top_k_text(src), comparison_questions(src)
    )
    doc.ref("In `results/{}` on `v0.2.1-gold`, after the augmentation", RETRIEVAL_RUN)
    pool = retrieval.ROUND1_POOL
    unjudged = retrieval.at_top_k(src, "unjudged")
    doc.check(
        "has an unjudged@10 of {}--{} (bm25 {}, dense {}, hybrid and hybrid_rerank {})",
        *[formatting.metric(v) for v in retrieval.metric_range(src, "comparison", pool, unjudged)],
        *[
            formatting.metric(v)
            for v in retrieval.category_metrics(src, "comparison", ("bm25", "dense"), unjudged)
        ],
        # Two methods written with one value: a second value fails the comparison.
        fmt(
            retrieval.shared_category_metric(
                src, "comparison", ("hybrid", "hybrid_rerank"), unjudged
            )
        ),
    )
    doc.check("augmented to depth {}", protocol.pool_depth_text(src))
    doc.check(
        "On the {} ambiguous questions, venue_split_oracle's unjudged@10 is {}",
        formatting.count(datasets.category_count(src, "ambiguous")),
        formatting.metric(retrieval.oracle_unjudged(src, "ambiguous")),
    )
    doc.ref("(`results/{}`), a result in [findings]", RETRIEVAL_RUN)
    values = gold_sheet.coverage_values(src, gold_sheet.comparisons(src))
    doc.check(
        "{} of the {} comparisons have a named venue recorded `missing_from_pool` ({}) or "
        "`absent_in_docs` ({})",
        formatting.count(datasets.one_sided_comparisons(src)),
        comparison_questions(src),
        formatting.count(values["missing_from_pool"]),
        formatting.count(values["absent_in_docs"]),
    )
    doc.check(
        "the labeled categories hold {}--{} questions",
        *[formatting.count(n) for n in gold_sheet.labelled_range(src, "comparison")],
    )
    doc.check(
        "the gold chunks of {} of the {} round-2 questions",
        formatting.count(datasets_rounds.exposed_sample_count(src)),
        round2_sample_text(src),
    )
    marks = datasets_rounds.source_chunk_marks(src)
    doc.check(
        "marked relevant {} times out of {}, other pool chunks {} times out of {}",
        *(
            formatting.count(n)
            for n in (
                marks.source_relevant,
                marks.source_shown,
                marks.other_relevant,
                marks.other_shown,
            )
        ),
    )
    registered = agreement.registered(src)
    doc.check(
        "reproduced by the same annotator at \u03ba {} [{}, {}] ({}% percentile bootstrap "
        "over questions;",
        formatting.metric(registered.report.kappa),
        *kappa_bounds(registered),
        Param(level(), INTERVAL_LEVEL),
    )
    doc.check(
        "{} questions labeled unanswerable have answers in the corpus:",
        formatting.count(len(agreement.recategorized(src))),
    )
    doc.claim(
        "no question shares {} or more consecutive words with any of its gold chunks, spans in "
        "backticks or double quotes excluded",
        formatting.count(LEAKAGE_NGRAM),
        holds=gold_sheet.longest_gold_run(src) < LEAKAGE_NGRAM,
        what="every shared run is shorter than the leakage threshold",
    )
    doc.check(
        "ccxt appears in {} questions, below",
        formatting.count(datasets.questions_naming(src, "ccxt")),
    )
    doc.given("the protocol's floor of {} per venue", protocol.VENUE_FLOOR, reason=REGISTERED)
