"""docs/results/findings.md, the gold set's pooling: venue detection and its pooling bias."""

from __future__ import annotations

from .... import formatting
from ....config_eval import COMPARISON_SCOPE, LABELLED_COMPARISON_SCOPE, MULTI_VENUE_SCOPE
from ....evaluation.labels import fmt
from ....models import COMPARISON, MULTI_VENUE
from ....retrieval.methods import ORACLE_METHODS
from ....retrieval.venue_detect import DETECTION_OUTCOMES, DetectionOutcome
from ...check import Doc
from ...compute import datasets, retrieval, retrieval_earlier, retrieval_tables
from ...run_ids import COVERAGE_FAILURE_RUN, PVALUE_RUN
from ...sources import Sources
from ..comparisons import (
    ORACLE,
    SPLIT,
    better_worse,
    comparison_questions,
    signed_parts,
    significance,
    top_k_text,
)
from ..methodology import protocol

# The detection outcome of a question that names no venue and has none detected; the
# detection table has no column for it.
UNNAMED: DetectionOutcome = "none"
# How the detection table names each group of questions.
DETECTION_LABELS = {
    COMPARISON_SCOPE: "comparison",
    MULTI_VENUE_SCOPE: f"naming >= {formatting.count(MULTI_VENUE)} venues",
    "all": "all",
}
# The methods whose top k was judged in venue_split's augmentation.
AUGMENTED = ("venue_split", *ORACLE_METHODS)


def detection(doc: Doc, src: Sources) -> None:
    """Venue detection, and what the oracle gains over it."""
    doc.claim(
        "| questions | exact | partial | missed | wrong |",
        holds=datasets.detection_counts(src, "all")[UNNAMED] == 0,
        what=f"no question has the detection outcome {UNNAMED}, which the table has no column for",
    )
    doc.table(
        "| questions | exact |",
        {
            f"{DETECTION_LABELS[r.group]} ({formatting.count(r.questions)})": [
                formatting.count(r.counts[o]) for o in DETECTION_OUTCOMES if o != UNNAMED
            ]
            for r in datasets.detection_rows(src)
        },
    )
    misses = datasets.detection_misses(src)
    doc.claim(
        "All {} missed questions name no venue in their text: the {} ambiguous questions by "
        "design, and {} that identify",
        formatting.count(misses.missed),
        formatting.count(misses.ambiguous),
        formatting.count(misses.other),
        holds=datasets.misses_are_unnamed(src),
        what="every ambiguous question is missed, and no missed question has a detected venue",
    )
    comparison = (*ORACLE, retrieval.at_top_k(src, "venue_coverage"), COMPARISON_SCOPE)
    multi = (*ORACLE, retrieval.at_top_k(src, "venue_coverage"), MULTI_VENUE_SCOPE)
    doc.ref("In run `results/{}`, venue_split_oracle", PVALUE_RUN)
    doc.claim(
        "venue_split_oracle - venue_split on venue_coverage@10: {} [{}, {}] on comparison ({} vs "
        "{}), p_Holm = {}, inconclusive at this size. On all {} multi-venue questions it is {} "
        "[{}, {}] ({} vs {}), p_Holm = {}, inconclusive at this size.",
        *signed_parts(retrieval.paired_interval(src, *comparison)),
        *better_worse(retrieval.paired_counts(src, *comparison)),
        formatting.p_value(retrieval.paired_p(src, *comparison, "holm_p")),
        formatting.count(retrieval.paired_questions(src, *multi)),
        *signed_parts(retrieval.paired_interval(src, *multi)),
        *better_worse(retrieval.paired_counts(src, *multi)),
        formatting.p_value(retrieval.paired_p(src, *multi, "holm_p")),
        holds=not retrieval.paired_significant(src, *comparison)
        and not retrieval.paired_significant(src, *multi),
        what=f"neither oracle comparison reaches p_Holm < {significance()}",
    )
    gained = retrieval_tables.oracle_gains(src)
    doc.check(
        "Of the {} questions the oracle gains, {} name no venue in their text and {} name only one",
        formatting.count(gained.questions),
        formatting.count(gained.naming_none),
        formatting.count(gained.naming_one),
    )


def augmentation_claims(doc: Doc, src: Sources) -> None:
    """Pooling bias: the chunks augmentation added and what they change."""
    aug = datasets.augmentation(src)
    depth = formatting.count(retrieval.pool_depth(src))
    doc.check(
        "`v0.2.1-gold` contains {} gold chunks found by judging the never-judged top {} of",
        formatting.count(aug.relevant),
        depth,
    )
    doc.check(
        "venue_split and venue_split_oracle on the {} comparison questions ({} chunks shown; {} "
        "of the {} at ranks {}--{};",
        comparison_questions(src),
        formatting.count(aug.shown),
        formatting.count(aug.relevant_lower),
        formatting.count(aug.relevant),
        formatting.count(retrieval.first_rank_below_top_k(src)),
        depth,
    )
    doc.table(
        "| method | against v0.2-gold |",
        {
            r.method: [
                formatting.metric(r.before),
                formatting.metric(r.after),
                formatting.signed(r.change),
            ]
            for r in retrieval.recall_before_after(src)
        },
    )
    doc.check(
        "Every method's comparison recall@10 changes by {} to {} with the {} chunks added.",
        *[formatting.signed(v) for v in retrieval.recall_change_range(src)],
        formatting.count(aug.relevant),
    )
    pool = retrieval.ROUND1_POOL
    doc.check(
        "whose comparison unjudged@10 is {}--{},",
        *[
            formatting.metric(v)
            for v in retrieval.metric_range(
                src, "comparison", pool, retrieval.at_top_k(src, "unjudged")
            )
        ],
    )
    doc.check(
        "and bm25 {}, dense {}, hybrid {} and hybrid_rerank {}: round 1 judged the fused top {} "
        "of the {} round-1 pool methods",
        *[
            formatting.metric(v)
            for v in retrieval.category_metrics(
                src, "comparison", pool, retrieval.at_top_k(src, "unjudged")
            )
        ],
        protocol.pool_display_text(src),
        formatting.count(len(retrieval.ROUND1_POOL)),
    )


def rescored_gap(doc: Doc, src: Sources) -> None:
    """venue_split's recall gap against both gold sets, and the unjudged share of each top k."""
    doc.claim(
        "gap to hybrid_rerank on recall@10 is {} [{}, {}] against `v0.2.1-gold`",
        *signed_parts(
            retrieval.paired_interval(
                src, *SPLIT, retrieval.at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE
            )
        ),
        holds=retrieval_tables.rescored_gap_agrees(src),
        what="the rescored gap against v0.2.1-gold equals the paired difference of the p-value run",
    )
    doc.claim(
        "Against `v0.2-gold` it is {} [{}, {}] in",
        *signed_parts(retrieval_earlier.earlier_gold_gap(src)),
        holds=retrieval_earlier.earlier_gold_run_matches(src)
        and formatting.signed(retrieval_earlier.earlier_gold_gap(src).value)
        == formatting.signed(retrieval_tables.recall_gap(src, augmented=False)),
        what="the run on v0.2-gold holds the retrieval run's top k, and its gap is the "
        "retrieval run's rescored against v0.2-gold",
    )
    doc.ref("in `results/{}`, the run of the same retrievals on `v0.2-gold`", COVERAGE_FAILURE_RUN)
    aug = datasets.augmentation(src)
    doc.check(
        "Of the {} chunks in venue_split's and venue_split_oracle's top {} that round 1 never "
        "judged, {} are relevant",
        formatting.count(aug.shown_top10),
        top_k_text(src),
        formatting.count(aug.relevant_top10),
    )
    doc.check("Share of each method's top {} that no judgment covers", top_k_text(src))
    doc.check(
        "on the {} labeled questions: bm25 {}, dense {}, hybrid {}, hybrid_rerank {}, "
        "venue_split {}, venue_split_oracle {}; on the {} comparison",
        formatting.count(datasets.labelled_count(src)),
        *[
            formatting.metric(v)
            for v in retrieval.summary_metrics(
                src, retrieval.METHODS, retrieval.at_top_k(src, "unjudged")
            )
        ],
        comparison_questions(src),
    )
    doc.check(
        "questions venue_split and venue_split_oracle are at {}, and bm25",
        # Two methods written with one value: a second value fails the comparison.
        fmt(
            retrieval.shared_category_metric(
                src, COMPARISON, AUGMENTED, retrieval.at_top_k(src, "unjudged")
            )
        ),
    )


def condensed(doc: Doc, src: Sources) -> None:
    """nDCG'@10 against nDCG@10: how nDCG changes when unjudged chunks are dropped."""
    doc.check(
        "nDCG'@10 - nDCG@10 on the {} labeled questions: bm25 {} ({} to {}), dense {} ({} to "
        "{}), hybrid {} ({} to {}), hybrid_rerank {} ({} to {}), venue_split {} ({} to {}), "
        "venue_split_oracle {} ({} to {}):",
        formatting.count(datasets.labelled_count(src)),
        *[
            v
            for r in retrieval_tables.condensed_ndcg(src)
            for v in (
                formatting.signed(r.difference),
                formatting.metric(r.plain),
                formatting.metric(r.condensed),
            )
        ],
    )
    bound = formatting.own_form(
        retrieval_tables.NDCG_PRIME_GAIN_BOUND, formatting.CODE_BOUND_DECIMALS
    )
    doc.claim(
        "under {} for every method.",
        bound,
        holds=retrieval_tables.condensed_gains_below_bound(src),
        what=f"nDCG'@{top_k_text(src)} - nDCG@{top_k_text(src)} below {bound} for every method",
    )
