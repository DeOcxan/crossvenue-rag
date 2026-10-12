"""evals/methodology.md, its first part: the opening, the decisions at a glance, sections 1 to 3."""

from __future__ import annotations

import math

from .... import formatting
from ....dataset.agreement import KAPPA_TARGET, MIN_RETEST_DAYS
from ....dataset.questions import MIN_QUESTIONS_PER_CATEGORY
from ....dataset.sampling import ROUND2_SIZE
from ....dataset.selection import TARGETS, target_size
from ....evaluation import mcnemar_power
from ....retrieval.methods import POOL_METHODS
from ...check import Doc
from ...compute import agreement, answers, corpus, datasets, retrieval
from ...outcomes import Param
from ...run_ids import RETRIEVAL_RUN
from ...sources import Sources
from .. import comparisons
from ..reasons import LIST_FOLLOWS, LIST_IN_SENTENCE, POWER_MODEL, REGISTERED
from . import protocol
from .protocol import FLOOR_SIZE, PSI, SMALLER_COMPARISON_SIZE
from .texts import (
    CONVENTION,
    DECISIONS,
    GENERATION_STUDIES,
    LITERATURE,
    POOLING_STUDIES,
    UNANSWERABLE_STUDIES,
)


def opening(doc: Doc, src: Sources) -> None:
    """The opening paragraph: the corpus."""
    doc.check("The corpus is {} chunks", formatting.count(corpus.chunk_count(src)))
    doc.check(
        "documentation from {} trading venues and the CCXT library",
        formatting.count(corpus.venue_count(src)),
    )
    doc.definition("Sections 1\u20137 cover general practice", reason=CONVENTION)


def decision_table(doc: Doc, src: Sources) -> None:
    """Decisions at a glance: sizes, the detectable difference and near-duplicates."""
    doc.definitions(DECISIONS, reason=REGISTERED)
    gaps = agreement.study(src).gaps.values()
    doc.check(
        "at least {} days after the last round-1 judgment",
        formatting.count(math.floor(min(gaps))),
    )
    _size_decisions(doc)
    comparison = TARGETS["comparison"]
    doc.check(
        "At n = {} the smallest detectable \u03b4 is {}; at {} it is {} (\u00a77)",
        Param(formatting.count(comparison), POWER_MODEL),
        formatting.design_parameter(mcnemar_power.detectable_delta(comparison, PSI)),
        Param(formatting.count(SMALLER_COMPARISON_SIZE), POWER_MODEL),
        formatting.design_parameter(mcnemar_power.detectable_delta(SMALLER_COMPARISON_SIZE, PSI)),
    )
    doc.given(
        "Chroma found naive generation producing identical and near-identical query pairs "
        "above {} cosine",
        "0.9",
        reason=LITERATURE + " (Chroma, Generative Benchmarking)",
    )
    _pooling_decision(doc, src)
    _later_decisions(doc, src)
    threshold = protocol.duplicate_threshold(src)
    doc.check("`eval.duplicate_threshold: {}` in `configs/default.yaml` |", threshold)
    doc.check("cosine \u2265 {} on `bge-base-en-v1.5` question embeddings", threshold)
    doc.check(
        "The expectation (\u00a76) is that {} catches",
        threshold,
    )
    resamples = protocol.bootstrap_resamples(src)
    doc.check(
        "Paired bootstrap over questions ({} resamples, seed {}) for intervals; paired "
        "randomization test (sign flip, {} permutations, same seed)",
        resamples,
        protocol.bootstrap_seed_text(src),
        resamples,
    )


def _size_decisions(doc: Doc) -> None:
    """Decisions at a glance: the set size, the category targets and the venue floor."""
    total = Param(formatting.count(target_size()), REGISTERED)
    comparison = Param(formatting.count(TARGETS["comparison"]), REGISTERED)
    unanswerable = Param(formatting.count(TARGETS["unanswerable"]), REGISTERED)
    least = Param(formatting.count(MIN_QUESTIONS_PER_CATEGORY), REGISTERED)
    doc.check(
        "Labeled set size | {} questions, hard floor {}, sized from the subgroups: comparison "
        ">= {} (all answerable), unanswerable >= {}, nothing below {} |",
        total,
        Param(formatting.count(FLOOR_SIZE), REGISTERED),
        comparison,
        unanswerable,
        least,
    )
    doc.check(
        "Category split | " + " \u00b7 ".join(f"{c} {{}}" for c in TARGETS) + " = {} |",
        *[Param(formatting.count(n), REGISTERED) for n in TARGETS.values()],
        total,
    )
    doc.check(
        "The {} constraints (comparison {}, unanswerable {}, nothing below {}) do not fit in "
        "{}; {} is what they sum to. A total of {}",
        Param("3", LIST_IN_SENTENCE),
        comparison,
        unanswerable,
        least,
        Param("100", REGISTERED),
        total,
        Param("100", REGISTERED),
    )
    doc.check(
        "Comparison is {}% of the set",
        formatting.percent_figure(TARGETS["comparison"], target_size()),
    )
    doc.check(
        "Venue coverage | every venue is the answer venue for >= {} questions; >= {}% DEX |",
        Param(protocol.VENUE_FLOOR, REGISTERED),
        Param("45", REGISTERED),
    )


def _later_decisions(doc: Doc, src: Sources) -> None:
    """Decisions at a glance: augmentation, the retest and the answer evaluation."""
    doc.check("has its unjudged top {} judged (augmentation)", protocol.pool_depth_text(src))
    doc.check(
        "Intra-rater reliability | {} questions re-annotated,",
        Param(formatting.count(ROUND2_SIZE), REGISTERED),
    )
    doc.check("after a >= {}-day gap", Param(formatting.count(MIN_RETEST_DAYS), REGISTERED))
    doc.check("target \u03ba >= {}", Param(formatting.own_form(KAPPA_TARGET), REGISTERED))
    doc.check(
        "Answer evaluation | {} generator configurations.",
        formatting.count(len(answers.configurations(src))),
    )
    doc.check(
        "reaches \u03ba >= {} (Spearman >= {} for coverage)",
        *[Param(t, REGISTERED) for t in protocol.judge_thresholds(src)],
    )


def _pooling_decision(doc: Doc, src: Sources) -> None:
    """Decisions at a glance: the pool's depth and what it leaves unjudged."""
    depth, display = protocol.pool_depth_text(src), protocol.pool_display_text(src)
    doc.check(
        "| Pooling | Depth {} from each of the {} pool methods",
        depth,
        formatting.count(len(POOL_METHODS)),
    )
    doc.check("top {} shown to the annotator", display)
    k = comparisons.top_k_text(src)
    doc.check("Depth must exceed `max(k_values)`={}", k)
    doc.check("how much of each pool method's top {} is judged", k)
    low, high = retrieval.metric_range(
        src, "comparison", retrieval.ROUND1_POOL, retrieval.at_top_k(src, "unjudged")
    )
    doc.check(
        "Round 1 judged the fused top {}; after the augmentation, each round-1 pool method's "
        "unjudged@10 on the {} comparison questions is {}\u2013{}. "
        "Augmented methods are judged to depth {}",
        display,
        comparisons.comparison_questions(src),
        formatting.metric(low),
        formatting.metric(high),
        depth,
    )
    doc.ref("The unjudged@10 range is from `results/{}`", RETRIEVAL_RUN)


def pooling(doc: Doc, src: Sources) -> None:
    """Section 1: pooling depth and incompleteness."""
    doc.definitions(POOLING_STUDIES, reason=LITERATURE)
    doc.check(
        "binary relevance over {} gold chunks on {} labeled questions",
        formatting.count(datasets.gold_chunk_total(src)),
        formatting.count(datasets.labelled_count(src)),
    )
    doc.definition("Two results define the boundaries.", reason="the two papers that follow")
    depth = protocol.pool_depth_text(src)
    doc.check(
        "The pool is depth {} across {} methods",
        depth,
        formatting.count(len(POOL_METHODS)),
    )
    methods = formatting.count(len(POOL_METHODS))
    doc.check(
        "{} systems is still a thin pool, and {} of them (hybrid, hybrid_rerank, venue_split) "
        "are built from the other {}, so their top {} is expected to overlap heavily with those "
        "of bm25 and dense. No saved run holds the {} methods' top {}, so the overlap is not "
        "measured. {} rules follow:",
        methods,
        Param("3", LIST_IN_SENTENCE),
        Param("2", LIST_IN_SENTENCE),
        depth,
        methods,
        depth,
        Param("3", LIST_FOLLOWS),
    )
    doc.check("has its never-judged top {} on that question judged", depth)
    k = comparisons.top_k_text(src)
    doc.check("Depth is the pool depth, not the top {} the metrics score", k)
    doc.check("(`crossvenue pool-run`, k = {}, no metrics", depth)
    doc.check("Condensed-list nDCG\u2032@{} is reported alongside nDCG@{}", k, k)
    doc.check(
        "the harness retrieves max(k) = {} deep, so the condensed list is the top {} with "
        "never-judged chunks removed; no rank below {} is retrieved, so nothing refills it.",
        k,
        k,
        k,
    )


def generation(doc: Doc) -> None:
    """Section 2: generated questions."""
    doc.definitions(GENERATION_STUDIES, reason=LITERATURE)
    doc.definition("Four separate failure modes, each with evidence:", reason=LIST_FOLLOWS)
    doc.definition(
        "The target median question length is in the real-query band (8\u201314 words for fact "
        "questions), below the 15\u201320-word synthetic band.",
        reason=REGISTERED + ", from the cited query-length studies",
    )


def unanswerable(doc: Doc) -> None:
    """Section 3: unanswerable questions."""
    doc.definitions(UNANSWERABLE_STUDIES, reason=LITERATURE)
    doc.definition("### The SQuAD 2.0 tradition", reason="the dataset's name")
    doc.definition("SQuAD 2.0's sense", reason="the dataset's name")
    target = Param(formatting.count(TARGETS["unanswerable"]), REGISTERED)
    doc.check(
        "At least {} questions are unanswerable ({} of the {}-question target, {}%)",
        target,
        target,
        Param(formatting.count(target_size()), REGISTERED),
        formatting.percent_figure(TARGETS["unanswerable"], target_size()),
    )
    doc.definition("for three reasons:", reason=LIST_FOLLOWS)
    doc.check("{} items is enough to make", target)
    doc.definition("They come in three kinds", reason=LIST_FOLLOWS)
