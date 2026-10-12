"""docs/results/findings.md: off-venue retrieval (methodology section 9b)."""

from __future__ import annotations

from .... import formatting
from ....evaluation.comparisons import TOP_ONE_OFF_VENUE
from ...check import Doc
from ...compute import datasets, retrieval
from ...compute import retrieval_off_venue as off_venue
from ...run_ids import RETRIEVAL_RUN
from ...sources import Sources
from ..comparisons import (
    comparison_questions,
    estimate,
    level,
    min_tested,
    p_bound,
    signed_parts,
    top_k_text,
)
from ..methodology.protocol import bootstrap_resamples, bootstrap_seed_text
from ..reasons import INTERVAL_LEVEL


def definitions(doc: Doc, src: Sources) -> None:
    """Section 9b's measures, as findings.md introduces them."""
    doc.ref("after the results of `results/{}` were seen", RETRIEVAL_RUN)
    doc.check("share of a question's top {} whose venue", top_k_text(src))
    doc.check("takes a slot in the top {}; it is not an error", top_k_text(src))
    doc.check(
        "All {} are computed from the run's saved `retrieved_venues`",
        formatting.count(len(off_venue.measure_names(src))),
    )


def measures(doc: Doc, src: Sources, category: str) -> None:
    """The off-venue measures of every method on the questions of `category`.

    Raises:
        ValueError: a method has no row on `category`.
    """
    size = formatting.count(datasets.category_count(src, category))
    doc.table(
        f"the {size} {category} questions: | method | off-venue share@{top_k_text(src)}",
        {
            r.method: [
                formatting.metric(r.share),
                formatting.metric(r.via),
                formatting.share(r.off_first, r.questions),
                formatting.share(r.via_first, r.questions),
                formatting.count(r.questions),
            ]
            for r in off_venue.measure_rows(src, category)
        },
    )
    doc.caption(
        f"Off-venue measures, run `results/{{run}}`, `v0.2.1-gold`, the {{}} {category} questions:",
        (RETRIEVAL_RUN,),
        size,
    )


def baseline(doc: Doc, src: Sources) -> None:
    """The figure's sizes, and why venue_split's adversarial rows equal hybrid_rerank's."""
    comparison = comparison_questions(src)
    adversarial = formatting.count(datasets.category_count(src, "adversarial"))
    doc.check(
        "stacked, on the {} comparison and the {} adversarial questions]", comparison, adversarial
    )
    doc.claim(
        "venue_split returns hybrid_rerank's top {} on all {} and the oracle on {}, so their rows "
        "equal hybrid_rerank's",
        top_k_text(src),
        formatting.count(off_venue.same_top_k(src, "venue_split", "adversarial")),
        formatting.count(off_venue.same_top_k(src, "venue_split_oracle", "adversarial")),
        holds=off_venue.equal_to_baseline(src, "adversarial"),
        what="venue_split equals hybrid_rerank on every adversarial question; the oracle's means "
        "equal it",
    )
    doc.caption(
        "Run `results/{run}`, `v0.2.1-gold`, {} comparison and {} adversarial questions. "
        "hybrid_rerank",
        (RETRIEVAL_RUN,),
        comparison,
        adversarial,
    )
    doc.given(
        "hybrid_rerank - each first-stage method, {}% paired bootstrap intervals",
        level(),
        reason=INTERVAL_LEVEL,
    )
    doc.check(
        "sign-flip randomization tests ({} resamples, seed {})",
        bootstrap_resamples(src),
        bootstrap_seed_text(src),
    )


def comparison_claims(doc: Doc, src: Sources) -> None:
    """hybrid_rerank against each first-stage method off venue."""
    doc.table(
        "| hybrid_rerank - b, scope |",
        {
            f"{r.b}, {r.scope}" + (" (descriptive)" if r.descriptive else ""): [
                estimate(r.share),
                formatting.optional(r.share_holm, formatting.p_value),
                estimate(r.first),
                formatting.optional(r.first_mcnemar, formatting.p_value),
                formatting.optional(r.first_holm, formatting.p_value),
            ]
            for r in off_venue.off_venue_comparison_rows(src)
        },
    )


def bm25(doc: Doc, src: Sources) -> None:
    """hybrid_rerank against BM25 off venue, and what the other comparisons show."""
    share = retrieval.at_top_k(src, "off_venue")
    doc.claim(
        "On the comparison questions hybrid_rerank - bm25 is {} [{}, {}] on off-venue share@{}, "
        "p_Holm < {}, and {} [{}, {}] on the top-1 off-venue rate, p_Holm = {} (both secondary)",
        *signed_parts(off_venue.off_venue_interval(src, "bm25", share, "comparison")),
        top_k_text(src),
        p_bound(),
        *signed_parts(off_venue.off_venue_interval(src, "bm25", TOP_ONE_OFF_VENUE, "comparison")),
        formatting.p_value(
            off_venue.off_venue_holm_p(src, "bm25", TOP_ONE_OFF_VENUE, "comparison")
        ),
        holds=off_venue.significant_against(src, "bm25", "comparison"),
        what="both bm25 comparisons on comparison questions are significant after Holm, "
        "intervals below 0",
    )
    split = off_venue.discordant(src, "bm25", share, "comparison")
    doc.claim(
        "Lower is better on both. hybrid_rerank has the smaller off-venue share on {} questions, "
        "the larger on none and the same on {}. Where bm25's first chunk is off-venue, its own "
        "is not on {} questions (a named venue on {}, a CCXT chunk on {}), with none the other "
        "way.",
        formatting.count(split.better),
        formatting.count(split.ties),
        formatting.count(off_venue.discordant(src, "bm25", TOP_ONE_OFF_VENUE, "comparison").better),
        *[formatting.count(v) for v in off_venue.first_rank_recovered(src, "bm25", "comparison")],
        holds=off_venue.never_worse(src, "bm25", "comparison"),
        what="counted in the metrics' direction, hybrid_rerank is worse than bm25 on no question",
    )
    doc.claim(
        "Every comparison against dense and hybrid on the comparison questions is inconclusive at "
        "this size.",
        holds=off_venue.others_include_zero(src, "bm25", "comparison"),
        what="every other comparison-scope off-venue interval includes 0",
    )
    doc.claim(
        "The {} adversarial questions are a category of fewer than {} questions, so under",
        formatting.count(off_venue.scope_questions(src, "adversarial")),
        min_tested(),
        holds=off_venue.scope_descriptive(src, "adversarial"),
        what="every adversarial off-venue row is descriptive, with no p-value",
    )
