"""Off-venue retrieval (methodology section 9b), from the retrieval run's saved venues.

The retrieval run's saved top k per question and method, re-scored for
off_venue@k and via_ccxt@k (`scored_runs`; no retrieval), and the paired
comparisons of hybrid_rerank with each first-stage method on those metrics
(`off_venue_paired_rows`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...config_eval import TOP_ONE_K
from ...evaluation.comparisons import (
    OFF_VENUE_SCOPES,
    TOP_ONE_OFF_VENUE,
    PairedRow,
    off_venue_comparisons,
)
from ...evaluation.evaluator import QueryRun, score
from ...evaluation.results import PER_QUERY_FILE
from ...formatting import SMALLEST_SHOWN_P
from ...stats import mean
from ..sources import Sources
from .datasets import category_count, category_ids
from .estimates import Estimate, is_significant
from .lookup import entry
from .memo import per_sources
from .retrieval import (
    BASELINE,
    FIRST_STAGE,
    METHODS,
    QuestionSplit,
    at_top_k,
    chunk_venues,
)

# The top-1 via-ccxt rate, the first-rank measure beside `TOP_ONE_OFF_VENUE`.
TOP_ONE_VIA_CCXT = f"via_ccxt@{TOP_ONE_K}"


@per_sources
def scored_runs(src: Sources) -> list[QueryRun]:
    """The retrieval run's saved top k per question and method, scored again.

    For off_venue@k and via_ccxt@k (methodology section 9b); no retrieval.
    """
    k_values = src.retrieval.config.eval.k_values
    venue_of = chunk_venues(src)
    return [
        score(entry(src.questions, r.question_id, "the gold set"), r.retrieval(venue_of), k_values)
        for r in src.retrieval.per_query
    ]


@per_sources
def off_venue_paired_rows(src: Sources) -> list[PairedRow]:
    """The paired comparisons of hybrid_rerank with each first-stage method off venue."""
    return off_venue_comparisons(
        src.retrieval.config, scored_runs(src), list(src.questions.values())
    )


def off_venue_means(src: Sources, method: str, category: str) -> tuple[int, dict[str, float]]:
    """(questions, mean of each off-venue metric) for one method on one category.

    Raises:
        ValueError: `method` has no row on a question of `category`.
    """
    runs = [
        r
        for r in scored_runs(src)
        if r.method == method and src.questions[r.question_id].effective_category == category
    ]
    if not runs:
        raise ValueError(f"{PER_QUERY_FILE} has no row of {method} on {category}")
    names = runs[0].off_venue_metrics
    return len(runs), {m: mean([r.off_venue_metrics[m] for r in runs]) for m in names}


def first_rank_count(src: Sources, metric: str, method: str, category: str) -> tuple[int, int]:
    """(questions whose first result `metric` counts, questions) of `method` on `category`.

    `metric` is a top-1 metric, `off_venue@1` or `via_ccxt@1`.

    Raises:
        ValueError: `metric` is no off-venue metric, or `method` has no row on `category`.
    """
    size, means = off_venue_means(src, method, category)
    return round(entry(means, metric, "the off-venue metrics") * size), size


def first_rank_recovered(src: Sources, b: str, category: str) -> tuple[int, int]:
    """Where `b`'s first chunk is off-venue and hybrid_rerank's is not: what it is instead.

    (questions where hybrid_rerank's first chunk is on a named venue, questions where it
    is a CCXT chunk), over the questions of `category`.
    """
    first = {
        (r.method, r.question_id): r.off_venue_metrics
        for r in scored_runs(src)
        if src.questions[r.question_id].effective_category == category
    }
    recovered = [
        entry(first, (BASELINE, q), PER_QUERY_FILE)
        for method, q in first
        if method == b
        and first[(b, q)][TOP_ONE_OFF_VENUE]
        and not entry(first, (BASELINE, q), PER_QUERY_FILE)[TOP_ONE_OFF_VENUE]
    ]
    via_ccxt = sum(1 for metrics in recovered if metrics[TOP_ONE_VIA_CCXT])
    return len(recovered) - via_ccxt, via_ccxt


def first_rank_counts(
    src: Sources, metric: str, category: str, methods: Sequence[str]
) -> list[int]:
    """`first_rank_count` of each of `methods` in turn, flattened: count, questions, ...

    Raises:
        ValueError: `metric` is no off-venue metric, or a method of `methods` has no row on
            `category`.
    """
    return [v for m in methods for v in first_rank_count(src, metric, m, category)]


def first_rank_spread(
    src: Sources, metric: str, category: str, methods: Sequence[str]
) -> tuple[int, int]:
    """(smallest, largest) `first_rank_count` over `methods`.

    Raises:
        ValueError: `methods` is empty, `metric` is no off-venue metric, or a method of `methods`
            has no row on `category`.
    """
    counts = [first_rank_count(src, metric, m, category)[0] for m in methods]
    return min(counts), max(counts)


def first_rank_at_most(
    src: Sources, metric: str, category: str, methods: Sequence[str], most: int
) -> bool:
    """Whether every one of `methods` has a `first_rank_count` of at most `most`.

    Raises:
        ValueError: `methods` is empty, `metric` is no off-venue metric, or a method of `methods`
            has no row on `category`.
    """
    return first_rank_spread(src, metric, category, methods)[1] <= most


def first_rank_adjacent(src: Sources, metric: str, category: str, methods: Sequence[str]) -> bool:
    """Whether the `first_rank_count`s of `methods` differ by at most one.

    Raises:
        ValueError: `methods` is empty, `metric` is no off-venue metric, or a method of `methods`
            has no row on `category`.
    """
    low, high = first_rank_spread(src, metric, category, methods)
    return high - low <= 1


def top_k_shares(src: Sources, category: str, methods: Sequence[str]) -> list[float]:
    """Per method in turn, via_ccxt@k then off_venue@k on `category`, flattened.

    Raises:
        ValueError: a method of `methods` has no row on a question of `category`.
    """
    names = (at_top_k(src, "via_ccxt"), at_top_k(src, "off_venue"))
    return [off_venue_means(src, m, category)[1][x] for m in methods for x in names]


def via_ccxt_exceeds_off_venue(src: Sources, category: str, methods: Sequence[str]) -> bool:
    """Whether, for each of `methods`, CCXT's pages take more of the top k than unnamed venues.

    Raises:
        ValueError: a method of `methods` has no row on a question of `category`.
    """
    via, off = at_top_k(src, "via_ccxt"), at_top_k(src, "off_venue")
    means = [off_venue_means(src, m, category)[1] for m in methods]
    return all(m[via] > m[off] for m in means)


def _off_venue_row(src: Sources, b: str, metric: str, scope: str) -> PairedRow:
    """The row of hybrid_rerank - `b` on an off-venue `metric` over `scope`."""
    for r in off_venue_paired_rows(src):
        if (r.b, r.metric, r.scope) == (b, metric, scope):
            return r
    raise ValueError(f"no off-venue comparison of {b} on {metric} over {scope}")


def off_venue_questions(src: Sources, b: str, metric: str, scope: str) -> int:
    """Questions paired in hybrid_rerank - `b` on an off-venue `metric` over `scope`.

    Raises:
        ValueError: there is no comparison of `b` on `metric` over `scope`.
    """
    return _off_venue_row(src, b, metric, scope).n


def off_venue_interval(src: Sources, b: str, metric: str, scope: str) -> Estimate:
    """hybrid_rerank - `b` on an off-venue metric, with its interval.

    Raises:
        ValueError: there is no comparison of `b` on `metric` over `scope`.
    """
    return Estimate.of(_off_venue_row(src, b, metric, scope))


def off_venue_holm_p(src: Sources, b: str, metric: str, scope: str) -> float:
    """p_Holm of hybrid_rerank - `b` on an off-venue metric.

    Raises:
        ValueError: there is no comparison of `b` on `metric` over `scope`, or the comparison is
            descriptive.
    """
    holm_p = _off_venue_row(src, b, metric, scope).holm_p
    if holm_p is None:
        raise ValueError(f"expected a p-value, got None: {b} {metric} {scope}")
    return holm_p


def off_venue_share_below_bound(src: Sources, b: str, scope: str) -> bool:
    """Whether hybrid_rerank - `b` on off_venue@k has p_Holm below `formatting.SMALLEST_SHOWN_P`.

    Its questions must be those of the top-1 comparison.

    Raises:
        ValueError: there is no off_venue@k or top-1 comparison of `b` over `scope`, or the
            off_venue@k comparison is descriptive.
    """
    share = at_top_k(src, "off_venue")
    return off_venue_holm_p(src, b, share, scope) < SMALLEST_SHOWN_P and off_venue_questions(
        src, b, share, scope
    ) == off_venue_questions(src, b, TOP_ONE_OFF_VENUE, scope)


def measure_names(src: Sources) -> list[str]:
    """The off-venue measures: off-venue and via-ccxt share at k, then at the first rank."""
    return [
        at_top_k(src, "off_venue"),
        at_top_k(src, "via_ccxt"),
        TOP_ONE_OFF_VENUE,
        TOP_ONE_VIA_CCXT,
    ]


@dataclass(frozen=True)
class MeasureRow:
    """One method's off-venue measures on one category."""

    method: str
    share: float  # off-venue share at k
    via: float  # via-ccxt share at k
    off_first: int  # questions whose first result is off venue
    via_first: int  # questions whose first result is via CCXT
    questions: int


def measure_rows(src: Sources, category: str) -> list[MeasureRow]:
    """Per method on `category`: its off-venue measures.

    Raises:
        ValueError: a method has no row on `category`.
    """
    share, via = at_top_k(src, "off_venue"), at_top_k(src, "via_ccxt")
    rows: list[MeasureRow] = []
    for m in METHODS:
        size, means = off_venue_means(src, m, category)
        off1 = first_rank_count(src, TOP_ONE_OFF_VENUE, m, category)[0]
        via1 = first_rank_count(src, TOP_ONE_VIA_CCXT, m, category)[0]
        rows.append(MeasureRow(m, means[share], means[via], off1, via1, size))
    return rows


def same_top_k(src: Sources, method: str, category: str) -> int:
    """Questions of `category` on which `method` retrieves hybrid_rerank's top k."""
    top = {(r.question_id, r.method): r.retrieved_ids for r in src.retrieval.per_query}
    return sum(
        entry(top, (q, method), PER_QUERY_FILE) == entry(top, (q, BASELINE), PER_QUERY_FILE)
        for q in category_ids(src, category)
    )


def equal_to_baseline(src: Sources, category: str) -> bool:
    """Whether venue_split and the oracle measure what hybrid_rerank does on `category`.

    venue_split retrieves hybrid_rerank's top k on every question; the oracle's
    means equal hybrid_rerank's.

    Raises:
        ValueError: venue_split_oracle or hybrid_rerank has no row on `category`.
    """
    size = category_count(src, category)
    oracle = off_venue_means(src, "venue_split_oracle", category)[1]
    baseline = off_venue_means(src, BASELINE, category)[1]
    return same_top_k(src, "venue_split", category) == size and all(
        oracle[m] == baseline[m] for m in measure_names(src)
    )


@dataclass(frozen=True)
class OffVenueComparisonRow:
    """hybrid_rerank - `b` over `scope` off venue; the p-values are None on a descriptive row."""

    b: str
    scope: str
    descriptive: bool
    share: Estimate  # the difference in off-venue share
    first: Estimate  # the difference in first-rank off-venue rate
    share_holm: float | None
    first_mcnemar: float | None
    first_holm: float | None


def off_venue_comparison_rows(src: Sources) -> list[OffVenueComparisonRow]:
    """hybrid_rerank against each first-stage method, per scope, off venue."""
    share = at_top_k(src, "off_venue")
    rows: list[OffVenueComparisonRow] = []
    for scope in OFF_VENUE_SCOPES:
        for b in FIRST_STAGE:
            s, f = (
                _off_venue_row(src, b, share, scope),
                _off_venue_row(src, b, TOP_ONE_OFF_VENUE, scope),
            )
            descriptive = s.role == "descriptive"
            rows.append(
                OffVenueComparisonRow(
                    b,
                    scope,
                    descriptive,
                    Estimate.of(s),
                    Estimate.of(f),
                    None if descriptive else s.holm_p,
                    None if descriptive else f.mcnemar_p,
                    None if descriptive else f.holm_p,
                )
            )
    return rows


def discordant(src: Sources, b: str, measure: str, scope: str) -> QuestionSplit:
    """How the questions of hybrid_rerank - `b` on `measure` split.

    Raises:
        ValueError: there is no comparison of `b` on `measure` over `scope`.
    """
    return QuestionSplit.of(_off_venue_row(src, b, measure, scope))


def significant_against(src: Sources, b: str, scope: str) -> bool:
    """Whether hybrid_rerank - `b` is significant after Holm on both off-venue measures.

    Off-venue share at k below `formatting.SMALLEST_SHOWN_P`, the first rank below the
    significance level, both intervals below 0.

    Raises:
        ValueError: either comparison of `b` over `scope` is missing or descriptive.
    """
    share = at_top_k(src, "off_venue")
    first = _off_venue_row(src, b, TOP_ONE_OFF_VENUE, scope)
    return (
        off_venue_holm_p(src, b, share, scope) < SMALLEST_SHOWN_P
        and is_significant(off_venue_holm_p(src, b, TOP_ONE_OFF_VENUE, scope))
        and _off_venue_row(src, b, share, scope).high < 0
        and first.high < 0
    )


def never_worse(src: Sources, b: str, scope: str) -> bool:
    """Whether hybrid_rerank is worse than `b` on no question, on both measures, lower better.

    Raises:
        ValueError: either comparison of `b` over `scope` is missing.
    """
    measures = (at_top_k(src, "off_venue"), TOP_ONE_OFF_VENUE)
    rows = [_off_venue_row(src, b, m, scope) for m in measures]
    return all(r.worse == 0 and not r.higher_is_better for r in rows)


def others_include_zero(src: Sources, b: str, scope: str) -> bool:
    """Whether every off-venue interval over `scope` against a method other than `b` includes 0."""
    return all(
        Estimate.of(r).includes_zero()
        for r in off_venue_paired_rows(src)
        if r.b != b and r.scope == scope
    )


def scope_questions(src: Sources, scope: str) -> int:
    """Questions of the first off-venue comparison over `scope`.

    Raises:
        ValueError: there is no comparison over `scope`.
    """
    n = next((r.n for r in off_venue_paired_rows(src) if r.scope == scope), None)
    if n is None:
        raise ValueError(f"no off-venue comparison over {scope}")
    return n


def scope_descriptive(src: Sources, scope: str) -> bool:
    """Whether every off-venue comparison over `scope` is descriptive, with no p-value."""
    return all(
        r.role == "descriptive" and r.randomisation_p is None and r.holm_p is None
        for r in off_venue_paired_rows(src)
        if r.scope == scope
    )
