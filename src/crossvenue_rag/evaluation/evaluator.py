"""Ablation harness: runs every configured method over the question set and scores them.

A question without gold is run (its latency counts and its top-k reaches `per_query.csv` as
labeling candidates) and left out of the gold-metric averages. The venue metrics need no gold
and cover every question that names enough venues (methodology section 9b).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..config import Config
from ..models import RETRIEVAL_STEPS, Question, Timings
from ..retrieval.filters import Filters
from ..retrieval.protocols import TimedRanks
from ..stats import P50, P95, mean, percentile
from .metrics import evaluate, ndcg_condensed_at_k, off_venue_at_k, venue_coverage_at_k

# Decimals a metric's mean is saved with in a run's tables, one past the three a person reads.
METRIC_DECIMALS = 4
# The decimals of a latency in `summary.json`.
LATENCY_DECIMALS = 1


@dataclass(slots=True)
class QueryRun:
    """One method's retrieval for one question, with its timings and every metric scored."""

    question_id: str
    method: str
    retrieved_ids: list[str]
    timings: Timings
    labelled: bool
    retrieved_venues: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    # venue_coverage@k; empty unless the question names two or more venues.
    venue_metrics: dict[str, float] = field(default_factory=dict)
    # unjudged@k (share of the top k no judgment covered) and ndcg_prime@k
    # (nDCG on the condensed list); empty unless the question is labeled and
    # its judgment history is available.
    judged_metrics: dict[str, float] = field(default_factory=dict)
    # off_venue@k and via_ccxt@k; empty unless the question names a venue.
    off_venue_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def multi_venue(self) -> bool:
        """Whether the question names two or more venues, so venue coverage was scored."""
        return bool(self.venue_metrics)

    def metric_value(self, source: str, name: str) -> float:
        """Metric `name` of the scores in field `source` (`metrics`, `venue_metrics`, ...).

        Raises:
            ValueError: `source` has no `name`, as when the run was scored at other k values.
        """
        scores: dict[str, float] = getattr(self, source)
        if name not in scores:
            raise ValueError(f"{self.method} on {self.question_id} has no {name} in {source}")
        return scores[name]


@dataclass(slots=True)
class MethodSummary:
    """One method's metric means and latency percentiles over the question set."""

    method: str
    questions: int
    labelled_questions: int
    metrics: dict[str, float]
    latency_p50_ms: float
    latency_p95_ms: float
    step_latency_p50_ms: dict[str, float]
    multi_venue_questions: int
    venue_metrics: dict[str, float]
    judged_metrics: dict[str, float]
    off_venue_metrics: dict[str, float]


class Evaluator:
    """Runs the configured retrieval methods over a question set and scores each retrieval."""

    def __init__(self, cfg: Config, retriever: TimedRanks) -> None:
        """An evaluator with the methods and k values of `cfg`, retrieving through `retriever`."""
        self.cfg = cfg
        self.retriever = retriever

    def run(
        self,
        questions: Sequence[Question],
        *,
        filter_by_question_venues: bool,
        judged: dict[str, set[str]] | None,
    ) -> list[QueryRun]:
        """The runs of every method over every question.

        `filter_by_question_venues` (`eval --venue-filter`) restricts retrieval to the
        venues a question names. The study measures without it whether retrieval finds
        the right venue on its own, which a pre-filter would decide in advance.
        `judged` holds the chunks round 1 judged, by question id; with None no run gets
        `unjudged@k` or `ndcg_prime@k`.

        Methods are the outer loop, so the cross-encoder's accelerator work
        cannot run into another method's `encode` and `search` timings.
        """
        runs: list[QueryRun] = []

        for method in self.cfg.eval.methods:
            for question in questions:
                timings = Timings()
                filters: Filters | None = (
                    {"venue": question.venues}
                    if filter_by_question_venues and question.venues
                    else None
                )
                results = self.retriever.retrieve(
                    question.question,
                    method=method,
                    k=max(self.cfg.eval.k_values),
                    filters=filters,
                    timings=timings,
                    venues_hint=question.venues,
                )
                retrieval = Retrieval(
                    method, [r.chunk_id for r in results], [r.venue for r in results], timings
                )
                run = score(question, retrieval, self.cfg.eval.k_values, judged)
                runs.append(run)
        return runs


@dataclass(frozen=True)
class Retrieval:
    """One method's ranked chunk ids for a question, their venues, and the step timings."""

    method: str
    ids: list[str]
    venues: list[str]
    timings: Timings = field(default_factory=Timings)


def score(
    question: Question,
    retrieval: Retrieval,
    k_values: Sequence[int],
    judged: dict[str, set[str]] | None = None,
) -> QueryRun:
    """Every metric of one retrieval.

    Shared by the harness and by `make reproduce`, which re-scores the retrievals
    a run saved in per_query.csv.
    """
    run = QueryRun(
        question_id=question.id,
        method=retrieval.method,
        retrieved_ids=retrieval.ids,
        timings=retrieval.timings,
        labelled=question.is_labelled,
        retrieved_venues=retrieval.venues,
    )
    if run.labelled and judged is not None and question.id in judged:
        seen = judged[question.id]
        run.judged_metrics = {
            **{f"unjudged@{k}": _unjudged(run.retrieved_ids, seen, k) for k in k_values},
            **{
                f"ndcg_prime@{k}": ndcg_condensed_at_k(
                    run.retrieved_ids, question.gold_chunk_ids, seen, k
                )
                for k in k_values
            },
        }
    if question.is_multi_venue:
        run.venue_metrics = {
            f"venue_coverage@{k}": venue_coverage_at_k(run.retrieved_venues, question.venues, k)
            for k in k_values
        }
    if question.venues:
        for k in k_values:
            off, via = off_venue_at_k(run.retrieved_venues, question.venues, k)
            run.off_venue_metrics[f"off_venue@{k}"] = off
            run.off_venue_metrics[f"via_ccxt@{k}"] = via
    if run.labelled:
        run.metrics = evaluate(run.retrieved_ids, question.gold_chunk_ids, k_values)
    return run


def metric_names_for(k_values: Sequence[int]) -> list[str]:
    """The relevance metric names scored at these k values, `mrr` first."""
    return ["mrr"] + [f"{n}@{k}" for k in k_values for n in ("recall", "precision", "ndcg")]


def venue_metric_names_for(k_values: Sequence[int]) -> list[str]:
    """The `venue_coverage@k` names for these k values."""
    return [f"venue_coverage@{k}" for k in k_values]


def off_venue_metric_names_for(k_values: Sequence[int]) -> list[str]:
    """The `off_venue@k` and `via_ccxt@k` names for these k values."""
    return [f"{n}@{k}" for k in k_values for n in ("off_venue", "via_ccxt")]


def _off_venue_means(runs: Sequence[QueryRun], k_values: Sequence[int]) -> dict[str, float]:
    named = [r for r in runs if r.off_venue_metrics]
    if not named:
        return {}
    return {
        name: round(
            mean([r.metric_value("off_venue_metrics", name) for r in named]), METRIC_DECIMALS
        )
        for name in off_venue_metric_names_for(k_values)
    }


def _unjudged(retrieved: Sequence[str], judged: set[str], k: int) -> float:
    """Share of the top k that no judgment covered.

    Each unjudged chunk counts as a miss whether or not it is relevant; the share
    does not measure the pooling bias itself. 0.0 for an empty list.
    """
    top = list(retrieved[:k])
    return mean([c not in judged for c in top]) if top else 0.0


def judged_metric_names_for(k_values: Sequence[int]) -> list[str]:
    """The `unjudged@k` names, then the `ndcg_prime@k` names, for these k values."""
    return [f"unjudged@{k}" for k in k_values] + [f"ndcg_prime@{k}" for k in k_values]


def _judged_means(runs: Sequence[QueryRun], k_values: Sequence[int]) -> dict[str, float]:
    covered = [r for r in runs if r.judged_metrics]
    if not covered:
        return {}
    return {
        name: round(
            mean([r.metric_value("judged_metrics", name) for r in covered]), METRIC_DECIMALS
        )
        for name in judged_metric_names_for(k_values)
    }


def _relevance_means(labelled: Sequence[QueryRun], names: Sequence[str]) -> dict[str, float]:
    """Each relevance metric's mean over the labeled runs; none when no run is labeled."""
    return {
        name: round(mean([r.metric_value("metrics", name) for r in labelled]), METRIC_DECIMALS)
        for name in names
        if labelled
    }


def _venue_means(runs: Sequence[QueryRun], k_values: Sequence[int]) -> dict[str, float]:
    multi = [r for r in runs if r.multi_venue]
    if not multi:
        return {}
    return {
        name: round(mean([r.metric_value("venue_metrics", name) for r in multi]), METRIC_DECIMALS)
        for name in venue_metric_names_for(k_values)
    }


def summarise(runs: Sequence[QueryRun], k_values: Sequence[int]) -> list[MethodSummary]:
    """One summary per method, in the order the methods first appear in `runs`.

    Saved precision: metric means at four decimals, latency percentiles at one.
    """
    by_method: dict[str, list[QueryRun]] = defaultdict(list)
    for run in runs:
        by_method[run.method].append(run)

    metric_names = metric_names_for(k_values)

    summaries: list[MethodSummary] = []
    for method, method_runs in by_method.items():
        labelled = [r for r in method_runs if r.labelled]
        totals = [r.timings.total_ms for r in method_runs]
        summaries.append(
            MethodSummary(
                method=method,
                questions=len(method_runs),
                labelled_questions=len(labelled),
                metrics=_relevance_means(labelled, metric_names),
                latency_p50_ms=round(percentile(totals, P50), LATENCY_DECIMALS),
                latency_p95_ms=round(percentile(totals, P95), LATENCY_DECIMALS),
                step_latency_p50_ms={
                    step: round(
                        percentile([getattr(r.timings, step) for r in method_runs], P50),
                        LATENCY_DECIMALS,
                    )
                    for step in (f"{s}_ms" for s in RETRIEVAL_STEPS)
                },
                multi_venue_questions=sum(1 for r in method_runs if r.multi_venue),
                venue_metrics=_venue_means(method_runs, k_values),
                judged_metrics=_judged_means(method_runs, k_values),
                off_venue_metrics=_off_venue_means(method_runs, k_values),
            )
        )
    return summaries


@dataclass(slots=True)
class SliceSummary:
    """Metrics for one method restricted to one slice of the question set."""

    dimension: str  # "venue" | "qtype" | "category"
    slice: str  # "bybit" | "comparison" | "adversarial"
    method: str
    questions: int
    labelled_questions: int
    metrics: dict[str, float]
    latency_p50_ms: float
    multi_venue_questions: int
    venue_metrics: dict[str, float]
    judged_metrics: dict[str, float]
    off_venue_metrics: dict[str, float]


def summarise_by(
    runs: Sequence[QueryRun],
    questions: Sequence[Question],
    k_values: Sequence[int],
    dimension: str,
) -> list[SliceSummary]:
    """Breaks the aggregate down along one dimension of the question set.

    The corpus is heavily unbalanced across venues (`make stats` gives the
    counts). Question types differ in difficulty by more than methods differ
    from each other, so an aggregate over both hides which of the two a
    headline number measures.

    A question naming several venues counts in each of them, so venue rows do
    not sum to the total: a comparison question exercises retrieval on every
    venue it names.

    Raises:
        ValueError: `dimension` is not one that `Question.slice_labels` defines.
    """
    by_id = {q.id: q for q in questions}
    metric_names = metric_names_for(k_values)

    grouped: dict[tuple[str, str], list[QueryRun]] = defaultdict(list)
    for run in runs:
        question = by_id.get(run.question_id)
        if question is None:
            continue
        for label in question.slice_labels(dimension):
            grouped[(label, run.method)].append(run)

    summaries: list[SliceSummary] = []
    for (label, method), slice_runs in sorted(grouped.items()):
        labelled = [r for r in slice_runs if r.labelled]
        summaries.append(
            SliceSummary(
                dimension=dimension,
                slice=label,
                method=method,
                questions=len(slice_runs),
                labelled_questions=len(labelled),
                metrics=_relevance_means(labelled, metric_names),
                latency_p50_ms=round(
                    percentile([r.timings.total_ms for r in slice_runs], P50), LATENCY_DECIMALS
                ),
                multi_venue_questions=sum(1 for r in slice_runs if r.multi_venue),
                venue_metrics=_venue_means(slice_runs, k_values),
                judged_metrics=_judged_means(slice_runs, k_values),
                off_venue_metrics=_off_venue_means(slice_runs, k_values),
            )
        )
    return summaries


# The steps whose p50 and p95 summary.csv holds: the retrieval steps, then their sum.
STEP_COLUMN_STEPS = (*RETRIEVAL_STEPS, "total")


def step_latency_columns(runs: Sequence[QueryRun]) -> dict[str, dict[str, float]]:
    """Per method, the nearest-rank p50 and p95 of every retrieval step, unrounded.

    The steps sum is included as `total`; the keys run `encode_p50_ms`,
    `encode_p95_ms`, ... `total_p95_ms`.
    """
    by_method: dict[str, list[QueryRun]] = defaultdict(list)
    for run in runs:
        by_method[run.method].append(run)
    return {
        method: {
            f"{step}_{name}_ms": percentile(
                [getattr(r.timings, f"{step}_ms") for r in method_runs], fraction
            )
            for step in STEP_COLUMN_STEPS
            for name, fraction in (("p50", P50), ("p95", P95))
        }
        for method, method_runs in by_method.items()
    }
