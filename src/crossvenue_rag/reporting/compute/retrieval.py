"""Retrieval runs: their config, metric tables and methods.

`src.retrieval` is the run the documents report (`run_ids.RETRIEVAL_RUN`),
`src.pvalues` the run whose paired.csv they quote (`run_ids.PVALUE_RUN`); a
function that takes `run` reads another run under `results/`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ... import formatting
from ...evaluation.comparisons import OFF_VENUE_PAIRS, PairedRow
from ...evaluation.metrics import recall_at_k
from ...evaluation.results import PerQueryRow, breakdown_file, read_summary
from ...models import Question
from ...retrieval.methods import METHODS as METHODS
from ...retrieval.methods import ORACLE_METHODS
from ...run_files import PAIRED_FILE, SUMMARY_FILE
from ...stats import P50, mean, percentile
from ..run_ids import RETRIEVAL_RUN
from ..sources import Sources
from . import datasets
from .distinct import shared_value
from .estimates import Estimate, is_significant
from .lookup import entry

# A ranking metric: (ranked ids, gold ids, k) -> value.
RankMetric = Callable[[Sequence[str], Sequence[str], int], float]

# The methods round 1 of labeling pooled (ADR 0005): a record of that pool as it was,
# fixed here rather than read from `retrieval.methods.POOL_METHODS`, which also holds
# venue_split, the pool method added after round 1.
ROUND1_POOL = ("bm25", "dense", "hybrid", "hybrid_rerank")
# The methods that retrieve only from the venues a question names: venue_split from those it
# detects in the text, the oracle from the labeled ones. With fewer than two such venues,
# each falls back to hybrid_rerank.
VENUE_FILTERED = ("venue_split", *ORACLE_METHODS)
# The methods whose top 10 an index rebuild can change: every method but BM25.
DENSE_DEPENDENT = tuple(m for m in METHODS if m != "bm25")
# The method every off-venue comparison is made against, hybrid_rerank, and the first-stage
# methods it is compared with, in the order of the pairs of methodology section 9b.
BASELINE = OFF_VENUE_PAIRS[0][0]
FIRST_STAGE = tuple(b for _, b in OFF_VENUE_PAIRS)


def top_k(src: Sources, run: str = RETRIEVAL_RUN) -> int:
    """The deepest cutoff of `run`'s `eval.k_values`, the depth its methods retrieved to."""
    return max(src.run(run).config.eval.k_values)


def at_top_k(src: Sources, metric: str) -> str:
    """The column of `metric` at the retrieval run's deepest cutoff: `recall@10`."""
    return f"{metric}@{top_k(src)}"


def chunk_venues(src: Sources) -> Mapping[str, str]:
    """`PerQueryRow.retrieval`'s chunk venues: the corpus's if a row saved none, else empty."""
    rows = src.retrieval.per_query
    return src.venue_of if any(r.retrieved_venues is None for r in rows) else {}


def dense_dependent_lists(src: Sources) -> int:
    """The top-k lists of the gold set that an index rebuild can change (`DENSE_DEPENDENT`)."""
    return datasets.question_count(src) * len(DENSE_DEPENDENT)


def pool_depth(src: Sources) -> int:
    """`eval.pool_depth_per_method` of the retrieval run: each method's depth in a pool."""
    return src.retrieval.config.eval.pool_depth_per_method


def first_rank_below_top_k(src: Sources) -> int:
    """The first rank a top-k metric does not score: `top_k` + 1."""
    return top_k(src) + 1


def method_count(src: Sources) -> int:
    """Retrieval methods the run compares."""
    return len(src.retrieval.methods)


def category_metric(src: Sources, category: str, method: str, metric: str) -> float:
    """`metric` of `method` on the questions of `category`, from by_category.csv.

    Raises:
        ValueError: by_category.csv has no column `metric`.
    """
    by_category = src.retrieval.by_category
    return entry(by_category, (category, method), breakdown_file("category")).value(metric)


def category_metrics(
    src: Sources, category: str, methods: Sequence[str], metric: str
) -> list[float]:
    """`category_metric` of `metric` of each of `methods` in turn.

    Raises:
        ValueError: by_category.csv has no column `metric`.
    """
    return [category_metric(src, category, m, metric) for m in methods]


def metric_range(
    src: Sources, category: str, methods: Sequence[str], metric: str
) -> tuple[float, float]:
    """(smallest, largest) `category_metric` of `metric` over `methods`.

    Raises:
        ValueError: `methods` is empty, or by_category.csv has no column `metric`.
    """
    values = category_metrics(src, category, methods, metric)
    return min(values), max(values)


def shared_category_metric(
    src: Sources, category: str, methods: Sequence[str], metric: str
) -> float | None:
    """The one `category_metric` of `metric` that all of `methods` have; None when they differ.

    Raises:
        ValueError: by_category.csv has no column `metric`.
    """
    return shared_value(category_metrics(src, category, methods, metric))


def oracle_unjudged(src: Sources, category: str) -> float:
    """venue_split_oracle's unjudged share of the top k (`unjudged@k`) on `category`."""
    return category_metric(src, category, "venue_split_oracle", at_top_k(src, "unjudged"))


def summary_metric(src: Sources, method: str, column: str) -> float:
    """A column of `method`'s row of summary.csv (`recall@10`, `p50_ms`, ...).

    Raises:
        ValueError: summary.csv has no row `method` or no column `column`.
    """
    return entry(src.retrieval.summary, method, SUMMARY_FILE).value(column)


def summary_metrics(src: Sources, methods: Sequence[str], column: str) -> list[float]:
    """`summary_metric` of `column` of each of `methods` in turn.

    Raises:
        ValueError: summary.csv has no column `column`.
    """
    return [summary_metric(src, m, column) for m in methods]


def step_p50(src: Sources, method: str, step: str) -> float:
    """p50 ms of one retrieval step of `method`, as the run's own table gives it.

    The table is `RetrievalRun.step_table`: the run's summary.csv, or its report.md for a
    run that predates the step columns. No p50 is recomputed from per_query.csv.

    Raises:
        ValueError: the table has no row `method` or no column `step`.
    """
    steps = entry(src.retrieval.step_table, method, "the step table")
    return float(entry(steps, step, f"the step table of {method}"))


def metric_rows(src: Sources, run: str) -> list[dict[str, str]]:
    """A retrieval run's summary.csv without its timing columns (`*_ms`)."""
    table = read_summary(src.run(run).dir)
    columns = [c for c in table.columns if not c.endswith("_ms")]
    return [{c: row.text(c) for c in columns} for row in table.rows]


def runs_agree_on_metrics(src: Sources, run: str, other: str) -> bool:
    """Whether two retrieval runs have the same summary on every metric column."""
    return metric_rows(src, run) == metric_rows(src, other)


def _comparison_rows(src: Sources, method: str) -> list[PerQueryRow]:
    """`method`'s rows of per_query.csv on the comparison questions."""
    return [
        r
        for r in src.retrieval.per_query
        if r.method == method and entry(src.questions, r.question_id, "the gold set").is_comparison
    ]


def covers_every_venue(src: Sources, row: PerQueryRow) -> bool:
    """Whether a row's top k holds every venue its question names: venue_coverage@k is 1."""
    return row.number(at_top_k(src, "venue_coverage")) == 1


def coverage_counts(src: Sources, method: str) -> tuple[int, int]:
    """(comparison questions with every named venue in the top k, comparison questions)."""
    rows = _comparison_rows(src, method)
    return sum(1 for r in rows if covers_every_venue(src, r)), len(rows)


def covered_share(src: Sources, method: str) -> float:
    """Share of the comparison questions with every named venue in `method`'s top k."""
    return coverage_counts(src, method)[0] / len(datasets.comparison_ids(src))


def coverage_by_detection(src: Sources) -> tuple[int, int]:
    """venue_split's covered comparison questions: (every named venue detected, otherwise)."""
    detected = covered_other = 0
    for r in _comparison_rows(src, "venue_split"):
        if not covers_every_venue(src, r):
            continue
        q = src.questions[r.question_id]
        if set(q.venues) <= set(src.detector.detect(q.question)):
            detected += 1
        else:
            covered_other += 1
    return detected, covered_other


def split_p50(src: Sources, method: str) -> float:
    """p50 total ms of `method` on the questions venue_split splits."""
    ids = set(datasets.split_ids(src))
    return percentile(
        [
            r.value("total_ms")
            for r in src.retrieval.per_query
            if r.method == method and r.question_id in ids
        ],
        P50,
    )


def _paired(src: Sources, a: str, b: str, metric: str, scope: str) -> PairedRow:
    return entry(src.pvalues.paired, (a, b, metric, scope), f"the p-value run's {PAIRED_FILE}")


def paired_interval(src: Sources, a: str, b: str, metric: str, scope: str) -> Estimate:
    """A - b on `metric` over `scope`, with its interval."""
    return Estimate.of(_paired(src, a, b, metric, scope))


def paired_includes_zero(src: Sources, a: str, b: str, metric: str, scope: str) -> bool:
    """Whether the interval of a - b on `metric` over `scope` includes 0."""
    return paired_interval(src, a, b, metric, scope).includes_zero()


def paired_p_below(src: Sources, a: str, b: str, metric: str, scope: str, column: str) -> bool:
    """Whether the p-value `column` (`mcnemar_p`, ...) of a - b is below the smallest shown.

    Raises:
        ValueError: paired.csv has no column `column`.
    """
    return _paired(src, a, b, metric, scope).value(column) < formatting.SMALLEST_SHOWN_P


def paired_p(src: Sources, a: str, b: str, metric: str, scope: str, column: str) -> float:
    """The p-value `column` (`holm_p`, `mcnemar_p`, ...) of a - b on `metric` over `scope`.

    Raises:
        ValueError: paired.csv has no column `column`.
    """
    return _paired(src, a, b, metric, scope).value(column)


@dataclass(frozen=True)
class QuestionSplit:
    """The questions a paired comparison a - b splits into, in the metric's direction."""

    better: int  # questions a is better on
    worse: int
    ties: int

    @classmethod
    def of(cls, row: PairedRow) -> QuestionSplit:
        """How the questions of a paired row split: its `better`, `worse` and `ties`."""
        return cls(row.better, row.worse, row.ties)


def paired_counts(src: Sources, a: str, b: str, metric: str, scope: str) -> QuestionSplit:
    """How the questions of a - b on `metric` over `scope` split."""
    return QuestionSplit.of(_paired(src, a, b, metric, scope))


def paired_questions(src: Sources, a: str, b: str, metric: str, scope: str) -> int:
    """Questions a - b is paired on for `metric` over `scope`."""
    return _paired(src, a, b, metric, scope).n


def paired_never_worse(src: Sources, a: str, b: str, metric: str, scope: str) -> bool:
    """Whether a is worse than b on no question of `scope` on `metric`."""
    return paired_counts(src, a, b, metric, scope).worse == 0


def paired_significant(src: Sources, a: str, b: str, metric: str, scope: str) -> bool:
    """Whether a - b on `metric` over `scope` has p_Holm below the significance level."""
    return is_significant(paired_p(src, a, b, metric, scope, "holm_p"))


def paired_gap(src: Sources, a: str, b: str, metric: str, scope: str) -> float:
    """The size of the difference a - b on `metric` over `scope`, without its sign."""
    return abs(paired_interval(src, a, b, metric, scope).value)


def excluded_loss(src: Sources, a: str, b: str, metric: str, scope: str) -> float:
    """The largest loss of a against b the interval excludes: minus its low end."""
    return -paired_interval(src, a, b, metric, scope).low


def labelled_scores(
    src: Sources, rows: Sequence[PerQueryRow], metric: RankMetric, gold: Mapping[str, Question]
) -> list[float]:
    """`metric` at the run's top k for each of `rows` whose question `gold` labels.

    Scored against the gold chunks of `gold`, a question set by id.
    """
    k = top_k(src)
    labelled = [(r, gold[r.question_id]) for r in rows if r.question_id in gold]
    return [metric(r.ids, q.gold_chunk_ids, k) for r, q in labelled if q.is_labelled]


def recall_against(src: Sources, method: str, gold: Mapping[str, Question]) -> float:
    """Mean recall@10 of `method` on the comparison questions `gold` labels, against `gold`."""
    rows = [
        r
        for r in src.retrieval.per_query
        if r.method == method and r.question_id in gold and gold[r.question_id].is_comparison
    ]
    return mean(labelled_scores(src, rows, recall_at_k, gold))


@dataclass(frozen=True)
class RecallChange:
    """A method's comparison recall@10 against the gold set before and after augmentation.

    Unrounded: a change is computed from these and rounded only when written.
    """

    method: str
    before: float
    after: float

    @property
    def change(self) -> float:
        """After - before."""
        return self.after - self.before


def recall_before_after(src: Sources) -> list[RecallChange]:
    """Per method of `METHODS`, comparison recall@10 before and after augmentation."""
    old, new = src.questions_before_augmentation, src.questions
    return [
        RecallChange(m, recall_against(src, m, old), recall_against(src, m, new)) for m in METHODS
    ]


def recall_drops(src: Sources) -> list[float]:
    """Per method, the drop of comparison recall@10 from augmentation (`recall_before_after`)."""
    return [r.before - r.after for r in recall_before_after(src)]


def recall_change_range(src: Sources) -> tuple[float, float]:
    """(largest, smallest) change of a method's comparison recall@10 from augmentation.

    Minus the smallest and minus the largest of `recall_drops`.
    """
    drops = recall_drops(src)
    return -min(drops), -max(drops)


def recall_lower_for_every_method(src: Sources) -> bool:
    """Whether every method's comparison recall@10 is lower against the augmented gold."""
    return min(recall_drops(src)) > 0
