"""The reranker bench: venue_split's cross-encoder called once per venue or once for all.

`src.bench` is the bench run (`run_ids.BENCH_RUN`). The paired differences are
recomputed from its per-question timings with the retrieval run's bootstrap settings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ...evaluation.bench import MODES, RAW_FILE, BenchRow
from ...evaluation.machine import MACHINE_FILE
from ...evaluation.paired import MIN_TESTED_QUESTIONS, paired_bootstrap
from ...run_files import SUMMARY_FILE
from ...stats import mean
from ..sources import Sources
from . import datasets, retrieval
from .estimates import Estimate
from .lookup import entry
from .memo import per_sources

# The bench's two modes (`evaluation.bench.MODES`): one cross-encoder call per venue, one for all.
PER_VENUE, ONE_CALL = MODES
# The p50 total time column of the bench summary.csv.
TOTAL_P50 = "total_p50_ms"
# The columns of the bench table, from summary.csv.
MODE_COLUMNS = (
    "calls_per_query",
    "pairs_per_call_mean",
    TOTAL_P50,
    "total_p95_ms",
    "total_mean_ms",
)


@dataclass(frozen=True)
class PairedBench:
    """Per-venue minus one-call, paired by question.

    `one_call_faster` counts the questions on which one call was faster in the
    rerank step. `outside_first` and `outside_second` are the mean time outside the
    rerank step of the mode timed first and of the mode timed second in each pair.
    """

    total_ms: Estimate
    rerank_ms: Estimate
    one_call_faster: int
    outside_first: float
    outside_second: float


def paired_bench(raw: Sequence[BenchRow], resamples: int, seed: int) -> PairedBench:
    """Per-venue minus one-call, paired by question, for total and rerank ms.

    Raises:
        ValueError: `resamples` is below 1.
    """
    by: dict[str, dict[str, BenchRow]] = defaultdict(dict)
    for r in raw:
        by[r.question_id][r.mode] = r
    pairs = [
        (entry(v, PER_VENUE, f"{RAW_FILE} of {q}"), entry(v, ONE_CALL, f"{RAW_FILE} of {q}"))
        for q, v in by.items()
    ]

    def difference(column: str) -> Estimate:
        a = [getattr(per_venue, column) for per_venue, _ in pairs]
        b = [getattr(one_call, column) for _, one_call in pairs]
        return Estimate.of_bootstrap(paired_bootstrap(a, b, resamples, seed))

    outside: dict[int, list[float]] = {1: [], 2: []}
    for r in raw:
        entry(outside, r.position, "the positions of a pair").append(r.total_ms - r.rerank_ms)
    return PairedBench(
        total_ms=difference("total_ms"),
        rerank_ms=difference("rerank_ms"),
        one_call_faster=sum(
            1 for per_venue, one_call in pairs if one_call.rerank_ms < per_venue.rerank_ms
        ),
        outside_first=mean(outside[1]),
        outside_second=mean(outside[2]),
    )


@per_sources
def paired(src: Sources) -> PairedBench:
    """`paired_bench` of the bench run, with the retrieval run's bootstrap resamples and seed."""
    return paired_bench(
        src.bench.raw,
        src.retrieval.config.eval.bootstrap_resamples,
        src.retrieval.config.eval.bootstrap_seed,
    )


def paired_difference(src: Sources, column: str) -> Estimate:
    """The mean of per-venue minus one-call on `column`, with its interval.

    `column` is `total_ms` or `rerank_ms`.
    """
    difference: Estimate = getattr(paired(src), column)
    return difference


def one_call_faster(src: Sources) -> int:
    """Questions on which one call was faster than one per venue in the rerank step."""
    return paired(src).one_call_faster


def outside_rerank(src: Sources) -> tuple[float, float]:
    """(mean ms outside the rerank step of the mode timed first, of the mode timed second)."""
    return paired(src).outside_first, paired(src).outside_second


def timed_question_count(src: Sources) -> int:
    """Questions the bench timed."""
    return len({r.question_id for r in src.bench.raw})


def times_split_questions(src: Sources) -> bool:
    """Whether the bench timed exactly the questions venue_split splits."""
    return {r.question_id for r in src.bench.raw} == set(datasets.split_ids(src))


def descriptive_with_saving(src: Sources) -> bool:
    """Whether the bench is a slice below the tested size and its rerank saving excludes 0."""
    return (
        timed_question_count(src) < MIN_TESTED_QUESTIONS
        and paired_difference(src, "rerank_ms").low > 0
    )


def mode_value(src: Sources, mode: str, column: str) -> float:
    """A column of `mode`'s row of the bench summary.csv.

    Raises:
        ValueError: summary.csv has no row `mode` or no column `column`.
    """
    return entry(src.bench.summary, mode, f"the bench {SUMMARY_FILE}").value(column)


def mode_rows(src: Sources) -> list[tuple[str, list[float]]]:
    """Per mode of `MODES`, its `MODE_COLUMNS`."""
    return [(m, [mode_value(src, m, c) for c in MODE_COLUMNS]) for m in MODES]


def modes_agree(src: Sources) -> bool:
    """Whether both modes score the same pairs and retrieve the same ids on every question."""
    rows = [entry(src.bench.summary, m, f"the bench {SUMMARY_FILE}") for m in MODES]
    pairs = {row.pairs_per_query_mean for row in rows}
    same = min(row.identical_retrievals for row in rows)
    return len(pairs) == 1 and same == timed_question_count(src)


def load_average(src: Sources) -> tuple[float, float]:
    """(median, maximum) of the 1-minute load average while timing.

    Raises:
        ValueError: machine.json records no load while timing.
    """
    load = src.bench.machine.load1_during
    if load.median is None or load.max is None:
        raise ValueError(f"{src.bench.dir / MACHINE_FILE} records no load while timing")
    return load.median, load.max


def cpu_count(src: Sources) -> int:
    """CPUs of the machine the bench ran on.

    Raises:
        ValueError: machine.json records no CPU count.
    """
    cpus = src.bench.machine.before.cpu_count
    if cpus is None:
        raise ValueError(f"{src.bench.dir / MACHINE_FILE} records no CPU count")
    return cpus


def discarded_pairs(src: Sources) -> int:
    """Timed pairs discarded because the machine failed the second check."""
    return src.bench.machine.discarded_pairs


def eval_below_bench(src: Sources) -> tuple[float, float]:
    """(how far venue_split's evaluation p50 is below the per-venue mode's, that p50), in ms.

    venue_split's p50 on the questions it splits in the retrieval run, against
    the per-venue mode's p50 in the bench.
    """
    bench_p50 = mode_value(src, PER_VENUE, TOTAL_P50)
    return bench_p50 - retrieval.split_p50(src, "venue_split"), bench_p50
