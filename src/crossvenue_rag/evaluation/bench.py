"""`crossvenue bench-rerank`: venue_split's cross-encoder calls timed both ways.

The two modes, one call per venue or one call in all, score the same (query, chunk) pairs,
and each question runs both back to back in one process, in alternating order. The bench
waits for a quiet machine before each pair, as `machine.busy_reasons` defines it (`--force`
skips the wait), and the load during timing is recorded.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import get_args

from .. import formatting, listing
from ..config import Config
from ..errors import UserError
from ..models import MULTI_VENUE, Question, Timings
from ..retrieval.protocols import BenchRetriever, RerankCalls, TimesVenueSplit
from ..retrieval.retriever import Retriever, venue_split_splits
from ..run_files import (
    SUMMARY_FILE,
    CsvRow,
    CsvTable,
    RunIndex,
    create_run_dir,
    question_fields,
    read_table,
    run_path,
    write_index,
    write_run_config,
    write_table,
)
from ..stats import P50, P95, mean, percentile
from .labels import markdown_table
from .machine import (
    COMPETITORS,
    LOAD_DECIMALS,
    MACHINE_FILE,
    LoadDuring,
    MachineRecord,
    MachineState,
    busy_reasons,
    machine_json,
    machine_state,
    recordable,
)

# One row per (question, round, mode).
RAW_FILE = "raw.csv"
# The decimals raw.csv and summary.csv keep, as the committed bench run has them: a raw
# timing, a summary timing, calls per query and pairs per call or per query.
RAW_MS_DECIMALS = 3
SUMMARY_MS_DECIMALS = 1
CALLS_DECIMALS = 2
PAIRS_DECIMALS = 1
BENCH_REPORT_FILE = "BENCH.md"
# Seconds between two checks for a quiet machine.
QUIET_WAIT_SECONDS = 5
# The checks one wait for a quiet machine makes before the bench gives up: an hour at
# QUIET_WAIT_SECONDS between checks.
MAX_QUIET_CHECKS = 720
# The timed pairs of one question discarded for a busy machine before the bench gives up.
MAX_PAIR_RETRIES = 100

# The two rerank modes in the order raw.csv, summary.csv and BENCH.md list them.
MODES: tuple[RerankCalls, ...] = get_args(RerankCalls)


class BenchRow(CsvRow):
    """One timed retrieval of one question in one rerank mode: a row of raw.csv."""

    question_id: str
    round: int
    position: int  # 1 if this mode ran first for this question in this round
    mode: str
    total_ms: float
    rerank_ms: float
    calls: int
    pairs_per_call: str  # space-separated
    identical: bool  # same retrieved ids as the other mode
    load1: float  # 1-minute load average before the pair


def split_questions(retriever: TimesVenueSplit, questions: Sequence[Question]) -> list[Question]:
    """The questions venue_split splits (two or more venues detected in the text).

    On the others venue_split is hybrid_rerank and the two modes are the same code.
    """
    return [q for q in questions if venue_split_splits(retriever.venue_detector.detect(q.question))]


@dataclass(frozen=True)
class Pacing:
    """How the bench waits for a quiet machine.

    With `quiet`, each question's pair of timings needs a quiet machine before
    and after it. The run checks `quiet()` up to `MAX_QUIET_CHECKS` times, calling
    `wait` after each busy check, and times the pair. It discards and repeats the
    pair if `quiet()` fails afterward, timing at most `MAX_PAIR_RETRIES` pairs.
    `load` gives the 1-minute load average recorded on each row. Without
    `quiet` the bench times without waiting.
    """

    quiet: Callable[[], bool] | None
    wait: Callable[[], None]
    load: Callable[[], float]


def run_bench(
    retriever: TimesVenueSplit,
    questions: Sequence[Question],
    rounds: int,
    pacing: Pacing,
) -> tuple[list[BenchRow], int]:
    """(rows, discarded pairs), both modes on every question in every round."""
    rows: list[BenchRow] = []
    discarded = 0
    for r in range(rounds):
        for i, q in enumerate(questions):
            order = MODES if (i + r) % len(MODES) == 0 else MODES[::-1]
            pair, retries = _quiet_pair(retriever, q, r + 1, order, pacing)
            rows += pair
            discarded += retries
    return rows, discarded


def _quiet_pair(
    retriever: TimesVenueSplit,
    q: Question,
    round_: int,
    order: Sequence[RerankCalls],
    pacing: Pacing,
) -> tuple[list[BenchRow], int]:
    """One question's pair of timings on a quiet machine, and the pairs discarded first."""
    for discarded in range(MAX_PAIR_RETRIES):
        _await_quiet(q, pacing)
        pair = _time_pair(retriever, q, round_, order, pacing.load())
        if pacing.quiet is None or pacing.quiet():
            return pair, discarded
    raise UserError(
        f"{q.id}: the machine was busy after each of "
        f"{formatting.plural(MAX_PAIR_RETRIES, 'timed pair')}",
        hint="stop the competing processes, or time anyway with --force",
    )


def _await_quiet(q: Question, pacing: Pacing) -> None:
    """Returns once `pacing.quiet()` holds; UserError after `MAX_QUIET_CHECKS` busy checks."""
    if pacing.quiet is None:
        return
    for _ in range(MAX_QUIET_CHECKS):
        if pacing.quiet():
            return
        pacing.wait()
    raise UserError(
        f"{q.id}: the machine was busy at each of "
        f"{formatting.plural(MAX_QUIET_CHECKS, 'check')} for a quiet machine",
        hint="stop the competing processes, or time anyway with --force",
    )


def _time_pair(
    retriever: TimesVenueSplit,
    q: Question,
    round_: int,
    order: Sequence[RerankCalls],
    load1: float,
) -> list[BenchRow]:
    # Saved precision: timings at three decimals of a millisecond, the load at two.
    done: dict[str, tuple[BenchRow, list[str]]] = {}
    for position, mode in enumerate(order, start=1):
        timings = Timings()
        pairs: list[int] = []
        results = retriever.venue_split_timed(q.question, mode, timings, pairs)
        row = BenchRow(
            question_id=q.id,
            round=round_,
            position=position,
            mode=mode,
            total_ms=round(timings.total_ms, RAW_MS_DECIMALS),
            rerank_ms=round(timings.rerank_ms, RAW_MS_DECIMALS),
            calls=len(pairs),
            pairs_per_call=" ".join(map(str, pairs)),
            identical=True,
            load1=round(load1, LOAD_DECIMALS),
        )
        done[mode] = (row, [x.chunk_id for x in results])
    same = done["per_venue"][1] == done["one_call"][1]
    return [done[mode][0].model_copy(update={"identical": same}) for mode in MODES]


class BenchSummaryRow(CsvRow):
    """One mode's latency percentiles and means, calls and pairs: a row of summary.csv."""

    mode: str
    n: int
    total_p50_ms: float
    total_p95_ms: float
    total_mean_ms: float
    rerank_p50_ms: float
    rerank_p95_ms: float
    rerank_mean_ms: float
    calls_per_query: float
    pairs_per_call_mean: float
    pairs_per_query_mean: float
    identical_retrievals: int


def summarise_modes(rows: Sequence[BenchRow]) -> list[BenchSummaryRow]:
    """One row per mode: latency percentiles and means, calls, pairs and identical retrievals.

    Saved precision: latency at one decimal of a millisecond, calls per query at two.

    Raises:
        ValueError: A mode of `MODES` has no row (`bench_rerank` times each at least once).
    """
    out = []
    for mode in MODES:
        mine = [r for r in rows if r.mode == mode]
        if not mine:
            raise ValueError(f"no bench row for mode {mode}")
        total = [r.total_ms for r in mine]
        rerank = [r.rerank_ms for r in mine]
        pairs = [int(p) for r in mine for p in r.pairs_per_call.split()]
        out.append(
            BenchSummaryRow(
                mode=mode,
                n=len(mine),
                total_p50_ms=round(percentile(total, P50), SUMMARY_MS_DECIMALS),
                total_p95_ms=round(percentile(total, P95), SUMMARY_MS_DECIMALS),
                total_mean_ms=round(mean(total), SUMMARY_MS_DECIMALS),
                rerank_p50_ms=round(percentile(rerank, P50), SUMMARY_MS_DECIMALS),
                rerank_p95_ms=round(percentile(rerank, P95), SUMMARY_MS_DECIMALS),
                rerank_mean_ms=round(mean(rerank), SUMMARY_MS_DECIMALS),
                calls_per_query=round(mean([r.calls for r in mine]), CALLS_DECIMALS),
                pairs_per_call_mean=round(mean(pairs), PAIRS_DECIMALS),
                pairs_per_query_mean=round(sum(pairs) / len(mine), PAIRS_DECIMALS),
                identical_retrievals=sum(r.identical for r in mine),
            )
        )
    return out


def read_bench_raw(run_dir: Path) -> CsvTable[BenchRow]:
    """raw.csv of a bench run; one format, the columns of `BenchRow`."""
    return read_table(run_dir / RAW_FILE, BenchRow)


def read_bench_summary(run_dir: Path) -> CsvTable[BenchSummaryRow]:
    """summary.csv of a bench run; one format, the columns of `BenchSummaryRow`."""
    return read_table(run_dir / SUMMARY_FILE, BenchSummaryRow)


@dataclass(frozen=True)
class BenchResult:
    """A finished bench: its rows, summary and the machine state around it.

    `discarded` counts the pairs thrown away because the machine became busy;
    `checked` is False when `--force` skipped the quiet-machine checks.
    """

    rows: Sequence[BenchRow]
    summary: Sequence[BenchSummaryRow]
    before: MachineState
    after: MachineState
    discarded: int
    checked: bool


def write_bench(
    cfg: Config,
    result: BenchResult,
    index_version: str,
    questions_path: Path,
) -> Path:
    """Writes raw.csv, summary.csv, machine.json, config.yaml, index.json and BENCH.md.

    raw.csv has one row per (question, round, mode); summary.csv has one row per
    mode, with nearest-rank percentiles as in every other run.
    """
    rows, summary = result.rows, result.summary
    run_dir = create_run_dir(run_path(cfg.paths.results, "_bench"))
    write_table(run_dir / RAW_FILE, CsvTable.of(rows))
    write_table(run_dir / SUMMARY_FILE, CsvTable.of(summary))
    record = MachineRecord(
        before=recordable(result.before),
        after=recordable(result.after),
        discarded_pairs=result.discarded,
        load1_during=_load_during(rows),
    )
    (run_dir / MACHINE_FILE).write_text(machine_json(record), encoding="utf-8")
    write_run_config(run_dir, cfg)
    index = RunIndex(index_version=index_version, **question_fields(questions_path))
    write_index(run_dir, index, final_newline=False)
    (run_dir / BENCH_REPORT_FILE).write_text(
        bench_report(rows, summary, result.before, result.discarded, checked=result.checked),
        encoding="utf-8",
    )
    return run_dir


@dataclass(frozen=True)
class BenchRequest:
    """The `bench-rerank` options."""

    questions_path: Path
    rounds: int
    _: KW_ONLY
    force: bool


@dataclass(frozen=True)
class MachineProbes:
    """What the bench reads of the machine and how it waits.

    `state` gives the machine state (`machine.machine_state`), `load1` the 1-minute load
    average a row records, and `sleep` waits so many seconds.
    """

    state: Callable[[], MachineState]
    load1: Callable[[], float]
    sleep: Callable[[float], None]


def _load1() -> float:
    return os.getloadavg()[0]


def _this_machine() -> MachineProbes:
    return MachineProbes(machine_state, _load1, time.sleep)


def bench_rerank(
    cfg: Config,
    request: BenchRequest,
    questions: Sequence[Question],
    *,
    notify: Callable[[str], None],
    probes: MachineProbes | None = None,
    open_retriever: Callable[[Config], AbstractContextManager[BenchRetriever]] | None = None,
) -> Path:
    """The run directory of venue_split's two rerank modes timed on the questions it splits.

    `notify` is called once each time the bench starts waiting for a quiet machine,
    with the reasons it is busy. `probes` read this machine unless given, and
    `open_retriever` builds the retriever from `cfg`; `Retriever` unless given.

    Raises:
        UserError: No question names two or more venues.
    """
    cfg = cfg.with_questions(request.questions_path)
    machine = probes or _this_machine()
    before = machine.state()
    pacing = Pacing(
        quiet=None if request.force else QuietCheck(notify, machine.state),
        wait=lambda: machine.sleep(QUIET_WAIT_SECONDS),
        load=machine.load1,
    )
    with (open_retriever or Retriever)(cfg) as retriever:
        version = retriever.require_current_index()
        chosen = split_questions(retriever, questions)
        if not chosen:
            raise UserError(
                f"no question names {formatting.count(MULTI_VENUE)} or more venues",
                hint="check --questions",
            )
        # Model loading and the first MPS kernels are not reranking time.
        retriever.warmup()
        for mode in MODES:
            retriever.venue_split_timed(chosen[0].question, mode, Timings(), [])
        rows, discarded = run_bench(retriever, chosen, request.rounds, pacing)
    result = BenchResult(
        rows, summarise_modes(rows), before, machine.state(), discarded, checked=not request.force
    )
    return write_bench(cfg, result, version, request.questions_path)


class QuietCheck:
    """`Pacing.quiet` from a machine state, announcing each wait once through `notify`."""

    def __init__(self, notify: Callable[[str], None], state: Callable[[], MachineState]) -> None:
        """A check of `state()` that calls `notify` with the reasons when a wait starts."""
        self.notify = notify
        self.state = state
        self.announced = False

    def __call__(self) -> bool:
        """Whether the machine is quiet at this check; a wait's first busy check is announced."""
        reasons = busy_reasons(self.state())
        if reasons and not self.announced:
            self.notify("; ".join(reasons))
            self.announced = True
        if not reasons:
            self.announced = False
        return not reasons


def _load_during(rows: Sequence[BenchRow]) -> LoadDuring:
    loads = [r.load1 for r in rows if r.mode == MODES[0]]
    return LoadDuring(median=percentile(loads, P50), max=max(loads)) if loads else LoadDuring()


def bench_report(
    rows: Sequence[BenchRow],
    summary: Sequence[BenchSummaryRow],
    before: MachineState,
    discarded: int,
    *,
    checked: bool,
) -> str:
    """The bench report as Markdown: the conditions it ran under and one table row per mode.

    With `checked` False (`--force`) it says the quiet-machine checks were skipped.
    """
    questions = len({r.question_id for r in rows})
    rounds = max(r.round for r in rows)
    load_before = listing.prose(
        [formatting.ratio(x, formatting.LOAD_AVERAGE_DECIMALS) for x in before.load_average]
    )
    during = _load_during(rows)
    load_during = (
        f"median {formatting.ratio(during.median, formatting.LOAD_AVERAGE_DECIMALS)}, "
        f"maximum {formatting.ratio(during.max, formatting.LOAD_AVERAGE_DECIMALS)}"
        if during.median is not None and during.max is not None
        else "not measured"
    )
    conditions = (
        "Each pair was timed between two checks that found none of the competing processes "
        f"({listing.prose(COMPETITORS)}) running and a 1-minute load average within the "
        f"bound; {formatting.plural(discarded, 'pair')} that failed the second check, on "
        f"either count, {'was' if discarded == 1 else 'were'} discarded and repeated."
        if checked
        else "The quiet-machine checks were skipped (`--force`): each pair was timed "
        "whatever else was running, and no pair was discarded."
    )
    header = [
        "mode",
        "n",
        "steps sum p50 ms",
        "steps sum p95 ms",
        "steps sum mean ms",
        "rerank p50 ms",
        "rerank p95 ms",
        "calls per query",
        "pairs per call",
        "identical retrievals",
    ]
    cells = [
        [
            s.mode,
            formatting.count(s.n),
            formatting.ms(s.total_p50_ms),
            formatting.ms(s.total_p95_ms),
            formatting.ms(s.total_mean_ms),
            formatting.ms(s.rerank_p50_ms),
            formatting.ms(s.rerank_p95_ms),
            formatting.ratio(s.calls_per_query, formatting.BENCH_SUMMARY_DECIMALS),
            formatting.ratio(s.pairs_per_call_mean, formatting.COUNT_MEAN_DECIMALS),
            formatting.count(s.identical_retrievals),
        ]
        for s in summary
    ]
    lines = [
        "# venue_split: one cross-encoder call per venue against one call for all venues",
        "",
        f"{formatting.plural(questions, 'question')} venue_split splits, "
        f"{formatting.plural(rounds, 'round')}, both modes per question "
        "back to back, order alternating; models loaded and warmed up before timing. "
        f"Load average before timing: {load_before} on "
        f"{formatting.cpus(before.cpu_count)}; "
        f"1-minute load during timing: {load_during}. {conditions} The steps sum is the sum "
        "of the measured retrieval steps.",
        "",
        markdown_table(header, cells, right=range(1, len(header))),
    ]
    return "\n".join(lines) + "\n"
