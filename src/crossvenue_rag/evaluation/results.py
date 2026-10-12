"""The files an evaluation run writes to `results/<timestamp>/`.

Beside the per-query rows, the aggregate and per-slice tables and the report, a run keeps
a copy of its config and the `index_version` it was computed against. A number without its
config is not a result, and a chunk id without its index is not a pointer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Config
from ..models import Question, Timings
from ..run_files import (
    PAIRED_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    CsvRow,
    CsvTable,
    OptionalFloat,
    OptionalInt,
    RunIndex,
    create_run_dir,
    question_fields,
    read_table,
    run_path,
    write_index,
    write_run_config,
    write_table,
)
from .comparisons import off_venue_comparisons, paired_comparisons
from .evaluator import (
    MethodSummary,
    QueryRun,
    Retrieval,
    SliceSummary,
    judged_metric_names_for,
    off_venue_metric_names_for,
    step_latency_columns,
    summarise_by,
    venue_metric_names_for,
)
from .report import RunTables, report_markdown

log = logging.getLogger(__name__)

PER_QUERY_FILE = "per_query.csv"
OFF_VENUE_FILE = "off_venue.csv"


def breakdown_file(dimension: str) -> str:
    """The file name of a retrieval run's table by `dimension`: by_<dimension>.csv."""
    return f"by_{dimension}.csv"


# The decimals of a step timing in per_query.csv.
TIMING_DECIMALS = 2


class PerQueryRow(CsvRow):
    """One question and method: a row of per_query.csv.

    The metric columns at each cutoff (`venue_coverage@10`, `unjudged@10`,
    `ndcg_prime@10`, ...) are read through `number`. `read_per_query` reads
    every format of the file:

    - 20260924T200525Z and 20260924T234623Z: without `retrieved_venues` and
      the metric columns;
    - the runs of 2026-09-25 from 002151Z on: with the metric columns each
      run computed;
    - the pool run 20260925T003721Z_pool: `question_id`, `method`, `k` and
      `retrieved_ids` only.
    """

    question_id: str
    method: str
    labelled: bool | None = None
    k: OptionalInt = None
    retrieved_ids: str
    retrieved_venues: str | None = None
    encode_ms: OptionalFloat = None
    search_ms: OptionalFloat = None
    fuse_ms: OptionalFloat = None
    rerank_ms: OptionalFloat = None
    total_ms: OptionalFloat = None

    @property
    def ids(self) -> list[str]:
        """The retrieved chunk ids, in rank order."""
        return self.retrieved_ids.split()

    def retrieval(self, venue_of: Mapping[str, str]) -> Retrieval:
        """The row as the retrieval it saved, for re-scoring.

        A run that saved no venues (`retrieved_venues`) has them resolved through
        `venue_of`, chunk id to venue, from the index its ids point into.

        Raises:
            ValueError: The row saved no venues and `venue_of` lacks one of its chunk ids,
                which the message names.
        """
        if self.retrieved_venues is not None:
            return Retrieval(self.method, self.ids, self.retrieved_venues.split())
        if missing := [c for c in self.ids if c not in venue_of]:
            raise ValueError(f"{self.method} on {self.question_id}: no venue for {missing[0]}")
        return Retrieval(self.method, self.ids, [venue_of[c] for c in self.ids])


class SummaryRow(CsvRow):
    """One method over the run: a row of summary.csv.

    The metric columns (`mrr`, `recall@10`, ...) and the per-step latency
    columns (`encode_p50_ms`, ...) are read through `number`. `read_summary`
    reads every format of the file:

    - 20260924T200525Z: counts and latency only;
    - 20260924T234623Z: with the ranking metrics;
    - the runs of 2026-09-25: with `multi_venue_questions` and the metric
      columns each run computed;
    - the later runs: with the per-step latency columns last.
    """

    method: str
    questions: int
    labelled_questions: int
    multi_venue_questions: OptionalInt = None
    p50_ms: float
    p95_ms: float


class BreakdownRow(CsvRow):
    """One method over one slice of a dimension: a row of by_<dimension>.csv.

    The slice is in the column named for the dimension (`category`), read
    through `text`; the metric columns are read through `number`.
    `read_breakdown` reads every format of the file, which are the formats
    of summary.csv without `p95_ms` and the per-step latency columns.
    """

    method: str
    questions: int
    labelled_questions: int
    multi_venue_questions: OptionalInt = None
    p50_ms: float


def read_per_query(run_dir: Path) -> CsvTable[PerQueryRow]:
    """per_query.csv of a retrieval or pool run, in any format `PerQueryRow` lists."""
    return read_table(run_dir / PER_QUERY_FILE, PerQueryRow)


def read_summary(run_dir: Path) -> CsvTable[SummaryRow]:
    """summary.csv of a retrieval run, in any format `SummaryRow` lists."""
    return read_table(run_dir / SUMMARY_FILE, SummaryRow)


def read_breakdown(run_dir: Path, dimension: str) -> CsvTable[BreakdownRow]:
    """by_<dimension>.csv of a retrieval run, in any format `BreakdownRow` lists."""
    return read_table(run_dir / breakdown_file(dimension), BreakdownRow)


def run_tables(
    cfg: Config,
    runs: Sequence[QueryRun],
    summaries: Sequence[MethodSummary],
    questions: Sequence[Question],
    venue_chunks: Mapping[str, int],
) -> RunTables:
    """Every table of an eval run: the breakdowns and the paired and off-venue comparisons.

    `venue_chunks` is the number of chunks of each venue in the corpus the runs
    retrieved from, which report.md states.
    """
    breakdowns: dict[str, list[SliceSummary]] = {
        dimension: summarise_by(runs, questions, cfg.eval.k_values, dimension)
        for dimension in cfg.eval.breakdowns
    }
    return RunTables(
        runs,
        summaries,
        breakdowns,
        paired_comparisons(cfg, runs, questions),
        off_venue_comparisons(cfg, runs, questions),
        venue_chunks,
    )


def write_eval_run(
    cfg: Config,
    tables: RunTables,
    questions: Sequence[Question],
    index_version: str,
) -> Path:
    """Writes an eval run's directory from its tables (`run_tables`) and returns its path.

    The directory holds per_query.csv, summary.csv, config.yaml, index.json,
    report.md and summary.json. The files by_<dimension>.csv, paired.csv and
    off_venue.csv are written only when their table is non-empty: a breakdown
    needs a slice of its dimension, and a comparison table a pair whose two
    methods both ran (`eval.paired_comparisons`, `OFF_VENUE_PAIRS`).
    """
    run_dir = create_run_dir(run_path(cfg.paths.results))
    runs, summaries = tables.runs, tables.summaries
    names = _ColumnNames.of(cfg.eval.k_values)
    _write_per_query(run_dir / PER_QUERY_FILE, runs, names)
    _write_summary(run_dir / SUMMARY_FILE, runs, summaries, names)
    for dimension, slices in tables.breakdowns.items():
        if slices:
            _write_breakdown(run_dir / breakdown_file(dimension), dimension, slices, names)
    write_run_config(run_dir, cfg)
    index = RunIndex(index_version=index_version, **question_fields(Path(cfg.eval.questions)))
    write_index(run_dir, index, final_newline=False)
    if tables.paired:
        write_table(run_dir / PAIRED_FILE, CsvTable.of(list(tables.paired)))
    if tables.off_venue:
        write_table(run_dir / OFF_VENUE_FILE, CsvTable.of(list(tables.off_venue)))
    text = report_markdown(cfg, run_dir.name, index_version, questions, tables)
    (run_dir / REPORT_FILE).write_text(text, encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2), encoding="utf-8"
    )
    log.info("wrote %s", run_dir)
    return run_dir


@dataclass(frozen=True, slots=True)
class _ColumnNames:
    """The metric column names every CSV of a run shares, for one set of cutoffs."""

    venue: list[str]
    judged: list[str]
    off_venue: list[str]

    @classmethod
    def of(cls, k_values: Sequence[int]) -> _ColumnNames:
        """The venue, judged and off-venue column names at `k_values`."""
        return cls(
            venue_metric_names_for(k_values),
            judged_metric_names_for(k_values),
            off_venue_metric_names_for(k_values),
        )


def _write_per_query(path: Path, runs: Sequence[QueryRun], names: _ColumnNames) -> None:
    # Saved precision: step timings at two decimals of a millisecond.
    header = [
        "question_id",
        "method",
        "labelled",
        "retrieved_ids",
        "retrieved_venues",
        *names.venue,
        *names.judged,
        *names.off_venue,
        *Timings().as_dict(),
    ]
    rows = (
        [
            run.question_id,
            run.method,
            run.labelled,
            " ".join(run.retrieved_ids),
            " ".join(run.retrieved_venues),
            *[run.venue_metrics.get(name, "") for name in names.venue],
            *[run.judged_metrics.get(name, "") for name in names.judged],
            *[run.off_venue_metrics.get(name, "") for name in names.off_venue],
            *[round(v, TIMING_DECIMALS) for v in run.timings.as_dict().values()],
        ]
        for run in runs
    )
    write_table(path, _table(PerQueryRow, header, rows))


def _table[RowT: CsvRow](
    row_type: type[RowT], header: Sequence[str], rows: Iterable[Sequence[object]]
) -> CsvTable[RowT]:
    """The rows as a table of `row_type` under `header`; an empty cell is an empty string."""
    return CsvTable(
        tuple(header),
        tuple(row_type.model_validate(dict(zip(header, row, strict=True))) for row in rows),
    )


def summary_cells(summary: MethodSummary | SliceSummary) -> dict[str, float]:
    """A summary's cells in summary.csv or a by_<dimension>.csv, by column.

    The columns from `questions` to the last off-venue metric: the question counts and every
    metric the summary has. A metric it lacks has no key, and its cell is empty.
    """
    return {
        "questions": summary.questions,
        "labelled_questions": summary.labelled_questions,
        **summary.metrics,
        "multi_venue_questions": summary.multi_venue_questions,
        **summary.venue_metrics,
        **summary.judged_metrics,
        **summary.off_venue_metrics,
    }


def _metric_columns(metric_names: Sequence[str], names: _ColumnNames) -> list[str]:
    """The columns `summary_cells` fills, in table order."""
    return [
        "questions",
        "labelled_questions",
        *metric_names,
        "multi_venue_questions",
        *names.venue,
        *names.judged,
        *names.off_venue,
    ]


def _cells(summary: MethodSummary | SliceSummary, columns: Sequence[str]) -> list[float | str]:
    cells = summary_cells(summary)
    return [cells.get(column, "") for column in columns]


def _write_summary(
    path: Path,
    runs: Sequence[QueryRun],
    summaries: Sequence[MethodSummary],
    names: _ColumnNames,
) -> None:
    """Writes summary.csv, one row per method.

    The per-step latency columns come last, unrounded, so a reader never has to
    parse them out of report.md.
    """
    columns = _metric_columns(sorted({name for s in summaries for name in s.metrics}), names)
    steps = step_latency_columns(runs)
    step_names = list(next(iter(steps.values()))) if steps else []
    header = ["method", *columns, "p50_ms", "p95_ms", *step_names]
    rows = (
        [
            summary.method,
            *_cells(summary, columns),
            summary.latency_p50_ms,
            summary.latency_p95_ms,
            *[_steps_of(steps, summary.method)[name] for name in step_names],
        ]
        for summary in summaries
    )
    write_table(path, _table(SummaryRow, header, rows))


def _steps_of(steps: Mapping[str, dict[str, float]], method: str) -> dict[str, float]:
    if method not in steps:
        raise ValueError(f"a summary names {method}, which no run of the tables has")
    return steps[method]


def _write_breakdown(
    path: Path, dimension: str, slices: Sequence[SliceSummary], names: _ColumnNames
) -> None:
    columns = _metric_columns(sorted({name for s in slices for name in s.metrics}), names)
    header = [dimension, "method", *columns, "p50_ms"]
    rows = (
        [slice_.slice, slice_.method, *_cells(slice_, columns), slice_.latency_p50_ms]
        for slice_ in slices
    )
    write_table(path, _table(BreakdownRow, header, rows))
