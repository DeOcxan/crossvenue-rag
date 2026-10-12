"""Re-deriving a committed retrieval run's tables from its per_query.csv, for `make reproduce`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..corpus.artifacts import load_chunk_venues
from ..dataset.journal import load_judged_sets
from ..dataset.questions import load_questions
from ..evaluation.comparisons import paired_comparisons, read_paired
from ..evaluation.evaluator import QueryRun, score, summarise, summarise_by
from ..evaluation.results import (
    PER_QUERY_FILE,
    breakdown_file,
    read_breakdown,
    read_per_query,
    read_summary,
    summary_cells,
)
from ..models import Question
from ..run_files import (
    PAIRED_FILE,
    SUMMARY_FILE,
    CsvRow,
    file_sha256,
    read_index,
    read_run_config,
)


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def rederive_retrieval_run(run_dir: Path, root: Path) -> tuple[int, list[str]]:
    """(cells compared, problems): the run's metric tables recomputed from its per_query.csv.

    The metric columns of `summary.csv`, `by_*.csv` and `paired.csv` are recomputed from
    the retrievals in per_query.csv with the evaluation code. Latency is not: per_query.csv
    rounds each timing. Skipped (0, []) when the run records no question file or that
    file does not match its recorded sha256.
    """
    questions_path = _unchanged_questions(run_dir, root)
    if questions_path is None:
        return 0, []
    cfg = read_run_config(run_dir)
    questions = load_questions(questions_path)
    runs = _rescored(run_dir, cfg, questions, load_judged_sets(questions_path))
    checked, problems = 0, []
    for table in _expected_tables(run_dir, cfg, runs, questions):
        cells, found = _compare(run_dir, table)
        checked += cells
        problems += found
    return checked, problems


def _unchanged_questions(run_dir: Path, root: Path) -> Path | None:
    """The question file the run read, if the run records it and it matches its recorded sha256."""
    recorded = read_index(run_dir)
    if recorded is None or recorded.questions is None:
        return None
    questions_path = root / recorded.questions
    if not questions_path.exists():
        return None
    if file_sha256(questions_path) != recorded.questions_sha256:
        return None
    return questions_path


def _rescored(
    run_dir: Path, cfg: Config, questions: Sequence[Question], judged: dict[str, set[str]] | None
) -> list[QueryRun]:
    """Every saved retrieval of per_query.csv, scored again."""
    by_id = {q.id: q for q in questions}
    venue_of = load_chunk_venues(cfg)

    return [
        score(
            by_id[row.question_id],
            row.retrieval(venue_of),
            cfg.eval.k_values,
            judged,
        )
        for row in read_per_query(run_dir).rows
    ]


@dataclass(frozen=True)
class _Expected:
    """A saved table of a run and its rows re-derived, each a mapping of column to value.

    A re-derived row holds every column its table's writer gives:
    `evaluation.results.summary_cells` for a summary or breakdown, and
    `PairedComparison.cells` for paired.csv. `_compare` skips a column the saved table
    lacks, such as the off-venue metrics of a run saved before methodology section 9b.
    """

    filename: str
    key: tuple[str, ...]
    rederived: list[dict[str, object]]
    saved_columns: tuple[str, ...]
    saved_rows: tuple[CsvRow, ...]


def _expected_tables(
    run_dir: Path, cfg: Config, runs: Sequence[QueryRun], questions: Sequence[Question]
) -> list[_Expected]:
    """Each saved table the run has, with its rows re-derived."""
    k_values = cfg.eval.k_values
    summary = read_summary(run_dir)
    tables = [
        _Expected(
            SUMMARY_FILE,
            ("method",),
            [{"method": s.method, **summary_cells(s)} for s in summarise(runs, k_values)],
            summary.columns,
            summary.rows,
        )
    ]
    for dimension in cfg.eval.breakdowns:
        if (run_dir / breakdown_file(dimension)).exists():
            slices = summarise_by(runs, questions, k_values, dimension)
            rows: list[dict[str, object]] = [
                {dimension: s.slice, "method": s.method, **summary_cells(s)} for s in slices
            ]
            saved = read_breakdown(run_dir, dimension)
            key = (dimension, "method")
            tables.append(
                _Expected(breakdown_file(dimension), key, rows, saved.columns, saved.rows)
            )
    if (run_dir / PAIRED_FILE).exists():
        paired = [row.cells() for row in paired_comparisons(cfg, runs, questions)]
        pairs = read_paired(run_dir / PAIRED_FILE)
        cell = ("a", "b", "metric", "scope")
        tables.append(_Expected(PAIRED_FILE, cell, paired, pairs.columns, pairs.rows))
    return tables


def _compare(run_dir: Path, table: _Expected) -> tuple[int, list[str]]:
    """(cells compared, problems): each re-derived row against the saved one.

    A saved row that nothing re-derives is a problem too, so the saved table and the
    re-derived one hold the same rows.
    """
    checked, problems = 0, []
    saved = {tuple(r.text(k) for k in table.key): r for r in table.saved_rows}
    rederived = {tuple(str(row[k]) for k in table.key) for row in table.rederived}
    problems += [
        f"{run_dir}/{table.filename}: row {ident} is saved but not re-derived"
        for ident in saved
        if ident not in rederived
    ]
    for row in table.rederived:
        ident = tuple(str(row[k]) for k in table.key)
        have = saved.get(ident)
        if have is None:
            problems.append(f"{run_dir}/{table.filename}: no row {ident}")
            continue
        for column, value in row.items():
            if column in table.key or column not in table.saved_columns:
                continue
            checked += 1
            if _cell(value) != have.text(column):
                problems.append(
                    f"{run_dir}/{table.filename} {ident} {column}: {have.text(column)!r} saved, "
                    f"{_cell(value)!r} re-derived from {PER_QUERY_FILE}"
                )
    return checked, problems
