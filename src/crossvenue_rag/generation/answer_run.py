"""Writes `results/<timestamp>_answers/` (or `_judged/`).

`answers.jsonl` (every record in full, each linking its cache entries),
`per_answer.csv`, `summary.csv` (metrics, tokens, cost and latency per
configuration), `paired.csv`, `report.md`, the resolved config and
`index.json` (the `index_version` the chunks came from and the question file).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..config import Config
from ..run_files import (
    PAIRED_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    CsvTable,
    RunIndex,
    create_run_dir,
    question_fields,
    run_path,
    write_index,
    write_run_config,
    write_table,
)
from .answer_pairs import paired_rows
from .answer_report import report
from .answer_rows import PER_ANSWER_FILE, per_answer_row, summary_rows
from .records import ANSWERS_FILE, AnswerRecord, write_answers


def write_answer_run(
    cfg: Config,
    records: Sequence[AnswerRecord],
    results_dir: Path,
    note: str,
    suffix: str = "_answers",
    *,
    index_version: str,
) -> Path:
    """Writes every file of an answer run into `results_dir/<timestamp><suffix>/`; returns it.

    The directory is created by this call and must not exist. The report is
    built first, so records it refuses (`answer_report.report`) leave no
    directory behind.

    Raises:
        UserError: `records` carry more than one judge.
    """
    configs = [m.name for m in cfg.generation.matrix]
    summary = summary_rows(records, configs)
    pairs = paired_rows(records, cfg)
    text = report(cfg, records, summary, pairs, note, index_version)
    run_dir = create_run_dir(run_path(results_dir, suffix))
    write_run_config(run_dir, cfg)
    index = RunIndex(index_version=index_version, **question_fields(Path(cfg.eval.questions)))
    write_index(run_dir, index, final_newline=True)
    write_answers(run_dir / ANSWERS_FILE, records)
    write_table(run_dir / PER_ANSWER_FILE, CsvTable.of([per_answer_row(r) for r in records]))
    write_table(run_dir / SUMMARY_FILE, CsvTable.of(summary))
    write_table(run_dir / PAIRED_FILE, CsvTable.of(pairs))
    (run_dir / REPORT_FILE).write_text(text, encoding="utf-8")
    return run_dir
