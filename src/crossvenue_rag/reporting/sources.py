"""The loaders of the saved artifacts the documents' numbers come from.

Each loader reads saved files under the repository root and makes no model call, no
retrieval and no database query.
"""

from __future__ import annotations

import math
from functools import cached_property
from pathlib import Path

from .. import formatting, listing
from ..config import SCHEMA_DEFAULTS, Config
from ..config_loading import DEFAULT_CONFIG, load_config
from ..corpus.artifacts import chunks_path, lowered_texts, read_chunk_venues, read_chunks
from ..corpus.manifest import ManifestRow, read_manifest
from ..dataset.journal import Journal, load_rounds
from ..dataset.questions import load_questions
from ..errors import UserError
from ..evaluation.bench import BenchRow, BenchSummaryRow, read_bench_raw, read_bench_summary
from ..evaluation.comparisons import PairedRow, read_paired
from ..evaluation.machine import MachineRecord, read_machine
from ..evaluation.power import PowerStudy, read_power_study
from ..evaluation.report import LATENCY_HEADING
from ..evaluation.results import (
    BreakdownRow,
    PerQueryRow,
    SummaryRow,
    read_breakdown,
    read_per_query,
    read_summary,
)
from ..generation.answer_pairs import ConfigComparison, read_config_comparisons
from ..generation.answer_rows import AnswerSummaryRow, read_answer_summary
from ..generation.calibration_file import Calibration, read_calibration
from ..generation.judge_sample import RATINGS_PATH, human_ratings, judgements_path, load_judgements
from ..generation.prompt_compare import PromptComparison, read_comparison
from ..generation.records import (
    ANSWERS_FILE,
    AnswerRecord,
    JudgementRow,
    read_answers,
    read_judgements,
)
from ..generation.schemas import Judgement
from ..models import RETRIEVAL_STEPS, Chunk, Question
from ..retrieval.venue_detect import VenueDetector
from ..run_files import PAIRED_FILE, REPORT_FILE, SUMMARY_FILE, read_run_config
from .patterns import TABLE_HEADER_LINES, table_cells, table_lines
from .run_ids import (
    ANSWERS_V1,
    ANSWERS_V2,
    BENCH_RUN,
    GOLD,
    GOLD_BEFORE_AUGMENTATION,
    JUDGED_V1,
    JUDGED_V2,
    NDCG_PRIME_RUN,
    POWER_RUN,
    PROMPTS_RUN,
    PVALUE_RUN,
    RETRIEVAL_RUN,
)

CHUNKS = chunks_path(SCHEMA_DEFAULTS)


def _plain_number(cell: str) -> str:
    """A report cell with thousands separators removed and U+2212 as `-`."""
    return cell.replace(",", "").replace(formatting.MINUS, "-")


def _is_number(cell: str) -> bool:
    """Whether `cell` is a finite number as text: `2505.3`, `0.0`, `-1`."""
    try:
        return math.isfinite(float(cell))
    except ValueError:
        return False


def _numeric_steps(
    path: Path, table: dict[str, dict[str, str]], hint: str
) -> dict[str, dict[str, str]]:
    """`table`, read from `path`, once every step cell of it is checked to be a number.

    Raises:
        UserError: A cell is empty or not a finite number; the message names the method and the
            step, and `hint` the fix.
    """
    for method, steps in table.items():
        for step, cell in steps.items():
            if not _is_number(cell):
                raise UserError(
                    f"{path}: the {step} p50 of {method} is no number: {cell!r}", hint=hint
                )
    return table


def _report_step_table(path: Path) -> dict[str, dict[str, str]]:
    """The per-step p50 table of a retrieval run's report.md, each cell a plain decimal.

    The only reader of latency from Markdown, for runs whose summary.csv
    predates the latency columns. per_query.csv rounds each timing to 0.01 ms,
    so a p50 recomputed from it can differ in the last digit. Two formats of the
    table are read:

    - the committed runs': one decimal for every step, no thousands separator
      (`2505.3`), the last column headed `total p50`;
    - `formatting.ms`, as `eval` writes the table and the tests write it: an
      integer from 10 ms with a thousands separator (`2,505`), one decimal below,
      U+2212 for a minus sign, the last column headed `steps sum p50`.

    Separators are removed and U+2212 becomes `-`, so a committed cell is
    returned exactly as written. Each method's row is keyed by the steps
    (`models.RETRIEVAL_STEPS`); the last column is not returned.

    Raises:
        UserError: report.md cannot be read or is not UTF-8, or has no latency section or no table
            in it. Or the table's header lacks a step, or a row has more or fewer cells than the
            header.
    """
    hint = f"run `crossvenue eval` again, which writes the step columns into {SUMMARY_FILE}"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise UserError(f"{path} cannot be read: {exc}", hint=hint) from exc
    _, found, section = text.partition(LATENCY_HEADING)
    rows = [line for _, line in table_lines(section, 0)]
    if not found or not rows:
        raise UserError(f"{path} has no table under {LATENCY_HEADING!r}", hint=hint)
    header = table_cells(rows[0])
    missing = [step for step in RETRIEVAL_STEPS if step not in header]
    if missing:
        raise UserError(
            f"{path}: the latency table lacks {formatting.plural(len(missing), 'step column')}: "
            f"{listing.prose(missing)}",
            hint=hint,
        )
    table = {}
    for row in rows[TABLE_HEADER_LINES:]:
        cells = [_plain_number(c) for c in table_cells(row)]
        if len(cells) != len(header):
            raise UserError(
                f"{path}: a row of the latency table has {formatting.plural(len(cells), 'cell')}, "
                f"its header {formatting.count(len(header))}",
                hint=hint,
            )
        by_column = dict(zip(header, cells, strict=True))
        table[cells[0]] = {step: by_column[step] for step in RETRIEVAL_STEPS}
    return _numeric_steps(path, table, hint)


def _check_record(path: Path, record: AnswerRecord) -> None:
    """Refuses a record whose judgment grades more claims than its answer has.

    The reports index a judgment's claims by the answer's. A claim citing a chunk number
    outside its context is refused earlier, by `read_answers`.

    Raises:
        UserError: the judgment grades more claims than the answer has; the message names the
            record.
    """
    generation, judge = record.generation, record.judge
    if generation is None or judge is None:
        return
    graded, claims = len(judge.judgement.claims), len(generation.answer.claims)
    if graded > claims:
        raise UserError(
            f"{path}: {record.answer_id}'s judgment grades {formatting.plural(graded, 'claim')} "
            f"of an answer with {formatting.count(claims)}",
            hint="judge the run again with `crossvenue judge-run`",
        )


class RetrievalRun:
    """results/<ts>/ from `crossvenue eval`: its config and metric tables."""

    def __init__(self, directory: Path) -> None:
        """The run saved in `directory`."""
        self.dir = directory

    @cached_property
    def config(self) -> Config:
        """config.yaml: the config the run was made with (`run_files.read_run_config`)."""
        return read_run_config(self.dir)

    @cached_property
    def by_category(self) -> dict[tuple[str, str], BreakdownRow]:
        """by_category.csv by (category, method)."""
        rows = read_breakdown(self.dir, "category").rows
        return {(r.text("category"), r.method): r for r in rows}

    @cached_property
    def summary(self) -> dict[str, SummaryRow]:
        """summary.csv by method."""
        return {r.method: r for r in read_summary(self.dir).rows}

    @cached_property
    def paired(self) -> dict[tuple[str, str, str, str], PairedRow]:
        """paired.csv by (a, b, metric, scope)."""
        return {(r.a, r.b, r.metric, r.scope): r for r in read_paired(self.dir / PAIRED_FILE).rows}

    @cached_property
    def per_query(self) -> tuple[PerQueryRow, ...]:
        """per_query.csv: one row per question and method."""
        return read_per_query(self.dir).rows

    @cached_property
    def methods(self) -> list[str]:
        """The methods of summary.csv, in its order."""
        return list(self.summary)

    @cached_property
    def step_table(self) -> dict[str, dict[str, str]]:
        """Per method, the p50 of each retrieval step as the run writes it, a number as text.

        Each method's row is keyed by the steps of `models.RETRIEVAL_STEPS` (`encode`,
        `search`, `fuse`, `rerank`) and by nothing else. Two formats are read:

        - `summary.csv` with the columns `<step>_p50_ms`, written by `eval` at
          full precision (`evaluation.evaluator.step_latency_columns`);
        - for runs whose summary.csv has none of those columns, the report.md table
          under "## Latency (p50, ms per step)" (`_report_step_table`).

        Raises:
            UserError: summary.csv has some of the step columns but not all.
        """
        summary = read_summary(self.dir)
        columns = {step: f"{step}_p50_ms" for step in RETRIEVAL_STEPS}
        missing = [c for c in columns.values() if c not in summary.columns]
        if len(missing) == len(columns):
            return _report_step_table(self.dir / REPORT_FILE)
        hint = "run `crossvenue eval` again, which writes every step column"
        if missing:
            raise UserError(
                f"{self.dir / SUMMARY_FILE} lacks "
                f"{formatting.plural(len(missing), 'step column')}: {listing.prose(missing)}",
                hint=hint,
            )
        table = {
            method: {step: row.text(column) for step, column in columns.items()}
            for method, row in self.summary.items()
        }
        return _numeric_steps(self.dir / SUMMARY_FILE, table, hint)


class BenchRun:
    """results/<ts>_bench/ from `crossvenue bench-rerank`."""

    def __init__(self, directory: Path) -> None:
        """The run saved in `directory`."""
        self.dir = directory

    @cached_property
    def summary(self) -> dict[str, BenchSummaryRow]:
        """summary.csv by mode (`per_venue`, `one_call`)."""
        return {r.mode: r for r in read_bench_summary(self.dir).rows}

    @cached_property
    def raw(self) -> tuple[BenchRow, ...]:
        """raw.csv: one row per question and mode."""
        return read_bench_raw(self.dir).rows

    @cached_property
    def machine(self) -> MachineRecord:
        """machine.json: the machine and its load while timing."""
        return read_machine(self.dir)


class PromptsRun:
    """results/<ts>_prompts/ from `crossvenue compare-prompts`."""

    def __init__(self, directory: Path) -> None:
        """The run saved in `directory`."""
        self.dir = directory

    @cached_property
    def comparison(self) -> dict[tuple[str, str], PromptComparison]:
        """comparison.csv by (config, metric)."""
        return {(r.config, r.metric): r for r in read_comparison(self.dir).rows}


class AnswerRun:
    """results/<ts>_answers/ or <ts>_judged/: generated and judged answers."""

    def __init__(self, directory: Path) -> None:
        """The run saved in `directory`."""
        self.dir = directory
        self._judgements: dict[str, list[JudgementRow]] = {}

    @cached_property
    def records(self) -> list[AnswerRecord]:
        """answers.jsonl: one record per configuration and question."""
        path = self.dir / ANSWERS_FILE
        records = read_answers(path)
        for record in records:
            _check_record(path, record)
        return records

    @cached_property
    def summary(self) -> dict[str, AnswerSummaryRow]:
        """summary.csv by configuration."""
        return {r.config: r for r in read_answer_summary(self.dir).rows}

    @cached_property
    def paired(self) -> dict[tuple[str, str, str, str], ConfigComparison]:
        """paired.csv by (a, b, metric, scope)."""
        rows = read_config_comparisons(self.dir).rows
        return {(r.a, r.b, r.metric, r.scope): r for r in rows}

    @cached_property
    def config(self) -> Config:
        """config.yaml: the config the run was made with (`run_files.read_run_config`)."""
        return read_run_config(self.dir)

    def judgements(self, judge: str, replicate: int) -> list[JudgementRow]:
        """The rows of `judge`'s judgment file in the run directory (`judgements_path`).

        Raises:
            UserError: `judge` is not a plain name.
        """
        path = judgements_path(self.dir, judge, replicate)
        if path.name not in self._judgements:
            self._judgements[path.name] = read_judgements(path)
        return self._judgements[path.name]


class Sources:
    """Every saved artifact the documents' numbers come from, under `root`."""

    def __init__(self, root: str | Path) -> None:
        """The artifacts under the repository root `root`."""
        self.root = Path(root)
        self._runs: dict[str, RetrievalRun] = {}

    def path(self, relative: str | Path) -> Path:
        """`relative` under the root."""
        return self.root / relative

    def run(self, run_id: str) -> RetrievalRun:
        """The retrieval or pool run `results/<run_id>/`, one loader per run id."""
        if run_id not in self._runs:
            self._runs[run_id] = RetrievalRun(self.path(SCHEMA_DEFAULTS.paths.results / run_id))
        return self._runs[run_id]

    @cached_property
    def retrieval(self) -> RetrievalRun:
        """The retrieval run the documents report (`run_ids.RETRIEVAL_RUN`)."""
        return self.run(RETRIEVAL_RUN)

    @cached_property
    def ndcg_prime(self) -> RetrievalRun:
        """The run that adds nDCG'@10 (`run_ids.NDCG_PRIME_RUN`)."""
        return self.run(NDCG_PRIME_RUN)

    @cached_property
    def pvalues(self) -> RetrievalRun:
        """The run whose paired.csv the documents quote (`run_ids.PVALUE_RUN`)."""
        return self.run(PVALUE_RUN)

    @cached_property
    def bench(self) -> BenchRun:
        """The reranker bench (`run_ids.BENCH_RUN`)."""
        return BenchRun(self.path(SCHEMA_DEFAULTS.paths.results / BENCH_RUN))

    @cached_property
    def judged_v1(self) -> AnswerRun:
        """The judged answers of prompt v1 (`run_ids.JUDGED_V1`)."""
        return AnswerRun(self.path(SCHEMA_DEFAULTS.paths.results / JUDGED_V1))

    @cached_property
    def judged_v2(self) -> AnswerRun:
        """The judged answers of prompt v2 (`run_ids.JUDGED_V2`)."""
        return AnswerRun(self.path(SCHEMA_DEFAULTS.paths.results / JUDGED_V2))

    @cached_property
    def prompts(self) -> PromptsRun:
        """The comparison of prompt v2 with v1 (`run_ids.PROMPTS_RUN`)."""
        return PromptsRun(self.path(SCHEMA_DEFAULTS.paths.results / PROMPTS_RUN))

    @cached_property
    def calibration(self) -> Calibration:
        """calibration.json of the prompt v1 answer run."""
        return read_calibration(self.v1.dir)

    @cached_property
    def v1(self) -> AnswerRun:
        """The answers of prompt v1 (`run_ids.ANSWERS_V1`)."""
        return AnswerRun(self.path(SCHEMA_DEFAULTS.paths.results / ANSWERS_V1))

    @cached_property
    def v2(self) -> AnswerRun:
        """The answers of prompt v2 (`run_ids.ANSWERS_V2`)."""
        return AnswerRun(self.path(SCHEMA_DEFAULTS.paths.results / ANSWERS_V2))

    @cached_property
    def power(self) -> PowerStudy:
        """power.json of the power run (`run_ids.POWER_RUN`)."""
        return read_power_study(self.path(SCHEMA_DEFAULTS.paths.results / POWER_RUN))

    @cached_property
    def human(self) -> dict[str, Judgement]:
        """The human's rating of each calibration answer, by answer id."""
        return human_ratings(self.root / RATINGS_PATH)

    def judge_grades(self, judge: str) -> dict[str, Judgement]:
        """`judge`'s judgment of each calibration answer it graded, by answer id.

        Raises:
            UserError: `judge` is not a plain name.
        """
        return load_judgements(judgements_path(self.v1.dir, judge))

    @cached_property
    def questions(self) -> dict[str, Question]:
        """The gold set (`run_ids.GOLD`) by question id."""
        return {q.id: q for q in load_questions(self.path(GOLD))}

    @cached_property
    def questions_before_augmentation(self) -> dict[str, Question]:
        """The gold set before augmentation (`run_ids.GOLD_BEFORE_AUGMENTATION`) by id."""
        return {q.id: q for q in load_questions(self.path(GOLD_BEFORE_AUGMENTATION))}

    @cached_property
    def rounds(self) -> Journal:
        """The labeling journal of the gold set (`dataset.journal.load_rounds`)."""
        return load_rounds(self.path(GOLD))

    @cached_property
    def rounds_before_augmentation(self) -> Journal:
        """The labeling journal of the gold set before augmentation (`load_rounds`)."""
        return load_rounds(self.path(GOLD_BEFORE_AUGMENTATION))

    @cached_property
    def chunks(self) -> list[Chunk]:
        """data/processed/chunks.jsonl (`corpus.artifacts.read_chunks`)."""
        return read_chunks(self.path(CHUNKS))

    @cached_property
    def lowered_texts(self) -> list[tuple[str, str]]:
        """(venue, lower-cased text) of each chunk of `chunks`, for counting terms."""
        return lowered_texts(self.chunks)

    @cached_property
    def default_config(self) -> Config:
        """configs/default.yaml, the config every command reads unless told otherwise."""
        return load_config(self.root / DEFAULT_CONFIG)

    @cached_property
    def doc_path_of(self) -> dict[str, str]:
        """The document of each chunk by chunk id, from `chunks`."""
        return {c.chunk_id: c.doc_path for c in self.chunks}

    @cached_property
    def venue_of(self) -> dict[str, str]:
        """The venue of each chunk by chunk id, from chunks.jsonl (`read_chunk_venues`)."""
        return read_chunk_venues(self.path(CHUNKS))

    @cached_property
    def manifest(self) -> list[ManifestRow]:
        """data/manifest.csv: one row per fetched document (`corpus.manifest.read_manifest`)."""
        return read_manifest(self.path(SCHEMA_DEFAULTS.paths.manifest))

    @cached_property
    def detector(self) -> VenueDetector:
        """The venue detector of data/sources.yaml."""
        return VenueDetector.from_file(self.path(SCHEMA_DEFAULTS.paths.sources))
