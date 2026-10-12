"""The runs the answer commands start: answer runs, judge runs, prompt comparisons.

`answer_eval` generates (and judges) every configuration of `generation.matrix`
over the question file and writes `results/<ts>_answers/`, and
`estimate_answer_eval` prices the same run without a model call. `judge_run`
judges a saved answer run again (`results/<ts>_judged/`), and `compare_prompt_runs`
compares the judged runs of two prompts (`results/<ts>_prompts/`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import formatting
from ..config import Config
from ..corpus.artifacts import chunks_by_id, load_chunk_venues, load_chunks
from ..dataset.questions import index_check, load_questions
from ..errors import UserError
from ..index_version import require_current_index
from ..models import Question
from ..retrieval.retriever import Retriever, checked_retriever
from ..run_files import RunSource, write_run_source
from .answer_run import write_answer_run as write_answer_results
from .answering import AnswerInputs, run_answers
from .calibration_sample import read_sample
from .estimate import EstimateRow, estimate_answers
from .judge_sample import judge_answers, judgements_path
from .judging import judge_records
from .llm import LLM, LLMFactory
from .prompt_compare import compare_prompts, side_effects
from .prompt_run import write_prompt_comparison
from .records import ANSWERS_FILE, read_answers, write_judgements
from .runs import require_run_index
from .schemas import FIRST_SCHEMA, PARTIAL_SCHEMA

Progress = Callable[[str], None]


def with_budget(cfg: Config, budget_usd: float | None) -> Config:
    """A copy of `cfg` whose `generation.budget_usd` is `budget_usd`, so the run records it.

    None keeps the configured budget.
    """
    if budget_usd is None:
        return cfg
    gen = cfg.generation.model_copy(update={"budget_usd": budget_usd})
    return cfg.model_copy(update={"generation": gen})


@dataclass(frozen=True)
class AnswerEvalJob:
    """The `answer-eval` options: which questions, which judge, how much to spend."""

    questions_path: Path
    question_ids: Sequence[str]
    judge: str | None
    budget: float | None
    note: str


def select_questions(
    questions: Sequence[Question], ids: Sequence[str], path: Path
) -> list[Question]:
    """The questions `ids` names, in file order; all of them when `ids` is empty.

    Raises:
        UserError: `ids` names a question `questions` lacks; the message names it.
    """
    if not ids:
        return list(questions)
    wanted = set(ids)
    missing = wanted - {q.id for q in questions}
    if missing:
        raise UserError(
            f"--question-id not in {path}: {', '.join(repr(i) for i in sorted(missing))}",
            hint="pass --question-id with ids of the question file",
        )
    return [q for q in questions if q.id in wanted]


def _prepare(cfg: Config, job: AnswerEvalJob) -> tuple[Config, list[Question]]:
    """The run's config and the questions to answer.

    Refuses an empty matrix or an unknown judge. The run's config is a copy of
    `cfg` that records the question file it read and the budget it ran under.
    """
    cfg = with_budget(cfg.with_questions(Path(job.questions_path)), job.budget)
    questions = select_questions(
        load_questions(job.questions_path), job.question_ids, job.questions_path
    )
    if not cfg.generation.matrix:
        raise UserError(
            "generation.matrix is empty",
            hint="list at least one configuration under generation.matrix in the --config "
            "file, or pass one with --set "
            "'generation.matrix=[{name: gemini@venue_split, generator: gemini, "
            "retriever: venue_split}]'",
        )
    if job.judge is not None:
        cfg.generation.judge_model(job.judge)  # an unknown judge is refused before any call
    return cfg, questions


def estimate_answer_eval(
    cfg: Config,
    job: AnswerEvalJob,
    *,
    make_llm: LLMFactory | None = None,
    open_retriever: Callable[[Config], Retriever] | None = None,
) -> list[EstimateRow]:
    """What `answer_eval` would cost, from the cache; makes no model call.

    `make_llm` builds the model client (`LLM` unless given) and `open_retriever`
    the retriever (`checked_retriever`).

    Raises:
        UserError: `generation.matrix` is empty, `job.question_ids` names a question the
            question file lacks, or `job.judge` is not a judge of `generation.judges`.
    """
    cfg, questions = _prepare(cfg, job)
    llm = (make_llm or LLM)(cfg.generation)
    with checked_retriever(cfg, index_check(job.questions_path), open_retriever=open_retriever) as (
        retriever,
        _version,
    ):
        return estimate_answers(cfg, retriever, questions, llm, job.judge)


def answer_eval(
    cfg: Config,
    job: AnswerEvalJob,
    *,
    progress: Progress,
    make_llm: LLMFactory | None = None,
    open_retriever: Callable[[Config], Retriever] | None = None,
) -> Path:
    """Generates every configuration on the questions; returns the run directory.

    report.md opens with its title and the `index_version` line, followed by the
    note, what the run spent of its budget, and why it stopped early if it did.
    `make_llm` builds the model client (`LLM` unless given) and `open_retriever`
    the retriever (`checked_retriever`).

    Raises:
        UserError: `generation.matrix` is empty, `job.question_ids` names a question the
            question file lacks, or `job.judge` is not a judge of `generation.judges`.
    """
    cfg, questions = _prepare(cfg, job)
    llm = (make_llm or LLM)(cfg.generation)
    with checked_retriever(cfg, index_check(job.questions_path), open_retriever=open_retriever) as (
        retriever,
        version,
    ):
        inputs = AnswerInputs(questions, retriever.by_id, llm, job.judge)
        batch = run_answers(cfg, retriever, inputs, progress=progress)
    lines = [job.note] if job.note else []
    lines.append(
        f"Spent this run: {formatting.cost(llm.spent_usd)} of a "
        f"{formatting.cost(llm.budget_usd)} budget."
    )
    if batch.stopped:
        planned = len(questions) * len(cfg.generation.matrix)
        lines.append(
            f"Stopped early: {batch.stopped}. Partial run: "
            f"{formatting.count(len(batch.records))} of {formatting.plural(planned, 'answer')}."
        )
    return write_answer_results(
        cfg, batch.records, cfg.paths.results, "\n\n".join(lines), index_version=version
    )


@dataclass(frozen=True)
class JudgeRunJob:
    """The `judge-run` options: the saved run, the judge, the questions, the budget."""

    run: Path
    questions_path: Path
    judge: str | None
    budget: float | None
    note: str


@dataclass(frozen=True)
class JudgeRunOutcome:
    """What a judge run wrote and spent.

    `judged` counts the answers that carry a judgment; `failed` those whose judgment
    failed. An answer whose generation failed, or one after a budget stop, is in neither.
    """

    run_dir: Path
    judged: int
    answers: int
    failed: int
    spent_usd: float
    stopped: str | None


def judge_run(
    cfg: Config, job: JudgeRunJob, *, progress: Progress, make_llm: LLMFactory | None = None
) -> JudgeRunOutcome:
    """Judges a saved answer run without generating again; writes `results/<ts>_judged/`.

    `source.json` beside the run records the source run, the judge, any budget
    stop and the spend, including the invalid attempts of failed judgments.
    `make_llm` builds the model client; `LLM` unless given.

    Raises:
        UserError: `job.judge` is None and `generation.judge` names no judge, or the judge is
            not one of `generation.judges`. Or `job.run` is not an answer or judged run, or
            records no index version or another one than the current index.
    """
    name = cfg.generation.chosen_judge(job.judge)
    version = require_run_index(job.run, require_current_index(cfg))
    cfg = with_budget(cfg.with_questions(Path(job.questions_path)), job.budget)
    source = read_answers(Path(job.run) / ANSWERS_FILE)
    llm = (make_llm or LLM)(cfg.generation)
    inputs = AnswerInputs(
        load_questions(cfg.eval.questions),
        chunks_by_id(load_chunks(cfg)),
        llm,
        name,
    )
    batch = judge_records(cfg, source, inputs, progress=progress)
    records = batch.records
    judged = sum(1 for r in records if r.judge is not None)
    header = f"Judged by {name} from `{job.run}`; answers copied unchanged." + (
        f" {job.note}" if job.note else ""
    )
    if batch.stopped:
        header += (
            f"\n\nStopped early: {batch.stopped}. Partial run: "
            f"{formatting.count(len(records))} of {formatting.plural(len(source), 'answer')} "
            f"copied, {formatting.count(judged)} judged."
        )
    run_dir = write_answer_results(
        cfg, records, cfg.paths.results, header, suffix="_judged", index_version=version
    )
    write_run_source(
        run_dir,
        RunSource(
            source_run=str(job.run),
            judge=name,
            stops=[batch.stopped] if batch.stopped else [],
            spent_usd=llm.spent_usd,
        ),
    )
    return JudgeRunOutcome(
        run_dir=run_dir,
        judged=judged,
        answers=len(source),
        failed=sum(1 for r in records if r.judge_error),
        spent_usd=llm.spent_usd,
        stopped=batch.stopped,
    )


@dataclass(frozen=True)
class JudgeSampleJob:
    """The `judge-sample` options: the judge, the sample file, the repeat, the budget."""

    judge: str
    sample_file: Path
    questions_path: Path
    replicate: int
    budget: float | None


@dataclass(frozen=True)
class JudgeSampleOutcome:
    """Where the judgments went, how many there are, and what they cost."""

    path: Path
    judged: int
    errors: int
    sampled: int
    spent_usd: float
    stopped: str | None


def judge_sample(
    cfg: Config, job: JudgeSampleJob, *, make_llm: LLMFactory | None = None
) -> JudgeSampleOutcome:
    """A judge candidate grades the calibration sample.

    Writes `judgements_<judge>.jsonl` in the sampled run (`judgements_<judge>_r<n>.jsonl`
    for replicate n) and returns counts only, so no judgment is printed before the human
    has rated. `make_llm` builds the model client; `LLM` unless given.

    Raises:
        UserError: `job.judge` is not a judge of `generation.judges` (the config takes only
            plain names as judges, as the judgments file name needs).
    """
    cfg.generation.judge_model(job.judge)
    run_dir, ids = read_sample(job.sample_file)
    require_run_index(Path(run_dir), require_current_index(cfg))
    llm = (make_llm or LLM)(with_budget(cfg, job.budget).generation)
    inputs = AnswerInputs(
        load_questions(job.questions_path),
        chunks_by_id(load_chunks(cfg)),
        llm,
        job.judge,
    )
    answers = read_answers(Path(run_dir) / ANSWERS_FILE)
    # Before any call: a judge name that is not a plain file-name part is refused for free.
    path = judgements_path(run_dir, job.judge, job.replicate)
    batch = judge_answers(cfg, answers, inputs, ids, replicate=job.replicate)
    write_judgements(path, batch.rows)
    errors = sum(row.error is not None for row in batch.rows)
    return JudgeSampleOutcome(
        path, len(batch.rows) - errors, errors, len(ids), llm.spent_usd, batch.stopped
    )


def compare_prompt_runs(cfg: Config, v1: Path, v2: Path) -> Path:
    """Compares the judged runs of prompt v1 and v2 per configuration; the run directory.

    Both runs must have been made against the current index, be two different runs,
    and hold answers of schema 1 (v1) and 2 (v2). Runs the side-effect checks on v2's
    partial answers, and writes results/<ts>_prompts/.

    Raises:
        UserError: `v1` and `v2` name the same run, or a run holds answers of another schema
            than its prompt's. Or a run is not an answer or judged run, or records no index
            version or another one than the current index.
    """
    if v1.resolve() == v2.resolve():
        raise UserError(
            f"--v1 and --v2 name the same run, {v1}",
            hint="pass --v1 the judged run of prompt v1 and --v2 the judged run of prompt v2",
        )
    version = require_run_index(v1, require_current_index(cfg))
    require_run_index(v2, version)
    configs = [m.name for m in cfg.generation.matrix]
    r1, r2 = read_answers(v1 / ANSWERS_FILE), read_answers(v2 / ANSWERS_FILE)
    pairs = (("--v1", v1, r1, FIRST_SCHEMA), ("--v2", v2, r2, PARTIAL_SCHEMA))
    for option, run, records, schema in pairs:
        schemas = {
            r.generation.answer_schema_version or FIRST_SCHEMA for r in records if r.generation
        }
        if schemas != {schema}:
            held = ", ".join(formatting.own_form(s) for s in sorted(schemas)) or "none"
            raise UserError(
                f"{option} takes answers of answer schema {formatting.own_form(schema)}, and {run} "
                f"holds schema {held}",
                hint="pass --v1 the judged run of prompt v1 (answer schema "
                f"{formatting.own_form(FIRST_SCHEMA)}) and --v2 the judged run of prompt v2 "
                f"(answer schema {formatting.own_form(PARTIAL_SCHEMA)})",
            )
    rows = compare_prompts(r1, r2, configs, cfg.eval.bootstrap_resamples, cfg.eval.bootstrap_seed)
    effects = side_effects(r2, load_chunk_venues(cfg))
    return write_prompt_comparison(cfg.paths.results, v1, v2, rows, effects, version)
