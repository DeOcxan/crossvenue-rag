"""A judge candidate's judgments of the calibration sample, set beside the human ratings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..confined_paths import plain_name_problem
from ..errors import UserError
from .answer import Submission, judge_answer
from .answering import AnswerInputs
from .grader_agreement import DimensionAgreement, compare_gradings
from .judging import require_questions, select_answers
from .llm import BudgetExceeded
from .openrouter import ModelCallError
from .records import AnswerRecord, JudgementRow, read_judgements, read_ratings
from .runs import require_answer_run
from .schemas import InvalidResponse, Judgement

RATINGS_PATH = Path("evals/annotations/judge_ratings.jsonl")


@dataclass(frozen=True)
class JudgeBatch:
    """The judgment rows a sample run made, in order, and the budget stop that ended it early."""

    rows: tuple[JudgementRow, ...]
    stopped: str | None


def judge_answers(
    cfg: Config,
    records: Sequence[AnswerRecord],
    inputs: AnswerInputs,
    answer_ids: Sequence[str],
    *,
    replicate: int,
) -> JudgeBatch:
    """Judge `inputs.judge_name` grades the listed answers; one record per answer.

    `replicate` 1 is the same judge again, for self-consistency; the replicate
    number enters the cache key, so a repeat is a fresh call. A judgment that
    stays invalid after the retries, or whose call fails after its retries, is
    recorded as `{"answer_id", "error"}` and the run goes on, so one unparseable
    answer leaves the others graded; the same failures `judge_records` records.
    Exceeding the budget stops the run; the answers graded before the stop are
    returned with the stop's message. Each claim cites numbers of its record's
    `chunk_ids`, as `read_answers` ensures.

    Raises:
        ValueError: `inputs.judge_name` is None: the caller names the judge.
        UserError: `records` holds no record of an id of `answer_ids`, the record of one
            holds a failed generation or answers a question `inputs.questions` lacks, or
            `inputs.judge_name` is not a judge of `generation.judges`.
    """
    if inputs.judge_name is None:
        raise ValueError("judge_answers needs a judge")
    chosen = select_answers(records, answer_ids)
    questions = require_questions(chosen, inputs.questions)
    out: list[JudgementRow] = []
    stopped: str | None = None
    for record in chosen:
        answer_id = record.answer_id
        submission = Submission(questions[record.question_id], record.generated(), inputs.chunks)
        try:
            graded = judge_answer(
                inputs.llm, cfg.generation, submission, inputs.judge_name, replicate=replicate
            )
        except BudgetExceeded as exc:
            stopped = exc.with_hint()
            break
        # As in `judging.judge_records`: only a failed judgment is recorded.
        except (InvalidResponse, ModelCallError) as exc:
            out.append(JudgementRow(answer_id=answer_id, error=str(exc)))
            continue
        out.append(JudgementRow(answer_id=answer_id, judge=graded))
    return JudgeBatch(tuple(out), stopped)


def judgements_path(run_dir: str | Path, judge_name: str, replicate: int = 0) -> Path:
    """The judgments file of `judge_name` in `run_dir`; `replicate` > 0 adds `_r<n>`.

    Raises:
        UserError: `judge_name` is not a plain name (`confined_paths.plain_name_problem`),
            as `x/y` or `..` would not be; the config refuses such names when it loads.
    """
    problem = plain_name_problem(judge_name)
    if problem is not None:
        raise UserError(
            f"judge name {judge_name!r} {problem}", hint="rename the judge under generation.judges"
        )
    suffix = f"_r{replicate}" if replicate else ""
    return Path(run_dir) / f"judgements_{judge_name}{suffix}.jsonl"


def load_judgements(path: Path) -> dict[str, Judgement]:
    """Judgments by answer id, from a judgments file (`judgements_path`).

    Rows whose judgment failed are left out; a missing file gives an empty dict.
    """
    if not path.exists():
        return {}
    return {row.answer_id: row.judge.judgement for row in read_judgements(path) if row.judge}


def human_ratings(path: str | Path = RATINGS_PATH) -> dict[str, Judgement]:
    """The last human rating in the file per answer id, in the judge's schema."""
    return {row.answer_id: row.rating for row in read_ratings(path)}


@dataclass(frozen=True)
class JudgeAgreements:
    """A judge candidate's agreement with the human ratings and with its own repeat.

    Each is None when there is nothing to compare with: no human rating, or no repeat.
    """

    with_humans: list[DimensionAgreement] | None
    with_repeat: list[DimensionAgreement] | None


def judge_agreements(cfg: Config, run: Path, judge: str) -> JudgeAgreements:
    """How `judge`'s judgments of the sample in `run` agree with the humans and with its repeat.

    Against the humans, the answers `generation.calibration_exclude_behaviour` names are left
    out of behaviour_correct, as select-judge and calibration.json count it.

    Raises:
        UserError: `run` is not an answer or judged run, holds no judgment by `judge`, or
            `judge` is not a plain name.
    """
    require_answer_run(run)
    first = load_judgements(judgements_path(run, judge))
    if not first:
        raise UserError(
            f"no judgments by {judge} in {run}",
            hint=f"run `crossvenue judge-sample --judge {judge}` first",
        )
    humans = human_ratings()
    repeat = load_judgements(judgements_path(run, judge, 1))
    excluded = cfg.generation.calibration_exclude_behaviour
    return JudgeAgreements(
        compare_gradings(first, humans, excluded) if humans else None,
        compare_gradings(first, repeat) if repeat else None,
    )
