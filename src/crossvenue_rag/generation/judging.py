"""Judging saved answers again, without generating them: one record per saved answer.

The judgment of each record is replaced; its generation is kept as saved. A failed
judgment is recorded in the record's `judge_error` and the run continues.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence

from .. import formatting, listing
from ..config import Config
from ..errors import UserError
from ..models import Question
from .answer import Submission, judge_answer
from .answering import AnswerBatch, AnswerInputs, until_budget
from .openrouter import ModelCallError
from .records import AnswerRecord
from .schemas import InvalidResponse

log = logging.getLogger(__name__)


def select_answers(
    records: Sequence[AnswerRecord], answer_ids: Sequence[str]
) -> list[AnswerRecord]:
    """The records of `answer_ids`, in that order, each with a generation.

    Raises:
        UserError: The run has no record of some of the ids, as when a sample file
            names another run's answers. Or some of the records hold a failed
            generation, which has nothing to judge or rate.
    """
    hint = "write the sample again from this run with `crossvenue rate-sample`"
    by_id = {r.answer_id: r for r in records}
    missing = [i for i in answer_ids if i not in by_id]
    if missing:
        raise UserError(
            f"{formatting.plural(len(missing), 'answer id')} not in the run: "
            f"{listing.first_items(missing)}",
            hint=hint,
        )
    failed = [i for i in answer_ids if by_id[i].generation is None]
    if failed:
        raise UserError(
            f"{formatting.plural(len(failed), 'answer id')} whose generation failed in the "
            f"run: {listing.first_items(failed)}",
            hint=f"{hint}, which draws only generated answers",
        )
    return [by_id[i] for i in answer_ids]


def require_questions(
    records: Sequence[AnswerRecord], questions: Sequence[Question]
) -> dict[str, Question]:
    """The questions by id, after checking every record's question is among them.

    Raises:
        UserError: A record answers a question the question file lacks.
    """
    by_id = {q.id: q for q in questions}
    missing = sorted({r.question_id for r in records if r.question_id not in by_id})
    if missing:
        raise UserError(
            f"{formatting.plural(len(missing), 'question')} of the run not in the question "
            f"file: {listing.first_items(missing)}",
            hint="pass the question file the run answered with --questions",
        )
    return by_id


def judge_records(
    cfg: Config,
    records: Sequence[AnswerRecord],
    inputs: AnswerInputs,
    *,
    progress: Callable[[str], None],
) -> AnswerBatch:
    """Judges saved answers without generating them again.

    Each record is copied with `judge` filled in. An answer shared by two
    configurations is one cached judge call. A budget stop ends the run with the
    records judged before it; a judgment that stays invalid after the retries, or
    whose call fails after its retries, is recorded in the record's `judge_error`.
    Each claim cites numbers of its record's `chunk_ids`, as `read_answers` ensures.

    Raises:
        ValueError: `inputs.judge_name` is None: the caller names the judge.
        UserError: A record of `records` answers a question `inputs.questions` lacks.
    """
    if inputs.judge_name is None:
        raise ValueError("judge_records needs a judge")
    questions = require_questions(records, inputs.questions)
    return until_budget(_judged(cfg, records, inputs, questions, inputs.judge_name), progress)


def _judged(
    cfg: Config,
    records: Sequence[AnswerRecord],
    inputs: AnswerInputs,
    questions: Mapping[str, Question],
    judge_name: str,
) -> Iterator[AnswerRecord]:
    """Each of `records` judged again, in order; one without a generation is copied unjudged.

    The category and expected behavior come from the question file, as the judge's prompt
    does: a set can recategorize a question after its answers were generated.
    """
    for record in records:
        question = questions[record.question_id]
        judged = record.model_copy(
            update={
                "judge": None,
                "judge_error": None,
                "category": question.effective_category,
                "expected_behavior": question.expected_behavior,
            }
        )
        if record.generation is not None:
            try:
                submission = Submission(question, record.generation, inputs.chunks)
                graded = judge_answer(inputs.llm, cfg.generation, submission, judge_name)
                judged = judged.model_copy(update={"judge": graded})
            # As in `answering.run_answers`: only a failed judgment is recorded on the answer.
            except (InvalidResponse, ModelCallError) as exc:
                judged = judged.model_copy(update={"judge_error": f"{type(exc).__name__}: {exc}"})
                log.warning("%s judge failed: %s", record.answer_id, judged.judge_error)
        yield judged
