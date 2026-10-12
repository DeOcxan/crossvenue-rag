"""The answer run: one record for each configuration of `generation.matrix` on each question.

Every generator on a retrieval method answers from the same top-k, so generators differ
only in the model. A failed generation or judgment is recorded with its error and the run
continues; a budget stop ends the run with the records made before it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from ..config import Config
from ..config_generation import MatrixEntry
from ..config_pipeline import RetrievalMethod
from ..models import Chunk, Question, Result
from ..retrieval.methods import ORACLE_METHODS
from ..retrieval.protocols import Ranks
from .answer import Submission, generate, judge_answer
from .llm import LLM, BudgetExceeded
from .openrouter import ModelCallError
from .records import AnswerRecord
from .schemas import InvalidResponse

log = logging.getLogger(__name__)

# Each method's top k per question, by (method, question id), retrieved once.
type Retrieved = dict[tuple[str, str], list[Result]]


def answer_id(config: str, question_id: str) -> str:
    """An answer's id: its configuration and its question, `config:question`."""
    return f"{config}:{question_id}"


def split_answer_id(answer: str) -> tuple[str, str]:
    """(configuration, question id) of an answer id, the inverse of `answer_id`."""
    config, _, question_id = answer.rpartition(":")
    return config, question_id


@dataclass(frozen=True)
class AnswerInputs:
    """What generating or judging answers reads, built once by the command.

    `chunks` are the ones the judge cites by id; `judge_name` is the judge
    candidate, `None` to generate only.
    """

    questions: Sequence[Question]
    chunks: Mapping[str, Chunk]
    llm: LLM
    judge_name: str | None


@dataclass(frozen=True)
class AnswerBatch:
    """The records a run made, in order, and the budget stop that ended it early, if any."""

    records: tuple[AnswerRecord, ...]
    stopped: str | None = None


def until_budget(records: Iterator[AnswerRecord], progress: Callable[[str], None]) -> AnswerBatch:
    """Every record of `records`, or those made before a budget stop and the stop's message.

    `progress` is called with each record's answer id as the record is made.
    """
    made: list[AnswerRecord] = []
    try:
        for record in records:
            made.append(record)
            progress(record.answer_id)
    except BudgetExceeded as exc:
        log.warning("stopped: %s", exc.with_hint())
        return AnswerBatch(tuple(made), exc.with_hint())
    return AnswerBatch(tuple(made))


def run_answers(
    cfg: Config,
    retriever: Ranks,
    inputs: AnswerInputs,
    *,
    progress: Callable[[str], None],
) -> AnswerBatch:
    """Generates every configuration on every question, and judges when asked.

    `inputs.judge_name` names the judge, `None` to generate only. A budget stop
    ends the run with the answers made before it.
    """
    return until_budget(_answers(cfg, retriever, inputs), progress)


def _answers(cfg: Config, retriever: Ranks, inputs: AnswerInputs) -> Iterator[AnswerRecord]:
    """One record per configuration and question, in order.

    Each method's top-k is retrieved once per question and shared by every
    generator on it.
    """
    gen = cfg.generation
    retrieved: Retrieved = {}
    for entry in gen.matrix:
        for question in inputs.questions:
            results = retrieve_once(retrieved, retriever, gen.top_k, entry.retriever, question)
            yield _answer(cfg, inputs, _new_record(entry, question), question, results)


def retrieve_for(
    retriever: Ranks, k: int, method: RetrievalMethod, question: Question
) -> list[Result]:
    """One method's top k for a question; the oracle reads the question's labeled venues."""
    hint = question.venues if method in ORACLE_METHODS else None
    return retriever.retrieve(question.question, method, k=k, venues_hint=hint)


def retrieve_once(
    retrieved: Retrieved, retriever: Ranks, k: int, method: RetrievalMethod, question: Question
) -> list[Result]:
    """`retrieve_for`, kept in `retrieved` so each (method, question) is retrieved once."""
    key = (method, question.id)
    if key not in retrieved:
        retrieved[key] = retrieve_for(retriever, k, method, question)
    return retrieved[key]


def _new_record(entry: MatrixEntry, question: Question) -> AnswerRecord:
    """An answer record before generation: the configuration and the question."""
    return AnswerRecord(
        answer_id=answer_id(entry.name, question.id),
        config=entry.name,
        generator=entry.generator,
        retriever=entry.retriever,
        question_id=question.id,
        question=question.question,
        category=question.effective_category,
        expected_behavior=question.expected_behavior,
        venues=list(question.venues),
        generation=None,
        judge=None,
        error=None,
    )


def _answer(
    cfg: Config,
    inputs: AnswerInputs,
    record: AnswerRecord,
    question: Question,
    chunks: Sequence[Result],
) -> AnswerRecord:
    """`record` with `generation` (and `judge`) filled; a failed call is recorded.

    A failed generation is recorded in `error`; a failed judgment of a generated
    answer in `judge_error`, the field `judging.judge_records` uses, and the generation is kept.
    """
    gen = cfg.generation
    model = gen.generators[record.generator]
    # A failed generation or judgment, after the retries, is InvalidResponse (from
    # `parse_answer` and `parse_judgement`) or ModelCallError, and is recorded on the
    # answer. Any other error, a budget stop or a prompt that is not the pinned one
    # among them, stops the run.
    try:
        generated = generate(inputs.llm, gen, model, question, chunks)
    except (InvalidResponse, ModelCallError) as exc:
        record = record.model_copy(update={"error": f"{type(exc).__name__}: {exc}"})
        log.warning("%s failed: %s", record.answer_id, record.error)
        return record
    record = record.model_copy(update={"generation": generated})
    if inputs.judge_name:
        submission = Submission(question, generated, inputs.chunks)
        try:
            graded = judge_answer(inputs.llm, gen, submission, inputs.judge_name)
        except (InvalidResponse, ModelCallError) as exc:
            record = record.model_copy(update={"judge_error": f"{type(exc).__name__}: {exc}"})
            log.warning("%s judge failed: %s", record.answer_id, record.judge_error)
            return record
        record = record.model_copy(update={"judge": graded})
    return record
