"""An answer run's files as typed models, each with its one reader, its one writer.

Three files hold them: `answers.jsonl` (one `AnswerRecord` per configuration
and question), `judgements_<judge>.jsonl` (one `JudgementRow` per sampled
answer; `judgements_<judge>_r<n>.jsonl` for repeat n) and the human ratings
(one `RatingRow` per rating); the response cache saves each call's `CallRecord`
as well. A field absent from a saved record stays
absent when the record is written back, and a field saved as null stays null, so
reading a file and writing it again gives the same bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .. import formatting, jsonl
from ..errors import UserError
from .context import Citation
from .metrics import AnswerMetrics
from .schemas import GeneratedAnswer, Judgement

ANSWERS_FILE = "answers.jsonl"


def fully_set[M: BaseModel](model: M, exclude: AbstractSet[str] = frozenset()) -> M:
    """`model` with every field but `exclude` marked as given, its defaults included.

    A field is written only when it was given (`SavedRecord.to_json`); a model
    parsed from a response that left a defaulted field out would otherwise lose
    that field from the saved record. A field in `exclude` is left unset, so it
    is not written.
    """
    return type(model).model_validate(model.model_dump(exclude=set(exclude)))


class SavedRecord(BaseModel):
    """A record saved as one JSON object: its fields in declaration order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_json(self) -> dict[str, Any]:
        """The JSON object the record is saved as; a field never given is left out."""
        return self.model_dump(exclude_unset=True)


class CallRecord(SavedRecord):
    """What one model call was and cost, as the cache and every answer record save it.

    The optional fields are absent from records of the earliest format, which
    carry `cost_source` instead; a record saved without them is written back
    without them.
    """

    requested_model: str
    actual_model: str
    provider: str
    input_tokens: int
    output_tokens: int  # answer tokens, reasoning excluded
    reasoning_tokens: int | None = None
    cost_usd: float | None
    cost_source: str | None = None
    latency_ms: float
    retries: int  # HTTP retries
    invalid_responses: int | None = None  # responses rejected by the schema check, then retried
    cache_key: str
    params: dict[str, Any] | None = None
    cached: bool = False

    def paid_usd(self) -> float:
        """What the call cost in USD.

        Raises:
            ValueError: The record has no cost; it is not counted as a free call.
        """
        if self.cost_usd is None:
            raise ValueError(f"call {self.cache_key} records no cost")
        return self.cost_usd

    @classmethod
    def from_cache(cls, saved: dict[str, Any]) -> CallRecord:
        """The record of a cache entry, marked cached.

        Only the fields `LLM.complete` writes into a call record are kept: an
        entry written by an earlier record format carries extra ones. An entry
        without `params` gets `{}`.
        """
        fields = {name: saved[name] for name in _WRITTEN_FIELDS if name in saved}
        return cls.model_validate({"params": {}, **fields, "cached": True})


# The fields `LLM.complete` writes into the call records it makes, in record order.
_WRITTEN_FIELDS = tuple(f for f in CallRecord.model_fields if f != "cost_source")


class GenerationRecord(SavedRecord):
    """The `generation` part of an answer record: the answer and how it was made.

    `citations` has one list per claim, one citation per chunk it cites;
    `chunk_ids` are the chunks the generator saw, numbered from 1.
    """

    answer: GeneratedAnswer
    citations: list[list[Citation]]
    chunk_ids: list[str]
    prompt_sha256: str
    # Absent from records of the earliest format and of answer schema v1.
    answer_schema_version: int | None = None
    # The generator parameters, in earliest-format records only; the other
    # formats keep them in `call.params`.
    params: dict[str, Any] | None = None
    call: CallRecord

    def numbered_chunk_ids(self) -> dict[int, str]:
        """The chunk ids by the number a claim cites them with, from 1."""
        return dict(enumerate(self.chunk_ids, start=1))


class JudgeRecord(SavedRecord):
    """The `judge` part of an answer record: one judge's grading of the answer."""

    # Absent from earliest-format records, whose runs used one judge.
    judge: str | None = None
    judgement: Judgement
    metrics: AnswerMetrics
    prompt_sha256: str
    # The judge parameters, in earliest-format records only.
    params: dict[str, Any] | None = None
    replicate: int
    call: CallRecord


class AnswerRecord(SavedRecord):
    """One configuration's answer to one question, as answers.jsonl saves it.

    `generation` and `judge` are None when that step failed or was not run.
    `error` says why generation failed; `judge_error` says why judging a
    generated answer failed, in a judged run or an answer run made with a judge.
    It is null in a judged run's record whose judging did not fail, and absent
    from an answer run's record whose judging did not fail or was not run.
    """

    answer_id: str
    config: str
    generator: str
    retriever: str
    question_id: str
    question: str
    category: str
    expected_behavior: str
    venues: list[str]
    generation: GenerationRecord | None
    judge: JudgeRecord | None
    error: str | None
    judge_error: str | None = None

    def generated(self) -> GenerationRecord:
        """The generation, for a caller that only handles generated answers.

        Raises:
            ValueError: The record's generation failed.
        """
        if self.generation is None:
            raise ValueError(f"{self.answer_id} has no generation: {self.error}")
        return self.generation

    def judged(self) -> JudgeRecord:
        """The judge's grading, for a caller that only handles judged answers.

        Raises:
            ValueError: The record was not judged.
        """
        if self.judge is None:
            raise ValueError(f"{self.answer_id} has no judgment: {self.judge_error}")
        return self.judge


def named_judges(records: Sequence[AnswerRecord]) -> list[str]:
    """The judges the records name, sorted; a record judged before judges were named adds none."""
    return sorted({r.judge.judge for r in records if r.judge is not None and r.judge.judge})


class JudgementRow(SavedRecord):
    """One row of `judgements_<judge>.jsonl`: the grading of one sampled answer, or its error."""

    answer_id: str
    judge: JudgeRecord | None = None
    error: str | None = None


class RatingRow(SavedRecord):
    """One human rating of an answer, in the judge's schema without rationales."""

    answer_id: str
    rater: str
    at: str
    rating: Judgement
    # The index the rated chunks were resolved against; absent when unknown.
    index_version: str | None = None


def read_answers(path: str | Path) -> list[AnswerRecord]:
    """Every record of an answers.jsonl file, in file order.

    Accepts every format an answer or judged run has been saved in, each
    written back with the same fields:

    - the earliest format, of trial runs no committed run keeps:
      `generation.params` holds the generator parameters, the call records
      carry `cost_source` and lack `reasoning_tokens`, `invalid_responses` and
      `params`, and `judge` has `params` and no `judge` name;
    - answer schema v1: no `answer_schema_version`, answers without `partial`
      and `missing`, call records with `reasoning_tokens`,
      `invalid_responses` and `params`;
    - answer schema v2: `answer_schema_version` and answers with `partial` and
      `missing`;
    - judged runs of either schema: `judge_error` last, the judge named in
      `judge.judge`;
    - a failed generation: `generation` and `judge` null, `error` a message.

    Every claim of a record cites numbers of its `chunk_ids`, so a reader can
    index `GenerationRecord.numbered_chunk_ids` by them.

    Raises:
        UserError: A claim cites a chunk number outside its record's `chunk_ids`. The error
            names the file and the line.
    """
    records: list[AnswerRecord] = []
    for number, record in jsonl.iter_models(path, AnswerRecord):
        generation = record.generation
        if generation is not None and (
            outside := generation.answer.numbers_outside(len(generation.chunk_ids))
        ):
            raise UserError(
                f"{formatting.location(path, number)}: {record.answer_id} cites chunk "
                f"{formatting.count(outside[0])} of a context of "
                f"{formatting.plural(len(generation.chunk_ids), 'chunk')}",
                hint="generate the answers again with `crossvenue answer-eval`",
            )
        records.append(record)
    return records


def write_answers(path: str | Path, records: Sequence[AnswerRecord]) -> None:
    """Writes answer records as answers.jsonl, replacing the file."""
    jsonl.write(path, [r.to_json() for r in records])


def read_judgements(path: str | Path) -> list[JudgementRow]:
    """Every row of a `judgements_<judge>.jsonl` file, in file order.

    Accepts the one format the calibration runs have: `answer_id` with either
    `judge`, a judge record as answers.jsonl saves it (the judge named, call
    records with `reasoning_tokens`, `invalid_responses` and `params`), or
    `error`. An `error` is the message of a judgment that stayed invalid or of a model
    call that failed after its retries.
    """
    return jsonl.read_models(path, JudgementRow)


def write_judgements(path: str | Path, rows: Sequence[JudgementRow]) -> None:
    """Writes judgment rows as a `judgements_<judge>.jsonl` file, replacing it."""
    jsonl.write(path, [r.to_json() for r in rows])


def read_ratings(path: str | Path) -> list[RatingRow]:
    """Every human rating in a ratings file, in file order; none when the file is absent.

    Accepts ratings with and without `index_version`; the committed ratings
    have none.
    """
    if not Path(path).exists():
        return []
    return jsonl.read_models(path, RatingRow)


def append_rating(path: str | Path, row: RatingRow) -> None:
    """Appends one rating to a ratings file, creating the file if it does not exist."""
    jsonl.append(path, [row.to_json()])
