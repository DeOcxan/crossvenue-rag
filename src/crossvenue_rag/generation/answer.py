"""A single model call as a validated record: a generator's answer, or the judge's verdicts on one.

`generate` returns a generator's answer with every claim's citations resolved to
an occurrence; `judge_answer` returns the judge's verdicts with the metrics derived
from them. Both records carry the call record, the parameters and the sha256 of
the prompt file, so a result names exactly what produced it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import formatting
from ..config_generation import GenerationConfig, ModelConfig
from ..models import Chunk, Question, Result
from .context import chunk_header, render_chunks, resolve_citation
from .llm import LLM
from .metrics import answer_metrics
from .openrouter import ModelRequest, SamplingParams
from .records import GenerationRecord, JudgeRecord, fully_set
from .runs import generator_prompt, judge_prompt
from .schemas import (
    ANSWER_SCHEMAS,
    FIRST_SCHEMA,
    JUDGEMENT_SCHEMA,
    V2_ANSWER_FIELDS,
    GeneratedAnswer,
    outcome,
    parse_answer,
    parse_judgement,
)


def generator_params(cfg: GenerationConfig, model: ModelConfig) -> SamplingParams:
    """The shared generator parameters.

    Temperature is None where the endpoint of `model` does not accept it.
    """
    params = cfg.generator_params
    temperature = params.temperature if model.temperature_supported else None
    return SamplingParams(params.max_tokens, params.reasoning_effort, temperature)


def judge_params(cfg: GenerationConfig) -> SamplingParams:
    """The judge's sampling parameters from the generation config."""
    return SamplingParams(cfg.judge_params.max_tokens, cfg.judge_params.effort)


def generator_message(question: Question, results: Sequence[Result]) -> str:
    """The generator's user message: the question and the numbered chunks."""
    return (
        f"Question: {question.question}\n\n"
        f"Documentation chunks ({len(results)}):\n\n{render_chunks(results)}"
    )


def generator_request(
    cfg: GenerationConfig, model: ModelConfig, question: Question, results: Sequence[Result]
) -> ModelRequest:
    """The generator's request for `question` over the chunks `results`, under `cfg`'s prompt."""
    system, _ = generator_prompt(cfg)
    return ModelRequest(
        model,
        system,
        generator_message(question, results),
        ANSWER_SCHEMAS[cfg.answer_schema_version],
        "answer",
        generator_params(cfg, model),
    )


def generate(
    llm: LLM,
    cfg: GenerationConfig,
    model: ModelConfig,
    question: Question,
    results: Sequence[Result],
) -> GenerationRecord:
    """The generator's answer to `question` from the chunks `results`, with its citations.

    The record holds the fields of the configured answer schema and no other: an
    answer of schema v1 has no `partial` or `missing` and the record no
    `answer_schema_version`, as every committed v1 run has them.
    """
    _, prompt_hash = generator_prompt(cfg)
    version = cfg.answer_schema_version
    request = generator_request(cfg, model, question, results)
    call = llm.complete(
        request, validate=lambda content: parse_answer(content, len(results), version)
    )
    answer = parse_answer(call.content, len(results), version)
    citations = [
        [resolve_citation(question.question, n, results[n - 1]) for n in claim.chunks]
        for claim in answer.claims
    ]
    v1 = version == FIRST_SCHEMA
    return GenerationRecord(
        answer=fully_set(answer, exclude=V2_ANSWER_FIELDS if v1 else frozenset()),
        citations=citations,
        chunk_ids=[r.chunk_id for r in results],
        prompt_sha256=prompt_hash,
        **({} if v1 else {"answer_schema_version": version}),
        call=call.record,
    )


def judge_message(
    question: Question,
    answer: GeneratedAnswer,
    results_by_number: Mapping[int, Result],
    gold: Mapping[str, Sequence[Chunk]],
) -> str:
    """The judge's user message for one answer.

    Holds the question, its venues and expected behavior, the answer with its
    flags, the claims, the cited chunks by their number in `results_by_number`,
    and the gold chunks of `gold`, grouped by venue. `results_by_number` numbers
    its chunks from 1, as `GenerationRecord.numbered_chunk_ids` does.

    Raises:
        ValueError: A claim cites a number `results_by_number` lacks; the message names it.
    """
    claims = answer.claims
    claim_lines = [
        f"({i}) [venue: {c.venue}] {c.text} -- cites chunks {c.chunks}"
        for i, c in enumerate(claims, start=1)
    ]
    cited = sorted({n for c in claims for n in c.chunks})
    if missing := answer.numbers_outside(len(results_by_number)):
        raise ValueError(
            f"a claim cites chunk {formatting.count(missing[0])}, which the numbered chunks lack"
        )
    cited_blocks = [
        f"{chunk_header(n, results_by_number[n])}\n{results_by_number[n].text.strip()}"
        for n in cited
    ]
    gold_blocks = []
    for venue in sorted(gold):
        for i, chunk in enumerate(gold[venue], start=1):
            gold_blocks.append(f"[gold {venue} {i}] venue: {venue}\n{chunk.text.strip()}")
    # `refused` as the metrics count it (a partial answer with no claim is a
    # refusal); `asked_clarification` as the model set it.
    flags = (
        f"refused: {str(outcome(answer) == 'refused').lower()}, "
        f"asked_clarification: {str(answer.asked_clarification).lower()}"
    )
    return "\n\n".join(
        [
            f"Question: {question.question}",
            f"Venues the question concerns: {', '.join(question.venues) or '-'}",
            "Venues with gold chunks (one coverage entry for each, no other): "
            + (", ".join(sorted(gold)) or "none"),
            f"Expected behaviour: {question.expected_behavior}",
            f"Answer ({flags}):\n{answer.answer}",
            "Claims:\n" + ("\n".join(claim_lines) if claim_lines else "(none)"),
            "Cited chunks:\n\n" + ("\n\n".join(cited_blocks) if cited_blocks else "(none)"),
            "Gold chunks:\n\n" + ("\n\n".join(gold_blocks) if gold_blocks else "(none)"),
        ]
    )


def gold_by_venue(gold_ids: Sequence[str], chunks: Mapping[str, Chunk]) -> dict[str, list[Chunk]]:
    """The gold chunks `chunks` holds, by venue, each venue's in `gold_ids` order.

    A gold id `chunks` lacks is left out.
    """
    gold: dict[str, list[Chunk]] = {}
    for chunk_id in gold_ids:
        if chunk_id in chunks:
            gold.setdefault(chunks[chunk_id].venue, []).append(chunks[chunk_id])
    return gold


@dataclass(frozen=True)
class Submission:
    """One generated answer as the judge sees it.

    Holds the question, the generation record, and the chunks by id (the cited
    ones and the gold ones).
    """

    question: Question
    generated: GenerationRecord
    chunks: Mapping[str, Chunk]


def judge_request(cfg: GenerationConfig, submission: Submission) -> tuple[str, str]:
    """The judge's (system, user) messages for one generated answer.

    Raises:
        ValueError: `submission.chunks` lacks a chunk the answer was generated from; the
            message names it.
    """
    question, generated, chunks = submission.question, submission.generated, submission.chunks
    system, _ = judge_prompt(cfg)
    if missing := [c for c in generated.chunk_ids if c not in chunks]:
        raise ValueError(f"the answer was generated from {missing[0]}, which the chunks lack")
    results_by_number = {
        n: Result.from_chunk(chunks[cid], 0.0) for n, cid in generated.numbered_chunk_ids().items()
    }
    gold = gold_by_venue(question.gold_chunk_ids, chunks)
    return system, judge_message(question, generated.answer, results_by_number, gold)


def judge_answer(
    llm: LLM,
    cfg: GenerationConfig,
    submission: Submission,
    judge_name: str,
    replicate: int = 0,
) -> JudgeRecord:
    """Grades one generated answer with judge `judge_name`.

    `replicate` > 0 is a deliberate repeat. The judge sees the question, the
    expected behavior, the claims, the chunks they cite and the question's gold
    chunks.

    Raises:
        UserError: `judge_name` is not a judge of `generation.judges`.
        ValueError: `submission.chunks` lacks a chunk the answer was generated from.
    """
    model = cfg.judge_model(judge_name)
    system, user = judge_request(cfg, submission)
    _, prompt_hash = judge_prompt(cfg)
    params = judge_params(cfg)
    answer = submission.generated.answer
    gold = set(gold_by_venue(submission.question.gold_chunk_ids, submission.chunks))
    claims = len(answer.claims)
    call = llm.complete(
        ModelRequest(model, system, user, JUDGEMENT_SCHEMA, "judgement", params),
        validate=lambda content: parse_judgement(content, claims, gold),
        replicate=replicate,
    )
    judgement = fully_set(parse_judgement(call.content, claims, gold))
    return JudgeRecord(
        judge=judge_name,
        judgement=judgement,
        metrics=answer_metrics(judgement),
        prompt_sha256=prompt_hash,
        replicate=replicate,
        call=call.record,
    )
