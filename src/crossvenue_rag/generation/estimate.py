"""The cost of an answer run before running it, from the cache; no model call."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import Config
from ..models import Question
from ..retrieval.protocols import Ranks
from .answer import generator_request
from .answering import Retrieved, retrieve_once
from .llm import LLM, observed_costs


@dataclass(slots=True)
class EstimateRow:
    """What one configuration's answers, or the judge's gradings, would cost.

    `cached` is None for the judge: its requests are not looked up in the cache.
    """

    item: str
    model: str
    calls: int
    cached: int | None
    mean_cost_usd: float | None
    estimate_usd: float | None


def estimate_answers(
    cfg: Config,
    retriever: Ranks,
    questions: Sequence[Question],
    llm: LLM,
    judge_name: str | None,
) -> list[EstimateRow]:
    """The cost of `run_answers`, before running it; makes no model call.

    Every generator request is built and looked up in the cache; each uncached
    one is priced at the mean cost of that model's cached calls. Judge calls are
    counted per answer, each priced at the mean cost of the judge model's cached
    calls, cached or not. A judge request holds the generated answer, which does
    not exist before the run. Each figure is an estimate at that
    mean and bounds the cost in neither direction: calls dearer than the cached
    ones cost more, and a judge call the cache already holds costs nothing.

    Raises:
        UserError: `judge_name` is not a judge of `generation.judges`.
    """
    gen = cfg.generation
    means = observed_costs(llm.cache)
    retrieved: Retrieved = {}
    rows: list[EstimateRow] = []
    answers = 0
    for entry in gen.matrix:
        model = gen.generators[entry.generator]
        seen: set[str] = set()
        cached = 0
        for question in questions:
            results = retrieve_once(retrieved, retriever, gen.top_k, entry.retriever, question)
            if llm.is_cached(generator_request(gen, model, question, results)):
                cached += 1
            seen.add(question.id)
        answers += len(seen)
        mean = means.get(model.model)
        uncached = len(seen) - cached
        rows.append(
            EstimateRow(
                entry.name,
                model.model,
                len(seen),
                cached,
                mean,
                None if mean is None else mean * uncached,
            )
        )
    if judge_name:
        model = gen.judge_model(judge_name)
        mean = means.get(model.model)
        rows.append(
            EstimateRow(
                f"judge {judge_name}",
                model.model,
                answers,
                None,
                mean,
                None if mean is None else mean * answers,
            )
        )
    return rows


def estimate_total(rows: Sequence[EstimateRow]) -> tuple[float, bool]:
    """The sum of the rows' known estimates, and whether any row's estimate is unknown."""
    known = [row.estimate_usd for row in rows if row.estimate_usd is not None]
    return sum(known), len(known) < len(rows)
