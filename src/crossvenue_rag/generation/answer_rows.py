"""The rows of an answer run's tables: one per answer, one per configuration.

`per_answer.csv` has one row per answer: the call records of the generator and
the judge, the outcome and the metrics. `summary.csv` has one row per
configuration: outcome counts, the mean of each metric per scope, tokens and
cost of both models, and the generator's latency.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from pydantic import ConfigDict

from ..config_generation import AnswerScope
from ..run_files import (
    SUMMARY_FILE,
    CsvRow,
    CsvTable,
    OptionalBool,
    OptionalFloat,
    OptionalInt,
    read_table,
)
from ..stats import P50, P95, mean_of_present, percentile
from .metrics import METRICS, AnswerMetrics
from .records import AnswerRecord, CallRecord
from .schemas import GeneratedAnswer, outcome

# The question scopes of the comparisons; `generation.primary_scope` is one of them.
SCOPES: tuple[AnswerScope, ...] = get_args(AnswerScope)
PER_ANSWER_FILE = "per_answer.csv"
# A cell of per_answer.csv as a row builder gives it; "" is an empty cell.
Cell = str | int | float | bool


class PerAnswerRow(CsvRow):
    """One answer: a row of per_answer.csv.

    Both call records, the outcome and the metrics; a cell is empty where the
    answer has no call, no judgment or the metric does not apply.
    It is the one format of the committed answer and judged runs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer_id: str
    config: str
    question_id: str
    category: str
    expected_behavior: str
    requested_model: str
    actual_model: str
    provider: str
    input_tokens: OptionalInt
    reasoning_tokens: OptionalInt
    output_tokens: OptionalInt
    cost_usd: OptionalFloat
    latency_ms: OptionalFloat
    retries: OptionalInt
    invalid_responses: OptionalInt
    cache_key: str
    outcome: str
    refused: OptionalBool
    asked_clarification: OptionalBool
    partial: OptionalBool
    claims: OptionalInt
    faithfulness: OptionalFloat
    venue_attribution_error: OptionalFloat
    coverage: OptionalFloat
    behaviour_correct: OptionalFloat
    judge: str
    judge_model: str
    judge_provider: str
    judge_input_tokens: OptionalInt
    judge_reasoning_tokens: OptionalInt
    judge_output_tokens: OptionalInt
    judge_cost_usd: OptionalFloat
    judge_latency_ms: OptionalFloat
    judge_cache_key: str
    error: str
    judge_error: str


def mean_column(metric: str, scope: str) -> str:
    """The summary.csv column of a metric's mean over a scope: `coverage:comparison`."""
    return f"{metric}:{scope}"


def _summary_column(field: str) -> str:
    """The summary.csv column of an `AnswerSummaryRow` field: `mean_column` for a metric mean."""
    for metric in METRICS:
        for scope in SCOPES:
            if field == f"{metric}_{scope}":
                return mean_column(metric, scope)
    return field


class AnswerSummaryRow(CsvRow):
    """One configuration over the run: a row of an answer run's summary.csv.

    The metric means are the columns `mean_column` names, `<metric>:<scope>`, read as
    the fields `<metric>_<scope>`.
    `read_answer_summary` reads two formats: the committed runs', which every
    answer and judged run from 20260925T175128Z_answers on wrote, and the same
    columns with `judge_errors`, failed judgments counted apart from `errors`.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True, alias_generator=_summary_column
    )

    config: str
    answers: int
    errors: int
    judge_errors: OptionalInt = None
    refused: int
    asked_clarification: int
    partial: int
    faithfulness_comparison: OptionalFloat
    venue_attribution_error_comparison: OptionalFloat
    coverage_comparison: OptionalFloat
    behaviour_correct_comparison: OptionalFloat
    faithfulness_all: OptionalFloat
    venue_attribution_error_all: OptionalFloat
    coverage_all: OptionalFloat
    behaviour_correct_all: OptionalFloat
    input_tokens: float
    reasoning_tokens: float
    output_tokens: float
    cost_usd: float
    cost_usd_new: float
    judge_input_tokens: float
    judge_reasoning_tokens: float
    judge_output_tokens: float
    judge_cost_usd: float
    latency_p50_ms: OptionalFloat
    latency_p95_ms: OptionalFloat
    http_retries: float
    invalid_responses: float
    judge_invalid_responses: float
    cached_calls: int
    params: str


def read_answer_summary(run_dir: Path) -> CsvTable[AnswerSummaryRow]:
    """summary.csv of an answer or judged run, in the format `AnswerSummaryRow` lists."""
    return read_table(run_dir / SUMMARY_FILE, AnswerSummaryRow)


def category_in_scope(category: str, scope: str) -> bool:
    """Whether a question of `category` counts in a scope: `all`, or that category."""
    return scope == "all" or category == scope


def in_scope(record: AnswerRecord, scope: str) -> bool:
    """Whether an answer counts in a scope (`category_in_scope` of its question's category)."""
    return category_in_scope(record.category, scope)


def _blank(value: Cell | None) -> Cell:
    """`value`, or an empty cell for a field the record does not have."""
    return "" if value is None else value


def _generator_call_columns(call: CallRecord | None) -> dict[str, Cell]:
    """The generator call's columns of per_answer.csv; empty cells without a call."""
    if call is None:
        return dict.fromkeys(_GENERATOR_CALL, "")
    return {
        "requested_model": call.requested_model,
        "actual_model": call.actual_model,
        "provider": call.provider,
        "input_tokens": call.input_tokens,
        "reasoning_tokens": _blank(call.reasoning_tokens),
        "output_tokens": call.output_tokens,
        "cost_usd": _blank(call.cost_usd),
        "latency_ms": call.latency_ms,
        "retries": call.retries,
        "invalid_responses": _blank(call.invalid_responses),
        "cache_key": call.cache_key,
    }


def _judge_call_columns(call: CallRecord | None) -> dict[str, Cell]:
    """The judge call's columns of per_answer.csv; empty cells without a judgment."""
    if call is None:
        return dict.fromkeys(_JUDGE_CALL, "")
    return {
        "judge_model": call.actual_model,
        "judge_provider": call.provider,
        "judge_input_tokens": call.input_tokens,
        "judge_reasoning_tokens": _blank(call.reasoning_tokens),
        "judge_output_tokens": call.output_tokens,
        "judge_cost_usd": _blank(call.cost_usd),
        "judge_latency_ms": call.latency_ms,
        "judge_cache_key": call.cache_key,
    }


def _answer_columns(answer: GeneratedAnswer | None) -> dict[str, Cell]:
    """The outcome columns of per_answer.csv; empty cells without an answer."""
    if answer is None:
        return dict.fromkeys(("outcome", "refused", "asked_clarification", "partial", "claims"), "")
    kind = outcome(answer)
    return {
        "outcome": kind,
        "refused": kind == "refused",
        "asked_clarification": answer.asked_clarification,
        "partial": _blank(answer.stated_partial()),
        "claims": len(answer.claims),
    }


def _metric_columns(metrics: AnswerMetrics | None) -> dict[str, Cell]:
    """One column per metric; empty where the metric does not apply or nothing was judged."""
    return {m: _blank(metrics.value(m) if metrics is not None else None) for m in METRICS}


# The columns of per_answer.csv from each call record, in file order.
_GENERATOR_CALL = (
    "requested_model",
    "actual_model",
    "provider",
    "input_tokens",
    "reasoning_tokens",
    "output_tokens",
    "cost_usd",
    "latency_ms",
    "retries",
    "invalid_responses",
    "cache_key",
)
_JUDGE_CALL = (
    "judge_model",
    "judge_provider",
    "judge_input_tokens",
    "judge_reasoning_tokens",
    "judge_output_tokens",
    "judge_cost_usd",
    "judge_latency_ms",
    "judge_cache_key",
)


def per_answer_row(record: AnswerRecord) -> PerAnswerRow:
    """One row of per_answer.csv: both call records, the outcome and the metrics."""
    gen, judged = record.generation, record.judge
    fields = {
        "answer_id": record.answer_id,
        "config": record.config,
        "question_id": record.question_id,
        "category": record.category,
        "expected_behavior": record.expected_behavior,
        **_generator_call_columns(gen.call if gen is not None else None),
        **_answer_columns(gen.answer if gen is not None else None),
        **_metric_columns(judged.metrics if judged is not None else None),
        "judge": _blank(judged.judge if judged is not None else None),
        **_judge_call_columns(judged.call if judged is not None else None),
        "error": record.error or "",
        "judge_error": record.judge_error or "",
    }
    return PerAnswerRow.model_validate(fields)


def _sum(values: Sequence[float | None]) -> float:
    return float(sum(v for v in values if isinstance(v, (int, float))))


def summary_rows(records: Sequence[AnswerRecord], configs: Sequence[str]) -> list[AnswerSummaryRow]:
    """One summary.csv row per configuration: outcomes, metric means, tokens, cost, latency."""
    rows = []
    for config in configs:
        mine = [r for r in records if r.config == config]
        ok = [r.generation for r in mine if r.generation is not None]
        judged = [r for r in mine if r.generation is not None and r.judge is not None]
        fields = {
            "config": config,
            **_outcome_counts(mine, [g.answer for g in ok]),
            **_metric_means(judged),
            **_usage([g.call for g in ok], [r.judged().call for r in judged]),
        }
        rows.append(AnswerSummaryRow.model_validate(fields))
    return rows


def _outcome_counts(
    mine: Sequence[AnswerRecord], answers: Sequence[GeneratedAnswer]
) -> dict[str, int]:
    """Answers, failed generations, failed judgments, and the answers of each non-answer outcome."""
    outcomes = [outcome(a) for a in answers]
    return {
        "answers": len(mine),
        "errors": sum(1 for r in mine if r.error),
        "judge_errors": sum(1 for r in mine if r.judge_error),
        "refused": outcomes.count("refused"),
        "asked_clarification": outcomes.count("asked"),
        "partial": outcomes.count("partial"),
    }


def _metric_means(judged: Sequence[AnswerRecord]) -> dict[str, float | None]:
    """The mean of every metric over the judged answers of each scope, `metric:scope`."""
    means: dict[str, float | None] = {}
    for scope in SCOPES:
        scoped = [r for r in judged if in_scope(r, scope)]
        for metric in METRICS:
            means[mean_column(metric, scope)] = mean_of_present(
                [r.judged().metrics.value(metric) for r in scoped]
            )
    return means


def _usage(
    calls: Sequence[CallRecord], jcalls: Sequence[CallRecord]
) -> dict[str, float | int | str | None]:
    """Tokens, cost and retries of the generator `calls` and judge `jcalls`; latency of `calls`."""
    # A cached call keeps the latency measured when it was made.
    latencies = [c.latency_ms for c in calls]
    both = [*calls, *jcalls]
    new = [c for c in calls if not c.cached]
    return {
        "input_tokens": _sum([c.input_tokens for c in calls]),
        "reasoning_tokens": _sum([c.reasoning_tokens for c in calls]),
        "output_tokens": _sum([c.output_tokens for c in calls]),
        # Cost of the answers; a cached answer carries the cost paid
        # when it was made. `cost_usd_new` is what this run paid.
        "cost_usd": _sum([c.cost_usd for c in calls]),
        "cost_usd_new": _sum([c.cost_usd for c in new]),
        "judge_input_tokens": _sum([c.input_tokens for c in jcalls]),
        "judge_reasoning_tokens": _sum([c.reasoning_tokens for c in jcalls]),
        "judge_output_tokens": _sum([c.output_tokens for c in jcalls]),
        "judge_cost_usd": _sum([c.cost_usd for c in jcalls]),
        "latency_p50_ms": percentile(latencies, P50) if latencies else None,
        "latency_p95_ms": percentile(latencies, P95) if latencies else None,
        "http_retries": _sum([c.retries for c in both]),
        "invalid_responses": _sum([c.invalid_responses for c in calls]),
        "judge_invalid_responses": _sum([c.invalid_responses for c in jcalls]),
        "cached_calls": sum(1 for c in both if c.cached),
        "params": json.dumps(calls[0].params, sort_keys=True) if calls else "",
    }
