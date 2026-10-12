"""report.md of an answer run: metrics, cost and latency, parameters, pairs."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import JsonValue, TypeAdapter

from .. import formatting
from ..config import Config
from ..config_generation import GenerationConfig
from ..errors import UserError
from ..evaluation.labels import fmt, markdown_table
from ..models import MS_PER_SECOND
from ..stats import INTERVAL_LEVEL_PERCENT
from .answer_pairs import ConfigComparison
from .answer_rows import AnswerSummaryRow
from .records import AnswerRecord, named_judges
from .runs import uncalibrated_metrics


def report(
    cfg: Config,
    records: Sequence[AnswerRecord],
    summary: Sequence[AnswerSummaryRow],
    pairs: Sequence[ConfigComparison],
    note: str,
    index_version: str,
) -> str:
    """report.md of an answer run: header, metrics, cost and latency, parameters, pairs.

    Raises:
        UserError: The records carry more than one judge, so no one calibration decides which
            metrics the report marks uncalibrated.
    """
    gen = cfg.generation
    judges = named_judges(records)
    if len(judges) > 1:
        raise UserError(
            f"the answers were judged by more than one judge: {', '.join(judges)}",
            hint="judge the run again with one judge (`crossvenue judge-run --judge NAME`)",
        )
    uncalibrated = uncalibrated_metrics(gen, judges[0] if judges else None)
    lines = [
        *_report_header(gen, records, judges, note, index_version),
        *(_uncalibrated_note(uncalibrated) if uncalibrated and judges else []),
        *_metrics_table(summary, uncalibrated),
        *_cost_table(summary),
        "",
        "## Parameters sent",
        "",
        *(f"- {row.config}: {_params_text(row.params)}" for row in summary),
        *_pairs_table(pairs),
    ]
    return "\n".join(lines) + "\n"


def _params_text(params: str) -> str:
    """The parameters a call sent, from their JSON object, as `key=value` items in code font.

    A nested object's keys are joined by dots (`provider.order`), a list's items by
    commas, and every value keeps its own form (`temperature=0.0`, `exclude=true`);
    no parameters (`""`) is a dash.
    """
    if not params:
        return formatting.MISSING
    sent = _JSON_OBJECT.validate_json(params)
    return ", ".join(f"`{key}={value}`" for key, value in _flat(sent))


_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _flat(params: dict[str, JsonValue], prefix: str = "") -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key, value in params.items():
        if isinstance(value, dict):
            items.extend(_flat(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            items.append((f"{prefix}{key}", ",".join(_scalar(v) for v in value)))
        else:
            items.append((f"{prefix}{key}", _scalar(value)))
    return items


def _scalar(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _report_header(
    gen: GenerationConfig,
    records: Sequence[AnswerRecord],
    judges: Sequence[str],
    note: str,
    index_version: str,
) -> list[str]:
    questions = {r.question_id for r in records}
    # A judge the config does not list is named with the model its records requested.
    recorded = {
        r.judge.judge: r.judge.call.requested_model for r in records if r.judge and r.judge.judge
    }
    judge_label = (
        ", ".join(
            f"{j} ({gen.judges[j].model if j in gen.judges else recorded[j]})" for j in judges
        )
        or "none (generation only)"
    )
    return [
        "# Answer evaluation",
        "",
        f"- `index_version`: `{index_version}`",
        "",
        *([note, ""] if note else []),
        f"{formatting.plural(len(questions), 'question')} \u00d7 "
        f"{formatting.plural(len(gen.matrix), 'configuration')}; "
        f"top-{formatting.count(gen.top_k)} chunks. Judge: {judge_label}.",
    ]


def _uncalibrated_note(uncalibrated: Sequence[str]) -> list[str]:
    return [
        "",
        "Uncalibrated: " + ", ".join(f"`{m}`" for m in uncalibrated) + ". The judge's "
        "dimension behind "
        "each failed its pre-registered agreement threshold with the human ratings "
        "(methodology section 9); these numbers are the judge's grades only and "
        "support no conclusion.",
    ]


def _metrics_table(summary: Sequence[AnswerSummaryRow], uncalibrated: Sequence[str]) -> list[str]:
    def head(metric: str, label: str) -> str:
        return f"{label} (uncalibrated)" if metric in uncalibrated else label

    header = [
        "config",
        "answers",
        "errors",
        "refused",
        "asked",
        head("faithfulness", "faithfulness"),
        head("venue_attribution_error", "venue attribution error"),
        head("coverage", "coverage"),
        head("behaviour_correct", "behavior"),
        head("coverage", "coverage (comparison)"),
        head("venue_attribution_error", "venue error (comparison)"),
    ]
    rows = [
        [
            row.config,
            formatting.count(row.answers),
            formatting.count(row.errors),
            formatting.count(row.refused),
            formatting.count(row.asked_clarification),
            fmt(row.faithfulness_all),
            fmt(row.venue_attribution_error_all),
            fmt(row.coverage_all),
            fmt(row.behaviour_correct_all),
            fmt(row.coverage_comparison),
            fmt(row.venue_attribution_error_comparison),
        ]
        for row in summary
    ]
    return [
        "",
        "## Metrics per configuration",
        "",
        markdown_table(header, rows, right=range(1, len(header))),
    ]


def latency_seconds(latency_ms: float) -> str:
    """A model-call duration saved in ms, as whole seconds (docs/style.md rule 11)."""
    return formatting.call_seconds(latency_ms / MS_PER_SECOND)


def _cost_table(summary: Sequence[AnswerSummaryRow]) -> list[str]:
    header = [
        "config",
        "tokens in",
        "reasoning",
        "answer",
        "generator USD",
        "judge USD",
        "p50 s",
        "p95 s",
        "invalid responses",
        "HTTP retries",
    ]
    rows = [
        [
            row.config,
            formatting.count(int(row.input_tokens)),
            formatting.count(int(row.reasoning_tokens)),
            formatting.count(int(row.output_tokens)),
            formatting.cost(row.cost_usd),
            formatting.cost(row.judge_cost_usd),
            formatting.optional(row.latency_p50_ms, latency_seconds),
            formatting.optional(row.latency_p95_ms, latency_seconds),
            formatting.count(int(row.invalid_responses)),
            formatting.count(int(row.http_retries)),
        ]
        for row in summary
    ]
    return [
        "",
        "## Cost and call duration per configuration",
        "",
        "Generator tokens: input, reasoning and answer counted separately. USD is the cost "
        "OpenRouter reported for each answer; a cached answer carries the cost paid when "
        "it was made. p50 and p95 are the duration of the generator's calls, in seconds.",
        "",
        markdown_table(header, rows, right=range(1, len(header))),
    ]


def _pairs_table(pairs: Sequence[ConfigComparison]) -> list[str]:
    header = [
        "a vs b",
        "metric",
        "scope",
        "n",
        "a \u2212 b",
        f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% CI",
        "randomization p",
        "role",
        "Holm p",
    ]
    rows = [
        [
            f"{p.a} vs {p.b}",
            f"`{p.metric}`",
            p.scope,
            formatting.count(p.n),
            formatting.signed(p.diff),
            formatting.interval(p.low, p.high),
            formatting.optional(p.randomisation_p, formatting.p_value),
            p.role if p.calibrated else p.role + ", uncalibrated",
            formatting.optional(p.holm_p, formatting.p_value),
        ]
        for p in pairs
    ]
    return [
        "",
        "## Paired comparisons",
        "",
        "a \u2212 b over questions both answered and judged; "
        f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% paired bootstrap interval, "
        "two-sided randomization test, Holm within the pre-registered comparison and within "
        "each other metric and scope.",
        "",
        markdown_table(header, rows, right=(3, 4, 6, 8)),
    ]
