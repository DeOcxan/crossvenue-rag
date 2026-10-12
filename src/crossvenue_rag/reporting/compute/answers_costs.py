"""The cost of the answer runs: dollars, call time."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import MS_PER_SECOND
from ...run_files import SUMMARY_FILE
from ...stats import P50, percentile
from ..sources import Sources
from .answers import PROMPT_V1, PROMPT_V2, PROMPTS, answer_run, configurations, judged_run
from .lookup import entry

# A generator call this slow or slower, in seconds, is a slow call.
SLOW_CALL_S = 120
# Mercury's configuration, whose slow calls and refusals the documents describe.
MERCURY = "mercury@venue_split"


def _change(src: Sources, config: str, column: str, *, judged: bool = False) -> tuple[float, float]:
    """(`column` of `config` under v1, under v2), from the answer or the judged runs."""
    run = judged_run if judged else answer_run
    return (
        entry(run(src, PROMPT_V1).summary, config, SUMMARY_FILE).value(column),
        entry(run(src, PROMPT_V2).summary, config, SUMMARY_FILE).value(column),
    )


@dataclass(frozen=True)
class CostRow:
    """One configuration's row of the cost table: each pair is (under v1, under v2)."""

    config: str
    generator_usd: tuple[float, float]
    judge_usd: tuple[float, float]
    # The generator's call duration, in seconds (docs/style.md rule 11).
    p50_s: tuple[float, float]
    p95_s: tuple[float, float]
    output_tokens: tuple[int, int]
    # Under v2 only.
    invalid_responses: int


def _count_change(src: Sources, config: str, column: str) -> tuple[int, int]:
    """(`column` of `config` under v1, under v2), a count in the answer runs' summary.csv.

    Raises:
        ValueError: an answer run's summary.csv has no row of `config`.
    """
    return (
        entry(answer_run(src, PROMPT_V1).summary, config, SUMMARY_FILE).count(column),
        entry(answer_run(src, PROMPT_V2).summary, config, SUMMARY_FILE).count(column),
    )


def _seconds(ms: tuple[float, float]) -> tuple[float, float]:
    return ms[0] / MS_PER_SECOND, ms[1] / MS_PER_SECOND


def cost_rows(src: Sources) -> list[CostRow]:
    """Per configuration: its cost, judge cost, call duration and output tokens, v1 and v2."""
    return [
        CostRow(
            config=c,
            generator_usd=_change(src, c, "cost_usd"),
            judge_usd=_change(src, c, "judge_cost_usd", judged=True),
            p50_s=_seconds(_change(src, c, "latency_p50_ms")),
            p95_s=_seconds(_change(src, c, "latency_p95_ms")),
            output_tokens=_count_change(src, c, "output_tokens"),
            invalid_responses=entry(src.v2.summary, c, SUMMARY_FILE).count("invalid_responses"),
        )
        for c in configurations(src)
    ]


def judge_cost_ratios(src: Sources) -> tuple[float, float]:
    """(smallest, largest) ratio of judge cost to generator cost of a configuration and prompt."""
    ratios = [
        judge / generator
        for c in configurations(src)
        for generator, judge in zip(
            _change(src, c, "cost_usd"), _change(src, c, "judge_cost_usd", judged=True), strict=True
        )
    ]
    return min(ratios), max(ratios)


@dataclass(frozen=True)
class JudgeCalls:
    """The judge calls of one judged run."""

    paid_usd: float  # the cost of the uncached calls
    cached: int
    p50_s: float  # the duration of an uncached call, in seconds


def judge_calls(src: Sources, prompt: int) -> JudgeCalls:
    """The judge calls of the judged run of `prompt`.

    Raises:
        ValueError: `prompt` is not 1 or 2, or the run made no paid judge call to time.
    """
    calls = [r.judge.call for r in judged_run(src, prompt).records if r.judge is not None]
    paid = [c for c in calls if not c.cached]
    if not paid:
        raise ValueError(f"the judged run of prompt v{prompt} made no paid call to time")
    return JudgeCalls(
        sum(c.paid_usd() for c in paid),
        len(calls) - len(paid),
        percentile([c.latency_ms for c in paid], P50) / MS_PER_SECOND,
    )


def invalid_judgements(src: Sources) -> int:
    """Answers of both judged runs whose judgment stayed invalid (`judge_error`)."""
    return sum(1 for p in PROMPTS for r in judged_run(src, p).records if r.judge_error)


def cached_calls(src: Sources, config: str, prompt: int) -> int:
    """Generator calls of `config` served from the cache in the answer run of `prompt`.

    Raises:
        ValueError: `prompt` is not 1 or 2, or the run's summary.csv has no row of `config`.
    """
    return entry(answer_run(src, prompt).summary, config, SUMMARY_FILE).cached_calls


def uncached_cost(src: Sources, prompt: int) -> float:
    """What the uncached generator calls of the answer run of `prompt` cost.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    summary = answer_run(src, prompt).summary
    return sum(entry(summary, c, SUMMARY_FILE).cost_usd_new for c in configurations(src))


@dataclass(frozen=True)
class SlowCalls:
    """`MERCURY`'s generator calls with an invalid response in one answer run."""

    over: int  # those taking `SLOW_CALL_S` or longer
    calls: int
    longest_s: float
    longest_invalid: int  # the invalid responses of the longest


def slow_calls(src: Sources, prompt: int) -> SlowCalls:
    """`MERCURY`'s generator calls with an invalid response, in the run of `prompt`.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    calls = [
        r.generated().call
        for r in answer_run(src, prompt).records
        if r.config == MERCURY and r.generated().call.invalid_responses
    ]
    seconds = [c.latency_ms / MS_PER_SECOND for c in calls]
    longest = max(calls, key=lambda c: c.latency_ms)
    over = [x for x in seconds if x >= SLOW_CALL_S]
    return SlowCalls(len(over), len(seconds), max(seconds), int(longest.invalid_responses or 0))
