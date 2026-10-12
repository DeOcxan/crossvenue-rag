"""Tests for `reporting.compute.answers_costs`, on runs written under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import answers_costs
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, ANSWERS_V2, JUDGED_V1, JUDGED_V2
from crossvenue_rag.reporting.sources import Sources
from tests.generation.factories import call_json, generation_json, judge_json, record_json
from tests.reporting.conftest import CONFIGS, complete, write_csv, write_judged_run

MERCURY = answers_costs.MERCURY


def summary(cost, judge_cost=0.0, **columns):
    """One summary.csv row per configuration of `CONFIGS`: cost `cost[i]` for the i-th."""
    return [
        {
            "config": c,
            "cost_usd": cost[i],
            "cost_usd_new": cost[i] / 2,
            "judge_cost_usd": judge_cost[i] if judge_cost else 0.0,
            "latency_p50_ms": 100 * (i + 1),
            "latency_p95_ms": 200 * (i + 1),
            "output_tokens": 1000,
            "invalid_responses": i,
            "cached_calls": 0,
            **columns,
        }
        for i, c in enumerate(CONFIGS)
    ]


def write_runs(root, *, v1, v2, judged_v1, judged_v2):
    """The two answer runs and the two judged runs, each with `summary` rows."""
    for run_id, rows in (
        (ANSWERS_V1, v1),
        (ANSWERS_V2, v2),
        (JUDGED_V1, judged_v1),
        (JUDGED_V2, judged_v2),
    ):
        write_judged_run(root, run_id)
        write_csv(root / "results" / run_id / "summary.csv", complete("answer_summary", rows))
    return Sources(root)


@pytest.fixture
def cost_src(tmp_path):
    return write_runs(
        tmp_path,
        v1=summary([1.0, 2.0, 4.0, 1.0]),
        v2=summary([2.0, 2.0, 2.0, 4.0]),
        judged_v1=summary([0.0] * 4, judge_cost=[0.5, 1.0, 1.0, 3.0]),
        judged_v2=summary([0.0] * 4, judge_cost=[1.0, 1.0, 1.0, 1.0]),
    )


def test_a_cost_row_pairs_each_column_under_v1_and_v2(cost_src):
    rows = answers_costs.cost_rows(cost_src)
    assert [r.config for r in rows] == list(CONFIGS)
    row = rows[2]
    assert row.config == "mercury@venue_split"
    assert (row.generator_usd, row.judge_usd) == ((4.0, 2.0), (1.0, 1.0))
    assert (row.p50_s, row.p95_s) == ((0.3, 0.3), (0.6, 0.6))
    assert row.output_tokens == (1000, 1000) and isinstance(row.output_tokens[0], int)
    assert row.invalid_responses == 2 and isinstance(row.invalid_responses, int)


def test_a_fractional_output_token_count_is_refused(tmp_path):
    src = write_runs(
        tmp_path,
        v1=summary([1.0] * 4, output_tokens=1000.5),
        v2=summary([1.0] * 4),
        judged_v1=summary([0.0] * 4),
        judged_v2=summary([0.0] * 4),
    )
    with pytest.raises(ValueError, match=r"output_tokens cell holds 1000\.5"):
        answers_costs.cost_rows(src)


def test_judge_cost_ratios_span_every_configuration_and_prompt(cost_src):
    # v1: 0.5/1, 1/2, 1/4, 3/1; v2: 1/2, 1/2, 1/2, 1/4.
    assert answers_costs.judge_cost_ratios(cost_src) == (0.25, 3.0)


def test_the_uncached_cost_sums_the_new_cost_of_every_configuration(cost_src):
    assert answers_costs.uncached_cost(cost_src, 1) == 4.0
    assert answers_costs.uncached_cost(cost_src, 2) == 5.0


def judged(question_id, *, cached, cost=0.01, latency=1000.0, judge_error=None):
    call = call_json(cached=cached, cost_usd=cost, latency_ms=latency)
    return record_json(
        question_id=question_id, judge=judge_json(call=call), judge_error=judge_error
    )


def test_judge_calls_count_the_cost_and_latency_of_uncached_calls_only(tmp_path):
    records = [
        judged("cv-1", cached=False, cost=0.2, latency=300.0),
        judged("cv-2", cached=False, cost=0.3, latency=100.0),
        judged("cv-3", cached=True, cost=5.0, latency=1.0),
        record_json(question_id="cv-4", judge_error="invalid JSON"),
    ]
    write_judged_run(tmp_path, JUDGED_V1, records=records)
    calls = answers_costs.judge_calls(Sources(tmp_path), 1)
    assert calls.paid_usd == pytest.approx(0.5)
    assert calls.cached == 1
    assert calls.p50_s == 0.1


def test_invalid_judgements_are_counted_over_both_judged_runs(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, records=[judged("cv-1", cached=False, judge_error="bad")])
    write_judged_run(
        tmp_path,
        JUDGED_V2,
        records=[
            record_json(question_id="cv-1", judge_error="bad"),
            record_json(question_id="cv-2", judge_error="bad"),
            judged("cv-3", cached=False),
        ],
    )
    assert answers_costs.invalid_judgements(Sources(tmp_path)) == 3


def mercury(question_id, *, seconds, invalid, config=MERCURY):
    call = call_json(latency_ms=seconds * 1000, invalid_responses=invalid)
    return record_json(config, question_id, generation=generation_json(call=call))


def test_slow_calls_are_mercurys_calls_with_an_invalid_response(tmp_path):
    records = [
        mercury("cv-1", seconds=130, invalid=1),
        mercury("cv-2", seconds=200, invalid=3),
        mercury("cv-3", seconds=40, invalid=1),
        # Slow, but no invalid response: not counted.
        mercury("cv-4", seconds=500, invalid=0),
        # Another configuration: not counted.
        mercury("cv-5", seconds=900, invalid=2, config="qwen@venue_split"),
    ]
    write_judged_run(tmp_path, ANSWERS_V1, records=records)
    slow = answers_costs.slow_calls(Sources(tmp_path), 1)
    assert slow == answers_costs.SlowCalls(over=2, calls=3, longest_s=200.0, longest_invalid=3)


def test_a_call_of_exactly_the_slow_bound_is_slow(tmp_path):
    bound = answers_costs.SLOW_CALL_S
    write_judged_run(tmp_path, ANSWERS_V2, records=[mercury("cv-1", seconds=bound, invalid=1)])
    assert answers_costs.slow_calls(Sources(tmp_path), 2).over == 1


def test_a_configuration_a_summary_lacks_is_a_value_error(tmp_path):
    src = write_runs(
        tmp_path,
        v1=summary([1.0] * 4),
        v2=summary([1.0] * 4)[:3],
        judged_v1=summary([0.0] * 4),
        judged_v2=summary([0.0] * 4),
    )
    for call in (
        answers_costs.cost_rows,
        answers_costs.judge_cost_ratios,
        lambda s: answers_costs.uncached_cost(s, 2),
        lambda s: answers_costs.cached_calls(s, CONFIGS[3], 2),
    ):
        with pytest.raises(ValueError, match=f"summary.csv has no '{CONFIGS[3]}'"):
            call(src)
