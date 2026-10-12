"""Tests for `generation.answer_rows`: the per-answer and per-configuration rows of a run."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.generation.answer_rows import in_scope, per_answer_row, summary_rows
from tests.conftest import assert_round_trip, committed_of_kind
from tests.generation.factories import (
    answer_json,
    call_json,
    generation_json,
    judge_json,
    metrics_json,
    record,
)

CONFIG = "gemini@venue_split"
OTHER = "qwen@venue_split"


def judged(question_id, *, metrics=None, call=None, judge_call=None, **fields):
    """A generated and judged record of `CONFIG`."""
    return record(
        CONFIG,
        question_id,
        generation=generation_json(call=call),
        judge=judge_json(metrics=metrics, call=judge_call),
        **fields,
    )


def test_an_answer_counts_in_all_and_in_its_own_category_only():
    fact = record(category="fact")
    assert in_scope(fact, "all")
    assert in_scope(fact, "fact")
    assert not in_scope(fact, "comparison")


def test_a_judged_answer_row_carries_its_outcome_metrics_and_both_calls():
    row = per_answer_row(
        judged(
            "cv-0001",
            metrics=metrics_json(coverage=0.5, faithfulness=None),
            call=call_json(reasoning_tokens=None, cost_usd=0.02),
            judge_call=call_json(actual_model="anthropic/opus", cost_usd=0.3),
        )
    )
    assert row.answer_id == f"{CONFIG}:cv-0001"
    assert row.outcome == "full"
    assert row.refused is False
    assert row.claims == 1
    assert row.coverage == 0.5
    # A metric that does not apply and a field the call lacks are empty cells.
    assert row.faithfulness is None
    assert row.reasoning_tokens is None
    assert row.cost_usd == 0.02
    assert row.judge == "opus"
    assert row.judge_model == "anthropic/opus"
    assert row.judge_cost_usd == 0.3
    assert row.error == ""


def test_a_refusal_row_is_marked_refused():
    generation = generation_json(answer=answer_json(refused=True, claims=0))
    row = per_answer_row(record(generation=generation))
    assert row.outcome == "refused"
    assert row.refused is True
    assert row.claims == 0


def test_a_failed_answer_row_has_empty_call_outcome_and_metric_cells():
    row = per_answer_row(record(error="HTTP 500"))
    assert row.error == "HTTP 500"
    for column in ("actual_model", "input_tokens", "outcome", "claims", "coverage", "judge"):
        assert row.text(column) == ""
    assert row.judge_model == ""


def test_summary_counts_answers_errors_and_each_non_answer_outcome():
    records = [
        record(CONFIG, "cv-1"),
        record(CONFIG, "cv-2", generation=generation_json(answer=answer_json(refused=True))),
        record(CONFIG, "cv-3", generation=generation_json(answer=answer_json(asked=True))),
        record(CONFIG, "cv-4", generation=generation_json(answer=answer_json(partial=True))),
        # Flagged partial without a claim: a refusal.
        record(
            CONFIG, "cv-5", generation=generation_json(answer=answer_json(partial=True, claims=0))
        ),
        record(CONFIG, "cv-6", error="timeout"),
        record(OTHER, "cv-1"),
    ]
    (row,) = summary_rows(records, [CONFIG])
    assert row.config == CONFIG
    assert (row.answers, row.errors) == (6, 1)
    assert (row.refused, row.asked_clarification, row.partial) == (2, 1, 1)


def test_summary_means_are_per_scope_over_judged_answers_with_a_value():
    records = [
        judged("cv-1", metrics=metrics_json(coverage=1.0), category="comparison"),
        judged("cv-2", metrics=metrics_json(coverage=0.0), category="fact"),
        # No coverage: not counted in the mean.
        judged("cv-3", metrics=metrics_json(coverage=None), category="comparison"),
        # Generated and never judged: not counted either.
        record(CONFIG, "cv-4", category="comparison"),
    ]
    (row,) = summary_rows(records, [CONFIG])
    assert row.number("coverage:comparison") == 1.0
    assert row.number("coverage:all") == 0.5
    assert row.number("faithfulness:all") == 1.0


def test_a_configuration_without_judged_answers_has_no_means():
    (row,) = summary_rows([record(CONFIG, "cv-1")], [CONFIG])
    assert row.number("coverage:all") is None
    assert row.number("behaviour_correct:comparison") is None


def test_summary_sums_tokens_and_cost_and_splits_off_the_new_cost():
    records = [
        judged("cv-1", call=call_json(cost_usd=0.01, latency_ms=100.0, retries=1)),
        judged("cv-2", call=call_json(cost_usd=0.02, latency_ms=300.0, cached=True)),
        judged("cv-3", call=call_json(cost_usd=0.04, latency_ms=200.0, reasoning_tokens=None)),
    ]
    (row,) = summary_rows(records, [CONFIG])
    assert row.input_tokens == 3000
    # A call without reasoning tokens adds none.
    assert row.reasoning_tokens == 60
    assert row.cost_usd == pytest.approx(0.07)
    # The cached call's cost was paid by an earlier run.
    assert row.cost_usd_new == pytest.approx(0.05)
    assert row.judge_cost_usd == pytest.approx(0.03)
    assert row.latency_p50_ms == 200.0
    assert row.latency_p95_ms == 300.0
    # Retries of the generator and judge calls; the one cached generator call.
    assert row.http_retries == 1
    assert row.cached_calls == 1
    assert json.loads(row.params) == {"model": "google/gemini-3.8-flash"}


def test_a_configuration_without_answers_has_zero_usage_and_no_latency():
    (row,) = summary_rows([record(CONFIG, "cv-1", error="timeout")], [CONFIG])
    assert row.cost_usd == 0.0
    assert row.latency_p50_ms is None
    assert row.params == ""


def test_summary_has_one_row_per_configuration_in_the_order_given():
    records = [record(CONFIG, "cv-1"), record(OTHER, "cv-1"), record(OTHER, "cv-2")]
    rows = summary_rows(records, [OTHER, CONFIG])
    assert [(r.config, r.answers) for r in rows] == [(OTHER, 2), (CONFIG, 1)]


ANSWER_RUNS = ("_answers", "_judged")


@pytest.mark.parametrize(
    "run", committed_of_kind("per_answer.csv", *ANSWER_RUNS), ids=lambda p: p.name
)
def test_every_committed_per_answer_table_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.generation.answer_rows import PER_ANSWER_FILE, PerAnswerRow
    from crossvenue_rag.run_files import read_table

    table = read_table(run / PER_ANSWER_FILE, PerAnswerRow)
    assert_round_trip(run, "per_answer.csv", table, tmp_path)


@pytest.mark.parametrize(
    "run", committed_of_kind("summary.csv", *ANSWER_RUNS), ids=lambda p: p.name
)
def test_every_committed_answer_summary_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.generation.answer_rows import read_answer_summary

    assert_round_trip(run, "summary.csv", read_answer_summary(run), tmp_path)


@pytest.mark.parametrize("run", committed_of_kind("paired.csv", *ANSWER_RUNS), ids=lambda p: p.name)
def test_every_committed_answer_paired_table_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.generation.answer_pairs import read_config_comparisons

    assert_round_trip(run, "paired.csv", read_config_comparisons(run), tmp_path)
