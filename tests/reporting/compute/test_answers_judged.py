"""Tests for `reporting.compute.answers_judged`, on judged runs written under tmp_path."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import pytest

from crossvenue_rag.reporting.compute import answers_judged as judged
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, ANSWERS_V2, JUDGED_V1, JUDGED_V2
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.generation.factories import (
    answer_json,
    generation_json,
    judge_json,
    judgement_json,
    metrics_json,
    record_json,
)
from tests.reporting.conftest import (
    CONFIGS,
    complete,
    write_chunks,
    write_csv,
    write_gold,
    write_judged_run,
    write_sources,
)

GEMINI, QWEN, MERCURY, RERANK = CONFIGS
# The primary pair the fixture config pre-registers (`generation.primary_pair`).
PRIMARY = (GEMINI, RERANK)


def paired(a, b, metric, **fields):
    """A paired.csv row of a against b; `fields` replace role, holm_p, mean_a and mean_b."""
    values = {"role": "secondary", "holm_p": 0.5, "mean_a": 0.5, "mean_b": 0.4, **fields}
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "scope": "comparison",
        "n": 50,
        "mean_a": values["mean_a"],
        "mean_b": values["mean_b"],
        "diff": values["mean_a"] - values["mean_b"],
        "low": -0.1,
        "high": 0.3,
        "randomisation_p": values["holm_p"] / 2,
        "holm_p": values["holm_p"],
        "role": values["role"],
    }


def every_pair(overrides=None):
    """Coverage and `behaviour_correct` rows of every pair of `CONFIGS`; coverage of
    `PRIMARY` is the primary comparison."""
    rows = []
    for a, b in combinations(CONFIGS, 2):
        role = "primary" if (a, b) == PRIMARY else "secondary"
        rows.append(paired(a, b, "coverage", role=role, **(overrides or {}).get((a, b), {})))
        rows.append(paired(a, b, "behaviour_correct"))
    return rows


def test_summary_rows_read_the_v1_judged_summary(tmp_path):
    write_judged_run(tmp_path, ANSWERS_V1)
    write_judged_run(tmp_path, JUDGED_V1)
    coverage = {GEMINI: 0.4, QWEN: 0.7, MERCURY: 0.1, RERANK: 0.5}
    rows = [
        {"config": c, "coverage:comparison": v, "behaviour_correct:all": 1}
        for c, v in coverage.items()
    ]
    write_csv(tmp_path / "results" / JUDGED_V1 / "summary.csv", complete("answer_summary", rows))
    src = Sources(tmp_path)
    columns = ("coverage:comparison", "behaviour_correct:all")
    assert judged.judged_summary_rows(src, columns) == [(c, [v, 1.0]) for c, v in coverage.items()]


def test_a_configuration_is_ahead_of_every_pair_only_with_each_difference_significant(tmp_path):
    # Every row of the fixture has a - b = +0.1: GEMINI, first in every pair it is in, leads.
    significant = {pair: {"holm_p": 0.001} for pair in combinations(CONFIGS, 2)}
    write_judged_run(tmp_path, JUDGED_V1, paired=every_pair(significant))
    src = Sources(tmp_path)
    assert judged.ahead_of_every_pair(src, GEMINI, "coverage", "comparison")
    assert not judged.ahead_of_every_pair(src, QWEN, "coverage", "comparison")
    write_judged_run(tmp_path / "other", JUDGED_V1, paired=every_pair())
    assert not judged.ahead_of_every_pair(
        Sources(tmp_path / "other"), GEMINI, "coverage", "comparison"
    )


def test_ahead_of_every_pair_on_a_metric_no_pair_has_is_a_value_error(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, paired=every_pair())
    with pytest.raises(ValueError, match=r"no pair of .* on coverge over comparison"):
        judged.ahead_of_every_pair(Sources(tmp_path), GEMINI, "coverge", "comparison")


def test_primary_rows_read_the_primary_pair_of_each_metric(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, paired=every_pair())
    (row,) = judged.primary_rows(Sources(tmp_path), ("coverage",), "comparison")
    assert (row.metric, row.questions) == ("coverage", 50)
    means = (row.mean_a, row.mean_b, row.randomisation_p, row.holm_p)
    assert means == pytest.approx((0.5, 0.4, 0.25, 0.5))
    difference = (row.difference.value, row.difference.low, row.difference.high)
    assert difference == pytest.approx((0.1, -0.1, 0.3))


def test_means_zero_needs_both_means_at_zero(tmp_path):
    rows = [
        paired(GEMINI, QWEN, "venue_attribution_error", mean_a=0, mean_b=0),
        paired(GEMINI, MERCURY, "venue_attribution_error", mean_a=0, mean_b=0.1),
    ]
    write_judged_run(tmp_path, JUDGED_V1, paired=rows)
    src = Sources(tmp_path)
    assert judged.means_zero(src, (GEMINI, QWEN), "venue_attribution_error", "comparison")
    assert not judged.means_zero(src, (GEMINI, MERCURY), "venue_attribution_error", "comparison")


def test_pair_rows_cover_every_pair_and_leave_out_the_primary_coverage(tmp_path):
    write_judged_run(tmp_path, ANSWERS_V1)
    write_judged_run(tmp_path, JUDGED_V1, paired=every_pair())
    rows = judged.pair_rows(Sources(tmp_path), "comparison")
    assert [(r.a, r.b) for r in rows] == list(combinations(CONFIGS, 2))
    # The primary comparison is reported elsewhere: no coverage values here.
    primary = [r for r in rows if r.coverage is None]
    assert [(r.a, r.b) for r in primary] == [PRIMARY]
    behaviour, coverage = primary[0].behaviour, rows[0].coverage
    assert coverage is not None
    for tested in (behaviour, coverage):
        d = tested.difference
        values = (d.value, d.low, d.high, tested.holm_p)
        assert values == pytest.approx((0.1, -0.1, 0.3, 0.5))


def test_any_comparison_below_the_level_primary_included_is_significant(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, paired=every_pair())
    assert not judged.any_pair_significant(Sources(tmp_path), "coverage")
    small_primary = every_pair({PRIMARY: {"holm_p": 0.001}})
    write_judged_run(tmp_path, JUDGED_V1, paired=small_primary)
    assert judged.any_pair_significant(Sources(tmp_path), "coverage")
    small_secondary = every_pair({(QWEN, MERCURY): {"holm_p": 0.04}})
    write_judged_run(tmp_path, JUDGED_V1, paired=small_secondary)
    assert judged.any_pair_significant(Sources(tmp_path), "coverage")
    assert not judged.any_pair_significant(Sources(tmp_path), "behaviour_correct")


def test_a_behaviour_difference_is_significant_only_below_the_level_after_holm(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, paired=every_pair())
    assert not judged.behaviour_significant(Sources(tmp_path), QWEN, MERCURY, "comparison")
    rows = [
        paired(QWEN, MERCURY, "behaviour_correct", holm_p=0.04)
        if (r["a"], r["b"], r["metric"]) == (QWEN, MERCURY, "behaviour_correct")
        else r
        for r in every_pair()
    ]
    write_judged_run(tmp_path, JUDGED_V1, paired=rows)
    assert judged.behaviour_significant(Sources(tmp_path), QWEN, MERCURY, "comparison")


def coverage_record(question_id, coverage, *, refused=False, config=GEMINI, category="comparison"):

    claims = 0 if refused else 1
    answer = answer_json(refused=refused, claims=claims)
    judgement = judgement_json(claims=claims)
    return record_json(
        config,
        question_id,
        generation=generation_json(answer=answer),
        judge=judge_json(judgement=judgement, metrics=metrics_json(coverage=coverage)),
        category=category,
    )


def test_coverage_is_averaged_apart_for_refusals_and_answers(tmp_path):
    records = [
        coverage_record("cv-1", 0.5, refused=True),
        coverage_record("cv-2", 0.0, refused=True),
        coverage_record("cv-3", 1.0),
        coverage_record("cv-4", 0.5),
        # Another configuration, another category, an unjudged answer: left out.
        coverage_record("cv-5", 0.0, config=QWEN),
        coverage_record("cv-6", 0.0, category="fact"),
        record_json(GEMINI, "cv-7", category="comparison"),
    ]
    write_judged_run(tmp_path, JUDGED_V1, records=records)
    src = Sources(tmp_path)
    assert judged.coverage_by_outcome(src, GEMINI) == judged.CoverageByOutcome(
        refused=judged.CoverageGroup(2, 0.25), answered=judged.CoverageGroup(2, 0.75)
    )
    assert judged.refusals_stating_facts(src, GEMINI) == 1


def test_a_configuration_that_never_refused_has_no_mean_for_its_refusals(tmp_path):
    records = [coverage_record("cv-1", 1.0), coverage_record("cv-2", 0.5)]
    write_judged_run(tmp_path, JUDGED_V1, records=records)
    by_outcome = judged.coverage_by_outcome(Sources(tmp_path), GEMINI)
    assert by_outcome.refused == judged.CoverageGroup(0, None)
    assert by_outcome.answered == judged.CoverageGroup(2, 0.75)


def test_a_configuration_that_refused_every_question_has_no_mean_for_its_answers(tmp_path):
    records = [
        coverage_record("cv-1", 0.5, refused=True),
        coverage_record("cv-2", 0.0, refused=True),
    ]
    write_judged_run(tmp_path, JUDGED_V1, records=records)
    by_outcome = judged.coverage_by_outcome(Sources(tmp_path), GEMINI)
    assert by_outcome.refused == judged.CoverageGroup(2, 0.25)
    assert by_outcome.answered == judged.CoverageGroup(0, None)


def test_a_judged_comparison_answer_without_coverage_is_an_error(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, records=[coverage_record("cv-1", None, refused=True)])
    with pytest.raises(ValueError, match="has no coverage"):
        judged.refusals_stating_facts(Sources(tmp_path), GEMINI)


def misattributed(config, question_id, *, venue, cited, partial=False):
    """An answer with one claim naming `venue` and citing chunk `cited`, judged misattributed."""
    return record_json(
        config,
        question_id,
        generation=generation_json(
            answer=answer_json(partial=partial, venue=venue), chunk_ids=(cited,)
        ),
        judge=judge_json(judgement=judgement_json(venue_correct=False)),
    )


@pytest.fixture
def mis_src(tmp_path):
    """Chunks of Binance, Bybit and CCXT; under v2 Gemini's partial answer names Binance
    citing CCXT, Qwen's names Bybit citing Binance, Mercury's partial answer is correct."""
    write_chunks(
        tmp_path,
        [
            make_chunk("b1", "binance", "x"),
            make_chunk("y1", "bybit", "y"),
            replace(make_chunk("x1", "ccxt", "z"), doc_path="wiki/exchanges/binance.md"),
        ],
    )
    write_gold(tmp_path, [{"id": "cv-0001"}, {"id": "cv-0002", "category": "unanswerable"}])
    write_judged_run(
        tmp_path, JUDGED_V1, records=[misattributed(GEMINI, "cv-0001", venue="bybit", cited="b1")]
    )
    partial_ok = record_json(
        MERCURY,
        "cv-0001",
        generation=generation_json(answer=answer_json(partial=True)),
        judge=judge_json(),
    )
    records = [
        misattributed(GEMINI, "cv-0001", venue="Binance", cited="x1", partial=True),
        misattributed(QWEN, "cv-0002", venue="bybit", cited="b1"),
        partial_ok,
    ]
    write_judged_run(tmp_path, JUDGED_V2, records=records)
    return Sources(tmp_path)


def test_attribution_rows_count_misattributions_by_kind_per_prompt(mis_src):
    rows = judged.attribution_rows(mis_src)
    assert [(r.config, r.prompt) for r in rows] == [(c, p) for p in ("v1", "v2") for c in CONFIGS]

    def counts(summary):
        kinds = summary.misattributed
        cells = [summary.answers, summary.claims, kinds["cross_exchange"], kinds["wrapper"]]
        return cells, summary.total_error

    by_key = {(r.config, r.prompt): counts(r.summary) for r in rows}
    assert by_key[(GEMINI, "v1")] == ([1, 1, 1, 0], 1.0)
    assert by_key[(GEMINI, "v2")] == ([1, 1, 0, 1], 1.0)
    assert by_key[(QWEN, "v2")] == ([1, 1, 1, 0], 1.0)
    assert by_key[(MERCURY, "v2")] == ([1, 1, 0, 0], 0.0)
    assert by_key[(RERANK, "v2")] == ([0, 0, 0, 0], None)


def test_misattributed_claims_are_those_of_v2_the_judge_marks(mis_src):
    mis = judged.misattributed_v2(mis_src)
    assert [(m.answer_id, m.cited) for m in mis] == [
        (f"{GEMINI}:cv-0001", {"ccxt"}),
        (f"{QWEN}:cv-0002", {"binance"}),
    ]
    # Two claims on two questions, one of them unanswerable.
    assert judged.misattributed_questions(mis_src) == judged.MisattributedQuestions(2, 2, 1)


def test_a_partial_answer_is_ccxt_alone_when_its_misattributed_claim_cites_only_ccxt(mis_src):
    # Gemini's and Mercury's partial answers; Gemini's is misattributed, citing CCXT.
    assert judged.partial_misattributed(mis_src) == judged.PartialMisattributed(2, 1, 1)


def partial_answer(config, question_id, *, venue, cited, partial=True):
    generation = generation_json(
        answer=answer_json(partial=partial, venue=venue), chunk_ids=(cited,)
    )
    return record_json(config, question_id, generation=generation)


def test_a_partial_answer_must_cite_the_exchange_it_names_or_ccxt(tmp_path):
    write_sources(tmp_path, ["binance", "bybit", "ccxt"])
    write_chunks(
        tmp_path,
        [make_chunk("b1", "binance", "x"), make_chunk("x1", "ccxt", "z")],
    )
    ok = [
        partial_answer(GEMINI, "cv-1", venue="Binance", cited="b1"),
        partial_answer(QWEN, "cv-1", venue="binance", cited="x1"),
        # A venue that is no exchange, and a full answer: not checked.
        partial_answer(MERCURY, "cv-1", venue="general", cited="b1"),
        partial_answer(RERANK, "cv-1", venue="bybit", cited="b1", partial=False),
    ]
    write_judged_run(tmp_path, ANSWERS_V2, records=ok)
    assert judged.partial_exchange_citations_ok(Sources(tmp_path))
    wrong = [*ok, partial_answer(GEMINI, "cv-2", venue="Bybit", cited="b1")]
    write_judged_run(tmp_path, ANSWERS_V2, records=wrong)
    assert not judged.partial_exchange_citations_ok(Sources(tmp_path))


def test_the_exchanges_are_the_venues_of_the_sources_file_but_ccxt(tmp_path):
    write_sources(tmp_path, ["binance", "okx", "ccxt"])
    write_chunks(tmp_path, [make_chunk("b1", "binance", "x")])
    misattributed = partial_answer(GEMINI, "cv-1", venue="OKX", cited="b1")
    write_judged_run(tmp_path, ANSWERS_V2, records=[misattributed])
    src = Sources(tmp_path)
    assert judged.exchanges(src) == {"binance", "okx"}
    assert not judged.partial_exchange_citations_ok(src)


def test_attribution_rows_give_each_pair_on_both_scopes_and_count_intervals_at_zero(tmp_path):
    rows = []
    for a, b in combinations(CONFIGS, 2):
        for scope, n in (("comparison", 20), ("all", 60)):
            row = paired(a, b, "venue_attribution_error")
            rows.append({**row, "scope": scope, "n": n, "low": 0.0, "high": 0.1})
    rows[1] = {**rows[1], "low": 0.01}
    write_judged_run(tmp_path, ANSWERS_V1)
    write_judged_run(tmp_path, JUDGED_V1, paired=rows)
    src = Sources(tmp_path)
    table = judged.attribution_pair_rows(src)
    assert [(r.a, r.b) for r in table] == list(combinations(CONFIGS, 2))
    assert (table[0].comparison_questions, table[0].all_questions) == (20, 60)
    assert table[0].all.difference.low == 0.01
    # An interval that ends at 0 holds 0; the first pair's interval on all questions is above it.
    assert judged.attribution_pairs_touching_zero(src, "comparison") == len(table)
    assert judged.attribution_pairs_touching_zero(src, "all") == len(table) - 1


def test_the_answers_behind_a_mean_are_those_that_carry_the_metric(tmp_path):
    records = [
        coverage_record("cv-1", 0.5),
        coverage_record("cv-2", None),
        coverage_record("cv-3", 1.0, category="fact"),
        coverage_record("cv-4", 1.0, config=QWEN),
        record_json(GEMINI, "cv-5", category="comparison"),
    ]
    write_judged_run(tmp_path, ANSWERS_V1)
    write_judged_run(tmp_path, JUDGED_V1, records=records)
    src = Sources(tmp_path)
    assert judged.metric_answer_range(src, "coverage", "comparison")[1] == 1
    assert judged.metric_answer_range(src, "coverage", "all")[1] == 2


def test_a_configuration_or_paired_row_the_judged_run_lacks_is_a_value_error(tmp_path):
    write_judged_run(tmp_path, ANSWERS_V1)
    write_judged_run(tmp_path, JUDGED_V1, paired=[paired(GEMINI, QWEN, "coverage")])
    rows = [{"config": GEMINI, "coverage:comparison": 0.4}]
    write_csv(tmp_path / "results" / JUDGED_V1 / "summary.csv", complete("answer_summary", rows))
    src = Sources(tmp_path)
    with pytest.raises(ValueError, match=f"summary.csv has no '{QWEN}'"):
        judged.judged_summary(src, 1, QWEN, "coverage:comparison")
    for call in (
        lambda: judged.primary_rows(src, ("coverage",), "comparison"),
        lambda: judged.means_zero(src, PRIMARY, "coverage", "comparison"),
        lambda: judged.pair_rows(src, "comparison"),
        lambda: judged.attribution_pair_rows(src),
        lambda: judged.behaviour_significant(src, GEMINI, QWEN, "comparison"),
    ):
        with pytest.raises(ValueError, match=r"paired\.csv has no"):
            call()


def test_a_cited_chunk_or_a_question_the_sources_lack_is_a_value_error(mis_src, tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}])
    with pytest.raises(ValueError, match="the gold set has no 'cv-0002'"):
        judged.misattributed_questions(Sources(tmp_path))
    write_chunks(tmp_path, [make_chunk("y1", "bybit", "y")])
    with pytest.raises(ValueError, match=r"chunks\.jsonl has no 'x1'"):
        judged.misattributed_v2(Sources(tmp_path))
    write_sources(tmp_path, ["binance", "bybit", "ccxt"])
    write_judged_run(
        tmp_path, ANSWERS_V2, records=[partial_answer(GEMINI, "cv-1", venue="Binance", cited="b1")]
    )
    with pytest.raises(ValueError, match=r"chunks\.jsonl has no 'b1'"):
        judged.partial_exchange_citations_ok(Sources(tmp_path))
