"""Tests for `reporting.compute.round3`, on committed datasets and runs or on `tmp_path` files."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.dataset.journal import rounds_path
from crossvenue_rag.reporting.compute import round3
from crossvenue_rag.reporting.run_ids import D23_ANSWER_CHUNKS, GOLD_ADJUDICATED
from crossvenue_rag.reporting.sources import Sources
from tests.published import ROOT
from tests.reporting.conftest import write_file, write_gold


@pytest.fixture(scope="module")
def src() -> Sources:
    return Sources(ROOT)


def test_the_gold_change_is_counted_in_chunks_and_questions(src):
    change = round3.gold_change(src)
    assert (change.before, change.after) == (282, 183)
    assert (change.added, change.removed) == (28, 127)
    assert change.before + change.added - change.removed == change.after
    assert (change.questions_changed, change.questions) == (56, 114)


def test_the_questions_with_gold_move_as_deviation_d27_lists(src):
    assert round3.left_gold(src) == ["cv-0006", "cv-0068", "cv-0092", "cv-0130"]
    assert round3.joined_gold(src) == ["cv-0132"]
    assert round3.stayed_empty(src) == ["cv-0067", "cv-0088", "cv-0090"]
    assert len(round3.shared_gold_questions(src)) == 88
    assert round3.flagged(src) == ["cv-0119"]


def test_rejudging_counts_faithfulness_and_attribution_changes(src):
    first = round3.rejudged(src.judged_v1, src.round3_judged_v1)
    second = round3.rejudged(src.judged_v2, src.round3_judged_v2)
    assert (first.answers, first.faithfulness, first.attribution) == (219, 3, 0)
    assert (second.answers, second.faithfulness, second.attribution) == (218, 13, 0)
    assert (first.first_verdicts, second.first_verdicts) == (1, 2)


def test_the_answer_primary_row_is_the_primary_coverage_row(src):
    row = round3.answer_primary_row(src, src.judged_v1)
    assert (row.a, row.b) == ("gemini@venue_split", "gemini@hybrid_rerank")
    assert (row.metric, row.scope, row.role) == ("coverage", "comparison", "primary")


def test_unjudged_at_k_has_one_cell_per_method_and_cutoff(src):
    assert round3.unjudged_mean_cells(src) == 18


def test_the_agreement_holds_only_the_pairs_both_rounds_judged(src):
    agreement = round3.rounds_agreement(src)
    assert agreement.total.pairs == 3382
    assert sum(v.pairs for _, v in agreement.by_category) == agreement.total.pairs
    assert sum(v.first_only for _, v in agreement.by_category) == agreement.total.first_only


def test_the_pace_of_a_round_reads_its_stamps(src):
    third = round3.pace(src, 3)
    assert (third.questions, third.decisions) == (114, 3385)
    assert third.session_seconds == 4476
    assert third.per_question == 30
    assert round3.pace(src, 2).questions == 25
    assert round3.pace(src, 1).longest_gap_seconds > round3.pace(src, 1).per_question


def test_the_median_pace_of_a_round_is_a_measured_gap(src):
    second = round3.pace(src, 2)
    assert second.per_question == 21
    assert second.per_decision == pytest.approx(0.68)


def test_round_1_counts_the_candidates_its_pools_showed_before_the_id_migration(src):
    assert round3.pace(src, 1).decisions == 2850


def test_retrieval_is_the_same_and_only_the_means_of_unjudged_move(src):
    assert round3.gold_free_equal(src)
    assert round3.unjudged_equal(src)
    assert round3.unjudged_means_differ(src) == 15


def test_the_judged_runs_change_the_rows_the_findings_list(src):
    v1 = round3.answer_changes(src.judged_v1, src.round3_judged_v1)
    v2 = round3.answer_changes(src.judged_v2, src.round3_judged_v2)
    assert (len(v1), len(v2)) == (7, 11)
    assert round3.tested_answer_rows(src.judged_v1) == 48


def test_the_cost_counts_the_calls_the_cache_did_not_serve(src):
    assert round(round3.new_judge_cost(src.round3_judged_v1), 3) == 5.985
    assert round(round3.new_judge_cost(src.round3_judged_v2), 3) == 6.475


def test_an_uncached_judge_call_that_records_no_cost_is_an_error(tmp_path):
    from crossvenue_rag.reporting.sources import AnswerRun
    from tests.generation.factories import call_json, judge_json, record_json
    from tests.reporting.conftest import write_judged_run

    cached = record_json(judge=judge_json(call=call_json(cost_usd=None, cached=True)))
    paid = record_json("qwen@venue_split", judge=judge_json(call=call_json(cost_usd=0.25)))
    write_judged_run(tmp_path, "run", records=[cached, paid])
    assert round3.new_judge_cost(AnswerRun(tmp_path / "results" / "run")) == 0.25
    unpaid = record_json("mercury@venue_split", judge=judge_json(call=call_json(cost_usd=None)))
    write_judged_run(tmp_path, "run", records=[cached, paid, unpaid])
    with pytest.raises(ValueError, match="records no cost"):
        round3.new_judge_cost(AnswerRun(tmp_path / "results" / "run"))


def test_round_3_counts_the_chunks_round_1_judged_plus_the_added_ones(tmp_path):
    write_gold(
        tmp_path,
        [{"id": "q1", "category": "fact", "qtype": "fact"}, {"id": "q2", "category": "noisy"}],
        path=GOLD_ADJUDICATED,
    )
    pool = {"pool": ["a", "b"], "relevant": [], "annotated_at": "2026-09-24T12:00:00Z"}
    write_file(
        tmp_path,
        str(rounds_path(GOLD_ADJUDICATED)),
        json.dumps({"q1": {"1": pool}, "q2": {"1": pool}}),
    )
    write_file(
        tmp_path,
        D23_ANSWER_CHUNKS,
        '{"id": "q1", "chunk_ids": ["a", "x"]}\n{"id": "q2", "chunk_ids": []}\n',
    )
    src = Sources(tmp_path)
    assert round3.round3_added(src) == {"q1": ["a", "x"]}
    sizes = round3.round3_sizes(src)
    assert list(sizes) == ["fact", "noisy"]
    assert (sizes["fact"].questions, sizes["fact"].decisions) == (1, 3)
    assert (sizes["noisy"].questions, sizes["noisy"].decisions) == (1, 2)
