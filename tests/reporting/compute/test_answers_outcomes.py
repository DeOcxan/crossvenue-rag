"""Tests for `reporting.compute.answers_outcomes`, on runs written under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import answers_outcomes as outcomes
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, ANSWERS_V2, JUDGED_V1, JUDGED_V2
from crossvenue_rag.reporting.sources import Sources
from tests.generation.factories import (
    answer_json,
    generation_json,
    judge_json,
    metrics_json,
    record_json,
)
from tests.reporting.conftest import (
    CONFIGS,
    answer_record,
    write_fixture,
    write_gold,
    write_judged_run,
    write_run,
)

GEMINI, QWEN, MERCURY, RERANK = CONFIGS


@pytest.fixture
def src(tmp_path):
    """`write_fixture`: Gemini refuses the comparison question under v1 and answers it
    in part under v2, where Mercury's claim-less partial answer counts as a refusal."""
    write_fixture(tmp_path)
    return Sources(tmp_path)


def test_a_category_expects_a_behavior_only_when_its_questions_agree(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "cv-1", "category": "fact", "expected_behavior": "answer"},
            {"id": "cv-2", "category": "ambiguous", "expected_behavior": "ask_clarification"},
            {"id": "cv-3", "category": "ambiguous", "expected_behavior": "answer"},
        ],
    )
    src = Sources(tmp_path)
    assert outcomes.expected_behaviour(src, "fact") == "answer"
    assert outcomes.expected_behaviour(src, "ambiguous") is None
    assert outcomes.expected_behaviour(src, "multihop") is None


def test_outcome_changes_pair_each_configurations_count_under_v1_and_v2(src):
    assert outcomes.outcome_counts(src, "comparison", "refused", 1) == [1, 0, 0, 0]
    assert outcomes.outcome_changes(src, "comparison", "refused") == [
        (1, 0),
        (0, 0),
        (0, 1),
        (0, 0),
    ]
    assert outcomes.outcome_changes(src, None, "partial") == [(0, 1), (0, 0), (0, 0), (0, 0)]
    # Every answer to the unanswerable question is a full answer.
    assert outcomes.outcome_counts(src, "unanswerable", "full", 2) == [1, 1, 1, 1]


def test_the_outcome_table_has_a_row_for_every_category_in_order(src):
    table = outcomes.outcome_table(src, "refused")
    assert [row.category for row in table] == list(outcomes.CATEGORIES)
    changes = [(1, 0), (0, 0), (0, 1), (0, 0)]
    assert table[0] == outcomes.OutcomeRow("comparison", 1, "answer", changes)
    assert (table[1].category, table[1].questions, table[1].expected) == (
        "unanswerable",
        1,
        "refuse",
    )
    # A category without questions: none, no expected behavior, no answers.
    assert table[3] == outcomes.OutcomeRow("fact", 0, None, [(0, 0)] * 4)


def test_no_outcome_holds_when_no_configuration_has_that_outcome(src):
    assert outcomes.no_outcome(src, "asked", 1)
    assert not outcomes.no_outcome(src, "refused", 1)
    assert outcomes.no_outcome(src, "partial", 1)
    assert not outcomes.no_outcome(src, "partial", 2)


@pytest.fixture
def asking_src(tmp_path):
    """Qwen asks on the ambiguous question under v1 and v2; Gemini also asks on the
    comparison question under v2."""
    write_gold(
        tmp_path,
        [
            {"id": "cv-0001", "expected_behavior": "answer"},
            {"id": "cv-0002", "category": "ambiguous", "expected_behavior": "ask_clarification"},
        ],
    )
    v1 = {(c, q): answer_record() for c in CONFIGS for q in ("cv-0001", "cv-0002")}
    v1[(QWEN, "cv-0002")] = answer_record(asked=True)
    v2 = {**v1, (GEMINI, "cv-0001"): answer_record(asked=True)}
    write_run(tmp_path, ANSWERS_V1, v1)
    write_run(tmp_path, ANSWERS_V2, v2)
    return Sources(tmp_path)


def test_clarifications_are_counted_per_prompt(asking_src):
    assert outcomes.clarifications(asking_src, QWEN) == (1, 1)
    assert outcomes.clarifications(asking_src, GEMINI) == (0, 1)


def test_clarifications_only_on_a_category_checks_every_v2_clarification(asking_src):
    assert outcomes.clarifications_only_on(asking_src, (QWEN,), "ambiguous")
    assert not outcomes.clarifications_only_on(asking_src, (QWEN, GEMINI), "ambiguous")
    # Configurations that never ask pass vacuously.
    assert outcomes.clarifications_only_on(asking_src, (MERCURY,), "comparison")


def test_never_clarify_holds_only_for_configurations_that_ask_nothing(asking_src):
    assert outcomes.never_clarify(asking_src, (MERCURY, RERANK))
    assert not outcomes.never_clarify(asking_src, (MERCURY, GEMINI))


def test_wrong_refusals_are_refusals_of_questions_that_expect_an_answer(src):
    # Gemini's v1 refusal and Mercury's claim-less v2 partial answer.
    assert outcomes.wrong_refusal_changes(src) == [(1, 0), (0, 0), (0, 1), (0, 0)]
    assert not outcomes.wrong_refusals_fall(src)


def test_wrong_refusals_fall_when_every_configuration_refuses_less(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001", "expected_behavior": "answer"}])
    write_run(tmp_path, ANSWERS_V1, {(c, "cv-0001"): answer_record(refused=True) for c in CONFIGS})
    write_run(tmp_path, ANSWERS_V2, {(c, "cv-0001"): answer_record() for c in CONFIGS})
    assert outcomes.wrong_refusals_fall(Sources(tmp_path))


def test_the_refusals_of_the_refusing_configuration_are_broken_down(src):
    # Gemini refused the comparison question under v1 and answered it in part under v2.
    assert outcomes.group_outcomes(src, "c") == {"full": 0, "partial": 1, "refused": 0}
    assert outcomes.group_outcomes(src, "a") == {"full": 0, "partial": 0, "refused": 0}
    assert outcomes.answered_refusals(src, QWEN) == 1
    assert outcomes.answered_refusals(src, GEMINI) == 0
    assert outcomes.citing_refusals(src) == 0
    assert outcomes.refusal_claims(src) == 0


def test_refusals_that_cite_a_chunk_or_carry_claims_are_counted(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}])
    refusal = answer_record(refused=True, claims=2)
    refusal["answer"] = "Bybit is not covered [1]."
    write_run(tmp_path, ANSWERS_V1, {(GEMINI, "cv-0001"): refusal})
    write_judged_run(tmp_path)
    src = Sources(tmp_path)
    assert outcomes.citing_refusals(src) == 1
    assert outcomes.refusal_claims(src) == 2


def test_non_refusals_are_the_category_questions_a_configuration_did_not_refuse(src):
    assert outcomes.non_refusals(src, GEMINI, "comparison", 1) == 0
    assert outcomes.non_refusals(src, GEMINI, "comparison", 2) == 1
    assert outcomes.non_refusals(src, MERCURY, "comparison", 2) == 0


def behaviour_record(question_id, value, *, config=GEMINI, category="comparison"):
    judge = judge_json(metrics=metrics_json(behaviour_correct=value))
    return record_json(config, question_id, judge=judge, category=category)


def test_judged_behaviour_sums_over_the_questions_both_runs_judged(tmp_path):
    write_judged_run(
        tmp_path,
        JUDGED_V1,
        records=[
            behaviour_record("cv-1", 1.0),
            behaviour_record("cv-2", 0.0),
            # Judged under v1 only.
            behaviour_record("cv-3", 1.0),
            behaviour_record("cv-1", 1.0, config=QWEN),
            behaviour_record("cv-4", 1.0, category="fact"),
        ],
    )
    write_judged_run(
        tmp_path,
        JUDGED_V2,
        records=[
            behaviour_record("cv-1", 1.0),
            behaviour_record("cv-2", 1.0),
            behaviour_record("cv-4", 0.0, category="fact"),
            record_json(GEMINI, "cv-3", category="comparison"),
        ],
    )
    src = Sources(tmp_path)
    assert outcomes.judged_behaviour(src, GEMINI, "comparison") == (1, 2)
    assert outcomes.judged_behaviour(src, GEMINI, "fact") == (1, 0)
    assert all(isinstance(n, int) for n in outcomes.judged_behaviour(src, GEMINI, "comparison"))


def test_a_behaviour_grade_other_than_0_or_1_is_refused(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, records=[behaviour_record("cv-1", 0.5)])
    write_judged_run(tmp_path, JUDGED_V2, records=[behaviour_record("cv-1", 1.0)])
    with pytest.raises(ValueError, match="neither 0 nor 1"):
        outcomes.judged_behaviour(Sources(tmp_path), GEMINI, "comparison")


def test_absence_claims_are_counted_among_the_v2_claims(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}, {"id": "cv-0002"}])
    absent = answer_json(claims=2, text="Bybit Does Not List a limit")
    write_run(
        tmp_path,
        ANSWERS_V2,
        {(GEMINI, "cv-0001"): absent, (QWEN, "cv-0001"): answer_json(claims=3)},
    )
    assert outcomes.absence_claims(Sources(tmp_path)) == (2, 5)


def test_the_claims_of_answers_whose_judgment_failed_are_counted_apart(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}, {"id": "cv-0002"}])
    refused = generation_json(answer=answer_json(claims=2, refused=True))
    unjudged = generation_json(answer=answer_json(claims=3))
    write_judged_run(
        tmp_path,
        JUDGED_V2,
        records=[
            record_json(GEMINI, "cv-0001", generation=refused, judge=judge_json()),
            record_json(QWEN, "cv-0001", generation=unjudged, judge_error="invalid JSON"),
        ],
    )
    assert outcomes.v2_unjudged_claims(Sources(tmp_path)) == 3


def test_an_outcome_or_question_the_runs_lack_is_a_value_error(src, tmp_path):
    with pytest.raises(ValueError, match="refusal groups has no 'd'"):
        outcomes.group_outcomes(src, "d")
    with pytest.raises(ValueError, match="v1 outcomes has no"):
        outcomes.answered_refusals(src, "absent@config")
    other = tmp_path / "other"
    write_gold(other, [{"id": "cv-0001"}])
    write_run(other, ANSWERS_V1, {(GEMINI, "cv-0001"): answer_record(refused=True, claims=0)})
    write_run(other, ANSWERS_V2, {(QWEN, "cv-0001"): answer_record()})
    write_judged_run(other)
    with pytest.raises(ValueError, match="v2 outcomes has no"):
        outcomes.group_outcomes(Sources(other), "c")


def test_a_clarification_on_a_question_the_gold_set_lacks_is_a_value_error(asking_src, tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}])
    with pytest.raises(ValueError, match="the gold set has no 'cv-0002'"):
        outcomes.clarifications_only_on(Sources(tmp_path), (QWEN,), "ambiguous")
