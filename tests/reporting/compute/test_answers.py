"""Tests for `reporting.compute.answers`, on a run written under tmp_path."""

from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.answer_rows import SCOPES
from crossvenue_rag.generation.metrics import METRICS
from crossvenue_rag.reporting.compute import answers
from crossvenue_rag.reporting.compute.estimates import Estimate
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, JUDGED_V1, JUDGED_V2
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.generation.factories import generation_json, judge_json, judgement_json, record_json
from tests.reporting.conftest import (
    CONFIGS,
    write_chunks,
    write_file,
    write_fixture,
    write_gold,
    write_judged_run,
)


def test_the_generator_settings_are_read_from_the_prompt_v1_answer_run_config(tmp_path):
    write_file(
        tmp_path,
        f"results/{ANSWERS_V1}/config.yaml",
        "generation:\n  max_invalid_retries: 3\n  generator_params: {max_tokens: 900}\n",
    )
    src = Sources(tmp_path)
    assert answers.max_invalid_retries(src) == 3
    assert answers.generator_max_tokens(src) == 900


def paired_row(a, b, metric, **values):
    row = {"a": a, "b": b, "metric": metric, "scope": "comparison"}
    return {**row, **{k: str(v) for k, v in values.items()}}


PAIR = ("gemini@venue_split", "gemini@hybrid_rerank")


def test_three_generators_answer_in_four_configurations(tmp_path):
    write_fixture(tmp_path)
    assert len(answers.generators(Sources(tmp_path))) == 3


def test_model_families_are_the_vendors_the_matrix_generators_model_ids_name(tmp_path):
    generators = {
        g: {"model": f"{vendor}/{g}"}
        for g, vendor in zip("abcd", ["v1", "v1", "v2", "v3"], strict=True)
    }
    matrix = [{"name": f"{g}@bm25", "generator": g, "retriever": "bm25"} for g in "abc"]
    config = {"generation": {"generators": generators, "matrix": matrix}}
    write_file(tmp_path, f"results/{ANSWERS_V1}/config.yaml", yaml.safe_dump(config))
    assert answers.model_families(Sources(tmp_path)) == 2


def test_the_judge_and_its_model_are_read_from_the_config(tmp_path):
    config = {"generation": {"judge": "opus", "judges": {"opus": {"model": "m/opus"}}}}
    write_judged_run(tmp_path, config=config)
    src = Sources(tmp_path)
    assert answers.judge_is(src, "opus", "m/opus")
    assert not answers.judge_is(src, "opus", "m/other")


def test_judged_comparisons_are_read_from_paired_csv(tmp_path):
    columns = {"mean_a": 0.4, "mean_b": 0.3, "holm_p": 0.02}
    v1 = [paired_row(*PAIR, "coverage", diff=0.1, low=0.05, high=0.2, **columns)]
    v2 = [
        paired_row(*PAIR, "coverage", diff=0.05, low=-0.1, high=0.2, randomisation_p=0.04),
        paired_row(*PAIR, "venue_attribution_error", diff=0, low=0, high=0, randomisation_p=0.3),
    ]
    write_judged_run(tmp_path, JUDGED_V1, paired=v1)
    write_judged_run(tmp_path, JUDGED_V2, paired=v2)
    src = Sources(tmp_path)
    assert answers.judged_means(src, PAIR, "coverage", "comparison") == (0.4, 0.3)
    assert answers.judged_interval(src, 1, PAIR, "coverage", "comparison") == Estimate(
        0.1, 0.05, 0.2
    )
    assert not answers.judged_includes_zero(src, 1, PAIR, "coverage", "comparison")
    assert answers.judged_includes_zero(src, 2, PAIR, "coverage", "comparison")
    assert answers.judged_p(src, 1, PAIR, "coverage", "comparison", "holm_p") == 0.02
    family = ("coverage", "venue_attribution_error")
    # Holm: the smaller p of two is doubled.
    assert answers.judged_holm(src, 2, PAIR, family, "comparison")[0] == 0.08


def test_the_primary_holm_p_is_adjusted_within_the_runs_primary_metrics(tmp_path):
    v2 = [
        paired_row(*PAIR, "coverage", randomisation_p=0.04),
        paired_row(*PAIR, "venue_attribution_error", randomisation_p=0.3),
    ]
    config = {"generation": {"primary_metrics": ["venue_attribution_error", "coverage"]}}
    write_judged_run(tmp_path, JUDGED_V2, paired=v2, config=config)
    src = Sources(tmp_path)
    assert answers.primary_holm(src, 2, PAIR, "coverage", "comparison") == 0.08
    with pytest.raises(ValueError, match="not a primary metric"):
        answers.primary_holm(src, 2, PAIR, "behaviour_correct", "comparison")


def judged_record(cited_chunk):
    generation = generation_json(chunk_ids=(cited_chunk,))
    judge = judge_json(judgement=judgement_json(venue_correct=False))
    return record_json("gemini@venue_split", generation=generation, judge=judge)


def test_misattributions_are_wrappers_when_they_cite_ccxt_on_the_named_exchange(tmp_path):
    write_chunks(
        tmp_path,
        [
            make_chunk("b1", "binance", "x"),
            replace(make_chunk("x1", "ccxt", "y"), doc_path="wiki/exchanges/binance.md"),
            make_chunk("y1", "bybit", "z"),
        ],
    )
    write_judged_run(tmp_path, JUDGED_V1, records=[judged_record("x1")])
    write_judged_run(tmp_path, JUDGED_V2, records=[])
    assert answers.misattributions_are_wrappers(Sources(tmp_path))
    assert answers.misattributed_count(Sources(tmp_path)) == 1
    write_judged_run(tmp_path, JUDGED_V2, records=[judged_record("y1")])
    assert not answers.misattributions_are_wrappers(Sources(tmp_path))
    assert answers.misattributed_count(Sources(tmp_path)) == 2


def test_the_answer_matrix_pairs_every_configuration_on_every_metric_and_scope(tmp_path):
    write_fixture(tmp_path)
    src = Sources(tmp_path)
    # 4 configurations of 3 generators: 6 pairs.
    assert answers.configurations(src) == list(CONFIGS)
    assert answers.generators(src) == ["gemini", "qwen", "mercury"]
    assert answers.refusing_config(src) == "gemini@venue_split"
    assert answers.configuration_pairs(src) == 6
    assert answers.paired_comparison_count(src) == 6 * len(METRICS) * len(SCOPES)
    # 48 x 0.05 = 2.4 false positives expected when every null holds.
    assert answers.expected_false_positives(src) == 2


def test_answers_and_refusals_of_the_prompt_v1_run(tmp_path):
    write_fixture(tmp_path)
    src = Sources(tmp_path)
    assert answers.answer_count(src) == 8
    assert (
        answers.outcome_count(src, "gemini@venue_split", "comparison", "refused", answers.PROMPT_V1)
        == 1
    )
    assert (
        answers.outcome_count(src, "qwen@venue_split", "comparison", "refused", answers.PROMPT_V1)
        == 0
    )
    assert answers.comparison_refusals(src, "gemini@venue_split") == 1


def test_an_answer_the_run_does_not_hold_is_a_value_error(tmp_path):
    write_fixture(tmp_path)
    with pytest.raises(ValueError, match="has no answer of gemini@venue_split to cv-9999"):
        answers.answer(Sources(tmp_path), "gemini@venue_split", "cv-9999")


def test_refusals_are_flattened_and_filtered_by_their_context(tmp_path):
    write_fixture(tmp_path)
    # The refusal's context, chunk c1, is Binance's; the question names Binance and Bybit.
    write_chunks(tmp_path, [make_chunk("c1", "binance", "x")])
    src = Sources(tmp_path)
    assert answers.refusal_ids(src) == ["cv-0001"]
    assert answers.refusals_with_every_venue(src) == []
    # No gold chunk at all in the context: group c.
    assert len(answers.refusal_groups(src)["a"]) == 0
    assert len(answers.refusal_groups(src)["c"]) == 1


def test_a_partial_answer_without_a_claim_is_claimless_under_prompt_v2(tmp_path):
    write_fixture(tmp_path)
    src = Sources(tmp_path)
    # Gemini's partial answer states a claim; Mercury's states none.
    assert answers.claimless_partial_configs(src) == ["mercury@venue_split"]
    assert answers.claimless_partials_all_from(src, "mercury@venue_split")
    assert not answers.claimless_partials_all_from(src, "gemini@venue_split")


def test_the_generator_temperature_is_read_from_the_prompt_v1_answer_run_config(tmp_path):
    config = f"results/{ANSWERS_V1}/config.yaml"
    write_file(tmp_path, config, "generation: {generator_params: {temperature: 0.3}}\n")
    assert answers.generator_temperature(Sources(tmp_path)) == 0.3
    write_file(tmp_path, config, "generation: {generator_params: {temperature: null}}\n")
    with pytest.raises(ValueError, match="temperature"):
        answers.generator_temperature(Sources(tmp_path))


def test_the_retried_statuses_are_the_clients_in_ascending_order():
    from crossvenue_rag.retries import RETRY_STATUSES

    statuses = answers.retried_statuses()
    assert statuses == sorted(statuses) and set(statuses) == RETRY_STATUSES


def test_a_prompt_version_or_paired_row_the_runs_lack_is_a_value_error(tmp_path):
    write_judged_run(tmp_path, JUDGED_V1, paired=[paired_row(*PAIR, "coverage")])
    src = Sources(tmp_path)
    with pytest.raises(ValueError, match="answer runs by prompt has no 3"):
        answers.answer_run(src, 3)
    with pytest.raises(ValueError, match="judged runs by prompt has no 3"):
        answers.judged_run(src, 3)
    for call in (answers.judged_interval, answers.judged_role, answers.judged_questions):
        with pytest.raises(ValueError, match=r"paired\.csv has no"):
            call(src, 1, PAIR, "coverage", "all")
    with pytest.raises(ValueError, match=r"paired\.csv has no"):
        answers.judged_means(src, PAIR, "coverage", "all")


def test_a_refused_question_the_gold_set_or_the_chunks_lack_is_a_value_error(tmp_path):
    write_fixture(tmp_path)
    write_gold(tmp_path, [{"id": "cv-0002"}])
    src = Sources(tmp_path)
    no_question = "the gold set has no 'cv-0001'"
    with pytest.raises(ValueError, match=no_question):
        answers.outcome_count(src, "gemini@venue_split", "comparison", "refused", answers.PROMPT_V1)
    with pytest.raises(ValueError, match=no_question):
        answers.wrong_refusals(src, "gemini@venue_split")
    with pytest.raises(ValueError, match=no_question):
        answers.refusal_groups(src)
    with pytest.raises(ValueError, match="the gold set has no 'cv-9999'"):
        answers.gold_less_venue(src, "cv-9999")
    write_gold(tmp_path, [{"id": "cv-0001", "gold_chunk_ids": ["c1"]}, {"id": "cv-0002"}])
    write_chunks(tmp_path, [make_chunk("c2", "binance", "x")])
    src = Sources(tmp_path)
    with pytest.raises(ValueError, match=r"chunks\.jsonl has no 'c1'"):
        answers.refusal_groups(src)
    with pytest.raises(ValueError, match=r"chunks\.jsonl has no 'c1'"):
        answers.context_venues(src, "cv-0001")


def test_a_refusal_group_other_than_a_b_or_c_is_a_value_error(tmp_path):
    write_fixture(tmp_path)
    write_chunks(tmp_path, [make_chunk("c1", "binance", "x")])
    with pytest.raises(ValueError, match="refusal groups has no 'd'"):
        answers.gold_less_refusals(Sources(tmp_path), "d")


def test_a_judged_run_config_without_a_primary_pair_is_a_user_error_naming_it(tmp_path):
    write_judged_run(tmp_path, config={"generation": {"matrix": []}})
    with pytest.raises(UserError, match=r"config\.yaml sets no generation\.primary_pair"):
        answers.primary_pair(Sources(tmp_path))
