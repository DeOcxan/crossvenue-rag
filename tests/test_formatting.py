"""Tests for `crossvenue_rag.formatting`."""

from __future__ import annotations

import pytest

from crossvenue_rag import formatting as fmt

M = "\u2212"


def test_metric_has_three_decimals():
    assert fmt.metric(0.62) == "0.620"
    assert fmt.metric(-0.051) == f"{M}0.051"


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.28, "+0.280"), (-0.0621, f"{M}0.062"), (0.0, "+0.000"), (-0.0001, "+0.000")],
)
def test_signed_uses_the_minus_sign_and_a_plus_for_zero(value, text):
    assert fmt.signed(value) == text


def test_interval_signs_both_bounds():
    assert fmt.interval(0.16, 0.4) == "[+0.160, +0.400]"
    assert fmt.interval(-0.14, 0.006) == f"[{M}0.140, +0.006]"


def test_p_value_has_three_decimals_and_a_floor():
    assert fmt.p_value(0.0241) == "0.024"
    assert fmt.p_value(0.0009) == "< 0.001"
    assert fmt.p_value(0.001) == "0.001"
    assert fmt.p_value(1.0) == "1.000"


def test_share_gives_the_count():
    assert fmt.share(49, 50) == "0.98 (49/50)"


def test_percent_has_one_decimal():
    assert fmt.percent(0.1083) == "10.8%"


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.14, "0.1"), (9.94, "9.9"), (10.0, "10"), (2531.3, "2,531"), (25.3, "25")],
)
def test_latency_is_an_integer_from_ten_ms(value, text):
    assert fmt.ms(value) == text


@pytest.mark.parametrize(
    ("value", "text"), [(83.4, "+83"), (-25.3, "\u221225"), (-2.54, "\u22122.5"), (-0.04, "+0.0")]
)
def test_a_latency_difference_is_signed_at_the_precision_of_a_latency(value, text):
    assert fmt.signed_ms(value) == text


def test_a_seed_a_config_value_or_an_identifier_keeps_its_own_form():
    assert fmt.own_form(20260925) == "20260925"
    assert fmt.own_form(60000) == "60000"
    assert fmt.own_form(0.5) == "0.5"


def test_a_config_value_of_a_kind_with_a_decimals_constant_is_written_at_its_decimals():
    assert fmt.own_form(0.7, fmt.CONFIG_THRESHOLD_DECIMALS) == "0.70"
    assert fmt.own_form(0.61, fmt.CONFIG_THRESHOLD_DECIMALS) == "0.61"
    assert fmt.own_form(-0.5, fmt.CONFIG_THRESHOLD_DECIMALS) == "\u22120.50"


def test_a_file_location_writes_its_line_number_as_an_identifier():
    assert fmt.location("data/questions.jsonl", 1234) == "data/questions.jsonl:1234"


def test_a_cost_is_in_dollars_at_three_decimals():
    assert fmt.cost(0.6571) == "$0.657"
    assert fmt.cost_figure(0.6571) == "0.657"


def test_a_count_has_a_thousands_separator():
    assert fmt.count(11934) == "11,934"


def test_a_number_keeps_its_decimals_and_writes_the_minus_sign_as_u2212():
    assert fmt.number(13.1, 2) == "13.10"
    assert fmt.number(-2.5, 1) == f"{M}2.5"


def test_a_plural_adds_s_or_takes_the_given_plural_and_formats_the_count():
    assert fmt.plural(1, "round") == "1 round"
    assert fmt.plural(2, "round") == "2 rounds"
    assert fmt.plural(1000, "pair") == "1,000 pairs"
    assert fmt.plural(2, "query", "queries") == "2 queries"


@pytest.mark.parametrize(
    "text",
    [
        "3 round(s)",
        "a -- b",
        "diff -0.062",
        "load [13.1, 11.17, 8.59]",
        "{'median': 4.09}",
        "p 0.0232",
        "labelled pool",
        "judgement",
        "randomisation",
        "favour",
        "behaviour",
        "No labels yet",
    ],
)
def test_the_output_defect_check_finds_each_kind(text):
    from tests.conftest import output_defects

    assert output_defects(text)


def test_formatted_text_passes_the_output_defect_check():
    from tests.conftest import output_defects

    line = (
        f"{fmt.signed(0.28)} {fmt.interval(-0.1, 0.2)} {fmt.p_value(0.0002)} "
        f"{fmt.ms(2531.3)} {fmt.cost(0.6571)} {fmt.plural(1, 'round')} behaviour_correct top-8"
    )
    assert output_defects(line) == []


def test_design_values_have_two_decimals_and_the_minus_sign():
    assert fmt.design_parameter(0.2) == "0.20"
    assert fmt.design_parameter(-0.125) == "\u22120.12"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (45, "45s"),
        (89.4, "89s"),
        (89.6, "1m"),
        (720, "12m"),
        (5399, "1h30m"),
        (7182, "2h00m"),
        (7500, "2h05m"),
    ],
)
def test_durations_switch_unit_at_ninety(seconds, text):
    assert fmt.duration(seconds) == text


def test_a_missing_value_is_a_dash():
    assert fmt.optional(None, fmt.metric) == "-"
    assert fmt.optional("", fmt.ms) == "-"
    assert fmt.optional("0.5", fmt.metric) == "0.500"


def test_fixed_decimals_have_no_separator_and_the_minus_sign():
    assert fmt.fixed(2505.25, 1) == "2505.2"
    assert fmt.fixed(-0.5, 2) == "\u22120.50"


def test_signed_takes_its_decimals():
    assert fmt.signed(12.4, 0) == "+12"
    assert fmt.signed(-0.0004, 3) == "+0.000"


def test_a_percent_figure_is_the_share_without_its_sign():
    assert fmt.percent_figure(1, 8) == "12.5"
    assert fmt.percent_figure(2, 3) == "66.7"


def test_a_cpu_count_is_pluralized_and_an_unknown_count_says_so():
    assert fmt.cpus(1) == "1 CPU" and fmt.cpus(10) == "10 CPUs"
    assert fmt.cpus(None) == "an unknown number of CPUs"


def test_a_value_that_rounds_to_zero_carries_no_minus_sign_in_any_formatter():
    assert fmt.metric(-0.0001) == "0.000"
    assert fmt.percent(-0.00001) == "0.0%"
    assert fmt.ms(-0.01) == "0.0"
    assert fmt.signed(-0.0001) == "+0.000"
    assert fmt.metric(-0.25) == "\u22120.250"


def test_a_ratio_or_mean_of_counts_takes_the_decimals_of_its_kind():
    assert fmt.ratio(2.4712, fmt.TOKENS_PER_WORD_DECIMALS) == "2.47"
    assert fmt.ratio(137.6, fmt.COST_RATIO_DECIMALS) == "138"
    assert fmt.ratio(1234.4, fmt.COUNT_MEAN_DECIMALS) == "1,234"


def test_a_call_duration_is_whole_seconds():
    assert fmt.call_seconds(242.6) == "243"


def test_a_shortened_hash_is_eight_hex_digits_and_an_ellipsis():
    assert fmt.short_hash("2825a7160c4d09bf" * 4) == "2825a716\u2026"


def test_a_model_revision_is_named_after_the_model_or_left_out():
    assert fmt.revision_text("abc123") == " at revision abc123"
    assert fmt.revision_text(None) == ""
