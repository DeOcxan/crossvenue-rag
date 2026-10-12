"""Tests for `evaluation.mcnemar_power`: the analytic power model of the exact McNemar test."""

from __future__ import annotations

import math

import pytest

from crossvenue_rag.evaluation import mcnemar_power as model
from crossvenue_rag.evaluation.paired import mcnemar_p
from crossvenue_rag.evaluation.power import SIGNIFICANCE_LEVEL


def test_the_binomial_pmf_is_the_probability_of_k_successes_in_n():
    assert model.binomial_pmf(1, 2, 0.5) == pytest.approx(0.5)
    assert model.binomial_pmf(0, 3, 0.1) == pytest.approx(0.729)


def test_rejections_are_the_splits_at_or_below_alpha():
    assert model.rejections(5) == ()
    assert model.rejections(6) == (0, 6)


def test_the_mcnemar_power_grows_with_the_difference_and_the_size():
    assert model.mcnemar_power(10, 0.5, 0.0) < 0.05
    assert model.mcnemar_power(50, 0.25, 0.10) < model.mcnemar_power(50, 0.25, 0.20)
    assert model.mcnemar_power(50, 0.25, 0.20) < model.mcnemar_power(90, 0.25, 0.20)
    # Every discordant question favors one method: power is P(enough discordant ones).
    assert model.mcnemar_power(20, 0.5, 0.5) == pytest.approx(
        sum(model.binomial_pmf(d, 20, 0.5) for d in range(6, 21))
    )


def test_the_detectable_delta_is_the_first_on_the_grid_to_reach_the_target():
    delta = model.detectable_delta(50, 0.25)
    assert model.mcnemar_power(50, 0.25, delta) >= 0.80
    assert model.mcnemar_power(50, 0.25, round(delta - 0.01, 2)) < 0.80
    # Five questions can never be significant.
    assert math.isnan(model.detectable_delta(5, 0.25))


def test_the_questions_needed_are_the_first_size_to_reach_the_target():
    size = model.questions_needed(0.25, 0.20)
    assert model.mcnemar_power(size, 0.25, 0.20) >= 0.80
    assert model.mcnemar_power(size - 1, 0.25, 0.20) < 0.80


def test_the_expected_split_rounds_the_mean_discordant_counts_and_its_p_is_exact():
    assert model.expected_split(120, 0.25, 0.10) == (21, 9)
    assert model.expected_split_p(120, 0.25, 0.10) == mcnemar_p(21, 30)


def test_the_rejected_splits_are_those_whose_p_is_within_alpha():
    for discordant in range(1, 120):
        by_definition = tuple(
            k for k in range(discordant + 1) if mcnemar_p(k, discordant) <= SIGNIFICANCE_LEVEL
        )
        assert model.rejections(discordant) == by_definition


def test_the_binomial_pmf_of_thousands_of_trials_does_not_overflow():
    # C(2000, 1000) is above the largest float; the normal approximation gives 1/sqrt(1000 pi).
    assert model.binomial_pmf(1000, 2000, 0.5) == pytest.approx(0.017839, abs=1e-6)
    assert model.binomial_pmf(0, 5, 0.0) == 1.0 and model.binomial_pmf(5, 5, 1.0) == 1.0
    assert model.binomial_pmf(1, 5, 0.0) == 0.0


def test_the_power_and_the_sample_size_of_a_large_study_are_computed_without_overflow():
    assert model.mcnemar_power(2000, 0.5, 0.1) == pytest.approx(1.0, abs=1e-4)
    size = model.questions_needed(0.3, 0.03)
    assert model.mcnemar_power(size, 0.3, 0.03) >= 0.80 > model.mcnemar_power(size - 1, 0.3, 0.03)
