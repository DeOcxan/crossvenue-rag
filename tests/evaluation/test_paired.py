"""Tests for `crossvenue_rag.evaluation.paired`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from crossvenue_rag.evaluation.paired import (
    holm,
    mcnemar_exact,
    mcnemar_p,
    paired_bootstrap,
    paired_randomisation,
)


def test_exact_mcnemar_uses_only_discordant_pairs():
    # 8 discordant pairs, all favoring a: p = 2 * (1/2)^8.
    a = [1] * 8 + [1] * 5 + [0] * 5
    b = [0] * 8 + [1] * 5 + [0] * 5
    assert mcnemar_exact(a, b) == pytest.approx(2 / 256)
    assert mcnemar_exact([1, 0], [1, 0]) == 1.0
    # Symmetric split: p = 1.
    assert mcnemar_exact([1, 0], [0, 1]) == 1.0


def test_paired_bootstrap_is_seeded_and_brackets_the_mean():
    a = [0.5, 1.0, 0.0, 1.0, 0.75, 0.25]
    b = [0.5, 0.5, 0.0, 0.5, 0.25, 0.25]
    first = paired_bootstrap(a, b, resamples=2000, seed=7)
    again = paired_bootstrap(a, b, resamples=2000, seed=7)
    assert first == again
    assert first.mean_diff == pytest.approx(0.25)
    assert first.low <= first.mean_diff <= first.high
    assert (first.better, first.worse, first.ties) == (3, 0, 3)


def test_randomisation_test_matches_exact_enumeration_on_a_small_sample():
    a = [1.0, 1.0, 1.0, 0.5, 0.0, 1.0]
    b = [0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    diff = np.subtract(a, b)
    observed = abs(diff.mean())
    flips = [np.array(s) for s in itertools.product([-1, 1], repeat=len(diff))]
    exact = sum(abs((s * diff).mean()) >= observed - 1e-12 for s in flips) / len(flips)
    p = paired_randomisation(a, b, permutations=20_000, seed=20260924)
    assert abs(p - exact) < 0.01


def test_randomisation_test_is_seeded_and_bounded():
    a = [0.9, 0.8, 0.7, 0.9, 0.6, 0.8, 0.9, 0.7] * 5
    b = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.1, 0.3] * 5
    p = paired_randomisation(a, b, permutations=10_000, seed=1)
    assert p == paired_randomisation(a, b, permutations=10_000, seed=1)
    assert p == 1 / 10_001  # no flip reaches a difference this consistent
    assert paired_randomisation(a, a, 10, 1) == 1.0
    assert paired_randomisation([], [], 10, 1) == 1.0


def test_holm_matches_the_step_down_definition():
    raw = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert holm(raw) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])
    assert holm([]) == []


def test_bootstrap_counts_better_in_the_metric_direction():

    # a is above b on two questions, below on one, equal on one.
    a, b = [0.1, 0.2, 0.5, 0.6], [0.3, 0.2, 0.4, 0.5]
    up = paired_bootstrap(a, b, 100, 0)
    down = paired_bootstrap(a, b, 100, 0, higher_is_better=False)
    assert (up.better, up.worse, up.ties) == (2, 1, 1)
    assert (down.better, down.worse, down.ties) == (1, 2, 1)
    assert paired_bootstrap([0.0], [0.5], 10, 0, higher_is_better=False).better == 1
    assert paired_bootstrap([0.0], [0.5], 10, 0).worse == 1
    assert up.mean_diff == down.mean_diff and (up.low, up.high) == (down.low, down.high)


def test_mcnemar_p_is_the_exact_two_sided_binomial_tail():
    # 0 of 5 discordant for one side: 2 * (1/2)^5.
    assert mcnemar_p(0, 5) == pytest.approx(0.0625)
    assert mcnemar_p(5, 5) == pytest.approx(0.0625)
    assert mcnemar_p(2, 4) == 1.0
