"""Tests for `crossvenue_rag.stats`."""

from __future__ import annotations

import numpy as np
import pytest

from crossvenue_rag.stats import (
    Interval,
    cohens_kappa,
    cohens_kappa_or_none,
    kappa_value,
    mean,
    mean_of_present,
    percentile,
    percentile_bounds,
)
from tests.conftest import REPO

CALIBRATION_RUN = REPO / "results/20260925T175128Z_answers"


def test_percentile_is_nearest_rank():
    assert percentile([10, 20, 30, 40], 0.5) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40


def test_the_percentile_of_no_values_is_an_error_like_the_mean():
    with pytest.raises(ValueError, match="percentile of no values"):
        percentile([], 0.5)


def test_the_mean_of_no_values_is_an_error_and_mean_of_present_is_none():
    with pytest.raises(ValueError, match="no values"):
        mean([])
    assert mean([1.0, 2.0]) == 1.5
    assert mean_of_present([]) is None
    assert mean_of_present([None, None]) is None
    assert mean_of_present([1.0, None, 3.0]) == 2.0


def test_cohens_kappa_corrects_for_chance():
    """Perfect agreement is 1.0; agreement no better than chance is 0.0."""
    result = cohens_kappa([True, False, True, False], [True, False, True, False])
    kappa, observed = result.kappa, result.observed
    assert kappa == pytest.approx(1.0) and observed == pytest.approx(1.0)

    # 50% observed agreement with balanced marginals is exactly chance.
    result = cohens_kappa([True, True, False, False], [True, False, True, False])
    kappa, observed, expected = result.kappa, result.observed, result.expected
    assert observed == pytest.approx(0.5)
    assert expected == pytest.approx(0.5)
    assert kappa == pytest.approx(0.0)

    # Degenerate case: both rounds called everything relevant.
    assert cohens_kappa([True, True], [True, True]).kappa == pytest.approx(1.0)


def test_kappa_or_none_is_none_where_kappa_is_undefined():
    assert cohens_kappa_or_none([]) is None
    assert cohens_kappa_or_none([(True, True), (True, True)]) is None
    pairs = [(True, True), (True, False), (False, False), (False, True), (True, True)]
    assert (
        cohens_kappa_or_none(pairs)
        == cohens_kappa([a for a, _ in pairs], [b for _, b in pairs]).kappa
    )


def _removed_kappa_or_none(verdicts):
    """The implementation `generation.metrics._kappa_or_none` had, kept as the reference."""
    if not verdicts:
        return None
    a = [x for x, _ in verdicts]
    b = [y for _, y in verdicts]
    n = len(a)
    observed = sum(x == y for x, y in verdicts) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def test_both_kappa_paths_agree_on_the_calibration_data():
    from crossvenue_rag.generation.grader_agreement import (
        BINARY_DIMENSIONS,
        binary_agreement,
        paired_verdicts,
    )
    from crossvenue_rag.generation.judge_sample import (
        RATINGS_PATH,
        human_ratings,
        judgements_path,
        load_judgements,
    )

    humans = human_ratings(REPO / RATINGS_PATH)
    rng = np.random.default_rng(0)
    for judge in ("sonnet", "opus"):
        pairs = paired_verdicts(load_judgements(judgements_path(CALIBRATION_RUN, judge)), humans)
        samples = [pairs] + [
            [pairs[i] for i in rng.integers(0, len(pairs), size=len(pairs))] for _ in range(200)
        ]
        for sample in samples:
            for dimension in BINARY_DIMENSIONS:
                if dimension == "behaviour_correct":
                    verdicts = [
                        p.behaviour_correct for p in sample if p.behaviour_correct is not None
                    ]
                else:
                    verdicts = [v for p in sample for v in getattr(p, dimension)]
                new = cohens_kappa_or_none(verdicts)
                assert new == _removed_kappa_or_none(verdicts)
                if new is not None:
                    direct = binary_agreement(
                        dimension, [a for a, _ in verdicts], [b for _, b in verdicts]
                    )
                    assert direct.kappa == new


def test_kappa_value_refuses_an_expected_agreement_of_one():
    assert kappa_value(0.75, 0.5) == 0.5
    with pytest.raises(ValueError, match="expected agreement is 1"):
        kappa_value(1.0, 1.0)
    with pytest.raises(ValueError, match="expected agreement is 1"):
        kappa_value(np.array([0.75, 1.0]), np.array([0.5, 1.0]))


def test_an_interval_of_no_draws_has_no_bounds_and_percentile_bounds_refuses_them():
    assert Interval.from_draws([], 3, 3) == Interval(None, None, 3, 3)
    assert Interval.from_draws([1.0, 2.0], 1, 3) == Interval(*percentile_bounds([1.0, 2.0]), 1, 3)
    with pytest.raises(ValueError, match="no values"):
        percentile_bounds([])
