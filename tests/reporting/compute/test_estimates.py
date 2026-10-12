import pytest

from crossvenue_rag.reporting.compute.estimates import Estimate, is_significant, is_small_effect


@pytest.mark.parametrize(
    ("low", "high", "includes"),
    [(-0.1, 0.2, True), (0.0, 0.2, True), (-0.2, 0.0, True), (0.1, 0.2, False)],
)
def test_an_interval_includes_zero_when_zero_lies_in_it_an_end_included(low, high, includes):
    assert Estimate(0.05, low, high).includes_zero() is includes


@pytest.mark.parametrize(
    ("difference", "p", "small"),
    [
        (Estimate(0.05, 0.01, 0.09), 0.5, True),  # the interval excludes 0
        (Estimate(0.05, -0.01, 0.09), 0.01, True),  # p below the level
        (Estimate(0.05, 0.0, 0.09), 0.5, False),  # an end at 0: no effect
        (Estimate(0.30, 0.10, 0.50), 0.01, False),  # an effect above delta
    ],
)
def test_a_small_effect_is_an_effect_whose_size_is_below_delta(difference, p, small):
    assert is_small_effect(difference, p, 0.20) is small


def test_a_p_is_significant_only_below_the_significance_level():
    from crossvenue_rag.evaluation.power import SIGNIFICANCE_LEVEL

    assert is_significant(SIGNIFICANCE_LEVEL / 2)
    assert not is_significant(SIGNIFICANCE_LEVEL)
