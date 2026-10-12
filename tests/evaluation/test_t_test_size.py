"""Tests for `evaluation.t_test_size`: the paired t-test sample size formula."""

from __future__ import annotations

import math

import pytest

from crossvenue_rag.evaluation import t_test_size


def test_paired_t_test_sizes_follow_n_equals_7_84_over_d_squared():
    assert t_test_size.questions_for_effect(0.5) == 32  # 7.84 / 0.25 = 31.36
    assert t_test_size.detectable_effect(90) == pytest.approx(math.sqrt(7.84 / 90))
