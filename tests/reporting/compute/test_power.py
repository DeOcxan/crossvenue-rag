"""Tests for `reporting.compute.power`: a power.json under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.evaluation.power import power_json
from crossvenue_rag.reporting.compute import power
from crossvenue_rag.reporting.run_ids import POWER_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import power_study_of
from tests.reporting.conftest import write_file

REPORT = {
    "delta_80_holm_first_step": 0.17,
    "delta_80_unadjusted": 0.15,
    "family": [["a", "b"], ["a", "c"], ["a", "d"]],
    "fractional_values": 12,
    "values": 150,
    "questions": 50,
    "discordant_pairs": 40,
    "question_pairs": 150,
    "psi": 0.267,
    "mean_magnitude": 0.41,
    "ceiling": 0.534,
    "points": [
        {"delta": 0.1, "reachable": True, "power_holm_first_step": 0.3, "power_unadjusted": 0.4},
        {"delta": 0.2, "reachable": True, "power_holm_first_step": 0.8, "power_unadjusted": 0.9},
    ],
}


def test_the_power_report_is_read_at_the_target_power(tmp_path):
    write_file(tmp_path, f"results/{POWER_RUN}/power.json", power_json(power_study_of(**REPORT)))
    src = Sources(tmp_path)
    assert power.detectable_gap(src, "holm_first_step") == 0.17
    assert power.detectable_gap(src, "unadjusted") == 0.15
    assert power.family_size(src) == 3
    assert power.power_at_gap(src, 0.2, "holm_first_step") == 0.8
    assert power.power_at_gap(src, 0.20, "unadjusted") == 0.9
    assert power.gap_ceiling(src) == 0.534


def test_the_family_compares_its_distinct_methods(tmp_path):
    write_file(tmp_path, f"results/{POWER_RUN}/power.json", power_json(power_study_of(**REPORT)))
    assert power.family_methods(Sources(tmp_path)) == 4


def test_a_gap_the_report_does_not_simulate_raises_a_value_error(tmp_path):
    write_file(tmp_path, f"results/{POWER_RUN}/power.json", power_json(power_study_of(**REPORT)))
    with pytest.raises(ValueError, match=r"simulates no delta 0\.3"):
        power.power_at_gap(Sources(tmp_path), 0.3, "holm_first_step")
