"""Tests for scripts/answer_figures.py: what a figure refuses before it draws."""

from __future__ import annotations

import pytest
import yaml
from answer_figures import candidate_colors, draw_calibration, draw_wrong_refusals

from crossvenue_rag.generation.calibration_file import calibration_json
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, PROMPTS_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import calibration_of
from tests.reporting.conftest import ANSWER_CONFIG, CONFIGS, write_file


def test_the_wrong_refusal_figure_refuses_a_comparison_without_a_configuration(tmp_path):
    write_file(tmp_path, f"results/{ANSWERS_V1}/config.yaml", yaml.safe_dump(ANSWER_CONFIG))
    write_file(tmp_path, f"results/{PROMPTS_RUN}/comparison.csv", "config,metric\n")
    with pytest.raises(ValueError, match=f"the wrong_refusal rows of .* has no '{CONFIGS[0]}'"):
        draw_wrong_refusals(Sources(tmp_path))


def test_the_calibration_figure_refuses_another_number_of_candidates_than_it_draws():
    with pytest.raises(ValueError, match="draws 2 judge candidates side by side; the run has 3"):
        candidate_colors(["opus", "sonnet", "haiku"])


def test_the_calibration_figure_refuses_a_comparison_without_an_interval(tmp_path):
    report = calibration_of(
        candidates={"opus": {}, "sonnet": {}},
        thresholds={"supported": 0.6},
        statistic={"supported": "kappa"},
        comparisons={"opus vs human": {"supported": {"n": 9, "kappa": 0.5}}},
    )
    write_file(tmp_path, f"results/{ANSWERS_V1}/calibration.json", calibration_json(report))
    with pytest.raises(
        ValueError, match="opus vs human records no interval for kappa on supported"
    ):
        draw_calibration(Sources(tmp_path))
