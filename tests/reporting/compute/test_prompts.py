"""Tests for `crossvenue_rag.reporting.compute.prompts`: prompt v2 against v1 per configuration."""

from __future__ import annotations

import pytest
import yaml

from crossvenue_rag.reporting.compute import prompts
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, PROMPTS_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import ANSWER_CONFIG, CONFIGS, write_csv, write_file


def comparison_row(config, metric, *, better=0, worse=0, p=1.0, n=40):
    return {
        "config": config,
        "metric": metric,
        "n": n,
        "v1": 0.5,
        "v2": 0.6,
        "diff": 0.1,
        "low": 0.0,
        "high": 0.2,
        "better": better,
        "worse": worse,
        "p": p,
        "test": "randomisation",
        "holm_p": p,
    }


def write_prompts(root, rows):
    """comparison.csv of the prompts run, and the v1 answer run's config naming `CONFIGS`."""
    write_file(root, f"results/{ANSWERS_V1}/config.yaml", yaml.safe_dump(ANSWER_CONFIG))
    write_csv(root / "results" / PROMPTS_RUN / "comparison.csv", rows)
    return Sources(root)


def test_change_rows_follow_the_configuration_order_with_both_metrics(tmp_path):
    rows = [comparison_row(c, m) for c in reversed(CONFIGS) for m in ("support", "refusal")]
    src = write_prompts(tmp_path, rows)
    table = prompts.change_rows(src, ("support", "refusal"))
    assert [row.config for row in table] == list(CONFIGS)
    first = table[0].first
    assert (first.v1, first.v2, first.holm_p) == pytest.approx((0.5, 0.6, 1.0))
    difference = (first.difference.value, first.difference.low, first.difference.high)
    assert difference == pytest.approx((0.1, 0.0, 0.2))


def test_questions_default_to_the_first_configuration(tmp_path):
    rows = [comparison_row(c, "support", n=40 + i) for i, c in enumerate(CONFIGS)]
    src = write_prompts(tmp_path, rows)
    assert prompts.questions(src, "support") == 40
    assert prompts.questions(src, "support", CONFIGS[2]) == 42


def test_changed_questions_add_better_and_worse_and_span_a_range(tmp_path):
    changes = [(1, 0), (0, 3), (2, 2), (0, 0)]
    rows = [
        comparison_row(c, "support", better=b, worse=w)
        for c, (b, w) in zip(CONFIGS, changes, strict=True)
    ]
    src = write_prompts(tmp_path, rows)
    assert prompts.changed_questions(src, "support") == [1, 3, 4, 0]
    assert prompts.change_range(src, "support") == prompts.ChangeRange(0, 4, 8)
    assert not prompts.v2_never_worse(src, "support")


def test_no_configuration_worse_under_v2_is_never_worse(tmp_path):
    rows = [comparison_row(c, "support", better=1) for c in CONFIGS]
    assert prompts.v2_never_worse(write_prompts(tmp_path, rows), "support")


def test_the_smallest_p_halves_with_each_changed_question():
    assert prompts.smallest_p(1) == 1.0
    assert prompts.smallest_p(3) == 0.25


def test_changes_all_against_v2_with_p_at_the_floor_are_recognised(tmp_path):
    # Each configuration has 3 worse questions, floor 0.25; p within the tolerance of it.
    at_floor = [comparison_row(c, "support", worse=3, p=0.25) for c in CONFIGS]
    assert prompts.changes_against_v2_at_floor(write_prompts(tmp_path / "a", at_floor), "support")
    one_better = [comparison_row(CONFIGS[0], "support", better=1, worse=2, p=0.25)]
    one_better += at_floor[1:]
    src = write_prompts(tmp_path / "b", one_better)
    assert not prompts.changes_against_v2_at_floor(src, "support")
    below_floor = [comparison_row(c, "support", worse=3, p=0.2) for c in CONFIGS]
    src = write_prompts(tmp_path / "c", below_floor)
    assert not prompts.changes_against_v2_at_floor(src, "support")
    far_above = [comparison_row(c, "support", worse=3, p=0.9) for c in CONFIGS]
    src = write_prompts(tmp_path / "d", far_above)
    assert not prompts.changes_against_v2_at_floor(src, "support")


@pytest.mark.parametrize(
    "call",
    [
        lambda src: prompts.change(src, CONFIGS[0], "refusal"),
        lambda src: prompts.questions(src, "refusal"),
        lambda src: prompts.v2_never_worse(src, "refusal"),
        lambda src: prompts.changed_questions(src, "refusal"),
        lambda src: prompts.changes_against_v2_at_floor(src, "refusal"),
    ],
)
def test_a_configuration_without_a_row_on_the_metric_is_a_value_error(tmp_path, call):
    src = write_prompts(tmp_path, [comparison_row(c, "support") for c in CONFIGS])
    with pytest.raises(ValueError, match=r"comparison\.csv has no"):
        call(src)
