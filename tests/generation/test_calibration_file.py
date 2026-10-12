"""Tests for `generation.calibration_file`: calibration.json as one model."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.generation.calibration_file import (
    CALIBRATION_FILE,
    Calibration,
    DimensionStatistics,
    calibration_json,
    read_calibration,
    write_calibration,
)
from tests.conftest import committed_with


@pytest.mark.parametrize("run", committed_with(CALIBRATION_FILE), ids=lambda p: p.name)
def test_a_committed_calibration_is_written_back_with_the_same_bytes(run):
    original = (run / CALIBRATION_FILE).read_bytes()
    assert calibration_json(read_calibration(run)).encode("utf-8") == original


def test_an_undefined_statistic_is_left_out_and_reads_back_as_none(tmp_path):
    row = DimensionStatistics(n=4, answers=4, agreement=0.5, agreement_ci=(0.25, 0.75))
    calibration = Calibration(
        run="r", ratings="x", raters=["a"], answers_rated=4, resamples=10, seed=1,
        thresholds={"supported": 0.6}, excluded_from_behaviour_correct=[],
        statistic={"supported": "kappa"}, comparisons={"j vs human": {"supported": row}},
        candidates={}, selected="j", reason="only candidate",
    )  # fmt: skip
    write_calibration(tmp_path, calibration)
    written = json.loads((tmp_path / CALIBRATION_FILE).read_text(encoding="utf-8"))
    assert list(written["comparisons"]["j vs human"]["supported"]) == [
        "n",
        "answers",
        "agreement",
        "agreement_ci",
    ]
    back = read_calibration(tmp_path).comparisons["j vs human"]["supported"]
    assert back.value("kappa") is None and back.interval("agreement") == (0.25, 0.75)
    assert back.undefined("agreement") == 0


def test_every_sibling_function_a_generation_docstring_names_exists():
    import importlib
    import pkgutil
    import re

    import crossvenue_rag.generation as package

    names = {m.name for m in pkgutil.iter_modules(package.__path__)}
    dangling = []
    for name in sorted(names):
        module = importlib.import_module(f"{package.__name__}.{name}")
        for sibling, attr in re.findall(r"`(\w+)\.(\w+)`", module.__doc__ or ""):
            # `llm.py` names a file.
            if sibling in names and attr != "py":
                target = importlib.import_module(f"{package.__name__}.{sibling}")
                if not hasattr(target, attr):
                    dangling.append(f"{name}: `{sibling}.{attr}`")
    assert dangling == []


@pytest.mark.parametrize("name", ["spearman_ci", "n", "tau"])
def test_a_name_that_is_no_statistic_is_a_value_error(name):
    """`getattr` alone would return `spearman_ci` or `n` as if they were estimates."""
    row = DimensionStatistics(n=4, answers=4, spearman=0.5, spearman_ci=(0.1, 0.9))
    with pytest.raises(ValueError, match="is not a statistic"):
        row.value(name)
    with pytest.raises(ValueError, match="is not a statistic"):
        row.interval(name)
    with pytest.raises(ValueError, match="is not a statistic"):
        row.undefined(name)
