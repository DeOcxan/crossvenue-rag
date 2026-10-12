"""calibration.json: a judge calibration as one model, in one file format.

`crossvenue select-judge` writes it into an answer run (`calibration_report.select_judge_files`)
in the one format of `results/20260925T175128Z_answers/calibration.json`. The fields of
`Calibration` come in order, indented by two, no final newline; each dimension holds only the
statistics that are defined, each followed by its interval and its undefined resamples.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..run_files import read_json_model

CALIBRATION_FILE = "calibration.json"
# The statistics a dimension can carry, in the order a record lists them.
STATISTICS = ("agreement", "kappa", "pearson", "spearman")


class DimensionStatistics(BaseModel):
    """One dimension of one comparison of two graders: n, estimates and bootstrap intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    answers: int
    agreement: float | None = None
    agreement_ci: tuple[float, float] | None = None
    agreement_undefined_resamples: int | None = None
    kappa: float | None = None
    kappa_ci: tuple[float, float] | None = None
    kappa_undefined_resamples: int | None = None
    pearson: float | None = None
    pearson_ci: tuple[float, float] | None = None
    pearson_undefined_resamples: int | None = None
    spearman: float | None = None
    spearman_ci: tuple[float, float] | None = None
    spearman_undefined_resamples: int | None = None

    def value(self, statistic: str) -> float | None:
        """The estimate of `statistic`, one of `STATISTICS`; None where it is undefined.

        Raises:
            ValueError: `statistic` is not one of `STATISTICS`.
        """
        value: float | None = getattr(self, _statistic(statistic))
        return value

    def interval(self, statistic: str) -> tuple[float, float] | None:
        """The 95% bootstrap interval of `statistic`, one of `STATISTICS`, if one was computed.

        Raises:
            ValueError: `statistic` is not one of `STATISTICS`.
        """
        interval: tuple[float, float] | None = getattr(self, f"{_statistic(statistic)}_ci")
        return interval

    def undefined(self, statistic: str) -> int:
        """Resamples in which `statistic` was undefined (0 when none or not resampled).

        Raises:
            ValueError: `statistic` is not one of `STATISTICS`.
        """
        count: int | None = getattr(self, f"{_statistic(statistic)}_undefined_resamples")
        return count or 0


def _statistic(name: str) -> str:
    if name not in STATISTICS:
        raise ValueError(f"{name!r} is not a statistic, one of {', '.join(STATISTICS)}")
    return name


class Candidate(BaseModel):
    """A judge candidate's dimensions at or above their threshold, the rest, and its cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passes: list[str]
    fails: list[str]
    cost_per_answer_usd: float


class Calibration(BaseModel):
    """calibration.json: every agreement the judge selection rests on, and the selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run: str
    ratings: str
    raters: list[str]
    answers_rated: int
    resamples: int
    seed: int
    thresholds: dict[str, float]
    excluded_from_behaviour_correct: list[str]
    statistic: dict[str, str]
    comparisons: dict[str, dict[str, DimensionStatistics]]
    candidates: dict[str, Candidate]
    selected: str
    reason: str


def human_comparison(judge: str) -> str:
    """The `comparisons` key of `judge` against the human ratings: `opus vs human`."""
    return f"{judge} vs human"


def judge_comparison(a: str, b: str) -> str:
    """The `comparisons` key of judge `a` against judge `b`: `opus vs sonnet`."""
    return f"{a} vs {b}"


def replicate_comparison(judge: str, replicate: int) -> str:
    """The `comparisons` key of `judge` against its replicate: `opus vs opus replicate 1`."""
    return f"{judge} vs {judge} replicate {replicate}"


def read_calibration(run_dir: Path) -> Calibration:
    """The calibration.json of an answer run."""
    return read_json_model(run_dir / CALIBRATION_FILE, Calibration)


def calibration_json(calibration: Calibration) -> str:
    """calibration.json as written: undefined statistics left out, indented by two."""
    return json.dumps(calibration.model_dump(mode="json", exclude_none=True), indent=2)


def write_calibration(run_dir: Path, calibration: Calibration) -> None:
    """Writes calibration.json into `run_dir`."""
    (run_dir / CALIBRATION_FILE).write_text(calibration_json(calibration), encoding="utf-8")
