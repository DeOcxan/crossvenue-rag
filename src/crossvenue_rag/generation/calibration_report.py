"""The calibration's files in the answer run: calibration.json and calibration.md."""

from __future__ import annotations

from pathlib import Path

from .. import formatting, listing
from ..config import Config
from ..evaluation.labels import markdown_table
from ..stats import INTERVAL_LEVEL_PERCENT
from .calibration import calibration_report
from .calibration_file import (
    Calibration,
    DimensionStatistics,
    write_calibration,
)

CALIBRATION_MARKDOWN = "calibration.md"


def calibration_markdown(report: Calibration) -> str:
    """calibration.md: the agreement tables and the selection.

    Every dimension a comparison of `report` lists with a threshold in `report.thresholds`
    has its statistic in `report.statistic`.

    Raises:
        ValueError: A dimension with a threshold has no statistic in `report.statistic`.
    """

    def cell(row: DimensionStatistics, statistic: str) -> str:
        value = row.value(statistic)
        if value is None:
            return formatting.MISSING
        bounds = row.interval(statistic)
        interval = (
            ""
            if bounds is None
            else f" [{formatting.metric(bounds[0])}, {formatting.metric(bounds[1])}]"
        )
        undefined = row.undefined(statistic)
        note = f", undefined in {formatting.plural(undefined, 'resample')}" if undefined else ""
        return f"{formatting.metric(value)}{interval}{note}"

    lines = [
        f"# Judge calibration: `{report.run}`",
        "",
        f"{formatting.plural(report.answers_rated, 'answer')} rated by "
        f"{listing.prose(list(report.raters))}; "
        f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% percentile bootstrap intervals "
        f"over answers, {formatting.plural(report.resamples, 'resample')}, "
        f"seed {formatting.own_form(report.seed)}."
        + (
            " Left out of `behaviour_correct` against the human: "
            f"{', '.join(report.excluded_from_behaviour_correct)}."
            if report.excluded_from_behaviour_correct
            else ""
        ),
        "",
    ]

    def bar(dimension: str) -> str:
        threshold = report.thresholds.get(dimension)
        if threshold is None:
            return formatting.MISSING
        if dimension not in report.statistic:
            raise ValueError(f"the report has a threshold on {dimension} but no statistic")
        bound = formatting.own_form(threshold, formatting.CONFIG_THRESHOLD_DECIMALS)
        return f"{report.statistic[dimension]} \u2265 {bound}"

    def dimensions(names: list[str]) -> str:
        return ", ".join(f"`{d}`" for d in names) or formatting.MISSING

    header = ["dimension", "n", "agreement", "kappa", "Spearman", "threshold"]
    for name, dims in report.comparisons.items():
        rows = [
            [
                f"`{dimension}`",
                formatting.count(row.n),
                cell(row, "agreement"),
                cell(row, "kappa"),
                cell(row, "spearman"),
                bar(dimension),
            ]
            for dimension, row in dims.items()
        ]
        lines += [f"## {name}", "", markdown_table(header, rows, right=range(1, len(header))), ""]
    selection = [
        [
            judge_name,
            dimensions(c.passes),
            dimensions(c.fails),
            formatting.cost(c.cost_per_answer_usd),
        ]
        for judge_name, c in report.candidates.items()
    ]
    lines += [
        "## Selection",
        "",
        markdown_table(["judge", "passes", "fails", "USD per answer"], selection, right=(3,)),
        "",
        f"Selected: {report.selected}. {report.reason}.",
        "",
    ]
    return "\n".join(lines)


def select_judge_files(cfg: Config, run_dir: Path) -> str:
    """Writes calibration.json and calibration.md into the run and returns the Markdown.

    Raises:
        UserError: `generation.judges` names no candidate, or a candidate with no judgments
            in the run or that graded none of the rated answers, or
            `generation.calibration_exclude_behaviour` names an answer the run does not have.
    """
    report = calibration_report(cfg, run_dir)
    write_calibration(run_dir, report)
    text = calibration_markdown(report)
    (run_dir / CALIBRATION_MARKDOWN).write_text(text, encoding="utf-8")
    return text
