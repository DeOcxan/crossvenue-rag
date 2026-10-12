"""The files of a prompt comparison, results/<ts>_prompts/, report.md's text among them.

`compare_prompts` and `side_effects` compute what this module writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .. import formatting, jsonl, listing
from ..evaluation.labels import markdown_table
from ..run_files import (
    REPORT_FILE,
    CsvTable,
    RunIndex,
    RunSource,
    create_run_dir,
    run_path,
    write_index,
    write_run_source,
    write_table,
)
from ..stats import INTERVAL_LEVEL_PERCENT
from .prompt_compare import COMPARISON_FILE, PromptComparison, SideEffect


def write_prompt_comparison(
    results_dir: Path,
    v1: Path,
    v2: Path,
    rows: Sequence[PromptComparison],
    effects: Sequence[SideEffect],
    index_version: str,
) -> Path:
    """Writes results/<ts>_prompts/ and returns its path.

    The directory holds comparison.csv, side_effects.jsonl, source.json,
    index.json and report.md (`prompt_report`).
    """
    run_dir = create_run_dir(run_path(results_dir, "_prompts"))
    write_table(run_dir / COMPARISON_FILE, CsvTable.of(rows))
    jsonl.write(run_dir / "side_effects.jsonl", [asdict(e) for e in effects], ascii_only=True)
    write_run_source(run_dir, RunSource(v1=str(v1), v2=str(v2)))
    write_index(run_dir, RunIndex(index_version=index_version), final_newline=True)
    report = prompt_report(v1, v2, rows, index_version)
    (run_dir / REPORT_FILE).write_text(report, encoding="utf-8")
    return run_dir


def prompt_report(v1: Path, v2: Path, rows: Sequence[PromptComparison], index_version: str) -> str:
    """report.md of a prompt comparison: v2 against v1 per configuration and metric."""
    lines = [
        f"# Prompt v1 against v2: `{v1}` and `{v2}`",
        "",
        f"- `index_version`: `{index_version}`",
        "",
        "v2 \u2212 v1 per configuration, paired on questions; Holm across configurations "
        "within each metric. Faithfulness is not compared (its judge dimension failed "
        "calibration).",
        "",
        markdown_table(
            [
                "config",
                "metric",
                "n",
                "v1",
                "v2",
                "v2 \u2212 v1",
                f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% CI",
                "better",
                "worse",
                "test",
                "p",
                "Holm p",
            ],
            [
                [
                    r.config,
                    f"`{r.metric}`",
                    formatting.count(r.n),
                    formatting.metric(r.v1),
                    formatting.metric(r.v2),
                    formatting.signed(r.diff),
                    formatting.interval(r.low, r.high),
                    formatting.count(r.better),
                    formatting.count(r.worse),
                    _test_label(r.test),
                    formatting.p_value(r.p),
                    formatting.optional(r.holm_p, formatting.p_value),
                ]
                for r in rows
            ],
            right=(2, 3, 4, 5, 7, 8, 10, 11),
        ),
        *_not_computable(rows),
    ]
    return "\n".join(lines) + "\n"


def _not_computable(rows: Sequence[PromptComparison]) -> list[str]:
    """A report line naming the configurations with no venue_attribution_error row.

    Such a configuration has no question where both versions' answers carry
    claims; no such configuration gives no line.
    """
    with_error = {r.config for r in rows if r.metric == "venue_attribution_error"}
    missing = [r.config for r in rows if r.metric == "wrong_refusal" and r.config not in with_error]
    if not missing:
        return []
    return [
        "",
        "`venue_attribution_error` not computable (no question where both versions' "
        f"answers carry claims): {listing.prose(missing)}.",
    ]


def _test_label(test: str) -> str:
    """A test name as report.md shows it; comparison.csv keeps the saved name."""
    return test.replace("randomisation", "randomization")
