"""Tests for `generation.answer_report`: the report.md of an answer run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.config_generation import MatrixEntry
from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.answer_pairs import ConfigComparison
from crossvenue_rag.generation.answer_report import report
from crossvenue_rag.generation.answer_rows import summary_rows
from crossvenue_rag.generation.calibration_file import calibration_json
from crossvenue_rag.generation.metrics import METRICS
from tests.conftest import calibration_of
from tests.generation.factories import (
    call_json,
    generation_json,
    judge_json,
    metrics_json,
    record,
)
from tests.generation.offline import (
    gen_config,
)

CONFIGS = ("gemini@venue_split", "gemini@hybrid_rerank")


def answer_config(tmp_path: Path, **patch: Any) -> Config:
    """A config whose matrix is gemini with each of two retrievers."""
    matrix = [MatrixEntry(name=c, generator="gemini", retriever=c.split("@")[1]) for c in CONFIGS]
    return Config(generation=gen_config(tmp_path, top_k=5, matrix=matrix, **patch))


def judged_records(judge="opus", metrics=None):
    return [
        record(c, q, judge=judge_json(judge=judge, metrics=metrics))
        for c in CONFIGS
        for q in ("cv-0001", "cv-0002")
    ]


def pair(**fields) -> ConfigComparison:
    base = {
        "a": CONFIGS[0],
        "b": CONFIGS[1],
        "metric": "coverage",
        "scope": "comparison",
        "n": 50,
        "mean_a": 0.6,
        "mean_b": 0.5,
        "diff": 0.1,
        "low": -0.05,
        "high": 0.25,
        "randomisation_p": 0.0004,
        "mcnemar_p": None,
        "role": "primary",
        "holm_p": 0.0008,
    }
    return ConfigComparison(**{**base, **fields})


def make_report(tmp_path, records, pairs=(), config=None, note="", index_version="v1"):
    config = config or answer_config(tmp_path)
    return report(config, records, summary_rows(records, CONFIGS), pairs, note, index_version)


def lines_starting(text, prefix):
    return [line for line in text.splitlines() if line.startswith(prefix)]


def test_the_header_counts_questions_and_configurations_and_names_the_judge(tmp_path):
    text = make_report(tmp_path, judged_records(), note="A note.", index_version="v7")
    assert text.startswith("# Answer evaluation\n")
    assert "- `index_version`: `v7`" in text
    assert "\nA note.\n" in text
    assert "2 questions \u00d7 2 configurations; top-5 chunks." in text
    assert "Judge: opus (anthropic/claude-opus-5.5)." in text
    assert text.endswith("|\n")


def test_a_run_without_a_judge_says_so_and_marks_no_metric_uncalibrated(tmp_path):
    records = [record(CONFIGS[0], "cv-0001")]
    text = make_report(tmp_path, records)
    assert "1 question \u00d7 2 configurations" in text
    assert "Judge: none (generation only)." in text
    assert "Uncalibrated" not in text
    assert "(uncalibrated)" not in text


def test_every_metric_is_uncalibrated_when_no_calibration_names_the_judge(tmp_path):
    text = make_report(tmp_path, judged_records())
    (note,) = lines_starting(text, "Uncalibrated: ")
    named = note.removeprefix("Uncalibrated: ").split(".")[0].split(", ")
    assert sorted(named) == sorted(f"`{m}`" for m in METRICS)
    (head,) = lines_starting(text, "| config | answers |")
    assert "| faithfulness (uncalibrated) |" in head
    assert "| coverage (comparison) (uncalibrated) |" in head


def test_only_the_metrics_of_a_failed_dimension_are_marked_uncalibrated(tmp_path):
    (tmp_path / "calibration.json").write_text(
        calibration_json(
            calibration_of(candidates={"opus": {"passes": ["coverage"], "fails": ["supported"]}})
        )
    )
    config = answer_config(tmp_path, calibration_run=tmp_path)
    text = make_report(tmp_path, judged_records(), config=config)
    (note,) = lines_starting(text, "Uncalibrated: ")
    assert note.startswith("Uncalibrated: `faithfulness`. ")
    (head,) = lines_starting(text, "| config | answers |")
    assert "| faithfulness (uncalibrated) |" in head
    assert "| coverage |" in head


def test_answers_graded_by_two_judges_are_refused_naming_both(tmp_path):
    records = [*judged_records("opus")[:2], *judged_records("sonnet")[2:]]
    with pytest.raises(UserError, match="more than one judge: opus, sonnet"):
        make_report(tmp_path, records)


def test_the_metric_and_cost_tables_have_a_row_per_configuration(tmp_path):
    text = make_report(tmp_path, judged_records(metrics=metrics_json(coverage=None)))
    rows = lines_starting(text, f"| {CONFIGS[0]} |")
    assert len(rows) == 2
    metrics, cost = rows
    # Two answers, no errors, refusals or questions; coverage absent everywhere.
    assert metrics.startswith(f"| {CONFIGS[0]} | 2 | 0 | 0 | 0 | 1.000 | 0.000 | - | 1.000 | - |")
    # 2 x 1000 tokens in, 2 x 30 reasoning, 2 x 170 answer, 2 x $0.01 for each model;
    # each call's 1,000 ms is 1 s.
    assert cost == f"| {CONFIGS[0]} | 2,000 | 60 | 340 | $0.020 | $0.020 | 1 | 1 | 0 | 0 |"
    assert "- gemini@venue_split: `model=google/gemini-3.8-flash`" in text


def test_the_parameters_sent_are_readable_text_with_no_raw_dict(tmp_path):
    """The params cell of summary.csv is JSON for programs; the report prints it as text."""
    sent = {
        "max_tokens": 4000,
        "provider": {"order": ["a/b", "c"], "require_parameters": True},
        "temperature": 0.0,
    }
    generation = generation_json(call=call_json(params=sent))
    records = [record(c, "cv-0001", generation=generation) for c in CONFIGS]
    text = make_report(tmp_path, records)
    (line,) = lines_starting(text, f"- {CONFIGS[0]}: ")
    assert line == (
        f"- {CONFIGS[0]}: `max_tokens=4000`, `provider.order=a/b,c`, "
        "`provider.require_parameters=true`, `temperature=0.0`"
    )
    assert "{" not in line


def test_a_paired_row_shows_its_interval_p_values_and_role(tmp_path):
    pairs = [
        pair(),
        pair(metric="faithfulness", role="secondary", calibrated=False, diff=-0.2, holm_p=None),
    ]
    text = make_report(tmp_path, judged_records(), pairs)
    primary, secondary = lines_starting(text, f"| {CONFIGS[0]} vs {CONFIGS[1]} |")
    assert primary == (
        f"| {CONFIGS[0]} vs {CONFIGS[1]} | `coverage` | comparison | 50 | +0.100 "
        "| [\u22120.050, +0.250] | < 0.001 | primary | < 0.001 |"
    )
    assert "| \u22120.200 |" in secondary
    assert secondary.endswith("| secondary, uncalibrated | - |")


def test_a_report_without_pairs_keeps_an_empty_paired_table(tmp_path):
    text = make_report(tmp_path, judged_records())
    assert "## Paired comparisons" in text
    assert lines_starting(text, f"| {CONFIGS[0]} vs ") == []


def test_a_judge_the_config_does_not_list_is_named_with_its_recorded_model(tmp_path):
    text = make_report(tmp_path, judged_records(judge="retired"))
    assert "Judge: retired (anthropic/claude-opus-5.5)." in text
    assert "none (generation only)" not in text


def test_a_latency_saved_in_ms_is_shown_in_whole_seconds():
    from crossvenue_rag.generation.answer_report import latency_seconds

    assert latency_seconds(242_600.0) == "243"
