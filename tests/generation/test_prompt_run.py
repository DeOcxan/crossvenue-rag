"""Tests for `generation.prompt_run`: the prompts run directory and its report."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.generation.prompt_compare import ComparedRuns, compare_prompts
from crossvenue_rag.generation.prompt_run import write_prompt_comparison
from tests.generation.factories import prompt_record

QUESTIONS = Path("evals/q.jsonl")


def test_the_prompt_report_has_no_output_defects(tmp_path):
    from tests.conftest import output_defects

    v1 = [
        prompt_record("a", "q1", "answer", vae=0.5),
        prompt_record("a", "q2", "answer", refused=True),
    ]
    v2 = [prompt_record("a", "q1", "answer", vae=0.0), prompt_record("a", "q2", "answer", vae=0.0)]
    rows = compare_prompts(v1, v2, ["a"], 200, 1)
    run_dir = write_prompt_comparison(
        tmp_path, ComparedRuns(tmp_path / "v1run", tmp_path / "v2run", QUESTIONS), rows, [], "abc"
    )
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "\u2212" in report  # both differences fall
    assert output_defects(report) == []


def test_a_configuration_without_claims_in_both_is_reported_not_computable(tmp_path):
    v1 = [
        prompt_record("a", "q1", "answer", vae=0.0),
        prompt_record("b", "q1", "answer", refused=True),
    ]
    v2 = [prompt_record("a", "q1", "answer", vae=0.5), prompt_record("b", "q1", "answer", vae=0.0)]
    rows = compare_prompts(v1, v2, ["a", "b"], 100, 0)
    runs = ComparedRuns(tmp_path / "v1", tmp_path / "v2", QUESTIONS)
    run_dir = write_prompt_comparison(tmp_path, runs, rows, [], "v")
    report = (run_dir / "report.md").read_text()
    assert "`venue_attribution_error` not computable" in report
    assert ": b." in report


def test_the_interval_column_of_the_prompt_report_is_right_aligned():
    from crossvenue_rag.generation.prompt_run import prompt_report

    rows = compare_prompts(
        [prompt_record("a", "q1", "answer", vae=0.5)],
        [prompt_record("a", "q1", "answer", vae=0.0)],
        ["a"],
        100,
        0,
    )
    runs = ComparedRuns(Path("r/v1"), Path("r/v2"), QUESTIONS)
    lines = prompt_report(runs, rows, "v7").splitlines()
    rule = lines[lines.index(next(line for line in lines if line.startswith("| config |"))) + 1]
    assert rule.split("|")[1:-1] == [
        "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---:", "---:",
    ]  # fmt: skip


def test_the_report_text_is_built_without_files_and_written_as_built(tmp_path):
    from crossvenue_rag.generation.prompt_run import prompt_report

    v1 = [prompt_record("a", "q1", "answer", vae=0.5)]
    v2 = [prompt_record("a", "q1", "answer", vae=0.0)]
    rows = compare_prompts(v1, v2, ["a"], 100, 0)
    runs = ComparedRuns(Path("r/v1"), Path("r/v2"), QUESTIONS)
    text = prompt_report(runs, rows, "v7")
    assert text.startswith("# Prompt v1 against v2: `r/v1` and `r/v2`\n\n- `index_version`: `v7`\n")
    run_dir = write_prompt_comparison(tmp_path, runs, rows, [], "v7")
    assert (run_dir / "report.md").read_text(encoding="utf-8") == text


def test_the_prompt_report_names_the_question_file_at_its_start_and_in_the_table_caption():
    from crossvenue_rag.generation.prompt_run import prompt_report

    rows = compare_prompts(
        [prompt_record("a", "q1", "answer", vae=0.5)],
        [prompt_record("a", "q1", "answer", vae=0.0)],
        ["a"],
        100,
        0,
    )
    runs = ComparedRuns(Path("r/v1"), Path("r/v2"), QUESTIONS)
    lines = prompt_report(runs, rows, "v7").splitlines()
    assert lines[3] == "- `eval.questions`: `evals/q.jsonl`"
    caption = lines[lines.index(next(line for line in lines if line.startswith("| config |"))) - 2]
    assert caption.startswith(
        "v2 \u2212 v1 per configuration on `evals/q.jsonl`, paired on the n questions of each "
        "row, with its 95% paired bootstrap interval;"
    )
