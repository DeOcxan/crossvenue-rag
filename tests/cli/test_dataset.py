"""Tests for `cli.dataset`: kappa, select and validate-evals from the command line."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from crossvenue_rag.cli import dataset as _under_test
from crossvenue_rag.errors import UserError
from tests.cli.conftest import invoke
from tests.conftest import write_journal


def test_kappa_refuses_a_retest_within_fourteen_days(tmp_path):
    from typer.testing import CliRunner

    from crossvenue_rag.cli import app
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.models import Question

    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact", category="fact")])
    entry = {"pool": ["c1"], "relevant": ["c1"], "index_version": "v"}
    write_journal(
        path,
        {"q1": {"1": {**entry, "annotated_at": "2026-09-24T12:00:00Z"},
                "2": {**entry, "annotated_at": "2026-09-30T12:00:00Z"}}},
    )  # fmt: skip
    refused = CliRunner().invoke(app, ["kappa", "-q", str(path)])
    assert refused.exit_code != 0 and "less than 14 days" in str(
        refused.exception or refused.output
    )
    marked = CliRunner().invoke(app, ["kappa", "-q", str(path), "--short-gap"])
    assert marked.exit_code == 0 and "not the registered measurement" in marked.output
    assert "agreement by category" in marked.output


def test_validate_evals_fails_on_an_empty_dataset(config_file, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    result = invoke(
        "validate-evals", "-c", str(config_file), "-d", str(empty), "--skip-near-duplicates"
    )
    assert result.exit_code == 1
    assert "failed" in result.output and "contains no questions" in result.output


def test_validate_evals_prints_the_stats_of_a_passing_dataset(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.dataset import validate

    got: list = []

    def fake(path, cfg, *, check_near_duplicates: bool = True):
        got.append((path, check_near_duplicates))
        stats = validate.ValidationStats(total=2, answerable=2, by_category={"fact": 2})
        return validate.ValidationReport(warnings=["one warning"], stats=stats)

    monkeypatch.setattr(_under_test, "validate_dataset", fake)
    path = tmp_path / "set.jsonl"
    result = invoke("validate-evals", "-c", str(config_file), "-d", str(path))
    assert result.exit_code == 0, result.output
    assert got == [(path, True)]
    assert "ok" in result.output and "answerable / unanswerable" in result.output
    assert "fact=2" in result.output and "one warning" in result.output


def test_validate_evals_without_datasets_says_so(config_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = invoke("validate-evals", "-c", str(config_file))
    assert result.exit_code == 0 and "no dataset files under evals/datasets" in result.output


def judged_questions(tmp_path) -> Path:
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.models import Question

    path = tmp_path / "q.jsonl"
    stamp = "2026-09-01T00:00:00Z"
    save_questions(
        path,
        [
            Question(f"q{i}", f"Fixture question number {i} here?", "gold", ["bybit"], "fact",
                     category="fact" if i < 4 else "noisy", annotated_at=stamp)
            for i in range(6)
        ],
    )  # fmt: skip
    return path


def test_round_sample_writes_a_stratified_sample(config_file, tmp_path):
    from crossvenue_rag.dataset.sampling import read_sample_file

    out = tmp_path / "sample.txt"
    questions = judged_questions(tmp_path)
    result = invoke(
        "round-sample", "-c", str(config_file), "-q", str(questions),
        "--n", "3", "--seed", "5", "-o", str(out),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "3 questions, seed 5" in result.output
    ids = read_sample_file(out)
    assert len(ids) == 3 and all(i in result.output for i in ids)
    assert "fact" in result.output and "noisy" in result.output


def test_round_sample_refuses_more_than_the_judged_questions(config_file, tmp_path):
    questions = judged_questions(tmp_path)
    result = invoke(
        "round-sample", "-c", str(config_file), "-q", str(questions), "--n", "9",
        "-o", str(tmp_path / "s.txt"),
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert "exceeds the 6 eligible questions" in str(result.exception)


def two_rounds(tmp_path, *, stamped: bool = True) -> Path:
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.models import Question

    path = tmp_path / "q.jsonl"
    save_questions(
        path, [Question(q, "a?", "gold", ["bybit"], "fact", category="fact") for q in ("q1", "q2")]
    )

    def entry(stamp: str) -> dict:
        base = {"pool": ["c1", "c2"], "relevant": ["c1"], "index_version": "v"}
        return {**base, "annotated_at": stamp} if stamped else base

    both = {"1": entry("2026-09-01T12:00:00Z"), "2": entry("2026-09-20T12:00:00Z")}
    write_journal(path, {"q1": both, "q2": both})
    return path


def test_kappa_reports_the_gap_and_the_readings(tmp_path):
    path = two_rounds(tmp_path)
    exposed = tmp_path / "exposed.txt"
    exposed.write_text("q1\n", encoding="utf-8")
    result = invoke("kappa", "-q", str(path), "--exposed", str(exposed))
    assert result.exit_code == 0, result.output
    assert "retest gap: 19.0 to 19.0 days" in result.output
    assert "not the registered measurement" not in result.output
    assert "Below the target of 0.61, methodology section 5 reports" in result.output


def test_kappa_prints_the_interval_direction_and_sources_after_the_readings(tmp_path):
    result = invoke("kappa", "-q", str(two_rounds(tmp_path)))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "bootstrap over questions: 10,000 resamples, seed 20260924" in out
    order = [
        "kappa by reading",
        "Below the target",
        "kappa interval",
        "direction of",
        "source chunks",
    ]
    assert [out.index(s) for s in order] == sorted(out.index(s) for s in order)


def test_kappa_refuses_an_exposed_list_that_does_not_exist(tmp_path):
    result = invoke("kappa", "-q", str(two_rounds(tmp_path)), "--exposed", "missing.txt")
    assert isinstance(result.exception, UserError)
    assert "missing.txt not found" in str(result.exception)


def test_kappa_prints_each_interval_in_brackets(tmp_path):
    result = invoke("kappa", "-q", str(two_rounds(tmp_path)))
    assert result.exit_code == 0, result.output
    intervals = result.output.split("kappa interval")[1].split("direction of")[0]
    assert "95% interval" in intervals and " to " not in intervals


def test_kappa_without_timestamps_prints_no_gap(tmp_path):
    result = invoke("kappa", "-q", str(two_rounds(tmp_path, stamped=False)))
    assert result.exit_code == 0, result.output
    assert "retest gap" not in result.output


def test_kappa_without_round_history_is_refused(tmp_path):
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.models import Question

    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact")])
    result = invoke("kappa", "-q", str(path))
    assert isinstance(result.exception, UserError)
    assert "no annotation history" in str(result.exception)


def candidate_pool(path: Path) -> Path:
    from tests.dataset.factories import record, save_records

    path.parent.mkdir(parents=True, exist_ok=True)
    save_records(path, [record(id="q1"), record(id="q2")])
    return path


@pytest.fixture
def sessions(monkeypatch) -> list:
    """The selection sessions `select` started, as (candidates, plan)."""

    got: list = []
    monkeypatch.setattr(
        _under_test, "run", lambda cfg, candidates, plan: got.append((candidates, plan))
    )
    return got


def test_select_runs_the_session_on_the_given_pool(config_file, tmp_path, sessions):
    pool = candidate_pool(tmp_path / "pool.jsonl")
    questions = tmp_path / "evals" / "q.jsonl"
    result = invoke(
        "select", "-c", str(config_file), "-d", str(pool), "-q", str(questions),
        "--category", "fact", "--include-skipped",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "(2 candidates)" in result.output
    [(candidates, plan)] = sessions
    assert [c.id for c in candidates] == ["q1", "q2"]
    assert plan.questions_path == str(questions)
    assert plan.selection_file == str(tmp_path / "evals" / "selection" / "selection.jsonl")
    assert plan.categories == ["fact"] and plan.include_skipped


def test_select_defaults_to_the_newest_candidate_pool(config_file, tmp_path, monkeypatch, sessions):
    monkeypatch.chdir(tmp_path)
    candidate_pool(tmp_path / "evals" / "datasets" / "v0.1-candidates.jsonl")
    candidate_pool(tmp_path / "evals" / "datasets" / "v0.10-candidates.jsonl")
    result = invoke("select", "-c", str(config_file), "-q", str(tmp_path / "q.jsonl"))
    assert result.exit_code == 0, result.output
    assert "v0.10-candidates.jsonl" in result.output and len(sessions) == 1


def test_select_refuses_an_unknown_category_before_the_session(config_file, tmp_path, sessions):
    pool = candidate_pool(tmp_path / "pool.jsonl")
    result = invoke("select", "-c", str(config_file), "-d", str(pool), "--category", "nope")
    assert isinstance(result.exception, UserError) and sessions == []


def finalize_with(monkeypatch, outcome) -> None:

    monkeypatch.setattr(_under_test, "finalize_selection", lambda *args: outcome)


def met_composition():
    from crossvenue_rag.dataset.selection import TARGETS, CompositionReport

    return CompositionReport(Counter(TARGETS), {}, TARGETS["comparison"], [])


def finalize_pool(config_file, tmp_path):
    pool = candidate_pool(tmp_path / "pool.jsonl")
    return invoke(
        "select", "-c", str(config_file), "-d", str(pool), "-q", str(tmp_path / "q.jsonl"),
        "--finalize",
    )  # fmt: skip


def test_select_finalize_reports_a_short_set_and_refuses(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.dataset.finalize import Finalized
    from crossvenue_rag.dataset.selection import TARGETS, CompositionReport

    short = CompositionReport(Counter({"fact": 5}), {"fact": 3}, 0, [])
    finalize_with(monkeypatch, Finalized(short))
    result = finalize_pool(config_file, tmp_path)
    assert isinstance(result.exception, UserError)
    assert "short" in result.output and f"target {TARGETS['fact']}" in result.output


def test_select_finalize_without_validation_stops_after_writing(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.dataset.finalize import Finalized

    finalize_with(monkeypatch, Finalized(met_composition(), written=105))
    result = finalize_pool(config_file, tmp_path)
    assert result.exit_code == 0, result.output
    assert "wrote 105 questions" in result.output and "validation" not in result.output


def test_select_finalize_prints_reasons_and_a_passing_validation(
    config_file, tmp_path, monkeypatch
):
    from crossvenue_rag.dataset.finalize import Finalized
    from crossvenue_rag.dataset.validate import ValidationReport

    outcome = Finalized(
        met_composition(),
        written=105,
        reasons=Counter({"not grounded": 2}),
        validation=ValidationReport(warnings=["a warning"]),
    )
    finalize_with(monkeypatch, outcome)
    result = finalize_pool(config_file, tmp_path)
    assert result.exit_code == 0, result.output
    assert "not grounded" in result.output and "a warning" in result.output
    assert "validation ok" in result.output


def test_select_finalize_exits_nonzero_on_a_failed_validation(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.dataset.finalize import Finalized
    from crossvenue_rag.dataset.validate import ValidationReport

    failed = ValidationReport(errors=["bad row"])
    finalize_with(monkeypatch, Finalized(met_composition(), written=105, validation=failed))
    result = finalize_pool(config_file, tmp_path)
    assert result.exit_code == 1
    assert "validation failed" in result.output and "bad row" in result.output


def test_validate_evals_prints_a_bracketed_path_and_venue_as_they_are(
    config_file, tmp_path, monkeypatch
):
    from crossvenue_rag.dataset import validate

    def fake(path, cfg, *, check_near_duplicates: bool = True):
        stats = validate.ValidationStats(total=1, by_venue={"[/x]": 1})
        return validate.ValidationReport(warnings=["venue [bold]v[/bold]"], stats=stats)

    monkeypatch.setattr(_under_test, "validate_dataset", fake)
    path = tmp_path / "c" / "[/x]" / "set.jsonl"
    result = invoke("validate-evals", "-c", str(config_file), "-d", str(path))
    assert result.exit_code == 0, (result.exception, result.output)
    flat = "".join(result.output.split())
    assert str(path) in flat and "[/x]=1" in flat and "venue[bold]v[/bold]" in flat


def test_kappa_with_round_3_is_refused_and_points_to_round_agreement(config_file, tmp_path):
    questions = judged_questions(tmp_path)
    result = invoke("kappa", "-c", str(config_file), "-q", str(questions), "--second", "3")
    assert isinstance(result.exception, UserError)
    assert "round-agreement" in result.exception.with_hint()


def test_round_agreement_prints_no_kappa_and_no_target(config_file, tmp_path):
    from crossvenue_rag.dataset.questions import load_questions

    questions = judged_questions(tmp_path)
    ids = [q.id for q in load_questions(questions)]
    entry = {"pool": ["a", "b"], "relevant": ["a"], "index_version": "v"}
    write_journal(questions, {q: {"1": dict(entry), "3": dict(entry, relevant=["b"])} for q in ids})
    result = invoke(
        "round-agreement", "-c", str(config_file), "-q", str(questions),
        "--metadata-from", str(questions),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "raw agreement" in result.output and "descriptive" in result.output
    assert "kappa" not in result.output.lower() and "target" not in result.output.lower()


def test_round3_gold_refuses_an_unfinished_round_and_writes_nothing(config_file, tmp_path):
    questions = judged_questions(tmp_path)
    out = tmp_path / "out.jsonl"
    base = Path("evals/datasets/v0.2.2-gold.jsonl").resolve()
    result = invoke(
        "round3-gold", "-c", str(config_file), "-q", str(questions),
        "--base", str(base), "-o", str(out),
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert "without a round-3 entry" in str(result.exception) and not out.exists()
