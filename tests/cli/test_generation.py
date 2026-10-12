"""Tests for `cli.generation`: the answer commands with fakes, no network, no database."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from crossvenue_rag.cli import app
from crossvenue_rag.cli import generation as _under_test
from crossvenue_rag.errors import UserError
from crossvenue_rag.generation import judge_sample as judge_sample_module
from crossvenue_rag.generation.records import AnswerRecord, write_answers
from crossvenue_rag.labeling import rate as rate_module
from tests.cli.conftest import current_version, invoke
from tests.generation.factories import (
    matrix_records,
    prompt_record,
    rated_in,
    write_judgements,
    write_ratings,
)


def answer_run(tmp_path: Path, version: str) -> Path:
    run = tmp_path / "results" / "20260101T000000Z_answers"
    run.mkdir(parents=True)
    write_answers(run / "answers.jsonl", matrix_records())
    (run / "index.json").write_text(json.dumps({"index_version": version}))
    return run


def test_rate_sample_writes_a_stratified_sample_of_the_run(config_file, tmp_path):
    run = answer_run(tmp_path, current_version(config_file))
    out = tmp_path / "sample.txt"
    # The fixture run answers none of the questions the default config excludes.
    result = CliRunner().invoke(
        app,
        ["rate-sample", "--run", str(run), "-o", str(out), "-c", str(config_file),
         "--set", "generation.calibration_exclude=[]"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert result.output.startswith("40 answers -> ")
    ids = [line.split()[0] for line in out.read_text().splitlines() if not line.startswith("#")]
    assert len(ids) == len(set(ids)) == 40


def test_rate_sample_prints_a_bracketed_output_path_as_written(config_file, tmp_path):
    run = answer_run(tmp_path, current_version(config_file))
    for name in ("[/x]", "[bold]"):
        out = tmp_path / name / "sample.txt"
        out.parent.mkdir(parents=True)
        result = CliRunner().invoke(
            app,
            ["rate-sample", "--run", str(run), "-o", str(out), "-c", str(config_file),
             "--set", "generation.calibration_exclude=[]"],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert name in " ".join(result.output.split())


def test_a_run_made_against_another_index_is_refused(config_file, tmp_path):
    run = answer_run(tmp_path, "0000000000000000")
    result = CliRunner().invoke(
        app,
        ["rate-sample", "--run", str(run), "-o", str(tmp_path / "s.txt"), "-c", str(config_file)],
    )
    assert isinstance(result.exception, UserError)
    assert "was made against index 0000000000000000" in str(result.exception)
    assert not (tmp_path / "s.txt").exists()


def test_judge_agreement_needs_the_judges_grades(config_file, tmp_path):
    run = answer_run(tmp_path, current_version(config_file))
    result = CliRunner().invoke(
        app, ["judge-agreement", "--run", str(run), "--judge", "opus", "-c", str(config_file)]
    )
    assert isinstance(result.exception, UserError)
    assert "no judgments by opus" in str(result.exception)


def test_ask_prints_the_claims_with_their_citations(config_file, monkeypatch):
    import crossvenue_rag.generation.ask as ask_module
    import crossvenue_rag.generation.llm as llm_module
    from tests.generation.offline import (
        ASK_ANSWER,
        SECRET,
        DetectingRetriever,
        FakeTransport,
        openrouter_payload,
    )

    class ContextRetriever(DetectingRetriever):
        def __init__(self, cfg):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def require_current_index(self):
            return "v"

    real_llm = llm_module.LLM
    transport = FakeTransport(respond=lambda url, body: openrouter_payload(ASK_ANSWER))
    monkeypatch.setattr(ask_module, "Retriever", ContextRetriever)
    monkeypatch.setattr(ask_module, "LLM", lambda gen: real_llm(gen, transport, api_key=SECRET))
    result = CliRunner().invoke(app, ["ask", "binance testnet ip ban?", "-c", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "bans scale 2m to 3d" in result.output
    assert "testnet/rest-api.md" in result.output
    assert SECRET not in result.output


def test_answer_eval_estimate_prints_the_cost_and_makes_no_run(config_file, monkeypatch):
    from crossvenue_rag.generation.estimate import EstimateRow

    jobs: list = []

    def estimate(cfg, job):
        jobs.append(job)
        return [EstimateRow("gemini@venue_split", "google/gemini-3.8-flash", 2, 1, 0.5, 0.5)]

    monkeypatch.setattr(_under_test, "estimate_answer_eval", estimate)
    monkeypatch.setattr(_under_test, "answer_eval", None)
    result = invoke(
        "answer-eval", "-c", str(config_file), "-q", "qs.jsonl", "--question-id", "cv-0001",
        "--question-id", "cv-0002", "--judge", "opus", "--budget", "0.5", "--estimate",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "cost estimate (no call made)" in result.output
    assert "$0.500" in result.output
    [job] = jobs
    assert job.questions_path == Path("qs.jsonl")
    assert (job.question_ids, job.judge, job.budget) == (("cv-0001", "cv-0002"), "opus", 0.5)


def test_answer_eval_prints_progress_the_report_and_the_run(config_file, tmp_path, monkeypatch):

    jobs: list = []

    def answer_eval(cfg, job, *, progress):
        jobs.append(job)
        progress("gemini@venue_split:cv-0001")
        run_dir = tmp_path / "run_answers"
        run_dir.mkdir()
        (run_dir / "report.md").write_text("# the run report\n", encoding="utf-8")
        return run_dir

    monkeypatch.setattr(_under_test, "answer_eval", answer_eval)
    result = invoke("answer-eval", "-c", str(config_file), "--note", "first try")
    assert result.exit_code == 0, result.output
    assert "gemini@venue_split:cv-0001" in result.output
    assert "# the run report" in result.output
    assert "results:" in result.output and "run_answers" in result.output
    [job] = jobs
    assert job.questions_path == Path("evals/questions.jsonl")
    assert (job.question_ids, job.judge, job.budget, job.note) == ((), None, None, "first try")


def rated_records() -> list[AnswerRecord]:
    """Two generated answers to QUESTIONS, each without a claim."""
    from tests.generation.factories import (
        answer_json,
        call_json,
        generation_json,
        record,
    )
    from tests.generation.offline import (
        QUESTIONS,
    )

    return [
        record(
            "gemini@venue_split",
            q.id,
            category=q.category,
            expected_behavior=q.expected_behavior,
            generation=generation_json(
                answer=answer_json(claims=0, refused=True),
                chunk_ids=(),
                call=call_json(cache_key=q.id),
            ),
        )
        for q in QUESTIONS
    ]


def test_rate_grades_the_sample_of_the_run(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.dataset.questions import save_questions
    from crossvenue_rag.generation.calibration_sample import write_calibration_sample
    from tests.generation.offline import (
        QUESTIONS,
    )

    version = current_version(config_file)
    run = answer_run(tmp_path, version)
    write_answers(run / "answers.jsonl", rated_records())
    sample = tmp_path / "sample.txt"
    write_calibration_sample(sample, run, rated_records(), 1, excluded=())
    questions = tmp_path / "qs.jsonl"
    save_questions(questions, QUESTIONS)
    calls: list = []

    def rate_session(views, ratings_path, rater, show, index_version=None):
        calls.append(([v.answer_id for v in views], rater, index_version))
        return len(views)

    monkeypatch.setattr(rate_module, "rate_session", rate_session)
    result = invoke(
        "rate", "-c", str(config_file), "-q", str(questions), "--sample-file", str(sample),
        "--rater", "tester",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "2 ratings written to" in result.output
    assert calls == [
        (["gemini@venue_split:cv-0001", "gemini@venue_split:cv-0002"], "tester", version)
    ]


def test_rate_needs_a_sample_that_names_its_run(config_file, tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("gemini@venue_split:cv-0001  x\n", encoding="utf-8")
    result = invoke("rate", "-c", str(config_file), "--sample-file", str(sample))
    assert isinstance(result.exception, UserError)
    assert "names no run" in str(result.exception)


def test_judge_sample_prints_counts_only(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.generation.jobs import JudgeSampleOutcome

    jobs: list = []

    def judge_sample(cfg, job):
        jobs.append(job)
        return JudgeSampleOutcome(
            tmp_path / "judgements_opus_r1.jsonl", 3, 1, 4, 0.0123, stopped=None
        )

    monkeypatch.setattr(_under_test, "judge_sample", judge_sample)
    result = invoke(
        "judge-sample", "--judge", "opus", "-c", str(config_file), "--sample-file", "s.txt",
        "--replicate", "1", "--budget", "2",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "3 of 4 answers judged, 1 judge error, $0.012 spent ->" in result.output
    [job] = jobs
    assert (job.judge, job.sample_file, job.replicate, job.budget) == (
        "opus",
        Path("s.txt"),
        1,
        2.0,
    )


def test_judge_agreement_compares_with_the_human_and_the_repeat(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.generation.judge_sample import human_ratings

    run = answer_run(tmp_path, current_version(config_file))
    write_judgements(run, "opus")
    write_judgements(run, "opus_r1", flip=True)
    humans = human_ratings(write_ratings(tmp_path / "r.jsonl"))
    monkeypatch.setattr(judge_sample_module, "human_ratings", lambda: humans)
    result = invoke("judge-agreement", "--run", str(run), "--judge", "opus", "-c", str(config_file))
    assert result.exit_code == 0, result.output
    assert "opus vs human" in result.output
    assert "opus vs opus (replicate 1)" in result.output
    assert "no human ratings" not in result.output


def test_judge_agreement_without_human_ratings_says_so(config_file, tmp_path, monkeypatch):
    run = answer_run(tmp_path, current_version(config_file))
    write_judgements(run, "opus")
    monkeypatch.setattr(judge_sample_module, "human_ratings", lambda: {})
    result = invoke("judge-agreement", "--run", str(run), "--judge", "opus", "-c", str(config_file))
    assert result.exit_code == 0, result.output
    assert "no human ratings" in result.output
    assert "replicate 1" not in result.output


def test_select_judge_writes_the_calibration_files(config_file, tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    write_judgements(run, "sonnet", cost=0.01)
    write_judgements(run, "opus", flip=True, cost=0.03)
    rated_in(monkeypatch, write_ratings(tmp_path / "ratings.jsonl"))
    result = invoke(
        "select-judge", "--run", str(run), "-c", str(config_file),
        "--set", "eval.bootstrap_resamples=50",
        "--set", "generation.calibration_exclude_behaviour=[]",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Selected: sonnet." in result.output
    assert "written:" in result.output
    assert json.loads((run / "calibration.json").read_text())["selected"] == "sonnet"
    assert (run / "calibration.md").exists()


def test_judge_run_prints_the_outcome_and_a_budget_stop(config_file, tmp_path, monkeypatch):
    from crossvenue_rag.generation.jobs import JudgeRunOutcome

    jobs: list = []

    def judge_run(cfg, job, *, progress):
        jobs.append(job)
        progress("gemini@venue_split:cv-0001")
        return JudgeRunOutcome(tmp_path / "x_judged", 1, 4, 0, 0.0042, "budget reached")

    monkeypatch.setattr(_under_test, "judge_run", judge_run)
    result = invoke(
        "judge-run", "--run", str(tmp_path / "src"), "--judge", "sonnet", "-c", str(config_file),
        "--note", "again",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "gemini@venue_split:cv-0001" in result.output
    assert "1 of 4 answers, 0 judge errors, $0.004 spent ->" in result.output
    assert "stopped: budget reached" in result.output
    [job] = jobs
    assert (job.run, job.judge, job.budget, job.note) == (tmp_path / "src", "sonnet", None, "again")


def test_compare_prompts_writes_the_comparison(config_file, tmp_path):
    version = current_version(config_file)
    runs = []
    for name, vae, schema in (("v1", 0.5, None), ("v2", 0.0, 2)):
        run = tmp_path / name
        run.mkdir()
        records = [
            prompt_record(config, f"q{i}", "answer", vae=vae, schema_version=schema)
            for config in ("gemini@venue_split", "qwen@venue_split", "mercury@venue_split",
                           "gemini@hybrid_rerank")
            for i in range(3)
        ]  # fmt: skip
        write_answers(run / "answers.jsonl", records)
        (run / "index.json").write_text(json.dumps({"index_version": version}))
        runs.append(run)
    result = invoke(
        "compare-prompts", "--v1", str(runs[0]), "--v2", str(runs[1]), "-c", str(config_file),
        "--set", "eval.bootstrap_resamples=200",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "results:" in result.output
    [out] = (tmp_path / "results").glob("*_prompts")
    assert (out / "report.md").exists()


def test_judge_sample_and_judge_run_print_a_bracketed_output_path_as_it_is(
    config_file, tmp_path, monkeypatch
):
    from crossvenue_rag.generation.jobs import JudgeRunOutcome, JudgeSampleOutcome

    out = tmp_path / "[/x]"
    monkeypatch.setattr(
        _under_test,
        "judge_sample",
        lambda cfg, job: JudgeSampleOutcome(out / "judgements.jsonl", 1, 0, 1, 0.0, "[/s]"),
    )
    monkeypatch.setattr(
        _under_test,
        "judge_run",
        lambda cfg, job, *, progress: JudgeRunOutcome(out / "judged", 1, 1, 0, 0.0, None),
    )
    sampled = invoke("judge-sample", "--judge", "opus", "-c", str(config_file))
    judged = invoke("judge-run", "--run", str(tmp_path), "-c", str(config_file))
    for result, path in ((sampled, out / "judgements.jsonl"), (judged, out / "judged")):
        assert result.exit_code == 0, (result.exception, result.output)
        assert str(path) in "".join(result.output.split())
    assert "stopped: [/s]" in sampled.output
