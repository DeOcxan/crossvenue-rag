"""Tests for `generation.jobs`: the answer and judge runs, offline, no database."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest

from crossvenue_rag import jsonl
from crossvenue_rag.config import Config
from crossvenue_rag.config_generation import GenerationConfig
from crossvenue_rag.config_loading import DEFAULT_CONFIG, load_config
from crossvenue_rag.dataset.questions import save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.generation import jobs as _under_test
from crossvenue_rag.generation.jobs import (
    AnswerEvalJob,
    JudgeRunJob,
    JudgeRunOutcome,
    JudgeSampleJob,
    answer_eval,
    estimate_answer_eval,
    judge_run,
    judge_sample,
    select_questions,
)
from crossvenue_rag.generation.llm import LLM, LLMFactory
from crossvenue_rag.generation.records import write_answers
from crossvenue_rag.index_version import require_current_index, write_index_meta
from crossvenue_rag.retrieval.retriever import Retriever
from tests.conftest import build_corpus, write_journal
from tests.generation.factories import (
    prompt_record,
)
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    FakeTransport,
    TwoChunkRetriever,
    calibration_config,
    openrouter_payload,
    respond,
)

VERSION = "v-test"


class ContextRetriever(TwoChunkRetriever):
    """The fake retriever as `Retriever(cfg)` is used: a context manager on an index."""

    def __init__(self, cfg):
        self.by_id = CHUNKS
        self.warmed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def require_current_index(self):
        return VERSION

    def warmup(self, *, load_reranker=True):
        self.warmed = True


# The jobs as their commands build them without options: every question, the configured
# judge, the config's budget, no note, the first grading.
answer_job = partial(AnswerEvalJob, question_ids=(), judge=None, budget=None, note="")
run_job = partial(JudgeRunJob, judge=None, budget=None, note="")
sample_job = partial(JudgeSampleJob, replicate=0, budget=None)


def questions_file(tmp_path: Path, version: str = VERSION) -> Path:
    """QUESTIONS in a file whose round-1 labels were made against `version`."""
    path = tmp_path / "evals" / "questions.jsonl"
    save_questions(path, QUESTIONS)
    write_journal(path, {q.id: {"1": {"index_version": version}} for q in QUESTIONS})
    return path


def fake_llm(answer=respond) -> FakeTransport:
    """A scripted transport for the model client `llm_on` builds."""
    return FakeTransport(respond=answer)


def llm_on(transport: FakeTransport) -> LLMFactory:
    """The jobs' `make_llm`: a model client on `transport`."""

    def make(gen: GenerationConfig) -> LLM:
        return LLM(gen, transport, api_key=SECRET)

    return make


def open_context(cfg: Config) -> Retriever:
    """The jobs' `open_retriever`: a `ContextRetriever`."""
    # The jobs read only what the fake has: chunks by id, ranks, the index and the warmup.
    return ContextRetriever(cfg)  # type: ignore[return-value]


def unparseable_judge(url, body):
    if body["model"].startswith("anthropic/"):
        return openrouter_payload("not json", model=body["model"], provider="Anthropic")
    return respond(url, body)


@pytest.fixture
def offline(monkeypatch):
    """No index build: the current index and the chunks are fakes."""
    monkeypatch.setattr(_under_test, "require_current_index", lambda cfg: VERSION)
    monkeypatch.setattr(_under_test, "load_chunks", lambda cfg: list(CHUNKS.values()))


def source_run(tmp_path: Path) -> Path:
    """A generated (unjudged) answer run under tmp_path/results."""
    job = answer_job(questions_path=questions_file(tmp_path))
    return answer_eval(
        calibration_config(tmp_path),
        job,
        progress=lambda _id: None,
        make_llm=llm_on(fake_llm()),
        open_retriever=open_context,
    )


def test_no_ids_selects_every_question():
    assert select_questions(QUESTIONS, (), Path("q.jsonl")) == QUESTIONS


def test_ids_select_their_questions_in_file_order():
    chosen = select_questions(QUESTIONS, ["cv-0002", "cv-0001"], Path("q.jsonl"))
    assert [q.id for q in chosen] == ["cv-0001", "cv-0002"]


def test_an_unknown_id_is_refused_by_name():
    with pytest.raises(UserError, match=r"not in q.jsonl: 'cv-9998', 'cv-9999'"):
        select_questions(QUESTIONS, ["cv-9999", "cv-0001", "cv-9998"], Path("q.jsonl"))


def test_answer_eval_generates_judges_and_writes_the_run(tmp_path, offline):
    transport = fake_llm()
    cfg = calibration_config(tmp_path)
    path = questions_file(tmp_path)
    seen: list[str] = []
    job = answer_job(questions_path=path, judge="sonnet", budget=1.5, note="a note")
    run_dir = answer_eval(
        cfg, job, progress=seen.append, make_llm=llm_on(transport), open_retriever=open_context
    )
    assert run_dir.parent == tmp_path / "results" and run_dir.name.endswith("_answers")
    assert seen == [
        "gemini@venue_split:cv-0001",
        "gemini@venue_split:cv-0002",
        "gemini@hybrid_rerank:cv-0001",
        "gemini@hybrid_rerank:cv-0002",
    ]
    records = jsonl.read(run_dir / "answers.jsonl")
    assert all(r["judge"]["judge"] == "sonnet" and r["error"] is None for r in records)
    assert json.loads((run_dir / "index.json").read_text())["index_version"] == VERSION
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "a note" in report and "of a $1.500 budget" in report
    assert "Stopped early" not in report
    assert cfg.eval.questions != path  # the caller's config is not changed
    assert f"questions: {path}" in (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert transport.calls and SECRET not in (run_dir / "answers.jsonl").read_text()


def test_answer_eval_answers_only_the_chosen_questions_unjudged(tmp_path, offline):
    transport = fake_llm()
    job = answer_job(questions_path=questions_file(tmp_path), question_ids=["cv-0002"])
    run_dir = answer_eval(
        calibration_config(tmp_path),
        job,
        progress=lambda _id: None,
        make_llm=llm_on(transport),
        open_retriever=open_context,
    )
    records = jsonl.read(run_dir / "answers.jsonl")
    assert {r["question_id"] for r in records} == {"cv-0002"}
    assert all(r["judge"] is None for r in records)


def test_answer_eval_records_a_budget_stop_in_the_report(tmp_path, offline):
    transport = fake_llm()
    job = answer_job(questions_path=questions_file(tmp_path), budget=0.001)
    run_dir = answer_eval(
        calibration_config(tmp_path),
        job,
        progress=lambda _id: None,
        make_llm=llm_on(transport),
        open_retriever=open_context,
    )
    assert len(jsonl.read(run_dir / "answers.jsonl")) == 1
    assert "Stopped early:" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_an_empty_matrix_is_refused(tmp_path, offline):
    transport = fake_llm()
    cfg = calibration_config(tmp_path)
    cfg.generation.matrix = []
    with pytest.raises(UserError, match=r"generation\.matrix is empty") as refused:
        answer_eval(
            cfg,
            answer_job(questions_path=questions_file(tmp_path)),
            progress=print,
            make_llm=llm_on(transport),
            open_retriever=open_context,
        )
    # The hint names the key and an override that fills it, which a config accepts. A
    # config whose matrix is empty names no primary pair, which would name a configuration.
    override = refused.value.hint.split("--set ")[1].strip("'")
    overrides = ["generation.primary_pair=[]", "generation.matrix=[]"]
    assert not load_config(DEFAULT_CONFIG, overrides).generation.matrix
    matrix = load_config(DEFAULT_CONFIG, [*overrides, override]).generation.matrix
    assert [m.name for m in matrix] == ["gemini@venue_split"]


def test_an_unknown_judge_is_refused_before_any_call(tmp_path, offline):
    transport = fake_llm()
    job = answer_job(questions_path=questions_file(tmp_path), judge="nobody")
    with pytest.raises(UserError, match=r"--judge 'nobody' is not a judge of generation\.judges"):
        answer_eval(
            calibration_config(tmp_path),
            job,
            progress=print,
            make_llm=llm_on(transport),
            open_retriever=open_context,
        )
    assert transport.calls == []


def test_labels_made_against_another_index_are_refused(tmp_path, offline):
    transport = fake_llm()
    job = answer_job(questions_path=questions_file(tmp_path, version="other"))
    with pytest.raises(UserError, match="were not made against the current index v-test"):
        answer_eval(
            calibration_config(tmp_path),
            job,
            progress=print,
            make_llm=llm_on(transport),
            open_retriever=open_context,
        )
    assert transport.calls == []
    assert not (tmp_path / "results").exists()


def test_estimate_makes_no_call_and_counts_cached_answers(tmp_path, offline):
    transport = fake_llm()
    cfg = calibration_config(tmp_path)
    job = answer_job(questions_path=questions_file(tmp_path), judge="sonnet")
    before = estimate_answer_eval(cfg, job, make_llm=llm_on(transport), open_retriever=open_context)
    assert transport.calls == []
    assert [(r.item, r.calls, r.cached) for r in before] == [
        ("gemini@venue_split", 2, 0),
        ("gemini@hybrid_rerank", 2, 0),
        ("judge sonnet", 4, None),
    ]
    assert all(r.mean_cost_usd is None and r.estimate_usd is None for r in before)

    answer_eval(
        cfg, job, progress=lambda _id: None, make_llm=llm_on(transport), open_retriever=open_context
    )
    after = estimate_answer_eval(cfg, job, make_llm=llm_on(transport), open_retriever=open_context)
    assert [r.cached for r in after] == [2, 2, None]
    assert after[0].estimate_usd == 0.0
    assert after[2].mean_cost_usd == pytest.approx(0.0012)


def test_judge_run_judges_a_saved_run_and_records_its_source(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm()
    seen: list[str] = []
    job = run_job(
        run=source, questions_path=questions_file(tmp_path), judge="opus", note="rejudged"
    )
    outcome = judge_run(
        calibration_config(tmp_path), job, progress=seen.append, make_llm=llm_on(transport)
    )
    assert isinstance(outcome, JudgeRunOutcome)
    assert (outcome.judged, outcome.answers, outcome.failed, outcome.stopped) == (4, 4, 0, None)
    assert outcome.spent_usd == pytest.approx(4 * 0.0012)
    assert len(seen) == 4 and len(transport.calls) == 4
    assert all(body["model"].startswith("anthropic/") for _, _, body in transport.calls)
    assert outcome.run_dir.name.endswith("_judged")
    recorded = json.loads((outcome.run_dir / "source.json").read_text())
    assert recorded["source_run"] == str(source) and recorded["judge"] == "opus"
    assert recorded["stops"] == [] and recorded["spent_usd"] == pytest.approx(outcome.spent_usd)
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    assert f"Judged by opus from `{source}`; answers copied unchanged. rejudged" in report
    judged = jsonl.read(outcome.run_dir / "answers.jsonl")
    source_records = jsonl.read(source / "answers.jsonl")
    assert [r["generation"] for r in judged] == [r["generation"] for r in source_records]


def test_judge_run_uses_the_configured_judge_by_default(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm()
    cfg = calibration_config(tmp_path)
    cfg.generation.judge = "sonnet"
    outcome = judge_run(
        cfg,
        run_job(run=source, questions_path=questions_file(tmp_path)),
        progress=print,
        make_llm=llm_on(transport),
    )
    assert json.loads((outcome.run_dir / "source.json").read_text())["judge"] == "sonnet"


def test_judge_run_without_a_judge_is_refused(tmp_path, offline):
    job = run_job(run=tmp_path / "run", questions_path=questions_file(tmp_path))
    with pytest.raises(UserError, match="no judge chosen"):
        judge_run(calibration_config(tmp_path), job, progress=print)


def test_judge_run_refuses_a_run_from_another_index(tmp_path, offline):
    source = source_run(tmp_path)
    (source / "index.json").write_text(json.dumps({"index_version": "old"}))
    job = run_job(run=source, questions_path=questions_file(tmp_path), judge="opus")
    with pytest.raises(UserError, match="was made against index old"):
        judge_run(calibration_config(tmp_path), job, progress=print)


def test_judge_run_counts_failed_judgements(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm(unparseable_judge)
    job = run_job(run=source, questions_path=questions_file(tmp_path), judge="opus")
    outcome = judge_run(
        calibration_config(tmp_path), job, progress=print, make_llm=llm_on(transport)
    )
    assert (outcome.judged, outcome.answers, outcome.failed) == (0, 4, 4)


def test_judge_run_does_not_count_an_answer_whose_generation_failed_as_judged(tmp_path, offline):
    from crossvenue_rag.generation.records import read_answers

    source = source_run(tmp_path)
    saved = read_answers(source / "answers.jsonl")
    failed = {"generation": None, "error": "ModelCallError: 500"}
    write_answers(
        source / "answers.jsonl",
        [r.model_copy(update=failed) if i < 2 else r for i, r in enumerate(saved)],
    )
    transport = fake_llm()
    job = run_job(run=source, questions_path=questions_file(tmp_path), judge="opus")
    outcome = judge_run(
        calibration_config(tmp_path), job, progress=print, make_llm=llm_on(transport)
    )
    assert (outcome.judged, outcome.answers, outcome.failed) == (2, 4, 0)


def test_judge_run_stops_at_the_budget(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm()
    job = run_job(run=source, questions_path=questions_file(tmp_path), judge="opus", budget=0.001)
    outcome = judge_run(
        calibration_config(tmp_path), job, progress=print, make_llm=llm_on(transport)
    )
    assert (outcome.judged, outcome.answers) == (1, 4)
    assert outcome.stopped is not None
    assert json.loads((outcome.run_dir / "source.json").read_text())["stops"] == [outcome.stopped]


def sample_file(tmp_path: Path, run: Path) -> Path:
    from crossvenue_rag.generation.calibration_sample import write_calibration_sample
    from crossvenue_rag.generation.records import read_answers

    records = [r for r in read_answers(run / "answers.jsonl") if r.config == "gemini@venue_split"]
    path = tmp_path / "sample.txt"
    write_calibration_sample(path, run, records, 1, ())
    return path


def test_judge_sample_writes_the_judgements_and_returns_counts(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm()
    job = sample_job(
        judge="sonnet",
        sample_file=sample_file(tmp_path, source),
        questions_path=questions_file(tmp_path),
    )
    outcome = judge_sample(calibration_config(tmp_path), job, make_llm=llm_on(transport))
    assert outcome.path == source / "judgements_sonnet.jsonl"
    assert (outcome.judged, outcome.errors, outcome.sampled) == (2, 0, 2)
    assert outcome.spent_usd == pytest.approx(2 * 0.0012) and len(transport.calls) == 2
    assert [r["answer_id"] for r in jsonl.read(outcome.path)] == [
        "gemini@venue_split:cv-0001",
        "gemini@venue_split:cv-0002",
    ]


def test_a_replicate_is_a_fresh_call_into_its_own_file(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm()
    sample, questions = sample_file(tmp_path, source), questions_file(tmp_path)
    judge_sample(
        calibration_config(tmp_path),
        sample_job("opus", sample, questions),
        make_llm=llm_on(transport),
    )
    transport = fake_llm()
    outcome = judge_sample(
        calibration_config(tmp_path),
        sample_job("opus", sample, questions, replicate=1),
        make_llm=llm_on(transport),
    )
    assert outcome.path.name == "judgements_opus_r1.jsonl"
    assert len(transport.calls) == 2


def test_judge_sample_counts_parse_errors(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm(unparseable_judge)
    job = sample_job("opus", sample_file(tmp_path, source), questions_file(tmp_path))
    outcome = judge_sample(calibration_config(tmp_path), job, make_llm=llm_on(transport))
    assert (outcome.judged, outcome.errors, outcome.sampled) == (0, 2, 2)


def test_judge_sample_refuses_an_unknown_judge(tmp_path, offline):
    job = sample_job("nobody", tmp_path / "sample.txt", tmp_path / "q.jsonl")
    with pytest.raises(UserError, match="'nobody' is not a judge"):
        judge_sample(calibration_config(tmp_path), job)


def test_judge_sample_reports_the_budget_stop_that_ended_it(tmp_path, offline):
    source = source_run(tmp_path)
    transport = fake_llm()
    job = sample_job(
        judge="sonnet",
        sample_file=sample_file(tmp_path, source),
        questions_path=questions_file(tmp_path),
        budget=0.001,
    )
    outcome = judge_sample(calibration_config(tmp_path), job, make_llm=llm_on(transport))
    assert (outcome.judged, outcome.sampled) == (1, 2)
    assert outcome.stopped is not None and "budget" in outcome.stopped


def prompt_runs(cfg, tmp_path, versions):
    """Judged runs v1 and v2, made against `versions`, each config answering three questions."""
    runs = []
    for name, vae, version, schema in zip(
        ("v1", "v2"), (0.5, 0.0), versions, (None, 2), strict=True
    ):
        run = tmp_path / name
        run.mkdir()
        records = [
            prompt_record(m.name, f"q{i}", "answer", vae=vae, schema_version=schema)
            for m in cfg.generation.matrix
            for i in range(3)
        ]
        write_answers(run / "answers.jsonl", records)
        (run / "index.json").write_text(json.dumps({"index_version": version}))
        runs.append(run)
    return runs


def test_prompt_runs_are_compared_and_written(cfg, tmp_path):
    write_index_meta(cfg, len(build_corpus(cfg)))
    cfg.eval.bootstrap_resamples = 200
    version = require_current_index(cfg)
    run_dir = _under_test.compare_prompt_runs(cfg, *prompt_runs(cfg, tmp_path, [version] * 2))
    assert run_dir.name.endswith("_prompts")
    assert (run_dir / "report.md").exists()


def test_a_prompt_run_made_against_another_index_is_refused(cfg, tmp_path):
    write_index_meta(cfg, len(build_corpus(cfg)))
    versions = [require_current_index(cfg), "0" * 16]
    with pytest.raises(UserError, match="0000000000000000"):
        _under_test.compare_prompt_runs(cfg, *prompt_runs(cfg, tmp_path, versions))


def test_prompt_runs_swapped_or_repeated_are_refused(cfg, tmp_path):
    write_index_meta(cfg, len(build_corpus(cfg)))
    v1, v2 = prompt_runs(cfg, tmp_path, [require_current_index(cfg)] * 2)
    with pytest.raises(UserError, match="--v1 and --v2 name the same run"):
        _under_test.compare_prompt_runs(cfg, v1, v1)
    with pytest.raises(UserError, match=r"--v1 takes answers of answer schema 1, and .* schema 2$"):
        _under_test.compare_prompt_runs(cfg, v2, v1)


def test_a_judge_run_stopped_by_its_budget_is_reported_partial_and_records_that_budget(
    tmp_path, offline
):
    source = source_run(tmp_path)
    transport = fake_llm()
    job = run_job(run=source, questions_path=questions_file(tmp_path), judge="opus", budget=0.002)
    outcome = judge_run(
        calibration_config(tmp_path), job, progress=print, make_llm=llm_on(transport)
    )
    assert outcome.stopped is not None and outcome.judged == 1
    report = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Stopped early: run budget" in report
    assert "Partial run: 1 of 4 answers copied, 1 judged." in report
    config = (outcome.run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "budget_usd: 0.002" in config
