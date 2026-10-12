"""Tests for `cli.evaluate`: argument errors are raised before any model or database is opened, and
every command runs end to end on the fixture corpus with an in-memory retriever."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from crossvenue_rag.cli import app
from crossvenue_rag.config import Config
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.corpus.artifacts import load_chunks
from crossvenue_rag.dataset.questions import save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.evaluation.machine import MachineState
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.index_version import require_current_index
from crossvenue_rag.models import Question
from tests.cli.conftest import current_version
from tests.conftest import InMemoryRetriever, write_journal
from tests.evaluation.factories import fake_run


def test_a_filter_without_a_value_is_refused(config_file):
    result = CliRunner().invoke(
        app, ["retrieve", "ban length", "-c", str(config_file), "--filter", "testnet"]
    )
    assert isinstance(result.exception, UserError)
    assert "bad --filter 'testnet'" in str(result.exception)


def test_an_unknown_method_is_refused(config_file):
    result = CliRunner().invoke(app, ["retrieve", "ban", "-c", str(config_file), "-m", "magic"])
    assert isinstance(result.exception, UserError)


class MemoryRetriever(InMemoryRetriever):
    """`Retriever` as the commands open it, over the fixture corpus: no model, no Postgres."""

    def __init__(self, cfg: Config) -> None:
        chunks = load_chunks(cfg)
        super().__init__(cfg, chunks, BM25Index.build(chunks))
        self.warmups: list[bool] = []

    def warmup(self, *, load_reranker: bool = True) -> None:
        self.warmups.append(load_reranker)

    def require_current_index(self) -> str:
        return require_current_index(self.cfg)


@pytest.fixture
def memory_retriever(monkeypatch):
    # Every module that builds a retriever: the commands, `compare_methods` and the bench.
    for module in ("retrieval.retriever", "evaluation.bench"):
        monkeypatch.setattr(f"crossvenue_rag.{module}.Retriever", MemoryRetriever)


def labelled_questions(config_file: Path, tmp_path: Path) -> Path:
    """A question file labeled against the current index, gold in the fixture corpus."""
    cfg = load_config(config_file)
    by_venue = {c.venue: c.chunk_id for c in load_chunks(cfg)}
    path = tmp_path / "evals" / "questions.jsonl"
    questions = [
        Question("cv-0001", "binance and bybit rate limits per second", "gold",
                 ["binance", "bybit"], "comparison", [by_venue["binance"], by_venue["bybit"]],
                 category="comparison"),
        Question("cv-0002", "uniswap fee tiers", "gold", ["uniswap"], "fact",
                 [by_venue["uniswap"]], category="fact"),
    ]  # fmt: skip
    save_questions(path, questions)
    entry = {"index_version": current_version(config_file), "pool": [], "relevant": []}
    write_journal(path, {q.id: {"1": entry} for q in questions})
    return path


def test_retrieve_prints_each_methods_results(config_file, memory_retriever):
    result = CliRunner().invoke(
        app, ["retrieve", "fee tiers", "-c", str(config_file), "-m", "all", "--k", "2",
              "--venue", "uniswap", "--venue", "pancakeswap", "--venue-type", "dex"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "bm25" in result.output and "hybrid_rerank" in result.output
    assert "binance" not in result.output


def test_eval_runs_every_method_and_writes_the_results(config_file, tmp_path, memory_retriever):
    questions = labelled_questions(config_file, tmp_path)
    result = CliRunner().invoke(
        app, ["eval", "-c", str(config_file), "-q", str(questions), "--venue-filter"]
    )
    assert result.exit_code == 0, result.output
    run_dir = Path(result.output.split("results:")[-1].strip().replace("\n", ""))
    assert (run_dir / "report.md").exists() and (run_dir / "per_query.csv").exists()
    # The resolved config names the question file that ran, which overrides the default.
    assert str(questions) in (run_dir / "config.yaml").read_text().replace("\n", "")


def test_eval_refuses_gold_the_corpus_does_not_contain(config_file, tmp_path, memory_retriever):
    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("cv-1", "x", "gold", ["binance"], "fact", ["missing"])])
    entry = {"index_version": current_version(config_file)}
    write_journal(path, {"cv-1": {"1": entry}})
    result = CliRunner().invoke(app, ["eval", "-c", str(config_file), "-q", str(path)])
    assert isinstance(result.exception, UserError)
    assert "1 gold chunk id not in the current corpus" in str(result.exception)


def test_pool_run_writes_a_pool_directory(config_file, tmp_path, memory_retriever):
    questions = labelled_questions(config_file, tmp_path)
    result = CliRunner().invoke(
        app, ["pool-run", "-m", "bm25", "-m", "venue_split", "--k", "12",
              "--category", "comparison", "-c", str(config_file), "-q", str(questions)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "2 retrievals, k=12" in result.output
    [run_dir] = (tmp_path / "results").iterdir()
    assert run_dir.name.endswith("_pool") and (run_dir / "per_query.csv").exists()


def test_pool_run_without_k_reports_the_configured_depth(config_file, tmp_path, memory_retriever):
    questions = labelled_questions(config_file, tmp_path)
    result = CliRunner().invoke(
        app, ["pool-run", "-m", "bm25", "--category", "comparison", "-c", str(config_file),
              "-q", str(questions), "--set", "eval.pool_depth_per_method=12"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "1 retrieval, k=12" in result.output


def test_pool_run_writes_its_depth_through_the_formatting_module_in_its_own_form(
    config_file, tmp_path, memory_retriever, monkeypatch
):
    questions = labelled_questions(config_file, tmp_path)
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    result = CliRunner().invoke(
        app, ["pool-run", "-m", "bm25", "--k", "12", "--category", "comparison",
              "-c", str(config_file), "-q", str(questions)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "1 retrieval, k=<12>" in result.output


def test_pool_run_refuses_an_unknown_method(config_file):
    result = CliRunner().invoke(app, ["pool-run", "-m", "magic", "-c", str(config_file)])
    assert isinstance(result.exception, UserError)
    assert "--method 'magic' is not a retrieval method" in str(result.exception)


def test_bench_rerank_times_the_split_questions(
    config_file, tmp_path, memory_retriever, monkeypatch
):
    idle = MachineState(
        measured_at="t",
        cpu_count=8,
        load_average=(0.1, 0.1, 0.1),
        competing_processes=(),
        busy_processes=(),
    )
    monkeypatch.setattr("crossvenue_rag.evaluation.bench.machine_state", lambda: idle)
    questions = labelled_questions(config_file, tmp_path)
    result = CliRunner().invoke(
        app, ["bench-rerank", "-c", str(config_file), "-q", str(questions), "--rounds", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "venue_split: one cross-encoder call per venue" in result.output
    [run_dir] = (tmp_path / "results").iterdir()
    assert run_dir.name.endswith("_bench") and (run_dir / "raw.csv").exists()


def test_power_prints_the_study_and_writes_it(config_file, tmp_path):
    run, _ = fake_run(tmp_path)
    result = CliRunner().invoke(
        app, ["power", "--run", str(run), "--studies", "5", "--permutations", "5",
              "-c", str(config_file)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "recall@10 on 6 comparison questions, family of 2" in result.output
    [run_dir] = (tmp_path / "results").iterdir()
    study = json.loads((run_dir / "power.json").read_text())
    assert study["studies"] == 5 and study["permutations"] == 5


def test_power_refuses_an_unknown_category_before_it_reads_the_run(config_file, tmp_path):
    missing = tmp_path / "no-such-run"
    result = CliRunner().invoke(
        app, ["power", "--run", str(missing), "--category", "magic", "-c", str(config_file)]
    )
    assert isinstance(result.exception, UserError)
    assert "--category 'magic' is not a question category" in str(result.exception)


def test_retrieve_without_k_reads_retrieval_k_from_the_config(config_file, monkeypatch):
    seen: list[int] = []

    def compare(cfg, query, methods, k, filters):
        seen.append(k)
        return []

    monkeypatch.setattr("crossvenue_rag.cli.evaluate.compare_methods", compare)
    CliRunner().invoke(app, ["retrieve", "q", "-c", str(config_file), "--set", "retrieval.k=7"])
    CliRunner().invoke(app, ["retrieve", "q", "-c", str(config_file), "--k", "3"])
    assert seen == [7, 3]


def test_power_without_k_reads_the_deepest_cutoff_of_eval_k_values(config_file, monkeypatch):
    seen: list[int] = []

    def study(request, questions, seed):
        seen.append(request.k)
        raise UserError("stop here")

    monkeypatch.setattr("crossvenue_rag.evaluation.power.power_study", study)
    monkeypatch.setattr("crossvenue_rag.cli.evaluate.load_questions", lambda path: [])
    monkeypatch.setattr("crossvenue_rag.cli.evaluate.run_questions_path", lambda run: run)
    primary = "eval.primary_comparison=[venue_split,hybrid_rerank,venue_coverage@4,comparison]"
    args = ["power", "--run", "r", "-c", str(config_file)]
    args += ["--set", "eval.k_values=[1,4]", "--set", primary]
    CliRunner().invoke(app, args)
    assert seen == [4]


def test_a_k_past_both_candidate_sides_is_refused_once_with_a_hint_that_clears_both(
    config_file, monkeypatch
):
    import re

    monkeypatch.setattr("crossvenue_rag.cli.evaluate.compare_methods", lambda *args: [])
    args = ["retrieve", "q", "-m", "hybrid", "--k", "60", "-c", str(config_file)]
    refused = CliRunner().invoke(app, args)
    assert isinstance(refused.exception, UserError), refused.output
    assert "retrieval.bm25_candidates (50) and retrieval.dense_candidates (50)" in str(
        refused.exception
    )
    sets = re.findall(r"--set (\S+?)(?=[ `])", refused.exception.hint)
    assert len(sets) == 2
    followed = CliRunner().invoke(app, [*args, *(a for s in sets for a in ("--set", s))])
    assert followed.exit_code == 0, (followed.exception, followed.output)


def test_a_refused_k_and_the_bounds_it_exceeds_keep_the_form_of_the_option_and_the_config(
    config_file, monkeypatch
):
    monkeypatch.setattr("crossvenue_rag.cli.evaluate.compare_methods", lambda *args: [])
    sides = ["--set", "retrieval.bm25_candidates=1200", "--set", "retrieval.dense_candidates=1200"]
    deep = ["--set", "reranker.top_n=1200"]
    args = ["retrieve", "q", "-m", "hybrid", "--k", "1500", "-c", str(config_file), *sides, *deep]
    refused = CliRunner().invoke(app, args)
    assert isinstance(refused.exception, UserError), refused.output
    assert str(refused.exception).startswith(
        "--k 1500 is deeper than retrieval.bm25_candidates (1200) and "
        "retrieval.dense_candidates (1200)"
    )
    assert refused.exception.hint.startswith("pass --k 1200 or less")
    assert "--set retrieval.bm25_candidates=1500" in refused.exception.hint
