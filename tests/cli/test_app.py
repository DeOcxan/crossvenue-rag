"""Tests for `cli.app`: config loading and method names."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.cli.app import load, methods_from
from crossvenue_rag.errors import UserError


def test_all_is_every_method_but_the_oracle():
    methods = methods_from("all")
    assert "venue_split" in methods and "venue_split_oracle" not in methods
    assert methods_from("bm25") == ["bm25"]


def test_a_typo_or_the_oracle_is_refused_before_any_model_loads():
    with pytest.raises(UserError, match="--method 'bm52' is not a retrieval method") as typo:
        methods_from("bm52")
    assert "venue_split_oracle" not in (typo.value.hint or "")
    with pytest.raises(UserError, match="--method venue_split_oracle: it reads") as oracle:
        methods_from("venue_split_oracle")
    assert "venue_split_oracle" not in (oracle.value.hint or "")


def test_set_overrides_reach_the_loaded_config(config_file):
    cfg = load(config_file, ["retrieval.k=20"])
    assert cfg.retrieval.k == 20


def test_the_top_level_help_names_the_commands_that_take_no_config():
    import importlib

    app_module = importlib.import_module("crossvenue_rag.cli.app")
    assert "except `reproduce` and `runs-manifest` takes `--config`" in " ".join(
        app_module.HELP.split()
    )


# Each option given a value outside its domain, and what the refusal names. `{c}` is the
# config, `{q}` a question file, `{run}` an answer run at the current index, `{dir}` a
# directory that is no run, `{none}` a path with nothing at it.
OUT_OF_DOMAIN = [
    (["retrieve", "  ", "-c", "{c}"], "the query is empty"),
    (["retrieve", "q", "--k", "0", "-c", "{c}"], "--k must be at least 1, got 0"),
    (["retrieve", "q", "--k", "-3", "-c", "{c}"], "--k must be at least 1, got \u22123"),
    (["retrieve", "q", "--method", "bogus", "-c", "{c}"], "--method 'bogus' is not"),
    (["retrieve", "q", "--k", "51", "-c", "{c}"],
     "--k 51 is deeper than retrieval.bm25_candidates"),
    (["retrieve", "q", "-m", "hybrid_rerank", "--k", "51", "-s", "retrieval.bm25_candidates=60",
      "-s", "retrieval.dense_candidates=60", "-c", "{c}"], "is deeper than reranker.top_n (50)"),
    (["retrieve", "q", "-m", "all", "--k", "51", "-c", "{c}"],
     "so hybrid, hybrid_rerank and venue_split would return fewer"),
    (["retrieve", "q", "--filter", "venue=bybit", "-c", "{c}"], "bad --filter 'venue=bybit'"),
    (["retrieve", "q", "--venue", "bogus", "-c", "{c}"], "--venue bogus: 1 unknown venue"),
    (["retrieve", "q", "--venue-type", "cefi", "-c", "{c}"],
     "--venue-type 'cefi' is not a venue type"),
    (["retrieve", "q", "--filter", "environment", "-c", "{c}"], "bad --filter 'environment'"),
    (["eval", "-q", "{dir}", "-c", "{c}"], "is a directory where a file is expected"),
    (["pool-run", "-m", "bm25", "--category", "bogus", "-q", "{q}", "-c", "{c}"],
     "--category 'bogus' is not a question category"),
    (["pool-run", "-m", "bm25", "--k", "5", "-q", "{q}", "-c", "{c}"], "--k 5: a pool depth of 5"),
    # The oracle may build augmentation candidates: it passes the method check.
    (["pool-run", "-m", "venue_split_oracle", "--k", "5", "-q", "{q}", "-c", "{c}"],
     "a pool depth of 5"),
    (["ask", " ", "-c", "{c}"], "the question is empty"),
    (["ask", "q", "--generator", "nope", "-c", "{c}"], "--generator 'nope' is not"),
    (["power", "--run", "{run}", "-s", "paths.results={q}", "-c", "{c}"],
     "cannot be a directory"),
    (["check", "-s", "store.hnsw_m=40", "-c", "{c}"], "invalid --set store.hnsw_m=40"),
    (["check", "-s", "retrieval.k=abc", "-s", "reranker.top_n=xyz", "-c", "{c}"],
     "invalid --set retrieval.k=abc, --set reranker.top_n=xyz"),
    (["check", "-s", "retrieval.k=abc", "-s", "store.hnsw_m=40", "-c", "{c}"],
     "invalid --set retrieval.k=abc, --set store.hnsw_m=40"),
    (["pool-run", "-m", "bm25", "-q", "{none}", "-c", "{c}"], "not found"),
    (["bench-rerank", "--rounds", "0", "-c", "{c}"], "--rounds must be at least 1"),
    (["power", "--run", "{dir}", "--k", "0", "-c", "{c}"], "--k must be at least 1"),
    (["power", "--run", "{dir}", "--studies", "0", "-c", "{c}"], "--studies must be at least 1"),
    (["power", "--run", "{dir}", "--permutations", "-1", "-c", "{c}"],
     "--permutations must be at least 1"),
    (["power", "--run", "{dir}", "-c", "{c}"], "has no index.json"),
    (["count-term", " ", "-c", "{c}"], "the term to count is empty"),
    (["count-term", "fee", "--venue", "bogus", "-c", "{c}"], "--venue 'bogus' is not a venue"),
    (["id-map", "{none}", "--out", "{dir}", "-c", "{c}"], "is a directory"),
    (["answer-eval", "--budget", "0", "-c", "{c}"], "--budget must be an amount of USD above 0"),
    (["answer-eval", "--budget", "-1", "-c", "{c}"], "above 0, got $\u22121.000"),
    (["answer-eval", "--budget", "nan", "-c", "{c}"], "--budget must be"),
    (["answer-eval", "--budget", "inf", "-c", "{c}"], "--budget must be"),
    (["answer-eval", "--judge", "", "-q", "{q}", "-c", "{c}"], "--judge '' is not a judge"),
    (["answer-eval", "--question-id", "cv-9999", "-q", "{q}", "-c", "{c}"],
     "--question-id not in"),
    (["rate-sample", "--run", "{dir}", "-c", "{c}"], "is not an answer run"),
    (["rate-sample", "--run", "{run}", "--out", "{dir}", "-c", "{c}"], "is a directory"),
    (["rate-sample", "--run", "{run}", "--out", "{q}/x.txt", "-c", "{c}"], "is a file"),
    (["rate", "--sample-file", "{none}", "-c", "{c}"], "not found"),
    (["judge-sample", "--judge", "opus", "--replicate", "2", "-c", "{c}"],
     "--replicate must be 0 or 1, got 2"),
    (["judge-sample", "--judge", "opus", "--replicate", "-1", "-c", "{c}"], "got \u22121"),
    (["judge-sample", "--judge", "opus", "--budget", "0", "-c", "{c}"], "--budget must be"),
    (["judge-sample", "--judge", "opus", "--sample-file", "{none}", "-c", "{c}"], "not found"),
    (["judge-agreement", "--run", "{run}", "--judge", "bogus", "-c", "{c}"],
     "--judge 'bogus' is not a judge"),
    (["judge-agreement", "--run", "{dir}", "--judge", "opus", "-c", "{c}"], "not an answer run"),
    (["select-judge", "--run", "{dir}", "-c", "{c}"], "is not an answer run"),
    (["judge-run", "--run", "{dir}", "-c", "{c}"], "is not an answer run"),
    (["judge-run", "--run", "{run}", "--budget", "inf", "-c", "{c}"], "--budget must be"),
    (["compare-prompts", "--v1", "{run}", "--v2", "{run}", "-c", "{c}"],
     "--v1 and --v2 name the same run"),
    (["check", "-c", "{dir}"], "is a directory where a file is expected"),
    (["check", "-c", "{none}"], "not found"),
]  # fmt: skip


@pytest.mark.parametrize(("args", "names"), OUT_OF_DOMAIN)
def test_an_option_outside_its_domain_is_refused_naming_it(config_file, tmp_path, args, names):
    from crossvenue_rag.generation.records import write_answers
    from crossvenue_rag.models import Question
    from crossvenue_rag.run_files import write_index
    from tests.cli.conftest import current_version, invoke
    from tests.generation.factories import matrix_records

    questions = tmp_path / "q.jsonl"
    save = [Question("cv-0001", "binance fees?", "gold", ["binance"], "fact")]
    questions.write_text("".join(f"{json.dumps(q.to_dict())}\n" for q in save), encoding="utf-8")
    run = tmp_path / "results" / "20260101T000000Z_answers"
    run.mkdir(parents=True)
    write_answers(run / "answers.jsonl", matrix_records())
    write_index(run, {"index_version": current_version(config_file)}, final_newline=True)
    (tmp_path / "plain").mkdir()
    paths = {
        "c": config_file, "q": questions, "run": run, "dir": tmp_path / "plain",
        "none": tmp_path / "absent.txt",
    }  # fmt: skip
    result = invoke(*[a.format(**{k: str(v) for k, v in paths.items()}) for a in args])
    assert isinstance(result.exception, UserError), (result.exception, result.output)
    assert names in str(result.exception)


def test_a_replicate_outside_0_and_1_is_refused_with_a_hint_naming_both_values(config_file):
    from tests.cli.conftest import invoke

    result = invoke("judge-sample", "--judge", "opus", "--replicate", "2", "-c", str(config_file))
    assert isinstance(result.exception, UserError), result.output
    assert result.exception.hint.startswith("pass --replicate 0 to grade the sample, or 1")


def test_reproduce_and_runs_manifest_refuse_to_run_outside_the_repository(tmp_path, monkeypatch):
    from tests.cli.conftest import invoke

    monkeypatch.chdir(tmp_path)
    for command in ("reproduce", "runs-manifest"):
        result = invoke(command)
        assert isinstance(result.exception, UserError), command
        assert "is not the repository root" in str(result.exception)
