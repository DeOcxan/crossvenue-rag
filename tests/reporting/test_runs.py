"""Tests for `reporting.runs`: the registry of committed runs and re-derived tables."""

from __future__ import annotations

import json
from pathlib import Path

from crossvenue_rag.reporting.runs import (
    build_registry,
    committed_runs,
    derive_answer_version,
    load_retrieval_tops,
    verify_registry,
)
from crossvenue_rag.run_files import registry_json


def test_the_registry_lists_committed_runs_and_catches_a_changed_file(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("results/*\n!results/a/\n", encoding="utf-8")
    run = tmp_path / "results" / "a"
    run.mkdir(parents=True)
    (run / "index.json").write_text(json.dumps({"index_version": "v1"}), encoding="utf-8")
    (run / "report.md").write_text("x\n", encoding="utf-8")
    assert committed_runs(tmp_path) == ["a"]
    assert verify_registry(tmp_path) == [
        "results/runs.json is missing; write it with `crossvenue runs-manifest`"
    ]
    (tmp_path / "results" / "runs.json").write_text(registry_json(build_registry(tmp_path)))
    assert verify_registry(tmp_path) == []
    (run / "report.md").write_text("y\n", encoding="utf-8")
    assert verify_registry(tmp_path) == [
        "results/a/report.md: sha256 differs from results/runs.json"
    ]


def test_an_answer_run_takes_the_version_of_the_retrieval_its_chunks_match(tmp_path: Path):
    retrieval = tmp_path / "r"
    retrieval.mkdir()
    (retrieval / "index.json").write_text(json.dumps({"index_version": "v1"}), encoding="utf-8")
    (retrieval / "per_query.csv").write_text(
        "question_id,method,retrieved_ids\nq1,dense,a b c\n", encoding="utf-8"
    )
    from tests.generation.factories import generation_json, record

    answer = record("gemini@dense", "q1", generation=generation_json(chunk_ids=("a", "b")))
    assert derive_answer_version([answer], load_retrieval_tops({"r": retrieval})) == ("v1", "r")
    answer = record("gemini@dense", "q1", generation=generation_json(chunk_ids=("b", "a")))
    assert derive_answer_version([answer], load_retrieval_tops({"r": retrieval})) is None
