"""Tests for `crossvenue_rag.labeling.candidates`."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.dataset.journal import load_rounds
from crossvenue_rag.dataset.questions import load_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.candidates import (
    augment_candidates,
    load_candidates,
    read_run_rankings,
)
from tests.conftest import output_defects
from tests.labeling.conftest import (
    VERSION,
    RankingRetriever,
    setup_augment,
)


def test_candidates_are_only_never_judged_chunks_best_rank_first(tmp_path):
    _, candidates = setup_augment(tmp_path)
    [entry] = load_candidates(candidates)
    assert [c.chunk_id for c in entry.chunks] == ["n2", "n1"]  # n2 rank 1 (oracle)
    assert entry.chunks[0].found_by == {"venue_split": 4, "venue_split_oracle": 1}
    assert entry.index_version == VERSION


def test_pool_run_writes_rankings_only_and_feeds_the_candidate_builder(cfg, tmp_path):
    from crossvenue_rag.labeling.candidates import PoolSpec, pool_run

    path, _ = setup_augment(tmp_path)
    run_dir = tmp_path / "20260925T000000Z_pool"
    spec = PoolSpec(path, ["venue_split", "venue_split_oracle"], 20, "comparison")
    rows = pool_run(cfg, RankingRetriever(), spec, VERSION, run_dir)
    assert rows == 2
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "POOL_RUN.md",
        "config.yaml",
        "index.json",
        "per_query.csv",
    ]  # no summary, report or metrics
    index = json.loads((run_dir / "index.json").read_text())
    assert "not an evaluation" in index["purpose"].lower() and index["k"] == 20
    pool_report = (run_dir / "POOL_RUN.md").read_text(encoding="utf-8")
    assert output_defects(pool_report) == []
    assert f"- `index_version`: `{VERSION}`" in pool_report

    entries = augment_candidates(
        read_run_rankings(run_dir),
        load_questions(path),
        load_rounds(path),
        ["venue_split", "venue_split_oracle"],
        "comparison",
        k=20,
    )
    assert [c.chunk_id for c in entries[0].chunks] == ["n2", "n1"]


def test_a_per_query_file_missing_a_column_is_refused(tmp_path):
    setup_augment(tmp_path)
    (tmp_path / "run" / "per_query.csv").write_text(
        "question_id,method\ncv-0001,venue_split\n", encoding="utf-8"
    )
    with pytest.raises(UserError, match=r"per_query\.csv:2.*retrieved_ids"):
        read_run_rankings(tmp_path / "run")


def test_a_line_that_is_not_a_candidate_list_is_refused_with_its_line(tmp_path):
    _, candidates = setup_augment(tmp_path)
    candidates.write_text(candidates.read_text(encoding="utf-8") + '{"id": "x"}\n', "utf-8")
    with pytest.raises(UserError, match=r"augment\.jsonl:2") as refused:
        load_candidates(candidates)
    assert "crossvenue augment-candidates" in refused.value.hint
