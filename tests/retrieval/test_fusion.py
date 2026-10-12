"""Tests for `crossvenue_rag.retrieval.fusion`."""

from __future__ import annotations

import pytest

from crossvenue_rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_matches_the_formula_and_records_component_ranks():
    fused = reciprocal_rank_fusion({"bm25": ["x", "y"], "dense": ["y"]}, k=60)
    scores = {f.chunk_id: f.score for f in fused}
    ranks = {f.chunk_id: f.ranks for f in fused}

    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    # Agreed-on documents win, and each method's rank stays for inspection.
    assert fused[0].chunk_id == "y"
    assert ranks["y"] == {"bm25": 2, "dense": 1}
    assert ranks["x"] == {"bm25": 1}


def test_rrf_ignores_a_duplicate_within_one_list():
    """A method returning the same id twice must not double its own vote."""
    once = reciprocal_rank_fusion({"a": ["x"]}, k=60)
    twice = reciprocal_rank_fusion({"a": ["x", "x"]}, k=60)
    assert once[0].score == twice[0].score


def test_rrf_is_deterministic_on_ties():
    """An ablation table whose rows shuffle between runs is not evidence."""
    lists = {"a": ["q", "p"], "b": ["p", "q"]}
    order = [f.chunk_id for f in reciprocal_rank_fusion(lists, k=60)]
    assert order == sorted(order)
