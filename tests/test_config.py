"""Tests for `crossvenue_rag.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from crossvenue_rag.config import Config
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError

DEFAULT = Path("configs/default.yaml")


def test_default_config_loads_and_round_trips_through_yaml():
    """Every results directory stores this; it has to reload into the same config."""
    cfg = load_config(DEFAULT)
    assert cfg.embedding.model_name == "BAAI/bge-base-en-v1.5"
    assert cfg.retrieval.rrf_k == 60

    reloaded = yaml.safe_load(cfg.to_yaml())
    assert reloaded["chunking"]["max_tokens"] == cfg.chunking.max_tokens
    assert reloaded["eval"]["methods"] == list(cfg.eval.methods)


def test_rerank_depth_must_cover_k():
    """top_n < k would silently return fewer results than the caller asked for."""
    with pytest.raises(UserError, match="top_n"):
        load_config(DEFAULT, ["reranker.top_n=3", "retrieval.k=10"])


def test_rerank_depth_must_cover_the_deepest_evaluated_k():
    """`eval` retrieves at max(k_values); retrieval.k is the depth of a call that names none."""
    overrides = [
        "reranker.top_n=50",
        "eval.k_values=[1, 5, 60]",
        "eval.pool_depth_per_method=80",
        "eval.pool_display=80",
        "eval.primary_comparison=[]",
    ]
    with pytest.raises(UserError, match=r"deepest list asked for \(80"):
        load_config(DEFAULT, overrides)


def test_reranking_covers_the_pool_and_the_generators_chunks():
    with pytest.raises(UserError, match=r"deepest list asked for \(60, from"):
        load_config(DEFAULT, ["eval.pool_depth_per_method=60", "eval.pool_display=60"])
    with pytest.raises(UserError, match=r"deepest list asked for \(80, from"):
        load_config(DEFAULT, ["generation.top_k=80"])


def test_each_side_of_a_hybrid_retrieval_gives_a_full_list():
    for side in ("bm25_candidates", "dense_candidates"):
        with pytest.raises(UserError, match=rf"retrieval\.{side} \(5\) is below the deepest k"):
            load_config(DEFAULT, [f"retrieval.{side}=5"])


def test_the_refused_depths_are_written_through_the_formatting_module(monkeypatch):
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(
        ValidationError, match=r"top_n \(<3>\) is below the deepest list asked for \(<20>,"
    ):
        Config(reranker={"top_n": 3})
    with pytest.raises(
        ValidationError, match=r"bm25_candidates \(<5>\) is below the deepest k asked for \(<10>,"
    ):
        Config(retrieval={"bm25_candidates": 5})


def test_depth_bounds_name_the_keys_that_cap_each_method():
    cfg = load_config(DEFAULT)
    assert cfg.depth_bounds("bm25") == cfg.depth_bounds("dense") == {}
    assert set(cfg.depth_bounds("hybrid")) == {
        "retrieval.bm25_candidates",
        "retrieval.dense_candidates",
    }
    assert cfg.depth_bounds("hybrid_rerank")["reranker.top_n"] == cfg.reranker.top_n
    assert cfg.depth_bounds("venue_split") == cfg.depth_bounds("hybrid_rerank")


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://u:secret@db:5432/x",
        "postgres://u@db/x?sslmode=require&password=secret",
        "host=db password=secret dbname=x",
    ],
)
def test_a_saved_config_holds_no_dsn_and_so_no_password(dsn):
    cfg = load_config(DEFAULT, [f"store.dsn={dsn}"])
    assert cfg.store.dsn == dsn
    text = cfg.to_yaml()
    assert "secret" not in text
    assert "dsn" not in yaml.safe_load(text)["store"]
