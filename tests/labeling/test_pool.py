"""Tests for `crossvenue_rag.labeling.pool`."""

from __future__ import annotations

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.dataset.journal import Judgment
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.pool import (
    RETEST_SHUFFLE_SEED,
    gather_candidates,
    pool_depth_of,
    retest_candidates,
)
from crossvenue_rag.models import Chunk, Question, Result
from tests.conftest import augmentation
from tests.labeling.conftest import VERSION, chunk, result


class OverlappingRetriever:
    """Every method returns the same ranking, so the fused pool is only as
    deep as one method: the case in which a pool can shrink below 25."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.by_id: dict[str, Chunk] = {}
        self.depths: list[int] = []

    def retrieve(self, query: str, method: str, *, k=None, venues_hint=None) -> list[Result]:
        assert k is not None
        self.depths.append(k)
        return [result(f"c{i}", f"text {i}") for i in range(k)]


def test_pool_is_topped_up_to_the_display_limit_when_methods_overlap(cfg: Config):
    retriever = OverlappingRetriever(cfg)
    question = Question("q1", "what is the limit", "gold", ["binance"], "fact")
    pool = gather_candidates(retriever, question, per_method=20, limit=25)
    assert len(pool) == 25
    assert len({r.chunk_id for r in pool}) == 25
    assert max(retriever.depths) == 40


MANY = {c.chunk_id: c for c in (chunk(f"k{i:02d}", "binance") for i in range(30))}


def round1_entry() -> Judgment:
    return Judgment.model_validate(
        {
            "annotator": "me",
            "annotated_at": "2026-09-24T12:00:00Z",
            "index_version": VERSION,
            "pool": [f"k{i:02d}" for i in range(25)],
            "relevant": ["k03"],
            "augmentations": [augmentation(["k25", "k26", "k03"], ["k26"])],
        }
    )


def test_candidates_are_the_round_1_pool_plus_augmentation_and_nothing_else():
    entry = round1_entry()
    shown = [r.chunk_id for r in retest_candidates(MANY, "cv-0001", entry, RETEST_SHUFFLE_SEED)]
    assert sorted(shown) == sorted(entry.judged_ids())
    assert len(shown) == len(set(shown)) == 27  # 25 pooled + 2 from augmentation
    assert all(r.component_ranks == {} for r in retest_candidates(MANY, "cv-0001", entry, 1))


def test_order_is_shuffled_and_reproducible_from_the_seed():
    entry = round1_entry()

    def order(question_id: str, seed: int) -> list[str]:
        return [r.chunk_id for r in retest_candidates(MANY, question_id, entry, seed)]

    first = order("cv-0001", RETEST_SHUFFLE_SEED)
    assert first == order("cv-0001", RETEST_SHUFFLE_SEED)
    assert first != entry.judged_ids()
    assert first != order("cv-0002", RETEST_SHUFFLE_SEED)
    assert first != order("cv-0001", RETEST_SHUFFLE_SEED + 1)


def test_a_round_1_chunk_missing_from_the_index_is_refused():
    entry = round1_entry()
    entry.pool.append("gone")
    with pytest.raises(UserError, match="1 round-1 chunk not in the current index"):
        retest_candidates(MANY, "cv-0001", entry, RETEST_SHUFFLE_SEED)


def test_a_pool_depth_override_is_held_to_the_config_bound(cfg: Config):
    assert pool_depth_of(cfg, None) == cfg.eval.pool_depth_per_method
    assert pool_depth_of(cfg, 30) == 30
    deepest = max(cfg.eval.k_values)
    with pytest.raises(UserError, match="does not exceed max") as refused:
        pool_depth_of(cfg, deepest)
    assert f"give a depth above {deepest}" in refused.value.hint


def test_retest_order_is_the_order_retest_candidates_shows():
    from crossvenue_rag.labeling.pool import retest_order

    chunks = {c: chunk(c, "binance") for c in ("a", "b", "c", "d", "e")}
    entry = Judgment.model_validate(
        {"pool": ["a", "b", "c"], "augmentations": [augmentation(["d", "e"])]}
    )
    order = retest_order("q1", entry, RETEST_SHUFFLE_SEED)
    shown = [r.chunk_id for r in retest_candidates(chunks, "q1", entry, RETEST_SHUFFLE_SEED)]
    assert order == shown and sorted(order) == ["a", "b", "c", "d", "e"]
