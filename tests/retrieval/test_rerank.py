"""Tests for `crossvenue_rag.retrieval.rerank`: one cross-encoder call for many lists."""

from __future__ import annotations

from crossvenue_rag.config_pipeline import RerankerConfig
from crossvenue_rag.models import Result
from crossvenue_rag.retrieval.rerank import Reranker


class FakeCrossEncoder:
    """Scores a pair by the number in its text; counts predict calls."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.calls.append(len(pairs))
        return [float(text.split()[-1]) for _, text in pairs]


def result(chunk_id: str, score: int) -> Result:
    return Result(chunk_id, 0.0, f"chunk {score}", "binance", "cex", "d.md", [], "", sources=[])


def reranker(top_n: int = 3) -> tuple[Reranker, FakeCrossEncoder]:
    model = FakeCrossEncoder()
    # The fake cross-encoder stands in for the model; no weights are loaded.
    return Reranker(RerankerConfig(top_n=top_n), None, model=model), model  # type: ignore[arg-type]


def test_rerank_many_scores_every_list_in_one_call_and_splits_the_scores_back():
    r, model = reranker(top_n=3)
    groups = [
        [result("a1", 1), result("a2", 9), result("a3", 5), result("a4", 99)],
        [result("b1", 7), result("b2", 2)],
    ]
    out = r.rerank_many("q", groups, k=2)
    assert model.calls == [5]  # 3 + 2 pairs, one predict call; a4 is past top_n
    assert [x.chunk_id for x in out[0]] == ["a2", "a3"]
    assert [x.chunk_id for x in out[1]] == ["b1", "b2"]


def test_rerank_many_matches_rerank_list_by_list():
    groups = [[result("a1", 3), result("a2", 8)], [result("b1", 4), result("b2", 6)]]
    together, _ = reranker()
    one_by_one, _ = reranker()
    merged = together.rerank_many("q", [list(g) for g in groups], k=2)
    separate = [one_by_one.rerank("q", list(g), k=2) for g in groups]
    assert [[x.chunk_id for x in g] for g in merged] == [[x.chunk_id for x in g] for g in separate]


def test_rerank_many_handles_empty_groups():
    r, model = reranker()
    assert r.rerank_many("q", [[], []], k=2) == [[], []]
    assert model.calls == []


def test_a_missing_sentence_transformers_is_a_user_error_naming_the_revision(monkeypatch):
    import sys

    import pytest

    from crossvenue_rag.errors import UserError

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    cfg = RerankerConfig(revision="abc123")
    with pytest.raises(UserError, match=r"cannot load reranker\.model_name .* at revision abc123:"):
        _ = Reranker(cfg, "cpu").model
