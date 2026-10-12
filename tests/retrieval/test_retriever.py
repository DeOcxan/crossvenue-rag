"""Tests for `crossvenue_rag.retrieval.retriever`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.config_pipeline import RetrievalMethod
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import Result, Timings
from crossvenue_rag.retrieval.filters import Filters
from crossvenue_rag.retrieval.retriever import Retriever, compare_methods
from tests.conftest import (
    AD_HOC_METHODS,
    CORPUS,
    VENUE_CHUNKS,
    InMemoryRetriever,
    build_corpus,
    make_chunk,
)


def test_bm25_finds_the_lexically_matching_chunk(cfg, bm25: BM25Index):
    results = Retriever(cfg, chunks=VENUE_CHUNKS, bm25=bm25).retrieve(
        "request weight per minute", method="bm25", k=2
    )
    assert results[0].chunk_id == "c-binance"
    # Everything a citation needs comes back with the hit.
    assert results[0].venue == "binance"
    assert results[0].heading_display == "API > Rate Limits"
    assert results[0].source_url


def test_unknown_method_is_rejected(cfg, bm25: BM25Index):
    with pytest.raises(ValueError, match="unknown method"):
        # A method outside RetrievalMethod on purpose: the call must refuse it.
        Retriever(cfg, chunks=VENUE_CHUNKS, bm25=bm25).retrieve("anything", method="magic", k=1)  # type: ignore[arg-type]


def test_bm25_method_reports_search_latency_only(cfg, bm25: BM25Index):
    """Per-step timings are the measurement; counting encode time toward BM25 would
    make the ablation table meaningless."""
    timings = Timings()
    results = Retriever(cfg, chunks=VENUE_CHUNKS, bm25=bm25).retrieve(
        "rate limits", method="bm25", k=2, timings=timings
    )
    assert results
    assert timings.search_ms > 0
    assert timings.encode_ms == 0 and timings.rerank_ms == 0


def test_venue_filter_separates_near_duplicate_pages(cfg: Config):
    """The control condition: the same sentence on two venues, one asked for."""
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))

    unfiltered = retriever.retrieve("four fee tiers 0.01% 0.05%", method="hybrid", k=4)
    assert {r.venue for r in unfiltered} & {"uniswap", "pancakeswap"}

    for method in AD_HOC_METHODS:
        filtered = retriever.retrieve(
            "four fee tiers 0.01% 0.05%",
            method=method,
            k=4,
            filters={"venue": ["pancakeswap"]},
        )
        assert filtered, method
        assert {r.venue for r in filtered} == {"pancakeswap"}, method

        by_type = retriever.retrieve(
            "rate limits and fees", method=method, k=4, filters={"venue_type": "cex"}
        )
        assert {r.venue_type for r in by_type} == {"cex"}, method


def test_venues_are_resolved_against_the_sources_yaml_the_options_are_checked_against(cfg: Config):
    """A venue of data/sources.yaml with no chunk here passes `--venue` and matches nothing;
    a venue outside it never reaches the retriever, which refuses it as a precondition."""
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    assert "dydx" not in {c.venue for c in chunks}
    assert retriever.retrieve("rate limits", method="bm25", k=4, filters={"venue": "dydx"}) == []
    with pytest.raises(ValueError, match="venues not in the venue map: bitmex"):
        retriever.retrieve("rate limits", method="bm25", k=4, filters={"venue": "bitmex"})


def test_an_environment_filter_matches_any_occurrence_of_a_merged_chunk(cfg: Config):
    """The merged chunk cites mainnet, but its text is on the testnet page too."""
    mainnet_text = CORPUS["binance/rest-api.md"][2]
    chunks = build_corpus(cfg, {"binance/testnet/rest-api.md": ("binance", "cex", mainnet_text)})
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    for method in AD_HOC_METHODS:
        results = retriever.retrieve(
            "weight budget per minute", method=method, k=5, filters={"environment": "testnet"}
        )
        assert results, method
        assert all(r.venue == "binance" for r in results)
        assert all("testnet" in {s.environment for s in r.sources} for r in results)


def test_venue_split_interleaves_one_ranked_list_per_named_venue(cfg: Config):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    query = "fee tiers on uniswap and pancakeswap"
    results = retriever.retrieve(query, method="venue_split", k=4)
    venues = [r.venue for r in results]
    assert venues[:2] == ["uniswap", "pancakeswap"]  # named order, round-robin
    assert len({r.chunk_id for r in results}) == len(results)

    oracle = retriever.retrieve(
        "what are the fee tiers",
        method="venue_split_oracle",
        k=4,
        venues_hint=["pancakeswap", "uniswap"],
    )
    assert [r.venue for r in oracle][:2] == ["pancakeswap", "uniswap"]


def test_venue_split_with_one_named_venue_is_hybrid_rerank(cfg: Config):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    for query in ("fee tiers on uniswap", "what are the fee tiers"):
        split = retriever.retrieve(query, method="venue_split", k=3)
        plain = retriever.retrieve(query, method="hybrid_rerank", k=3)
        assert [r.chunk_id for r in split] == [r.chunk_id for r in plain]


def test_oracle_without_venues_is_an_error(cfg: Config):

    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    with pytest.raises(ValueError, match="venues_hint"):
        retriever.retrieve("fees", method="venue_split_oracle", k=3)


def test_one_call_and_per_venue_reranking_give_the_same_retrieval(cfg: Config):
    """bench-rerank times the two against each other; they must differ only in
    how the pairs are split into cross-encoder calls."""
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    query = "fee tiers on uniswap and pancakeswap"
    one_pairs: list[int] = []
    per_pairs: list[int] = []
    one = retriever.venue_split_timed(query, "one_call", Timings(), one_pairs)
    per = retriever.venue_split_timed(query, "per_venue", Timings(), per_pairs)
    assert [r.chunk_id for r in one] == [r.chunk_id for r in per]
    assert [r.chunk_id for r in one] == [
        r.chunk_id for r in retriever.retrieve(query, method="venue_split", k=cfg.retrieval.k)
    ]
    assert len(one_pairs) == 1 and len(per_pairs) == 2
    assert one_pairs == [sum(per_pairs)]


def test_venue_split_drops_a_named_venue_outside_the_venue_type_filter(cfg: Config):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    query = "fee tiers on uniswap and binance"
    split = retriever.retrieve(query, method="venue_split", k=3, filters={"venue_type": "dex"})
    plain = retriever.retrieve(query, method="hybrid_rerank", k=3, filters={"venue_type": "dex"})
    assert [r.chunk_id for r in split] == [r.chunk_id for r in plain]
    assert {r.venue for r in split} <= {"uniswap", "pancakeswap"}


class MethodNameRetriever:
    """Returns the method's name as its only chunk and records every call."""

    opened: ClassVar[list[MethodNameRetriever]] = []

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.warmups: list[bool] = []
        self.calls: list[tuple[str, str, int, object]] = []
        self.closed = False
        MethodNameRetriever.opened.append(self)

    def warmup(self, *, load_reranker: bool = True) -> None:
        self.warmups.append(load_reranker)

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int | None = None,
        filters: Filters | None = None,
        timings: Timings | None = None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        assert k is not None and timings is not None
        self.calls.append((query, method, k, filters))
        timings.search_ms += 1.0
        return [Result.from_chunk(VENUE_CHUNKS[0], 1.0)]

    def __enter__(self) -> MethodNameRetriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


def test_each_method_runs_after_one_warmup_with_its_own_timings():
    filters: Filters = {"venue": ["binance"]}
    out = compare_methods(
        Config(), "rate limits", ["bm25", "hybrid"], 3, filters, open_retriever=MethodNameRetriever
    )
    retriever = MethodNameRetriever.opened[-1]
    assert retriever.warmups == [False]  # no hybrid_rerank: the reranker stays unloaded
    assert retriever.calls == [
        ("rate limits", "bm25", 3, filters),
        ("rate limits", "hybrid", 3, filters),
    ]
    assert [run.method for run in out] == ["bm25", "hybrid"]
    assert all(run.timings.search_ms == 1.0 for run in out)
    assert out[0].timings is not out[1].timings
    assert retriever.closed


def test_a_reranking_method_alone_loads_the_reranker_in_the_warmup():
    for method in ("hybrid_rerank", "venue_split", "venue_split_oracle"):
        compare_methods(Config(), "q", [method], 3, None, open_retriever=MethodNameRetriever)
        assert MethodNameRetriever.opened[-1].warmups == [True], method


def test_hybrid_rerank_loads_the_reranker_during_the_warmup():
    compare_methods(Config(), "q", ["hybrid_rerank"], 5, None, open_retriever=MethodNameRetriever)
    assert MethodNameRetriever.opened[-1].warmups == [True]


class _RecordingRetriever(Retriever):
    """`Retriever` as `checked_retriever` opens it, recording each step in `events`."""

    events: ClassVar[list[object]] = []

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, chunks=[make_chunk(c, "bybit", "text") for c in ("c1", "c2")])

    def __exit__(self, *exc: object) -> None:
        self.events.append("close")

    def require_current_index(self) -> str:
        self.events.append("index")
        return "v1"

    def warmup(self, *, load_reranker: bool = True) -> None:
        self.events.append(("warmup", load_reranker))


def test_a_checked_retriever_checks_the_index_before_it_warms_up_and_closes_after(cfg):
    from crossvenue_rag.retrieval import retriever as module

    events = _RecordingRetriever.events
    events.clear()

    def check(version: str, chunk_ids: set[str]) -> None:
        events.append(("check", version, frozenset(chunk_ids)))

    with module.checked_retriever(
        cfg, check, load_reranker=False, open_retriever=_RecordingRetriever
    ) as (opened, version):
        events.append("body")
    assert isinstance(opened, _RecordingRetriever) and version == "v1"
    assert events == [
        "index",
        ("check", "v1", frozenset({"c1", "c2"})),
        ("warmup", False),
        "body",
        "close",
    ]


def test_a_failed_check_loads_no_model_and_closes_the_retriever(cfg):
    from crossvenue_rag.errors import UserError
    from crossvenue_rag.retrieval import retriever as module

    events = _RecordingRetriever.events
    events.clear()

    def refuse(version: str, chunk_ids: set[str]) -> None:
        raise UserError("labeled against another index")

    with (
        pytest.raises(UserError),
        module.checked_retriever(cfg, refuse, open_retriever=_RecordingRetriever),
    ):
        events.append("body")
    assert events == ["index", "close"]
