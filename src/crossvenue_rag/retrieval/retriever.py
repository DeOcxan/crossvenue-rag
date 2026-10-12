"""The retriever: one entry point, `Retriever.retrieve`, for the six methods in `METHODS`.

An ablation is a loop over method names rather than parallel code paths. psycopg and
pgvector are imported when the Postgres connection first opens, for the index check, a
dense query or `Retriever.warmup`, and bm25s when the BM25 index is first loaded.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import Config
from ..config_pipeline import RERANKED_METHODS, RetrievalMethod
from ..corpus.artifacts import chunks_by_id, load_chunks
from ..corpus.sources import venue_types
from ..index.bm25_index import INDEX_DIRNAME, BM25Index
from ..index.embeddings import Embedder
from ..index.store import ChunkStore
from ..index_version import require_current_index
from ..models import MS_PER_SECOND, MULTI_VENUE, Chunk, Result, Timings
from .filters import Filters, resolve_metadata, resolve_venues
from .fusion import reciprocal_rank_fusion
from .methods import METHODS
from .protocols import RerankCalls, WarmTimedRanks
from .rerank import Reranker
from .venue_detect import VenueDetector

if TYPE_CHECKING:
    # Imported where first used: psycopg only when the connection opens (`Retriever.conn`).
    import psycopg

# Raises unless what a caller retrieves for may use this index: (`index_version`, chunk ids).
IndexCheck = Callable[[str, set[str]], None]


@dataclass(frozen=True)
class RerankBench:
    """How `bench-rerank` runs venue_split.

    The cross-encoder calls to make, and the list that receives the number of
    pairs of each call.
    """

    rerank_calls: RerankCalls
    pairs_per_call: list[int]


def venue_split_splits(venues: Sequence[str]) -> bool:
    """Whether venue_split splits a query whose venues are `venues`.

    `venues` are the venues the query names (detected in its text, or its labeled
    venues for the oracle) that a filter leaves. With fewer than `MULTI_VENUE` of
    them, venue_split is hybrid_rerank.
    """
    return len(venues) >= MULTI_VENUE


def _interleave(per_venue: Sequence[Sequence[Result]], k: int) -> list[Result]:
    """Round-robin over the venues' ranked lists, first occurrence of a chunk kept, up to k.

    Each kept Result is updated in place: its venue rank is added to
    `component_ranks`.
    """
    merged: list[Result] = []
    seen: set[str] = set()
    for rank in range(k):
        for results in per_venue:
            if rank < len(results) and results[rank].chunk_id not in seen:
                result = results[rank]
                result.component_ranks = {**result.component_ranks, "venue_rank": rank + 1}
                merged.append(result)
                seen.add(result.chunk_id)
        if len(merged) >= k:
            break
    return merged[:k]


class Retriever:
    """Holds the indexes; opens the Postgres connection on first use.

    The first use is the index check, a dense query or `warmup`, which opens it for every
    method, BM25 included.
    """

    def __init__(
        self,
        cfg: Config,
        chunks: Sequence[Chunk] | None = None,
        bm25: BM25Index | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """A retriever for `cfg`; each component not given is built or loaded from `cfg`."""
        self.cfg = cfg
        self._chunks = list(chunks) if chunks is not None else None
        self._by_id: dict[str, Chunk] | None = None
        self._venue_types: dict[str, str] | None = None
        self._bm25 = bm25
        self._embedder = embedder or Embedder(cfg.embedding)
        self._reranker = reranker or Reranker(cfg.reranker, cfg.embedding.device)
        self._store = ChunkStore(cfg)
        self._conn: psycopg.Connection | None = None
        self._venue_detector: VenueDetector | None = None

    @property
    def chunks(self) -> list[Chunk]:
        """Every chunk of the corpus, read from chunks.jsonl on first access."""
        if self._chunks is None:
            self._chunks = load_chunks(self.cfg)
        return self._chunks

    @property
    def by_id(self) -> dict[str, Chunk]:
        """Every chunk, keyed by chunk id."""
        if self._by_id is None:
            self._by_id = chunks_by_id(self.chunks)
        return self._by_id

    def _venues_allowed(self, filters: Filters | None) -> list[str] | None:
        """The venue allow-list of `filters`, resolved against data/sources.yaml."""
        if not filters:
            return None
        if self._venue_types is None:
            self._venue_types = venue_types(self.cfg.paths.sources)
        return resolve_venues(filters, self._venue_types)

    @property
    def bm25(self) -> BM25Index:
        """The BM25 index, loaded from the index directory on first access."""
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.cfg.paths.index / INDEX_DIRNAME)
        return self._bm25

    @property
    def conn(self) -> psycopg.Connection:
        """The Postgres connection, opened on first access and kept until `close`."""
        if self._conn is None:
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self.cfg.store.dsn)
            register_vector(self._conn)
        return self._conn

    def warmup(self, *, load_reranker: bool = True) -> None:
        """Loads the models and opens the connection before anything is timed.

        Otherwise the weight loading (and, on MPS, kernel compilation) is timed as part
        of the first query's `encode` step, which makes the reported latency of
        whichever method runs first meaningless.
        """
        self._embedder.encode_query("warmup")
        _ = self.conn
        if load_reranker:
            self._reranker.model.predict([("warmup", "warmup")], show_progress_bar=False)

    @property
    def venue_detector(self) -> VenueDetector:
        """The venue detector for the configured sources file, built on first access."""
        if self._venue_detector is None:
            self._venue_detector = VenueDetector.from_file(self.cfg.paths.sources)
        return self._venue_detector

    def _venue_split(
        self,
        query: str,
        venues_named: Sequence[str],
        k: int,
        filters: Filters | None,
        timings: Timings,
        bench: RerankBench | None = None,
    ) -> list[Result]:
        """hybrid_rerank per named venue, interleaved round-robin up to k.

        Venues come in the order they were named, so the first-named venue
        takes rank 1. A caller's `venue` and `venue_type` filters apply too: a
        named venue outside them is dropped, and with fewer than two venues
        left the result is `hybrid_rerank`. Retrieval and fusion run per venue; the
        cross-encoder scores every venue's candidates in one call
        (`Reranker.rerank_many`), so reranking takes one pass whatever the number of venues.

        `bench` (for `crossvenue bench-rerank`) can make one cross-encoder call
        per venue instead, with the same pairs and the same result, and
        receives the number of pairs of each call.
        """
        allowed = self._venues_allowed(filters)
        wanted = [v for v in venues_named if allowed is None or v in allowed]
        if not venue_split_splits(wanted):
            return self.retrieve(query, "hybrid_rerank", k, filters, timings)

        fused_per_venue = [
            self._fused(query, self.cfg.reranker.top_n, {**(filters or {}), "venue": [v]}, timings)
            for v in wanted
        ]
        per_venue = self._rerank_venues(query, fused_per_venue, k, timings, bench)
        start = time.perf_counter()
        merged = _interleave(per_venue, k)
        timings.fuse_ms += (time.perf_counter() - start) * MS_PER_SECOND
        return merged

    def _rerank_venues(
        self,
        query: str,
        fused_per_venue: list[list[Result]],
        k: int,
        timings: Timings,
        bench: RerankBench | None,
    ) -> list[list[Result]]:
        """Each venue's top k after the cross-encoder.

        The cross-encoder is called once unless `bench` says otherwise; the time
        goes to `timings.rerank_ms`.
        """
        top_n = self.cfg.reranker.top_n
        start = time.perf_counter()
        if bench is None or bench.rerank_calls == "one_call":
            per_venue = self._reranker.rerank_many(query, fused_per_venue, k)
            calls = [sum(min(len(g), top_n) for g in fused_per_venue)]
        else:
            per_venue = [self._reranker.rerank(query, g, k) for g in fused_per_venue]
            calls = [min(len(g), top_n) for g in fused_per_venue]
        timings.rerank_ms += (time.perf_counter() - start) * MS_PER_SECOND
        if bench is not None:
            bench.pairs_per_call.extend(calls)
        return per_venue

    def venue_split_timed(
        self,
        query: str,
        rerank_calls: RerankCalls,
        timings: Timings,
        pairs_per_call: list[int],
    ) -> list[Result]:
        """venue_split over the venues detected in `query`, for `bench-rerank`.

        The cross-encoder is called as `rerank_calls` says.
        """
        return self._venue_split(
            query,
            self.venue_detector.detect(query),
            self.cfg.retrieval.k,
            None,
            timings,
            RerankBench(rerank_calls, pairs_per_call),
        )

    def require_current_index(self) -> str:
        """The `index_version` both indexes carry."""
        return require_current_index(self.cfg, self._store.stored_index_versions(self.conn))

    def close(self) -> None:
        """Closes the Postgres connection if one is open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Retriever:
        """This retriever, closed when the `with` block exits."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Closes the connection; an exception from the block propagates."""
        self.close()

    def _bm25_search(
        self,
        query: str,
        k: int,
        venues: list[str] | None,
        timings: Timings,
        allowed_ids: set[str] | None,
    ) -> list[Result]:
        start = time.perf_counter()
        hits = self.bm25.search(query, k, venues, allowed_ids)
        timings.search_ms += (time.perf_counter() - start) * MS_PER_SECOND
        return [
            Result.from_chunk(self.by_id[chunk_id], score)
            for chunk_id, score in hits
            if chunk_id in self.by_id
        ]

    def _dense_search(
        self,
        query: str,
        k: int,
        venues: list[str] | None,
        timings: Timings,
        allowed_ids: set[str] | None,
    ) -> list[Result]:
        start = time.perf_counter()
        vector = self._embedder.encode_query(query)
        timings.encode_ms += (time.perf_counter() - start) * MS_PER_SECOND

        start = time.perf_counter()
        rows = self._store.search(
            self.conn, vector, k, venues, sorted(allowed_ids) if allowed_ids is not None else None
        )
        timings.search_ms += (time.perf_counter() - start) * MS_PER_SECOND

        return [
            Result(
                chunk_id=hit.chunk_id,
                score=hit.similarity,
                text=hit.metadata["text"],
                venue=hit.metadata["venue"],
                venue_type=hit.metadata["venue_type"],
                doc_path=hit.metadata["doc_path"],
                heading_path=hit.metadata["heading_path"],
                source_url=hit.metadata["source_url"],
                sources=hit.metadata["sources"],
            )
            for hit in rows
        ]

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        k: int,
        filters: Filters | None = None,
        timings: Timings | None = None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        """Top k for `query`. `venues_hint` is read only by `venue_split_oracle`.

        `venue_split` retrieves per venue the question names and interleaves the lists.
        `venue_split_oracle` runs it with the question's labeled venues in place of the
        detected ones, which a deployed system does not have. It bounds what better venue
        detection can add to venue coverage and is no upper bound on recall, precision or nDCG.

        Raises:
            UserError: A metadata filter of `filters` names a value no chunk carries.
            ValueError: `method` is not in `METHODS`, or `venue_split_oracle` has no
                `venues_hint`.
        """
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")

        timings = timings if timings is not None else Timings()
        if method == "venue_split":
            return self._venue_split(query, self.venue_detector.detect(query), k, filters, timings)
        if method == "venue_split_oracle":
            if venues_hint is None:
                raise ValueError(
                    "venue_split_oracle reads the question's labeled venues; pass venues_hint"
                )
            venues_named = list(dict.fromkeys(venues_hint))
            return self._venue_split(query, venues_named, k, filters, timings)
        if method == "bm25":
            venues = self._venues_allowed(filters)
            return self._bm25_search(
                query, k, venues, timings, resolve_metadata(filters, self.chunks)
            )
        if method == "dense":
            venues = self._venues_allowed(filters)
            return self._dense_search(
                query, k, venues, timings, resolve_metadata(filters, self.chunks)
            )

        # hybrid and hybrid_rerank share the fusion stage; only the depth differs,
        # because reranking is pointless unless it is given more than it returns.
        depth = self.cfg.reranker.top_n if method == "hybrid_rerank" else k
        results = self._fused(query, depth, filters, timings)

        if method == "hybrid":
            return results[:k]

        start = time.perf_counter()
        reranked = self._reranker.rerank(query, results, k)
        timings.rerank_ms += (time.perf_counter() - start) * MS_PER_SECOND
        return reranked

    def _fused(
        self, query: str, depth: int, filters: Filters | None, timings: Timings
    ) -> list[Result]:
        """BM25 and dense candidates fused by RRF, the top `depth` of them."""
        venues = self._venues_allowed(filters)
        allowed = resolve_metadata(filters, self.chunks)
        sparse = self._bm25_search(
            query, self.cfg.retrieval.bm25_candidates, venues, timings, allowed
        )
        dense = self._dense_search(
            query, self.cfg.retrieval.dense_candidates, venues, timings, allowed
        )

        start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"bm25": [r.chunk_id for r in sparse], "dense": [r.chunk_id for r in dense]},
            k=self.cfg.retrieval.rrf_k,
        )
        # Both sides carry the same chunk metadata, so which one wins a
        # duplicate does not matter: only the RRF score decides the order.
        known = {r.chunk_id: r for r in [*dense, *sparse]}
        results: list[Result] = []
        for item in fused[:depth]:
            source = known[item.chunk_id]
            results.append(
                Result(
                    chunk_id=item.chunk_id,
                    score=item.score,
                    text=source.text,
                    venue=source.venue,
                    venue_type=source.venue_type,
                    doc_path=source.doc_path,
                    heading_path=source.heading_path,
                    source_url=source.source_url,
                    component_ranks=item.ranks,
                    sources=source.sources,
                )
            )
        timings.fuse_ms += (time.perf_counter() - start) * MS_PER_SECOND
        return results


@dataclass(frozen=True)
class MethodRun:
    """One method's top k for one query, with its step timings."""

    method: RetrievalMethod
    results: list[Result]
    timings: Timings


def compare_methods(
    cfg: Config,
    query: str,
    methods: Sequence[RetrievalMethod],
    k: int,
    filters: Filters | None,
    *,
    open_retriever: Callable[[Config], AbstractContextManager[WarmTimedRanks]] | None = None,
) -> list[MethodRun]:
    """Each method's top k for one query, with its step timings, after a warmup.

    Without the warmup the first query's `encode` step reports the model loading
    as if it were encoding latency. `open_retriever` builds the
    retriever from `cfg`; `Retriever` unless given.

    Raises:
        UserError: A metadata filter of `filters` names a value no chunk carries.
        ValueError: A method of `methods` is not in `METHODS`, or is `venue_split_oracle`,
            which needs the question's labeled venues and gets none here.
    """
    out: list[MethodRun] = []
    with (open_retriever or Retriever)(cfg) as retriever:
        retriever.warmup(load_reranker=any(m in RERANKED_METHODS for m in methods))
        for name in methods:
            timings = Timings()
            results = retriever.retrieve(query, method=name, k=k, filters=filters, timings=timings)
            out.append(MethodRun(name, results, timings))
    return out


@contextmanager
def checked_retriever(
    cfg: Config,
    check: IndexCheck,
    *,
    load_reranker: bool = True,
    open_retriever: Callable[[Config], Retriever] | None = None,
) -> Iterator[tuple[Retriever, str]]:
    """The warmed-up retriever for `cfg` and the `index_version` it reads, once `check` passes.

    `check` receives the `index_version` and the corpus's chunk ids after the two indexes
    are found current, and before any model is loaded; the reranker is loaded only when
    `load_reranker`. The connection closes when the `with` block exits. `open_retriever`
    builds the retriever from `cfg`; `Retriever` unless given.
    """
    with (open_retriever or Retriever)(cfg) as retriever:
        version = retriever.require_current_index()
        check(version, set(retriever.by_id))
        # Warmed up before anything is timed: model loading is not retrieval latency.
        retriever.warmup(load_reranker=load_reranker)
        yield retriever, version
