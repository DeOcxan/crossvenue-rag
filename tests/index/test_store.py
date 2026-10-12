"""Tests for `index.store`: the SQL a vector search sends, with no database."""

from __future__ import annotations

import json

import numpy as np
import pytest
from psycopg import sql

from crossvenue_rag.config import Config
from crossvenue_rag.index.embeddings import Embedder
from crossvenue_rag.index.store import ChunkStore

APPROXIMATE = "ORDER BY embedding <=> %s\n"
EXACT = "ORDER BY (embedding <=> %s) + 0, chunk_id\n"


class Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    """Records each statement as text and its parameters; returns one row."""

    def __init__(self):
        self.statements: list[tuple[str, list]] = []

    def execute(self, statement, params=None):
        text = statement.as_string(None) if isinstance(statement, sql.Composable) else statement
        self.statements.append((text, params))
        source = {"venue": "bybit", "doc_path": "a.md", "heading_path": ["H"],
                  "source_url": "u", "position": 0}  # fmt: skip
        row = ("c1", 0.9, "bybit", "cex", "a.md", ["H"], "u", "text", 3, json.dumps([source]))
        return Result([row])


def search(k=5, venues=None, chunk_ids=None):
    conn = FakeConnection()
    store = ChunkStore(Config())
    # search takes a psycopg connection; the fake records what it is sent.
    hits = store.search(conn, np.zeros(3), k, venues, chunk_ids)  # type: ignore[arg-type]
    return conn.statements, hits


@pytest.mark.parametrize(
    ("kwargs", "order"),
    [
        ({}, APPROXIMATE),
        ({"venues": ["bybit"]}, APPROXIMATE),
        ({"venues": ["bybit", "binance"]}, EXACT),
        ({"chunk_ids": ["c1", "c2"]}, EXACT),
        ({"venues": ["bybit"], "chunk_ids": ["c1"]}, EXACT),
        ({"k": 101}, EXACT),
    ],
)
def test_only_a_filter_hnsw_can_serve_uses_the_index(kwargs, order):
    statements, _ = search(**kwargs)
    (setting, _), (query, params) = statements
    assert setting == "SET LOCAL hnsw.ef_search = 100"
    assert order in query
    assert params[-1] == kwargs.get("k", 5)


def test_filters_become_where_clauses_with_their_parameters():
    statements, _ = search(venues=["bybit", "binance"], chunk_ids=["c1"])
    query, params = statements[1]
    assert "WHERE venue = ANY(%s) AND chunk_id = ANY(%s)" in query
    assert params[1:3] == [["bybit", "binance"], ["c1"]]


def test_a_row_comes_back_as_a_hit_with_its_similarity_and_metadata():
    _, hits = search()
    [hit] = hits
    assert (hit.chunk_id, hit.similarity) == ("c1", 0.9)
    meta = hit.metadata
    assert meta["heading_path"] == ["H"] and meta["sources"][0].doc_path == "a.md"


class FakeEmbedder(Embedder):
    """Encodes every text as a zero vector of `width` dimensions."""

    width = 3

    def encode_documents(self, texts, *, show_progress=True):
        return np.zeros((len(texts), self.width))

    def model_fingerprint(self):
        return "fake"


class FakeWriteStore(ChunkStore):
    def connect_for_writing(self):
        import contextlib

        return contextlib.nullcontext(FakeWriteConnection())

    def drop(self, conn):
        pass

    def create_schema(self, conn):
        pass

    def insert(self, conn, chunks, vectors, model, version):
        return len(chunks)


class FakeWriteConnection:
    def commit(self):
        pass


def test_a_dimension_the_model_does_not_produce_is_refused_with_the_value_to_set():
    from crossvenue_rag.errors import UserError
    from crossvenue_rag.index import store
    from tests.conftest import make_chunk

    cfg = Config()
    cfg.embedding.dimension = 1024
    chunks = [make_chunk("c1", "bybit", "text")]
    embedder = FakeEmbedder(cfg.embedding)
    with pytest.raises(UserError, match=r"sets embedding\.dimension to 1024, but") as refused:
        store.index_chunks(cfg, chunks, "v", embedder=embedder, store=FakeWriteStore(cfg))
    assert "produced 3 dimensions" in str(refused.value)
    assert "`embedding.dimension: 3`" in refused.value.hint


def test_the_refused_dimension_and_the_one_to_set_are_written_through_the_formatting_module(
    monkeypatch,
):
    from crossvenue_rag.errors import UserError
    from crossvenue_rag.index import store
    from tests.conftest import make_chunk

    cfg = Config()
    cfg.embedding.dimension = 1024
    chunks = [make_chunk("c1", "bybit", "text")]
    embedder = FakeEmbedder(cfg.embedding)
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(UserError, match=r"sets embedding\.dimension to <1024>, but") as refused:
        store.index_chunks(cfg, chunks, "v", embedder=embedder, store=FakeWriteStore(cfg))
    assert "`embedding.dimension: <3>`" in refused.value.hint


def test_the_index_log_line_counts_chunks_in_words(caplog):
    import logging

    from crossvenue_rag.index import store
    from tests.conftest import make_chunk

    cfg = Config()
    cfg.embedding.dimension = FakeEmbedder.width
    chunks = [make_chunk("c1", "bybit", "text")]
    embedder = FakeEmbedder(cfg.embedding)
    with caplog.at_level(logging.INFO, logger="crossvenue_rag.index.store"):
        assert (
            store.index_chunks(cfg, chunks, "v", embedder=embedder, store=FakeWriteStore(cfg)) == 1
        )
    assert "indexed 1 chunk at index_version v" in caplog.messages


class RefusingStore(ChunkStore):
    def connect_read_only(self):
        raise OSError("connection refused")


def test_an_unreachable_table_is_reported_as_the_reason_it_cannot_be_read():
    from crossvenue_rag.index.store import store_state

    cfg = Config()
    assert store_state(cfg, store=RefusingStore(cfg)) == "connection refused"
