"""Tests for `index.build`: the version check, the BM25 index and the dense step.

`index_chunks` is replaced in every test: the real one writes the configured
Postgres table.
"""

from __future__ import annotations

import pytest

from crossvenue_rag.corpus.artifacts import chunks_path, read_corpus_stats, stats_path
from crossvenue_rag.errors import UserError
from crossvenue_rag.index import build
from crossvenue_rag.index_version import compute_index_version
from tests.conftest import build_corpus


@pytest.fixture
def dense(monkeypatch):
    written = []

    def fake_index_chunks(cfg, chunks, version):
        written.append(version)
        return len(chunks)

    monkeypatch.setattr(build, "index_chunks", fake_index_chunks)
    return written


def test_both_indexes_are_built_at_the_config_version(cfg, dense):
    chunks = build_corpus(cfg)
    built = build.build_indexes(cfg, skip_dense=False)
    assert built.chunks == len(chunks) and built.dense_rows == len(chunks)
    assert built.bm25_dir.exists()
    assert dense == [built.index_version] == [compute_index_version(cfg)]


def test_skip_dense_writes_no_table(cfg, dense):
    build_corpus(cfg)
    assert build.build_indexes(cfg, skip_dense=True).dense_rows is None
    assert dense == []


def test_chunks_ingested_under_another_config_are_refused(cfg, dense):
    build_corpus(cfg)
    stats = read_corpus_stats(cfg).model_copy(update={"index_version": "0" * 16})
    stats_path(cfg).write_text(stats.model_dump_json(), encoding="utf-8")
    with pytest.raises(UserError, match="ingested as index_version 0000000000000000") as refused:
        build.build_indexes(cfg, skip_dense=False)
    assert str(refused.value).startswith(f"{chunks_path(cfg)} was ingested")
    assert dense == []


def test_an_empty_chunks_file_is_refused_before_bm25_is_built(cfg, dense):
    build_corpus(cfg)
    chunks_path(cfg).write_text("", encoding="utf-8")
    with pytest.raises(UserError, match="holds no chunk") as refused:
        build.build_indexes(cfg, skip_dense=False)
    assert "crossvenue ingest" in refused.value.hint and dense == []
    assert f"if {cfg.paths.raw} is empty" in refused.value.hint
