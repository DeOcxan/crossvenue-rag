"""Both indexes as `crossvenue index` builds them, from chunks.jsonl.

`build_indexes` refuses chunks ingested under another config, writes the BM25
index and index_meta.json, and then, unless told to skip it, the pgvector table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..corpus.artifacts import chunks_path, load_chunks, read_corpus_stats
from ..errors import UserError
from ..index_version import compute_index_version, write_index_meta
from .bm25_index import INDEX_DIRNAME, BM25Index
from .store import index_chunks


@dataclass(frozen=True)
class IndexBuild:
    """What `build_indexes` wrote: the chunks, the BM25 directory and the version.

    `dense_rows` is the number of rows written to pgvector, None when it was skipped.
    """

    chunks: int
    bm25_dir: Path
    index_version: str
    dense_rows: int | None


def build_indexes(cfg: Config, *, skip_dense: bool) -> IndexBuild:
    """Builds the BM25 index and, unless `skip_dense`, the pgvector table.

    Raises:
        UserError: chunks.jsonl holds no chunk, which leaves BM25 nothing to index, or the
            chunks were ingested under a config that implies another `index_version`, which
            would stamp the wrong version on every row.
    """
    chunks = load_chunks(cfg)
    if not chunks:
        raise UserError(
            f"{chunks_path(cfg)} holds no chunk, so there is nothing to index",
            hint=f"run `crossvenue ingest` first, after `crossvenue fetch` if {cfg.paths.raw} is "
            "empty",
        )
    version = compute_index_version(cfg)
    ingested = read_corpus_stats(cfg).index_version
    if ingested != version:
        raise UserError(
            f"{chunks_path(cfg)} was ingested as index_version {ingested}, this config implies "
            f"{version}",
            hint="re-run `crossvenue ingest` with the same config first",
        )
    bm25_dir = cfg.paths.index / INDEX_DIRNAME
    BM25Index.build(chunks).save(bm25_dir)
    write_index_meta(cfg, len(chunks))
    dense_rows = None if skip_dense else index_chunks(cfg, chunks, version)
    return IndexBuild(len(chunks), bm25_dir, version, dense_rows)
