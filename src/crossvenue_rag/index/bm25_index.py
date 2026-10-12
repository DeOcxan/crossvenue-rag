"""Sparse BM25 index over the same chunks Postgres holds, built with `bm25s`.

Lexical retrieval matches the tokens that distinguish venues and that dense
models smooth away: `/v5/order/create`, `postOnly`, `X-MBX-APIKEY`, `1200`.
`bm25s` keeps the index as a sparse matrix on disk and needs no service; it and
PyStemmer are imported where first used, so importing the retriever does not
load them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..errors import UserError
from ..models import Chunk
from ..run_files import read_json_model

if TYPE_CHECKING:
    import bm25s
    import Stemmer
    from bm25s.tokenization import Tokenized


INDEX_DIRNAME = "bm25"
IDS_FILENAME = "chunk_ids.json"


class IndexIds(BaseModel):
    """chunk_ids.json: the chunk id and venue of each row of the BM25 matrix, in row order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_ids: list[str]
    venues: list[str]


def _tokenize(texts: Sequence[str]) -> Tokenized:
    """bm25s tokenizer with English stopwords and Snowball stemming.

    Stemming matters for this corpus: "rate limiting", "rate limits" and
    "rate-limited" appear in the same paragraph across venues.
    """
    import bm25s

    return bm25s.tokenize(list(texts), stopwords="en", stemmer=_stemmer(), show_progress=False)


@cache
def _stemmer() -> Stemmer.Stemmer:
    """The English Snowball stemmer, built once per process rather than once per query."""
    import Stemmer

    return Stemmer.Stemmer("english")


class BM25Index:
    """Wraps a bm25s retriever plus the chunk ids and venues it indexed."""

    def __init__(self, retriever: bm25s.BM25, chunk_ids: list[str], venues: list[str]) -> None:
        """An index over `retriever`, whose i-th document is `chunk_ids[i]` on `venues[i]`."""
        self.retriever = retriever
        self.chunk_ids = chunk_ids
        self.venues = venues

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> BM25Index:
        """An index over `chunks`.

        Raises:
            ValueError: There are no chunks.
        """
        import bm25s

        if not chunks:
            raise ValueError("cannot build a BM25 index over zero chunks")
        tokens = _tokenize([c.text for c in chunks])
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        return cls(retriever, [c.chunk_id for c in chunks], [c.venue for c in chunks])

    def save(self, directory: Path) -> None:
        """Writes the index and its chunk ids and venues into `directory`."""
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory), corpus=None)
        ids = IndexIds(chunk_ids=self.chunk_ids, venues=self.venues)
        (directory / IDS_FILENAME).write_text(json.dumps(ids.model_dump()), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        """The index saved in `directory`.

        Raises:
            UserError: No index is there.
        """
        import bm25s

        if not (directory / IDS_FILENAME).exists():
            raise UserError(f"no BM25 index at {directory}", hint="run `crossvenue index` first")
        retriever = bm25s.BM25.load(str(directory), load_corpus=False)
        ids = read_json_model(
            directory / IDS_FILENAME, IndexIds, hint="run `crossvenue index` to rebuild it"
        )
        return cls(retriever, ids.chunk_ids, ids.venues)

    def search(
        self,
        query: str,
        k: int,
        venues: Sequence[str] | None,
        allowed_ids: set[str] | None,
    ) -> list[tuple[str, float]]:
        """Top-k `(chunk_id, bm25_score)` among chunks sharing a query token (score above 0).

        Venue and id filters rank the whole corpus and then drop non-matching
        hits: bm25s has no filtered-search API, and re-indexing per filter would
        change the IDF statistics and make filtered and unfiltered runs
        incomparable. A fixed over-fetch depth would return fewer than k hits
        for a selective filter such as `environment=testnet`; ranking the whole
        corpus takes milliseconds.
        """
        tokens = _tokenize([query])
        filtered = bool(venues) or allowed_ids is not None
        fetch = len(self.chunk_ids) if filtered else min(k, len(self.chunk_ids))
        if fetch == 0:
            return []

        indices, scores = self.retriever.retrieve(tokens, k=fetch, show_progress=False, n_threads=0)
        indices = np.asarray(indices)[0]
        scores = np.asarray(scores)[0]

        allowed = set(venues) if venues else None
        out: list[tuple[str, float]] = []
        for index, score in zip(indices, scores, strict=True):
            # bm25s ranks every document, so past the ones sharing a query token it returns
            # score-0 ones; the scores are sorted, so the first of those ends the hits.
            if score <= 0:
                break
            if allowed is not None and self.venues[index] not in allowed:
                continue
            if allowed_ids is not None and self.chunk_ids[index] not in allowed_ids:
                continue
            out.append((self.chunk_ids[index], float(score)))
            if len(out) == k:
                break
        return out
