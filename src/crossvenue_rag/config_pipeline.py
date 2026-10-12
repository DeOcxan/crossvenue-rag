"""The pipeline config sections: paths, fetching, chunking, encoders, retrieval and store.

Each section is a `config_section.Section`; `config.Config` holds them with the
evaluation and generation sections.
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from . import formatting
from .config_section import Section, named
from .confined_paths import DirectoryPath

RetrievalMethod = Literal[
    "bm25", "dense", "hybrid", "hybrid_rerank", "venue_split", "venue_split_oracle"
]
# A torch device: cpu, cuda, mps or xpu, with an optional device index.
DEVICE_PATTERN = r"^(cpu|cuda|mps|xpu)(:\d+)?$"
# Retries a request gets at most: the wait doubles on each, so the tenth is 512 times the first.
MAX_RETRIES = 10
# The longest table name whose index names (`<table>_embedding_idx`, 14 more characters)
# stay within Postgres's 63-byte identifier limit; a table name is ASCII, one byte a character.
MAX_TABLE_NAME = 49
# A table name: ASCII letters, digits and underscores.
TABLE_NAME_PATTERN = r"[A-Za-z0-9_]+"
# The methods whose fused list the cross-encoder reranks (a warmup loads it), and all those
# whose list is fused from the BM25 and dense candidates (`retrieval.retriever`).
RERANKED_METHODS = frozenset({"hybrid_rerank", "venue_split", "venue_split_oracle"})
FUSED_METHODS = RERANKED_METHODS | {"hybrid"}
# pgvector requires an HNSW index's ef_construction to be at least this many times its m.
EF_CONSTRUCTION_PER_M = 2
# pgvector's bounds on the HNSW parameters, smallest and largest each.
HNSW_M_MIN, HNSW_M_MAX = 2, 100
HNSW_EF_CONSTRUCTION_MIN, HNSW_EF_CONSTRUCTION_MAX = 4, 1000
HNSW_EF_SEARCH_MIN, HNSW_EF_SEARCH_MAX = 1, 1000


def package_version() -> str:
    """The installed version of crossvenue-rag, or "unknown" from a bare checkout."""
    try:
        return version("crossvenue-rag")
    except PackageNotFoundError:
        return "unknown"


def default_user_agent() -> str:
    """The User-Agent header the fetcher sends, naming the package version."""
    return f"crossvenue-rag/{package_version()} (research corpus builder)"


class PathsConfig(Section):
    """Where the pipeline reads its inputs and writes its artifacts, relative to the repository."""

    sources: Path = Path("data/sources.yaml")
    manifest: Path = Path("data/manifest.csv")
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    index: Path = Path("data/index")
    results: DirectoryPath = Path("results")


class ChunkingConfig(Section):
    """Token budgets of the chunker, counted by the embedding model's tokenizer."""

    max_tokens: int = Field(450, gt=0)
    overlap_tokens: int = Field(60, ge=0)
    min_tokens: int = Field(40, ge=0)
    keep_code_blocks_intact: bool = True
    max_code_block_tokens: int = Field(1200, gt=0)

    @model_validator(mode="after")
    def _overlap_fits(self) -> ChunkingConfig:
        overlap = formatting.own_form(self.overlap_tokens)
        most = formatting.own_form(self.max_tokens)
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap}) must be smaller than max_tokens ({most}); "
                "otherwise chunking never advances"
            )
        if self.min_tokens > self.max_tokens:
            least = formatting.own_form(self.min_tokens)
            raise ValueError(f"min_tokens ({least}) must not exceed max_tokens ({most})")
        return self


class EmbeddingConfig(Section):
    """The dense encoder: model, vector dimension, prefixes and batching."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    dimension: int = Field(768, gt=0)
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    batch_size: int = Field(32, gt=0)
    # None picks mps, cuda or cpu, whichever the machine has.
    device: str | None = Field(None, pattern=DEVICE_PATTERN)
    # Hugging Face commit of the model and its tokenizer. Not part of
    # `index_version` (docs/adr/0004-index-version.md); recorded in index_meta.json.
    revision: str | None = Field(None, min_length=1)

    @field_validator("model_name")
    @classmethod
    def _names_a_model(cls, v: str) -> str:
        return named(v, "model_name")


class RerankerConfig(Section):
    """The cross-encoder and how many fused candidates it rescores."""

    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = Field(50, gt=0)
    batch_size: int = Field(16, gt=0)
    # Hugging Face commit of the cross-encoder.
    revision: str | None = Field(None, min_length=1)

    @field_validator("model_name")
    @classmethod
    def _names_a_model(cls, v: str) -> str:
        return named(v, "model_name")


class RetrievalConfig(Section):
    """Results returned, candidates per retrieval side, and the RRF constant."""

    k: int = Field(10, gt=0)
    dense_candidates: int = Field(50, gt=0)
    bm25_candidates: int = Field(50, gt=0)
    rrf_k: int = Field(60, gt=0)


class StoreConfig(Section):
    """The pgvector store: connection, table and HNSW parameters."""

    # Left out of every dump, so no saved file holds a password (`Config.to_yaml`).
    dsn: str = Field("postgresql://crossvenue:crossvenue@localhost:5433/crossvenue", exclude=True)
    table: str = Field("chunks", max_length=MAX_TABLE_NAME)
    # Within pgvector's HNSW bounds; ef_construction must also be at least twice m.
    hnsw_m: int = Field(16, ge=HNSW_M_MIN, le=HNSW_M_MAX)
    hnsw_ef_construction: int = Field(64, ge=HNSW_EF_CONSTRUCTION_MIN, le=HNSW_EF_CONSTRUCTION_MAX)
    hnsw_ef_search: int = Field(100, ge=HNSW_EF_SEARCH_MIN, le=HNSW_EF_SEARCH_MAX)

    @field_validator("dsn")
    @classmethod
    def _connection_string(cls, v: str) -> str:
        """A libpq URI or key=value string, checked in shape only: nothing is connected to."""
        if not (re.match(r"^postgres(ql)?://", v) or "=" in v):
            raise ValueError(
                "dsn must be a libpq URI (postgresql://user:password@host:port/db) or "
                "key=value pairs (host=... dbname=...)"
            )
        return v

    @model_validator(mode="after")
    def _ef_construction_covers_m(self) -> StoreConfig:
        if self.hnsw_ef_construction < EF_CONSTRUCTION_PER_M * self.hnsw_m:
            raise ValueError(
                f"hnsw_ef_construction ({formatting.own_form(self.hnsw_ef_construction)}) must "
                f"be at least {formatting.count(EF_CONSTRUCTION_PER_M)} times hnsw_m "
                f"({formatting.own_form(self.hnsw_m)}), as pgvector requires"
            )
        return self

    @field_validator("table")
    @classmethod
    def _plain_identifier(cls, v: str) -> str:
        # The table name is interpolated into DDL, so it may not be arbitrary text; ASCII
        # only, so the length limit in characters is the identifier limit in bytes.
        if not re.fullmatch(TABLE_NAME_PATTERN, v):
            raise ValueError(f"table must be ASCII letters, digits and underscores, got {v!r}")
        return v


class FetchConfig(Section):
    """How the fetcher makes HTTP requests: pacing, timeout, User-Agent and retries."""

    request_delay_seconds: float = Field(0.5, ge=0, allow_inf_nan=False)
    timeout_seconds: float = Field(30.0, gt=0, allow_inf_nan=False)
    # Names the package version, so a site operator can tell which build fetched.
    user_agent: str = Field(default_factory=default_user_agent)
    max_retries: int = Field(3, ge=0, le=MAX_RETRIES)
    # The first wait before a retry, doubling on each; apart from the pacing delay, so
    # pacing can be 0 without a failed request being tried again at once.
    backoff_seconds: float = Field(0.5, gt=0, allow_inf_nan=False)

    @field_validator("user_agent")
    @classmethod
    def _names_the_fetcher(cls, v: str) -> str:
        """A site operator reads the User-Agent to tell who fetched, so it may not be blank."""
        return named(v, "user_agent")
