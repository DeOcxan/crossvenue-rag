"""`index_version`: one short hash naming the exact index a label or result belongs to.

A gold label is a chunk id, which means something only against the corpus, chunker,
schema and model that produced it: scored against another index, it would point at
different text without failing. `eval` and `label` refuse to run when the versions disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .clock import utc_now
from .config import Config
from .config_pipeline import ChunkingConfig
from .errors import UserError, require_file
from .hashes import text_sha256
from .models import CHUNK_SCHEMA_VERSION, ID_HASH_DIGITS
from .run_files import file_sha256, read_json_model

INDEX_META_FILENAME = "index_meta.json"
# What a damaged index_meta.json is told to do: `crossvenue index` writes it.
INDEX_META_HINT = "rebuild the index with `make ingest index`"
# What a command that needs the built index is told when there is none.
BUILD_INDEX_HINT = "build the index with `make ingest index`"


class EmbeddingComponents(BaseModel):
    """The embedding settings that change a vector: model, dimension, prefix, normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    dimension: int
    document_prefix: str
    normalize: bool


class IndexComponents(BaseModel):
    """Everything that can move a chunk id or a vector, which the index version hashes.

    The sha256 of data/manifest.csv (any source document changing), the chunking
    config, the embedding settings and the chunk schema version
    (`models.CHUNK_SCHEMA_VERSION`). index_meta.json saves it as `components`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_sha256: str
    chunking: ChunkingConfig
    embedding: EmbeddingComponents
    chunk_schema_version: str


def index_components(cfg: Config) -> IndexComponents:
    """The index components of this config and the manifest it names."""
    manifest = require_file(cfg.paths.manifest, hint="run `crossvenue fetch` first")
    return IndexComponents(
        manifest_sha256=file_sha256(manifest),
        chunking=cfg.chunking,
        embedding=EmbeddingComponents(
            model_name=cfg.embedding.model_name,
            dimension=cfg.embedding.dimension,
            document_prefix=cfg.embedding.document_prefix,
            normalize=cfg.embedding.normalize,
        ),
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
    )


def version_of(components: IndexComponents) -> str:
    """The index version of `components`: the first 16 hex digits of their canonical-JSON sha256.

    The JSON is the components' fields with sorted keys, so the version does not
    depend on field order.
    """
    fields = components.model_dump(mode="json")
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False)
    return text_sha256(payload)[:ID_HASH_DIGITS]


def compute_index_version(cfg: Config) -> str:
    """The version an index built from this config and manifest would have."""
    return version_of(index_components(cfg))


def index_meta_path(cfg: Config) -> Path:
    """The path of index_meta.json in the configured index directory."""
    return cfg.paths.index / INDEX_META_FILENAME


def model_revisions(cfg: Config) -> dict[str, str | None]:
    """The Hugging Face commits of the encoder (and its tokenizer) and the reranker."""
    return {"embedding": cfg.embedding.revision, "reranker": cfg.reranker.revision}


def write_index_meta(cfg: Config, chunks: int) -> str:
    """Writes index_meta.json for an index of `chunks` chunks and returns its index version."""
    components = index_components(cfg)
    version = version_of(components)
    path = index_meta_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "index_version": version,
                "built_at": utc_now(),
                "chunks": chunks,
                "components": components.model_dump(mode="json"),
                # Outside `components`, so not part of the version (ADR 0004).
                "model_revisions": model_revisions(cfg),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return version


class IndexMeta(BaseModel):
    """The part of index_meta.json a reader needs: the `index_version` it records.

    The other keys `write_index_meta` writes (`built_at`, `chunks`, `components`,
    `model_revisions`) are ignored here.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    index_version: str


def built_index_version(cfg: Config) -> str | None:
    """The index version recorded in index_meta.json; None when no index is built."""
    return recorded_index_version(index_meta_path(cfg))


def recorded_index_version(meta_path: Path) -> str | None:
    """The `index_version` an index_meta.json at `meta_path` records; None when there is no file."""
    if not meta_path.exists():
        return None
    return read_json_model(meta_path, IndexMeta, hint=INDEX_META_HINT).index_version


def require_built_index_version(meta_path: Path) -> str:
    """The `index_version` the index_meta.json at `meta_path` records, which a built index has.

    Raises:
        UserError: There is no index_meta.json at `meta_path`, so no index is built; the hint
            names the command that builds it.
    """
    version = recorded_index_version(meta_path)
    if version is None:
        raise UserError(
            f"no index is built: no index metadata at {meta_path}", hint=BUILD_INDEX_HINT
        )
    return version


def require_current_index(cfg: Config, store_versions: list[str] | None = None) -> str:
    """The built index version, after checking it is the one this config implies.

    `store_versions` is what Postgres rows carry (`None` skips that check, for
    callers with no database).

    Raises:
        UserError: The built version is not the one this config implies, or
            `store_versions` is not the built version alone. The hint names the fix.
    """
    built = require_built_index_version(index_meta_path(cfg))
    expected = compute_index_version(cfg)
    if built != expected:
        raise UserError(
            f"the index on disk is {built}, but this config and manifest imply {expected}",
            hint="the manifest, chunking config, embedding model or chunk schema changed "
            "since the last build; rebuild with `make ingest index`",
        )
    if store_versions is not None and store_versions != [built]:
        raise UserError(
            f"Postgres rows carry index_version {', '.join(store_versions) or 'none'}, "
            f"the BM25 side is {built}",
            hint="the two indexes were built from different corpora; rebuild with `make index`",
        )
    return built
