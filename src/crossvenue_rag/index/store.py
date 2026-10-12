"""pgvector-backed chunk store: vectors, metadata, model and `index_version` per chunk.

psycopg and pgvector are imported where a method first needs them, so importing
the retriever, which builds a store, does not load them.

A venue filter is a `WHERE` clause evaluated by the engine that runs the ANN
search rather than a post-filter that shrinks k, because the venue filter is the
control condition of the cross-venue confusion experiment.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import formatting
from ..config import Config
from ..errors import UserError
from ..models import Chunk, ChunkSource
from .embeddings import Embedder

if TYPE_CHECKING:
    import psycopg
    from psycopg import sql

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoreState:
    """What the Postgres table holds: its rows, and the models and index versions on them."""

    rows: int
    models: list[str]
    index_versions: list[str]


def store_state(cfg: Config, *, store: ChunkStore | None = None) -> StoreState | str:
    """The table's state, read on a read-only connection, or why it cannot be read.

    An unreachable server is the normal state before `make db`, so it is returned as a
    value to report. `store` is the table read; `ChunkStore(cfg)` unless given.
    """
    import psycopg

    try:
        store = store or ChunkStore(cfg)
        with store.connect_read_only() as conn:
            return StoreState(
                store.count(conn), store.stored_models(conn), store.stored_index_versions(conn)
            )
    # psycopg.Error: the server refuses the connection or the login, or the table is
    # missing. OSError: the socket fails before psycopg reaches a server.
    except (psycopg.Error, OSError) as exc:
        return str(exc)


class ChunkStore:
    """The chunk table in Postgres: schema, bulk insert, inspection and vector search."""

    def __init__(self, cfg: Config) -> None:
        """A store for the DSN, table and HNSW parameters in `cfg`."""
        self.cfg = cfg

    @property
    def table(self) -> sql.Identifier:
        """The chunk table's name as an SQL identifier."""
        from psycopg import sql

        return sql.Identifier(self.cfg.store.table)

    @contextmanager
    def connect_for_writing(self) -> Iterator[psycopg.Connection]:
        """A connection for building the index: ensures the extension exists.

        The only place in the package that issues DDL is `crossvenue index`,
        through this and `create_schema` / `drop`.
        """
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(self.cfg.store.dsn) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(conn)
            yield conn

    @contextmanager
    def connect_read_only(self) -> Iterator[psycopg.Connection]:
        """A connection that cannot change the database.

        `read_only` makes every transaction READ ONLY, so Postgres itself
        refuses DDL and writes; used by `crossvenue check`, which must be safe to
        run against any database, including one someone else is indexing.
        """
        import psycopg
        from pgvector.psycopg import register_vector

        with psycopg.connect(self.cfg.store.dsn) as conn:
            conn.read_only = True
            register_vector(conn)
            yield conn

    def create_schema(self, conn: psycopg.Connection) -> None:
        """Creates the table, its venue index and its HNSW index where they do not exist."""
        from psycopg import sql

        dim = self.cfg.embedding.dimension
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id        text PRIMARY KEY,
                    venue           text NOT NULL,
                    venue_type      text NOT NULL,
                    doc_path        text NOT NULL,
                    heading_path    jsonb NOT NULL,
                    source_url      text NOT NULL,
                    text            text NOT NULL,
                    token_count     integer NOT NULL,
                    position        integer NOT NULL,
                    sources         jsonb NOT NULL,
                    environment     text,
                    api_version     text,
                    product         text,
                    doc_type        text,
                    embedding_model text NOT NULL,
                    index_version   text NOT NULL,
                    embedding       vector({dim}) NOT NULL
                )
                """
            ).format(table=self.table, dim=sql.Literal(dim))
        )
        conn.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {table} (venue)").format(
                name=sql.Identifier(f"{self.cfg.store.table}_venue_idx"), table=self.table
            )
        )
        conn.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {name} ON {table}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {efc})
                """
            ).format(
                name=sql.Identifier(f"{self.cfg.store.table}_embedding_idx"),
                table=self.table,
                m=sql.Literal(self.cfg.store.hnsw_m),
                efc=sql.Literal(self.cfg.store.hnsw_ef_construction),
            )
        )
        conn.commit()

    def drop(self, conn: psycopg.Connection) -> None:
        """Drops the table, if it exists, with its indexes."""
        from psycopg import sql

        conn.execute(sql.SQL("DROP TABLE IF EXISTS {table}").format(table=self.table))
        conn.commit()

    def insert(
        self,
        conn: psycopg.Connection,
        chunks: Sequence[Chunk],
        vectors: np.ndarray,
        embedding_model: str,
        index_version: str,
    ) -> int:
        """Inserts one row per chunk with its vector, commits, and returns the row count.

        `vectors[i]` is the vector of `chunks[i]`.

        Raises:
            ValueError: `chunks` and `vectors` differ in length.
        """
        from psycopg import sql

        if len(chunks) != len(vectors):
            raise ValueError(
                f"{formatting.plural(len(chunks), 'chunk')} but "
                f"{formatting.plural(len(vectors), 'vector')}"
            )

        rows: Iterable[tuple[object, ...]] = (
            (
                chunk.chunk_id,
                chunk.venue,
                chunk.venue_type,
                chunk.doc_path,
                json.dumps(chunk.heading_path),
                chunk.source_url,
                chunk.text,
                chunk.token_count,
                chunk.position,
                json.dumps([asdict(source) for source in chunk.sources], ensure_ascii=False),
                chunk.environment,
                chunk.api_version,
                chunk.product,
                chunk.doc_type,
                embedding_model,
                index_version,
                vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )

        statement = sql.SQL(
            """
            INSERT INTO {table} (chunk_id, venue, venue_type, doc_path, heading_path,
                                 source_url, text, token_count, position, sources,
                                 environment, api_version, product, doc_type,
                                 embedding_model, index_version, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        ).format(table=self.table)

        with conn.cursor() as cur:
            cur.executemany(statement, list(rows))
        conn.commit()
        return len(chunks)

    def count(self, conn: psycopg.Connection) -> int:
        """The number of rows in the table."""
        from psycopg import sql

        result = conn.execute(sql.SQL("SELECT count(*) FROM {table}").format(table=self.table))
        row = result.fetchone()
        return int(row[0]) if row else 0

    def stored_index_versions(self, conn: psycopg.Connection) -> list[str]:
        """The distinct index versions the rows carry, sorted."""
        from psycopg import sql

        result = conn.execute(
            sql.SQL("SELECT DISTINCT index_version FROM {table}").format(table=self.table)
        )
        return sorted(row[0] for row in result.fetchall())

    def stored_models(self, conn: psycopg.Connection) -> list[str]:
        """The distinct embedding-model fingerprints the rows carry."""
        from psycopg import sql

        result = conn.execute(
            sql.SQL("SELECT DISTINCT embedding_model FROM {table}").format(table=self.table)
        )
        return [row[0] for row in result.fetchall()]

    def search(
        self,
        conn: psycopg.Connection,
        query_vector: np.ndarray,
        k: int,
        venues: Sequence[str] | None,
        chunk_ids: Sequence[str] | None,
    ) -> list[SearchHit]:
        """Cosine-similarity search, optionally restricted to venues and/or ids.

        `chunk_ids` carries the metadata filters: the retriever resolves them to
        an id allow-list once, in Python, so BM25 and Postgres apply the same
        set rather than two implementations of the same rule.

        A hit carries the similarity, because pgvector's `<=>` yields cosine
        *distance*, which would sort backward if passed to a fusion step
        expecting scores.
        """
        from psycopg import sql

        # SET takes no bind parameters, so the value is spliced as a literal.
        conn.execute(
            sql.SQL("SET LOCAL hnsw.ef_search = {}").format(
                sql.Literal(self.cfg.store.hnsw_ef_search)
            )
        )

        where, filter_params = _where(venues, chunk_ids)
        exact = self._exact(venues, chunk_ids, k)
        order = sql.SQL("(embedding <=> %s) + 0, chunk_id" if exact else "embedding <=> %s")
        # Placeholder order: similarity expression, optional filters in clause
        # order, ORDER BY expression, LIMIT.
        params: list[object] = [query_vector, *filter_params, query_vector, k]

        statement = sql.SQL(
            """
            SELECT chunk_id, 1 - (embedding <=> %s) AS similarity,
                   venue, venue_type, doc_path, heading_path, source_url, text, token_count,
                   sources
            FROM {table}
            {where}
            ORDER BY {order}
            LIMIT %s
            """
        ).format(table=self.table, where=where, order=order)

        result = conn.execute(statement, params)
        return [_search_hit(row) for row in result.fetchall()]

    def _exact(self, venues: Sequence[str] | None, chunk_ids: Sequence[str] | None, k: int) -> bool:
        """Whether a search must be exact rather than served by HNSW.

        HNSW visits about `ef_search` rows and filters afterward, so a filter
        it cannot push down, or a k beyond `ef_search`, would return fewer
        than k rows. Those searches are exact: ordering by an expression the
        index cannot serve makes Postgres scan and sort the allowed rows.
        A single venue is left to the planner (venue_split and the oracle take
        this path); on the built index it chooses the venue btree index, an exact
        scan, but the choice follows the table statistics.
        """
        return (
            (venues is not None and len(venues) > 1)
            or chunk_ids is not None
            or k > self.cfg.store.hnsw_ef_search
        )


def _where(
    venues: Sequence[str] | None, chunk_ids: Sequence[str] | None
) -> tuple[sql.Composable, list[list[str]]]:
    """The WHERE clause of a search and its parameters, in clause order."""
    from psycopg import sql

    clauses: list[sql.Composable] = []
    params: list[list[str]] = []
    if venues:
        clauses.append(sql.SQL("venue = ANY(%s)"))
        params.append(list(venues))
    if chunk_ids is not None:
        clauses.append(sql.SQL("chunk_id = ANY(%s)"))
        params.append(list(chunk_ids))
    where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
    return where, params


@dataclass(frozen=True)
class SearchHit:
    """One chunk a vector search returns."""

    chunk_id: str
    similarity: float  # cosine similarity, in [-1, 1]
    metadata: dict[str, Any]  # the chunk's stored columns, as `Result` takes them


def _search_hit(row: Sequence[Any]) -> SearchHit:
    """One result row as a `SearchHit`."""
    chunk_id, similarity, venue, venue_type, doc_path, heading, url, text, tokens, raw = row
    srcs = raw if isinstance(raw, list) else json.loads(raw)
    return SearchHit(
        chunk_id,
        float(similarity),
        {
            "venue": venue,
            "venue_type": venue_type,
            "doc_path": doc_path,
            "heading_path": heading if isinstance(heading, list) else json.loads(heading),
            "source_url": url,
            "text": text,
            "token_count": tokens,
            "sources": [ChunkSource(**src) for src in srcs],
        },
    )


def index_chunks(
    cfg: Config,
    chunks: Sequence[Chunk],
    index_version: str,
    *,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
) -> int:
    """Encodes every chunk and loads it into a freshly created table.

    The table is dropped and recreated on every call: every chunk is re-encoded
    anyway, and a rebuild is the only way a schema change cannot leave
    half-migrated rows behind. `embedder` and `store` are built from `cfg` unless given.

    Raises:
        UserError: The encoder produces vectors of another dimension than
            `embedding.dimension`.
    """
    embedder = embedder or Embedder(cfg.embedding)
    store = store or ChunkStore(cfg)

    vectors = embedder.encode_documents([c.text for c in chunks], show_progress=True)
    if vectors.shape[1] != cfg.embedding.dimension:
        raise UserError(
            f"the config sets embedding.dimension to "
            f"{formatting.own_form(cfg.embedding.dimension)}, but {cfg.embedding.model_name} "
            f"produced {formatting.plural(int(vectors.shape[1]), 'dimension')}",
            hint=f"set `embedding.dimension: {formatting.own_form(vectors.shape[1])}` "
            "in the config",
        )

    with store.connect_for_writing() as conn:
        store.drop(conn)
        store.create_schema(conn)
        written = store.insert(conn, chunks, vectors, embedder.model_fingerprint(), index_version)
        conn.commit()
    # basicConfig prints this line to the terminal, so its count is formatted.
    log.info("indexed %s at index_version %s", formatting.plural(written, "chunk"), index_version)
    return written
