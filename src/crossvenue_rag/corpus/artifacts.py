"""Processed-corpus artifacts: the chunk file and the statistics beside it.

`ingest` writes these files and everything downstream reads them through this
module, so retrieval reads chunks.jsonl without importing the chunker, the
normalizer or the tokenizer loader.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .. import formatting, jsonl, listing
from ..config import Config
from ..errors import UserError, not_blank, require_file
from ..models import Chunk, ChunkSource, VenueType, missing_fields, wrongly_typed_fields
from ..run_files import read_json_model

CHUNKS_FILENAME = "chunks.jsonl"
STATS_FILENAME = "corpus_stats.json"
# What a missing chunks.jsonl is told to do: `ingest` writes it.
INGEST_HINT = "run `crossvenue ingest` first"


def chunks_path(cfg: Config) -> Path:
    """The path of chunks.jsonl under the configured processed directory."""
    return cfg.paths.processed / CHUNKS_FILENAME


def stats_path(cfg: Config) -> Path:
    """The path of corpus_stats.json under the configured processed directory."""
    return cfg.paths.processed / STATS_FILENAME


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CorpusTotals(_Frozen):
    """Documents, chunks and token lengths over the whole corpus."""

    documents: int
    chunks: int
    tokens: int
    tokens_p50: int
    tokens_p95: int


class VenueStats(_Frozen):
    """Documents, chunks and token lengths of one venue."""

    venue: str
    venue_type: VenueType
    documents: int
    chunks: int
    tokens_total: int
    tokens_mean: float
    tokens_p50: int
    tokens_p95: int
    tokens_max: int


class CrossVenueExample(_Frozen):
    """One text found on more than one venue, left unmerged."""

    venues: list[str]
    chunk_ids: list[str]
    doc_paths: list[str]


class DedupSummary(_Frozen):
    """What exact deduplication merged and what it left apart.

    `groups` counts groups of two or more occurrences merged into one chunk,
    all within one venue; `testnet_mirror_groups` those spanning testnet and
    non-testnet pages.
    """

    chunks_before: int
    chunks_after: int
    removed: int
    groups: int
    occurrences_in_groups: int
    cross_venue_groups: int
    cross_venue_examples: list[CrossVenueExample]
    groups_by_venue: dict[str, int]
    testnet_mirror_groups: int


class FieldCoverage(_Frozen):
    """How often one structural field was assigned, its values, and where it was not."""

    assigned: int
    share: float
    values: dict[str, int]
    unassigned_by_venue: dict[str, int]


class MetadataCoverage(_Frozen):
    """The share of chunks each structural field was assigned on."""

    chunks: int
    fields: dict[str, FieldCoverage]


class CorpusStats(_Frozen):
    """corpus_stats.json: the corpus by venue, chunk lengths, deduplication and metadata.

    Every format `ingest` has written: two-space indent, non-ASCII characters
    unescaped, no final newline; `index_version`, `dedup` and
    `metadata_coverage` are absent from files written before each was added.
    """

    totals: CorpusTotals
    by_venue: list[VenueStats]
    chunk_length_histogram: dict[str, int]
    index_version: str | None = None
    dedup: DedupSummary | None = None
    metadata_coverage: MetadataCoverage | None = None


def corpus_stats_json(stats: CorpusStats) -> str:
    """corpus_stats.json's text for `stats`."""
    fields = stats.model_dump(mode="json", exclude_unset=True)
    return json.dumps(fields, indent=2, ensure_ascii=False)


def write_corpus_stats(cfg: Config, stats: CorpusStats) -> None:
    """Writes `stats` to corpus_stats.json beside the chunks."""
    stats_path(cfg).write_text(corpus_stats_json(stats), encoding="utf-8")


def read_corpus_stats(cfg: Config) -> CorpusStats:
    """corpus_stats.json of the last ingest.

    Raises:
        UserError: There is no corpus_stats.json.
    """
    path = stats_path(cfg)
    if not path.exists():
        raise UserError(f"no corpus statistics at {path}", hint=INGEST_HINT)
    return read_json_model(path, CorpusStats, hint="run `crossvenue ingest` again")


def save_chunks(cfg: Config, chunks: Sequence[Chunk]) -> Path:
    """Writes `chunks` to chunks.jsonl and returns its path."""
    path = chunks_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write(path, (c.to_dict() for c in chunks))
    return path


def load_chunks(cfg: Config) -> list[Chunk]:
    """Every chunk in the configured chunks.jsonl, in file order."""
    return read_chunks(require_file(chunks_path(cfg), hint=INGEST_HINT))


def read_chunks(path: Path) -> list[Chunk]:
    """Every chunk of a chunks.jsonl at `path`, in file order.

    Raises:
        UserError: A record is not a chunk record: it, or one of its `sources`, lacks a
            field (`chunk_id` among them), holds a value of the wrong type in one, or has an
            unknown one. The message names the line.
    """
    chunks: list[Chunk] = []
    for number, row in jsonl.iter_rows(path):
        if problems := _chunk_record_problems(row):
            raise UserError(
                f"{formatting.location(path, number)}: the record {listing.prose(problems)}",
                hint="run `crossvenue ingest` again",
            )
        chunks.append(Chunk.from_dict(row))
    return chunks


def _chunk_record_problems(row: Mapping[str, Any]) -> list[str]:
    """What keeps `row` from being a `Chunk` record (`Chunk.from_dict`): none for a valid one."""
    sources = row.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        return [*_field_problems(row, Chunk), "has `sources` that are not a list of objects"]
    in_sources = (
        f"has a source that {p}" for s in sources for p in _field_problems(s, ChunkSource)
    )
    return [*_field_problems(row, Chunk), *dict.fromkeys(in_sources)]


def _field_problems(record: Mapping[str, Any], model: type[Chunk | ChunkSource]) -> list[str]:
    """What `record` gets wrong against the dataclass `model`: lacking, mistyped and unknown keys.

    `sources` is not type-checked here: the caller checks each source against `ChunkSource`.
    """
    lacking = missing_fields(record, model)
    typed = {name: value for name, value in record.items() if name != "sources"}
    wrong = wrongly_typed_fields(typed, model)
    known = {f.name for f in fields(model)}
    unknown = [key for key in record if key not in known]
    return [
        *([f"has no {listing.prose(lacking)}"] if lacking else []),
        *(f"has a wrong {name}: {problem}" for name, problem in wrong.items()),
        *(
            [f"has {formatting.plural(len(unknown), 'unknown field')} {listing.prose(unknown)}"]
            if unknown
            else []
        ),
    ]


def chunks_by_id(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    """`chunks` keyed by chunk id."""
    return {chunk.chunk_id: chunk for chunk in chunks}


@dataclass(frozen=True)
class TermCount:
    """Chunks containing a term: in total, and per venue with at least one, in corpus order."""

    total: int
    by_venue: Mapping[str, int]


def count_term(cfg: Config, term: str, venue: str | None) -> TermCount:
    """Chunks containing `term`, per venue.

    The counting convention of this project: the unit is the chunk, the match
    is a case-insensitive substring, and the corpus is
    `data/processed/chunks.jsonl`. Counting occurrences instead of chunks, or
    matching case, gives another number for the same term, and these numbers
    are cited in `notes` fields and CHANGELOG entries.

    Raises:
        UserError: `term` is empty or only whitespace, which every chunk or none
            contains, or `venue` is not a venue of the corpus.
    """
    not_blank("the term to count", term)
    texts = lowered_texts(load_chunks(cfg))
    venues = sorted({v for v, _ in texts})
    if venue is not None and venue not in venues:
        raise UserError(
            f"--venue {venue!r} is not a venue of the corpus",
            hint=f"pass --venue one of: {', '.join(venues)}",
        )
    return count_term_in(texts, term, venue)


def lowered_texts(chunks: Iterable[Chunk]) -> list[tuple[str, str]]:
    """(venue, lower-cased text) of each chunk, the input of `count_term_in`."""
    return [(chunk.venue, chunk.text.lower()) for chunk in chunks]


def count_term_in(
    texts: Iterable[tuple[str, str]], term: str, venue: str | None = None
) -> TermCount:
    """`count_term` over (venue, lower-cased text) pairs from `lowered_texts`."""
    needle = term.lower()
    counts: dict[str, int] = {}
    total = 0
    for chunk_venue, text in texts:
        if venue and chunk_venue != venue:
            continue
        if needle in text:
            counts[chunk_venue] = counts.get(chunk_venue, 0) + 1
            total += 1
    return TermCount(total, counts)


def load_chunk_venues(cfg: Config) -> dict[str, str]:
    """`chunk_id` -> `venue`, for checks of the venue a label points to.

    Keeps two fields per chunk instead of building every chunk, which is all a
    check that a dataset's pointers resolve needs.
    """
    return read_chunk_venues(chunks_path(cfg))


def venue_chunk_counts(cfg: Config) -> dict[str, int]:
    """The number of chunks of each venue in chunks.jsonl."""
    return dict(Counter(load_chunk_venues(cfg).values()))


def read_chunk_venues(path: Path) -> dict[str, str]:
    """`chunk_id` -> `venue` of a chunks.jsonl at `path`."""
    return read_chunk_field(path, "venue")


def read_chunk_field(path: Path, name: str) -> dict[str, str]:
    """`chunk_id` -> the string field `name` (`venue`, `text`) of each record of a chunks.jsonl.

    A record without the field maps to "". It builds no `Chunk`, so it reads a record
    of any format `ingest` has written.

    Raises:
        UserError: Nothing is at `path`, or a directory is. A record lacks a string `chunk_id`, or
            holds a `name` that is not a string; the message names its line.
    """
    fields: dict[str, str] = {}
    for number, row in jsonl.iter_rows(require_file(path, hint=INGEST_HINT)):
        record = {name: "", **row}
        if problem := _string_problem(path, number, record, ("chunk_id", name)):
            raise UserError(problem, hint="run `crossvenue ingest` again")
        fields[record["chunk_id"]] = record[name]
    return fields


@dataclass(frozen=True)
class ChunkKey:
    """What identifies a chunk in a chunks.jsonl record: its id, venue and text."""

    chunk_id: str
    venue: str
    text: str


def read_chunk_keys(path: str | Path, *, hint: str) -> list[ChunkKey]:
    """The id, venue and text of each record of a chunks.jsonl at `path`, in file order.

    It builds no `Chunk`, so it reads the file of an earlier index whatever fields
    its records have beside these three.

    Raises:
        UserError: A record lacks a string `chunk_id`, `venue` or `text`, with `hint` as
            the fix.
    """
    rows = list(jsonl.iter_rows(path))
    keys: list[ChunkKey] = []
    for number, row in rows:
        if problem := _string_problem(path, number, row, ("chunk_id", "venue", "text")):
            raise UserError(problem, hint=hint)
        keys.append(ChunkKey(row["chunk_id"], row["venue"], row["text"]))
    return keys


def _string_problem(
    path: str | Path, number: int, row: Mapping[str, Any], names: Sequence[str]
) -> str | None:
    """Why a chunks.jsonl record cannot be read, or None when its fields `names` are strings."""
    lacking = [name for name in names if not isinstance(row.get(name), str)]
    if not lacking:
        return None
    return (
        f"{formatting.location(path, number)}: the record lacks a string {listing.prose(lacking)}"
    )
