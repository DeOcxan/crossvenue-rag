"""Raw pages -> normalized documents -> chunks -> deduplicated chunks on disk.

Also writes corpus statistics, the deduplication report and metadata coverage
beside the chunks, so `make stats` can show all three without re-ingesting.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from .. import formatting, listing
from ..config import Config
from ..errors import UserError
from ..index_version import compute_index_version
from ..models import Chunk, Document
from ..run_files import file_sha256
from ..stats import P50, P95, mean, percentile
from .artifacts import CorpusStats, CorpusTotals, VenueStats, save_chunks, write_corpus_stats
from .chunking import Chunker, TokenCounter
from .dedup import deduplicate
from .doc_metadata import MetadataRules, coverage
from .manifest import ManifestRow, read_manifest
from .normalize import normalize_markdown
from .tokens import token_counter_for

# The decimals of a venue's mean chunk length in corpus_stats.json.
TOKENS_MEAN_DECIMALS = 1
# The lower edges, in tokens, of the chunk-length histogram's nine coarse bins, the last
# open-ended: enough to see a chunker regression in a terminal.
LENGTH_BIN_EDGES = (0, 50, 100, 200, 300, 400, 500, 750, 1000)
# Each bin as (lower edge, upper edge excluded), in order; the last has no upper edge.
LENGTH_BINS: tuple[tuple[int, int | None], ...] = tuple(
    zip(LENGTH_BIN_EDGES, [*LENGTH_BIN_EDGES[1:], None], strict=True)
)

log = logging.getLogger(__name__)


def doc_path_of(row: ManifestRow) -> str:
    """A manifest row's document path: its `local_path` without the first segment, the venue.

    The venue is already a field of its own; a `local_path` with no `/` is returned whole.
    """
    return row.local_path.split("/", 1)[1] if "/" in row.local_path else row.local_path


def load_document(row: ManifestRow, raw_root: Path) -> Document | None:
    """One manifest row read into a normalized Document.

    None when the file normalizes to nothing, and None with a logged warning when it
    cannot be read or is not UTF-8: that document is left out of the corpus. `ingest`
    has already refused a file whose bytes differ from the manifest.
    """
    path = raw_root / row.local_path
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("unreadable %s: %s", path, exc)
        return None

    text = normalize_markdown(raw)
    if not text.strip():
        return None

    return Document(
        venue=row.venue,
        venue_type=row.venue_type,
        doc_path=doc_path_of(row),
        source_url=row.source_url,
        text=text,
    )


def unverified_files(rows: list[ManifestRow], raw_root: Path) -> list[str]:
    """Manifest rows whose file is missing or differs from the recorded sha256."""
    bad = []
    for row in rows:
        path = raw_root / row.local_path
        if not path.is_file() or file_sha256(path) != row.sha256:
            bad.append(row.local_path)
    return bad


def ingest(cfg: Config, count_tokens: TokenCounter | None = None) -> list[Chunk]:
    """Normalizes and chunks everything in the manifest; writes chunks.jsonl.

    Every raw file must match the sha256 the manifest records: `index_version`
    hashes the manifest, not the files, so a changed file would change chunk
    ids under an unchanged version.

    Raises:
        UserError: The manifest is empty or missing, or a raw file is missing or differs
            from it.
    """
    rows = read_manifest(cfg.paths.manifest)
    if not rows:
        raise UserError(
            f"{cfg.paths.manifest} is empty or missing", hint="run `crossvenue fetch` first"
        )
    bad = unverified_files(rows, cfg.paths.raw)
    if bad:
        raise UserError(
            f"{formatting.plural(len(bad), 'raw file')} missing or different from "
            f"{cfg.paths.manifest}: {listing.first_items(bad)}",
            hint="run `crossvenue fetch`, which checks out the recorded sources and verifies them",
        )

    if count_tokens is None:
        count_tokens = token_counter_for(
            cfg.embedding.model_name, cfg.embedding.revision, cfg.chunking.max_tokens
        )

    chunker = Chunker(cfg.chunking, count_tokens)
    rules = _metadata_rules(cfg)
    occurrences: list[Chunk] = []
    documents = 0

    for row in rows:
        doc = load_document(row, cfg.paths.raw)
        if doc is None:
            continue
        documents += 1
        for chunk in chunker.chunk(doc):
            for source in chunk.sources:
                rules.annotate(source)
            occurrences.append(chunk)

    chunks, dedup_report = deduplicate(occurrences)
    save_chunks(cfg, chunks)
    stats = corpus_stats(chunks, documents_by_venue(rows)).model_copy(
        update={
            "index_version": compute_index_version(cfg),
            "dedup": dedup_report,
            "metadata_coverage": coverage(chunks),
        }
    )
    write_corpus_stats(cfg, stats)
    # basicConfig prints this line to the terminal, so its numbers are formatted.
    log.info(
        "ingested %s into %s (%s before exact dedup, %s, %s left separate)",
        formatting.plural(documents, "document"),
        formatting.plural(len(chunks), "chunk"),
        formatting.count(dedup_report.chunks_before),
        formatting.plural(dedup_report.groups, "group"),
        formatting.plural(dedup_report.cross_venue_groups, "cross-venue group"),
    )
    return chunks


def _metadata_rules(cfg: Config) -> MetadataRules:
    if not Path(cfg.paths.sources).exists():
        log.warning("%s not found; chunks get no structural metadata", cfg.paths.sources)
        return MetadataRules([])
    return MetadataRules.from_file(cfg.paths.sources)


def documents_by_venue(rows: list[ManifestRow]) -> dict[str, int]:
    """Manifest documents per venue: the `documents` column of corpus_stats.json."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.venue] += 1
    return dict(counts)


def _percentile(values: list[int], fraction: float) -> int:
    # A corpus without chunks has no chunk lengths; its statistics report 0 tokens.
    return int(percentile(values, fraction)) if values else 0


def corpus_stats(chunks: list[Chunk], doc_counts: dict[str, int]) -> CorpusStats:
    """Per-venue document/chunk counts and the chunk-length distribution.

    Saved precision: mean tokens per chunk at one decimal.
    """
    by_venue: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_venue[chunk.venue].append(chunk)

    venues: list[VenueStats] = []
    for venue, venue_chunks in sorted(by_venue.items()):
        lengths = sorted(c.token_count for c in venue_chunks)
        venues.append(
            VenueStats(
                venue=venue,
                venue_type=venue_chunks[0].venue_type,
                documents=doc_counts.get(venue, 0),
                chunks=len(venue_chunks),
                tokens_total=sum(lengths),
                tokens_mean=round(mean(lengths), TOKENS_MEAN_DECIMALS),
                tokens_p50=_percentile(lengths, P50),
                tokens_p95=_percentile(lengths, P95),
                tokens_max=lengths[-1],
            )
        )

    all_lengths = sorted(c.token_count for c in chunks)
    histogram = {}
    for low, high in LENGTH_BINS:
        label = f"{low}-{high}" if high is not None else f"{low}+"
        histogram[label] = sum(
            1 for length in all_lengths if low <= length and (high is None or length < high)
        )

    totals = CorpusTotals(
        documents=sum(doc_counts.values()),
        chunks=len(chunks),
        tokens=sum(all_lengths),
        tokens_p50=_percentile(all_lengths, P50),
        tokens_p95=_percentile(all_lengths, P95),
    )
    return CorpusStats(totals=totals, by_venue=venues, chunk_length_histogram=histogram)


__all__ = ["corpus_stats", "ingest", "load_document", "unverified_files"]
