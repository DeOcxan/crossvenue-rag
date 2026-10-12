"""Exact deduplication: one canonical chunk per byte-identical text within a venue.

Only exact text equality merges, so a chunk that differs by one character stays
separate. Identical text on two venues stays one chunk per venue and is
reported, because which venue says something is what this corpus measures.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..hashes import text_sha256
from ..models import Chunk, ChunkSource
from .artifacts import CrossVenueExample, DedupSummary

# The cross-venue groups a report shows as examples.
CROSS_VENUE_EXAMPLES = 10

# Environment order for choosing the primary occurrence. Anything that is not
# testnet (mainnet, or no environment at all) is cited before testnet.
_ENVIRONMENT_RANK = {"testnet": 1}


def primary_key(source: ChunkSource) -> tuple[int, str, int]:
    """Sort key choosing which occurrence a canonical chunk is cited as.

    Non-testnet before testnet, then doc path lexicographically, then position
    within the document. Deterministic, and it makes a Binance chunk shared by
    `rest-api.md` and `testnet/rest-api.md` cite the production page.
    """
    return (_ENVIRONMENT_RANK.get(source.environment or "", 0), source.doc_path, source.position)


def deduplicate(chunks: Sequence[Chunk]) -> tuple[list[Chunk], DedupSummary]:
    """The chunks with those of the same id (same venue, same text) merged into one.

    Byte-identical copies have nothing that distinguishes them and would
    occupy top-k slots that another relevant passage could hold, so each text
    is stored once and every occurrence is kept in `Chunk.sources`.

    Output order is the order of each group's first occurrence in the input, so
    an unchanged corpus produces an unchanged file. Each canonical chunk's
    citation fields and metadata come from its primary occurrence.
    """
    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.chunk_id, []).append(chunk)

    canonical: list[Chunk] = []
    merged = occurrences_in_groups = testnet_mirror_groups = 0
    groups_by_venue: dict[str, int] = {}
    for chunk_id, members in groups.items():
        sources = sorted((s for m in members for s in m.sources), key=primary_key)
        primary = sources[0]
        first = members[0]
        canonical.append(
            Chunk(
                chunk_id=chunk_id,
                venue=first.venue,
                venue_type=first.venue_type,
                doc_path=primary.doc_path,
                heading_path=list(primary.heading_path),
                source_url=primary.source_url,
                text=first.text,
                token_count=first.token_count,
                position=primary.position,
                sources=sources,
                environment=primary.environment,
                api_version=primary.api_version,
                product=primary.product,
                doc_type=primary.doc_type,
            )
        )
        if len(members) > 1:
            merged += 1
            occurrences_in_groups += len(members)
            groups_by_venue[first.venue] = groups_by_venue.get(first.venue, 0) + 1
            environments = {s.environment == "testnet" for s in sources}
            testnet_mirror_groups += environments == {True, False}

    by_text: dict[str, list[Chunk]] = {}
    for chunk in canonical:
        by_text.setdefault(text_sha256(chunk.text), []).append(chunk)
    cross_venue = [members for members in by_text.values() if len(members) > 1]
    examples = [
        CrossVenueExample(
            venues=sorted(m.venue for m in members),
            chunk_ids=[m.chunk_id for m in members],
            doc_paths=[m.doc_path for m in members],
        )
        for members in cross_venue[:CROSS_VENUE_EXAMPLES]
    ]
    summary = DedupSummary(
        chunks_before=len(chunks),
        chunks_after=len(canonical),
        removed=len(chunks) - len(canonical),
        groups=merged,
        occurrences_in_groups=occurrences_in_groups,
        cross_venue_groups=len(cross_venue),
        cross_venue_examples=examples,
        groups_by_venue=dict(sorted(groups_by_venue.items())),
        testnet_mirror_groups=testnet_mirror_groups,
    )
    return canonical, summary
