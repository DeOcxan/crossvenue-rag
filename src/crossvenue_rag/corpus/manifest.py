"""The corpus manifest: one row per fetched document, on disk as CSV.

The manifest and `data/sources.yaml` are the committed part of `data/`, and the
manifest makes the corpus reproducible from a clone (URLs, commit SHAs and content
hashes, with no vendor text redistributed). It is separate from `fetch`, so ingestion, dataset
validation and the CLI read it without importing `httpx` and `subprocess`.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .. import formatting
from ..clock import utc_now
from ..errors import UserError, require_file
from ..models import VENUE_TYPES, VenueType
from ..run_files import file_sha256, write_csv

# `local_path` is relative to `paths.raw`, that is `<venue>/<path inside the source>`,
# so the manifest stays valid if the raw directory is moved.
MANIFEST_FIELDS = [
    "venue",
    "venue_type",
    "source_url",
    "local_path",
    "fetched_at",
    "commit_sha",
    "sha256",
]


@dataclass(slots=True)
class ManifestRow:
    """Provenance for one fetched document."""

    venue: str
    venue_type: VenueType
    source_url: str
    local_path: str
    fetched_at: str
    # Empty for GitBook sources: there is no commit to pin to, so the content
    # hash is the only anchor.
    commit_sha: str
    sha256: str


def read_manifest(path: Path) -> list[ManifestRow]:
    """The manifest's rows, or an empty list if the file does not exist.

    Raises:
        UserError: A directory is at `path`. The file is not UTF-8.
    """
    if not path.exists():
        return []
    hint = _restore_hint(path)
    try:
        return _rows(require_file(path, hint=hint), hint)
    except UnicodeDecodeError as exc:
        raise UserError(f"{path} is not UTF-8 text: {exc.reason}", hint=hint) from exc


def _restore_hint(path: Path) -> str:
    return f"restore {path}, or rewrite it with `crossvenue fetch --follow-branch`"


def _rows(path: Path, hint: str) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in MANIFEST_FIELDS if f not in (reader.fieldnames or ())]
        if missing:
            raise UserError(f"{path} has no column {', '.join(missing)}", hint=hint)
        rows: list[ManifestRow] = []
        for row in reader:
            where = formatting.location(path, reader.line_num)
            if any(row[f] is None for f in MANIFEST_FIELDS):
                raise UserError(f"{where}: the row has fewer cells than the header", hint=hint)
            venue_type = next((t for t in VENUE_TYPES if t == row["venue_type"]), None)
            if venue_type is None:
                raise UserError(
                    f"{where}: venue_type must be {' or '.join(VENUE_TYPES)}, "
                    f"got {row['venue_type']!r}",
                    hint=hint,
                )
            rows.append(
                ManifestRow(
                    venue=row["venue"],
                    venue_type=venue_type,
                    source_url=row["source_url"],
                    local_path=row["local_path"],
                    fetched_at=row["fetched_at"],
                    commit_sha=row["commit_sha"],
                    sha256=row["sha256"],
                )
            )
        return rows


def write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    """Writes the manifest, sorted, so it diffs cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r.venue, r.local_path))
    write_csv(path, MANIFEST_FIELDS, ([asdict(row)[f] for f in MANIFEST_FIELDS] for row in ordered))


def preserve_timestamps(
    new_rows: list[ManifestRow], old_rows: Sequence[ManifestRow]
) -> list[ManifestRow]:
    """Carries `fetched_at` over for documents whose content hash and commit did not change.

    This is what makes the manifest a diffable artifact: re-fetching an unchanged
    corpus produces an unchanged file, so a real diff always means real drift.
    """
    previous = {(row.venue, row.local_path): row for row in old_rows}
    for row in new_rows:
        old = previous.get((row.venue, row.local_path))
        if old and old.sha256 == row.sha256 and old.commit_sha == row.commit_sha:
            row.fetched_at = old.fetched_at
    return new_rows


class CorpusSnapshot(BaseModel):
    """The corpus an artifact was built against, as a dataset's `corpus_snapshot` records it.

    `index_version` and `chunks` are recorded from dataset v0.1.5 on and absent before.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_path: str
    manifest_sha256: str
    documents: int
    recorded_at: str
    index_version: str | None = None
    chunks: int | None = None


def fingerprint(manifest_path: str | Path) -> CorpusSnapshot:
    """The identity of the corpus an artifact was built against.

    The manifest hash is the right anchor: it changes when any source document
    changes, so a dataset recording it can later be checked for whether its
    `source_chunk_ids` still mean anything. `documents` counts the rows `read_manifest`
    reads.
    """
    path = Path(manifest_path)
    return CorpusSnapshot(
        manifest_path=str(path),
        manifest_sha256=file_sha256(path),
        documents=len(read_manifest(path)),
        recorded_at=utc_now(),
    )
