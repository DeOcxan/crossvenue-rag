"""Carries chunk-id pointers from one index to another, verifying that nothing else changes.

A dataset, a question file and the judgment journal point into the corpus by
chunk id. An index change that renames ids without changing text is therefore
carried across mechanically: `build_id_map`, then `remap_jsonl` for a JSONL file;
`dataset.remap` carries the judgment journal.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from .. import formatting, jsonl, listing
from ..config import Config
from ..corpus.artifacts import CHUNKS_FILENAME, load_chunks, read_chunk_keys
from ..errors import UserError, require_file
from ..index_version import index_meta_path, require_built_index_version
from ..models import Chunk, make_chunk_id
from ..run_files import file_sha256, read_json_model


class IdMapFile(BaseModel):
    """The map file `IdMap.save` writes; its `old_ids` and `new_ids` counts are not read."""

    map: dict[str, str]
    old_chunks_sha256: str
    to_index_version: str


@dataclass(slots=True)
class IdMap:
    """Each chunk id of the earlier index to its rebuilt id, with both sides' provenance."""

    mapping: dict[str, str]
    # Provenance of the two sides, recorded in the map file.
    old_chunks_sha256: str
    to_index_version: str

    @property
    def canonical_ids(self) -> int:
        """The distinct rebuilt ids the old ids map onto."""
        return len(set(self.mapping.values()))

    def save(self, path: str | Path) -> None:
        """Writes the map, its provenance and its id counts to `path` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "old_chunks_sha256": self.old_chunks_sha256,
                    "to_index_version": self.to_index_version,
                    "old_ids": len(self.mapping),
                    "new_ids": self.canonical_ids,
                    "map": self.mapping,
                },
                indent=1,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> IdMap:
        """The map saved at `path` by `save`."""
        saved = read_json_model(
            Path(path), IdMapFile, hint="pass --map a map written by `crossvenue id-map`"
        )
        return cls(saved.map, saved.old_chunks_sha256, saved.to_index_version)


def build_id_map(
    old_chunks_path: str | Path, new_chunks: Sequence[Chunk], to_index_version: str
) -> tuple[IdMap, list[str]]:
    """Each chunk id of `old_chunks_path` mapped onto the same venue and text in `new_chunks`.

    Several old ids map to one id exactly when the old chunks were
    byte-identical copies. Returns the map and the old ids whose text has no
    counterpart in `new_chunks`; a non-empty list means the index changed more
    than ids, and the migration is not mechanical.
    """
    new_ids = {c.chunk_id for c in new_chunks}
    mapping: dict[str, str] = {}
    orphans: list[str] = []
    olds = read_chunk_keys(
        old_chunks_path, hint=f"pass the {CHUNKS_FILENAME} of the index before the rebuild"
    )
    for old in olds:
        new_id = make_chunk_id(old.venue, old.text)
        if new_id in new_ids:
            mapping[old.chunk_id] = new_id
        else:
            orphans.append(old.chunk_id)
    return IdMap(mapping, file_sha256(Path(old_chunks_path)), to_index_version), orphans


def write_id_map(cfg: Config, old_chunks_path: Path, out: Path) -> IdMap:
    """The map of `old_chunks_path`'s chunk ids onto the built index's chunks, written to `out`.

    Raises:
        UserError: An old chunk has no identical text in the current corpus.
    """
    built = require_built_index_version(index_meta_path(cfg))
    mapping, orphans = build_id_map(old_chunks_path, load_chunks(cfg), built)
    if orphans:
        raise UserError(
            f"{formatting.plural(len(orphans), 'old chunk')} without identical text in the "
            f"current corpus, for example {listing.first_items(orphans)}",
            hint="the rebuild changed text as well as ids; pointers into it need re-judging",
        )
    mapping.save(out)
    return mapping


def map_ids(ids: list[str], mapping: dict[str, str], unmapped: set[str]) -> list[str]:
    """Maps a list of ids, collapsing duplicates the map creates, keeping order."""
    out: list[str] = []
    for chunk_id in ids:
        new_id = mapping.get(chunk_id)
        if new_id is None:
            unmapped.add(chunk_id)
            new_id = chunk_id
        if new_id not in out:
            out.append(new_id)
    return out


def require_known_ids(only: set[str] | None, ids: Collection[str], path: str | Path) -> None:
    """Refuses an `only` id that is not among `ids`, the record ids of the file at `path`.

    Raises:
        UserError: An `only` id names no record of `path`.
    """
    absent = sorted(only - set(ids)) if only is not None else []
    if absent:
        raise UserError(
            f"no record of {path} has the id {', '.join(absent)}",
            hint="pass --only ids of records in the file",
        )


@dataclass(slots=True)
class RemapReport:
    """What a remap changed: record and id counts, and the ids the map lacks."""

    records: int = 0
    records_changed: int = 0
    ids_before: int = 0
    ids_after: int = 0
    unmapped: set[str] = field(default_factory=set)


def _id_lists(record: dict[str, object], fields: Sequence[str], where: str) -> dict[str, list[str]]:
    """The chunk ids in each field of `fields`; refuses a missing field or one holding others."""
    missing = [name for name in fields if name not in record]
    if missing:
        raise UserError(
            f"{where}: record lacks {formatting.plural(len(missing), 'field')} "
            f"to remap: {', '.join(missing)}",
            hint="pass --field with the id-list fields every record has",
        )
    lists: dict[str, list[str]] = {}
    not_ids: list[str] = []
    for name in fields:
        value = record[name]
        if isinstance(value, list) and all(isinstance(i, str) for i in value):
            lists[name] = [str(i) for i in value]
        else:
            not_ids.append(name)
    if not_ids:
        raise UserError(
            f"{where}: {', '.join(not_ids)} is not a list of chunk ids",
            hint="pass --field only for fields that hold chunk ids",
        )
    return lists


def remap_jsonl(
    source: str | Path,
    target: str | Path,
    id_map: IdMap,
    fields: Sequence[str],
    only: set[str] | None,
) -> RemapReport:
    """Writes `target` = `source` with `fields` mapped through `id_map`.

    `only` restricts the check of `fields` and the rewrite to records with those
    ids; every other record is copied verbatim, with its `fields` unchecked.
    Every field outside `fields` is verified unchanged, and
    a line that does not round-trip through `json.dumps` is refused rather than
    re-serialized. Ids missing from the map are left as they are and reported in
    `RemapReport.unmapped`: they were dangling before the migration, and the
    migration gives them no target. Every check runs before writing.

    Raises:
        UserError: Nothing is at `source`, or a directory is. The source is not UTF-8, or `only`
            names an id no record has.
        ValueError: A field outside `fields` changed, which would be a defect here.
    """
    fields = list(dict.fromkeys(fields))  # a field named twice is rewritten and counted once
    report = RemapReport()
    lines_out: list[str] = []
    seen: set[str] = set()
    hint = "pass the dataset to remap"
    try:
        original = require_file(source, hint=hint).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UserError(f"{source} is not UTF-8 text: {exc.reason}", hint=hint) from exc
    for number, line in enumerate(original.splitlines(), start=1):
        if not line.strip():
            lines_out.append(line)
            continue
        record = _record(line, source, number)
        seen.add(str(record.get("id")))
        report.records += 1
        if only is not None and record.get("id") not in only:
            lines_out.append(line)
            continue
        id_lists = _id_lists(record, fields, formatting.location(source, number))
        updated = dict(record)
        for name in fields:
            after = map_ids(id_lists[name], id_map.mapping, report.unmapped)
            report.ids_before += len(id_lists[name])
            report.ids_after += len(after)
            updated[name] = after
        new_line = json.dumps(updated, ensure_ascii=False)

        # The check is on the serialized line rather than the dict it came from:
        # parsed back, it has the same keys in the same order, and every field
        # outside `fields` equals the original's.
        reparsed = json.loads(new_line)
        if list(reparsed) != list(record) or any(
            reparsed[key] != record[key] for key in record if key not in fields
        ):
            raise ValueError(
                f"{formatting.location(source, number)}: fields other than "
                f"{listing.prose(list(fields))} changed"
            )
        if new_line != line:
            report.records_changed += 1
        lines_out.append(new_line)

    require_known_ids(only, seen, source)
    trailing = "\n" if original.endswith("\n") else ""
    Path(target).write_text("\n".join(lines_out) + trailing, encoding="utf-8")
    return report


def _record(line: str, source: str | Path, number: int) -> dict[str, object]:
    """The JSON object on line `number` (`jsonl.json_object`), which must round-trip."""
    record: dict[str, object] = jsonl.json_object(line, source, number)
    if json.dumps(record, ensure_ascii=False) != line:
        raise UserError(
            f"{formatting.location(source, number)} does not round-trip through json.dumps; "
            "refusing to re-serialize it, since that could change bytes outside the id fields",
            hint="remap a file written by crossvenue: one JSON object per line",
        )
    return record
