"""Carrying a question file's judgment journal, or a dataset, through an id map.

`remap-ids` remaps a JSONL dataset into another version (`index.migrate.remap_jsonl`) and a
judgment journal in place, here, since the journal model is of the dataset layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .. import formatting, listing
from ..errors import UserError
from ..index.migrate import IdMap, RemapReport, map_ids, remap_jsonl, require_known_ids
from .journal import (
    Adjudication,
    AdjudicationMigration,
    Decision,
    Judgment,
    Migration,
    Relabel,
    read_journal,
)


def remap_rounds(
    path: str | Path, id_map: IdMap, index_version: str, only: set[str] | None = None
) -> RemapReport:
    """Carries every chunk id of the judgment journal across, in place.

    A round's pool and verdict, round 1's corrections and augmentations, the adjudicated
    decisions and the relabel's pool and verdict are all mapped. Each migrated entry is
    stamped with `index_version` and keeps what it held before under `migrated_from`, which
    keeps the migration before it. The pool the annotator was shown (with its duplicate
    copies) so stays on record through every migration. Refuses, before writing, an `only` id
    that names no question of the journal, any id not in the map, and a map that merges two
    chunks the adjudication decided differently.

    Raises:
        UserError: An `only` id names no question of the journal, an id is not in the map, or
            two merged chunks carry different adjudicated verdicts.
    """
    carry = _Carry(id_map.mapping, index_version)
    journal = read_journal(Path(path))
    require_known_ids(only, journal.questions.keys(), path)
    conflicts: list[str] = []
    for question_id, entries in journal.questions.items():
        if only is not None and question_id not in only:
            continue
        for entry in entries.rounds.values():
            carry.judgment(entry)
        if entries.adjudicated is not None:
            merged = carry.adjudication(entries.adjudicated)
            conflicts.extend(f"{question_id} {chunk_id}" for chunk_id in merged)
        if entries.relabeled is not None:
            carry.relabel(entries.relabeled)
    report = carry.report
    if report.unmapped:
        # A judgment stamped with `index_version` must point only into that
        # index; nothing is written.
        raise UserError(
            f"{formatting.plural(len(report.unmapped), 'id')} in {path} are not in the map: "
            f"{listing.first_items(sorted(report.unmapped))}",
            hint="rebuild the map with `crossvenue id-map`, or re-judge the questions that "
            "hold them with `crossvenue label --all`; the journal was not changed",
        )
    if conflicts:
        raise UserError(
            f"the map merges chunks the adjudication decided differently: "
            f"{listing.first_items(conflicts)}",
            hint="adjudicate the question again after the index change; the journal was not "
            "changed",
        )
    Path(path).write_text(journal.to_text(), encoding="utf-8")
    return report


@dataclass
class _Carry:
    """Maps the chunk ids of journal entries onto one index, counting into `report`."""

    mapping: dict[str, str]
    index_version: str
    report: RemapReport = field(default_factory=RemapReport)

    def ids(self, ids: list[str]) -> list[str]:
        """`ids` mapped, duplicates the map creates collapsed; a missing id is reported."""
        self.report.ids_before += len(ids)
        out = map_ids(ids, self.mapping, self.report.unmapped)
        self.report.ids_after += len(out)
        return out

    def _counted(self) -> None:
        self.report.records += 1
        self.report.records_changed += 1

    def judgment(self, entry: Judgment) -> None:
        """A round's lists, and round 1's corrections and augmentations, mapped and stamped."""
        entry.migrated_from = Migration(
            pool=entry.pool,
            relevant=entry.relevant,
            index_version=entry.index_version,
            corrections=deepcopy(entry.corrections),
            augmentations=deepcopy(entry.augmentations),
            migrated_from=entry.migrated_from,
        )
        entry.pool = self.ids(entry.pool)
        entry.relevant = self.ids(entry.relevant)
        for correction in entry.corrections or []:
            correction.previous_relevant = self.ids(correction.previous_relevant)
            correction.relevant = self.ids(correction.relevant)
            correction.index_version = self.index_version
        for augmentation in entry.augmentations or []:
            augmentation.shown = self.ids(augmentation.shown)
            augmentation.relevant = self.ids(augmentation.relevant)
            augmentation.index_version = self.index_version
        entry.index_version = self.index_version
        self._counted()

    def relabel(self, relabel: Relabel) -> None:
        """A relabel's pool and verdict, mapped and stamped."""
        relabel.migrated_from = Migration(
            pool=relabel.pool,
            relevant=relabel.relevant,
            index_version=relabel.index_version,
            migrated_from=relabel.migrated_from,
        )
        relabel.pool = self.ids(relabel.pool)
        relabel.relevant = self.ids(relabel.relevant)
        relabel.index_version = self.index_version
        self._counted()

    def adjudication(self, adjudication: Adjudication) -> list[str]:
        """The decisions re-keyed by mapped chunk, and stamped.

        Returns the mapped chunks onto which the map merges two decisions that differ.
        """
        adjudication.migrated_from = AdjudicationMigration(
            decisions=deepcopy(adjudication.decisions),
            index_version=adjudication.index_version,
            migrated_from=adjudication.migrated_from,
        )
        decisions: dict[str, Decision] = {}
        conflicts: list[str] = []
        for chunk_id, decision in adjudication.decisions.items():
            new_id = map_ids([chunk_id], self.mapping, self.report.unmapped)[0]
            held = decisions.setdefault(new_id, decision)
            if held.relevant != decision.relevant and new_id not in conflicts:
                conflicts.append(new_id)
        self.report.ids_before += len(adjudication.decisions)
        self.report.ids_after += len(decisions)
        adjudication.decisions = decisions
        adjudication.index_version = self.index_version
        self._counted()
        return conflicts


DEFAULT_REMAP_FIELDS = ("source_chunk_ids", "gold_chunk_ids")


@dataclass(frozen=True)
class RemapRequest:
    """The `remap-ids` options: the map, the file to remap and where the result goes.

    A `.rounds.json` journal is remapped in place and takes neither `target` nor
    `fields`; a JSONL dataset is written to `target`, a path that must not exist,
    rewriting `fields` (None: `DEFAULT_REMAP_FIELDS`). `only` limits both to some record ids.
    """

    map_path: Path
    source: Path
    target: Path | None
    fields: Sequence[str] | None
    only: Sequence[str]


def remap_file(request: RemapRequest, current_index: str) -> tuple[RemapReport, Path]:
    """The report of chunk-id pointers carried through an id map, and the file written.

    Refuses a map built for another index than `current_index`, and a target or
    fields given for a journal, which is rewritten in place and always in every
    chunk-id list it holds. Refuses a dataset target that is missing or
    already exists too: datasets are immutable.

    Raises:
        UserError: The map targets another index than `current_index`. A journal gets a
            target or fields. A dataset's target is missing, exists, or lies in a missing
            directory. An `only` id names no record of the source.
    """
    mapping = IdMap.load(request.map_path)
    if mapping.to_index_version != current_index:
        raise UserError(
            f"the map targets index {mapping.to_index_version}, the built index is {current_index}",
            hint="rebuild the map with `crossvenue id-map` against the current index",
        )
    only = set(request.only) if request.only else None
    if request.source.name.endswith(".rounds.json"):
        _require_journal_options(request)
        return remap_rounds(request.source, mapping, current_index, only), request.source
    target = request.target
    if target is None:
        raise UserError(
            "a JSONL remap needs a target path",
            hint="pass the target, a path that does not exist yet, after the source: datasets "
            "are immutable, so the remap writes another version",
        )
    if target.exists():
        raise UserError(f"{target} exists", hint="datasets are immutable; write another version")
    if not target.parent.is_dir():
        raise UserError(
            f"{target.parent}, the directory of the target, does not exist",
            hint="pass a target in an existing directory",
        )
    fields = list(DEFAULT_REMAP_FIELDS if request.fields is None else request.fields)
    return remap_jsonl(request.source, target, mapping, fields, only), target


def _require_journal_options(request: RemapRequest) -> None:
    """Refuses a target or fields for a journal, which would otherwise be ignored."""
    if request.target is not None:
        raise UserError(
            f"{request.source} is a judgment journal, which is remapped in place; "
            f"the target {request.target} would not be written",
            hint="leave out the target to remap the journal in place",
        )
    if request.fields is not None:
        raise UserError(
            f"{request.source} is a judgment journal, whose every chunk-id list is "
            f"always remapped; --field {', '.join(request.fields)} would be ignored",
            hint="leave out --field for a journal",
        )
