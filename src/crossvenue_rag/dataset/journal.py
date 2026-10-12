"""The judgment journal of a question file, as typed records.

`annotations/<name>.rounds.json` holds, per question, every judgment made of it: the numbered
labeling rounds, the adjudication of the pairs two rounds disagree on and a relabel under the
amended criterion. A record keeps the key order of the file it was read from, so a journal read
and written again is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from ..errors import UserError
from ..models import VenueCoverage

# The journal keys of the judgments made after round 2 under the criterion amended then
# (methodology section 5a). One adjudicates the pairs the two rounds disagree on; the other
# records a question judged again on its candidates.
ADJUDICATED = "adjudicated"
RELABELED = "relabeled"
# The round whose judgments are the gold the later passes start from.
FIRST_ROUND = 1


class JournalRecord(BaseModel):
    """A record of the journal: unknown keys refused, the file's key order kept on writing.

    An optional field is written only when it holds a value; a field without a default is
    always written, `None` included.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    _order: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="wrap")
    @classmethod
    def _remember_order(cls, data: Any, handler: Any) -> Any:
        record = handler(data)
        if isinstance(data, dict) and isinstance(record, JournalRecord):
            record._order = tuple(data)
        return record

    @model_serializer(mode="wrap")
    def _ordered(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped: dict[str, Any] = handler(self)
        fields = type(self).model_fields
        keys = [
            (field.alias or name)
            for name, field in fields.items()
            if field.is_required() or getattr(self, name) is not None
        ]
        ordered = [k for k in self._order if k in keys] + [k for k in keys if k not in self._order]
        return {key: dumped[key] for key in ordered}


class Correction(JournalRecord):
    """A change to a round-1 verdict made by review, or a value derived by rule."""

    at: str
    annotator: str | None = None
    derived: bool | None = None
    rule: str | None = None
    index_version: str
    previous_relevant: list[str]
    previous_venue_coverage: dict[str, VenueCoverage]
    relevant: list[str]
    venue_coverage: dict[str, VenueCoverage]


class CoverageChange(JournalRecord):
    """A venue_coverage value an augmentation changed."""

    venue: str
    old: VenueCoverage = Field(alias="from")
    new: VenueCoverage = Field(alias="to")
    previous_assessment_wrong: bool


class Augmentation(JournalRecord):
    """Chunks judged after round 1 for a question, with the ones marked relevant."""

    at: str
    annotator: str
    index_version: str
    source: str
    shown: list[str]
    relevant: list[str]
    coverage_changes: list[CoverageChange]


class Migration(JournalRecord):
    """A judgment's chunk-id lists before they were carried to another index.

    `corrections` and `augmentations` are round 1's as they stood; `migrated_from` is the
    migration before this one, so every pool the annotator was shown stays on record.
    """

    pool: list[str]
    relevant: list[str]
    index_version: str | None
    corrections: list[Correction] | None = None
    augmentations: list[Augmentation] | None = None
    migrated_from: Migration | None = None


class Judgment(JournalRecord):
    """One labeling round's judgment of a question: the pool shown and the chunks marked.

    Round 1 also carries its corrections, made by review or derived by rule, and its
    augmentations; a later round carries the seed its candidates were shuffled with. Every
    round carries its id migrations.
    """

    annotator: str | None = None
    annotated_at: str | None = None
    index_version: str | None = None
    pool: list[str] = []
    relevant: list[str] = []
    corrections: list[Correction] | None = None
    venue_coverage: dict[str, VenueCoverage] | None = None
    augmentations: list[Augmentation] | None = None
    shuffle_seed: int | None = None
    migrated_from: Migration | None = None

    def judged_ids(self) -> list[str]:
        """Every chunk this judgment covers, in display order, without repeats.

        That is the pool it displayed, then any chunks augmentation showed. A chunk in
        here was looked at and is judged-irrelevant unless it is in `relevant`; a chunk not
        in here is unjudged, which is not the same thing.
        """
        seen: dict[str, None] = dict.fromkeys(self.pool)
        for augmentation in self.augmentations or []:
            seen.update(dict.fromkeys(augmentation.shown))
        return list(seen)

    def first_pass_relevant(self) -> set[str]:
        """The verdicts before review.

        These are the `previous_relevant` of the earliest correction (`relevant` when the
        judgment was never corrected), plus what augmentation marked, which review never
        touched.
        """
        base = self.corrections[0].previous_relevant if self.corrections else self.relevant
        augmented = {c for a in self.augmentations or [] for c in a.relevant}
        return set(base) | augmented


class Decision(JournalRecord):
    """The adjudicated verdict on one chunk."""

    relevant: bool
    annotated_at: str


class Adjudication(JournalRecord):
    """The adjudication of the pairs on which round 1 and round 2 disagree, per chunk."""

    annotator: str
    criterion: str
    index_version: str
    decisions: dict[str, Decision] = {}
    annotated_at: str | None = None
    migrated_from: AdjudicationMigration | None = None


class AdjudicationMigration(JournalRecord):
    """An adjudication's decisions before their chunk ids were carried to another index."""

    decisions: dict[str, Decision]
    index_version: str
    migrated_from: AdjudicationMigration | None = None


class Relabel(JournalRecord):
    """A question judged again under the amended criterion, on its round-1 candidates and more."""

    annotator: str
    annotated_at: str
    criterion: str
    index_version: str
    pool: list[str]
    relevant: list[str]
    venue_coverage: dict[str, VenueCoverage] | None = None
    migrated_from: Migration | None = None


class QuestionJournal(BaseModel):
    """Every judgment of one question, under the keys the file uses.

    Numbered rounds are keyed `"1"`, `"2"`; the adjudication and the relabel under
    `ADJUDICATED` and `RELABELED`.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rounds: dict[int, Judgment] = {}
    adjudicated: Adjudication | None = None
    relabeled: Relabel | None = None

    @model_validator(mode="before")
    @classmethod
    def _split_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        rounds = {int(k): v for k, v in data.items() if isinstance(k, str) and k.isdigit()}
        rest = {k: v for k, v in data.items() if not (isinstance(k, str) and k.isdigit())}
        return {"rounds": rounds, **rest}

    @model_serializer(mode="plain")
    def _keyed(self) -> dict[str, Any]:
        out: dict[str, Any] = {str(n): j.model_dump(by_alias=True) for n, j in self.rounds.items()}
        if self.adjudicated is not None:
            out[ADJUDICATED] = self.adjudicated.model_dump(by_alias=True)
        if self.relabeled is not None:
            out[RELABELED] = self.relabeled.model_dump(by_alias=True)
        return out

    def judgment(self, number: int) -> Judgment | None:
        """The judgment of round `number`, None when the round did not judge the question."""
        return self.rounds.get(number)

    @property
    def first(self) -> Judgment | None:
        """Round 1's judgment."""
        return self.rounds.get(FIRST_ROUND)

    def entries(self) -> list[tuple[str, str | None]]:
        """(key, `index_version`) of every entry in key order: the rounds, then the later passes."""
        keyed: dict[str, str | None] = {str(n): j.index_version for n, j in self.rounds.items()}
        if self.adjudicated is not None:
            keyed[ADJUDICATED] = self.adjudicated.index_version
        if self.relabeled is not None:
            keyed[RELABELED] = self.relabeled.index_version
        return sorted(keyed.items())

    def resolved_relevant(self) -> list[str]:
        """The relevant chunks the entries resolve to: the question's gold.

        A relabel replaces round 1. Otherwise each adjudicated decision is applied to round
        1's corrected set: a chunk decided not relevant is dropped, and one decided relevant
        is appended in the order of the decisions.
        """
        if self.relabeled is not None:
            return list(self.relabeled.relevant)
        relevant = list(self.first.relevant) if self.first is not None else []
        decisions = self.adjudicated.decisions if self.adjudicated is not None else {}
        for chunk_id, decision in decisions.items():
            if decision.relevant and chunk_id not in relevant:
                relevant.append(chunk_id)
            elif not decision.relevant and chunk_id in relevant:
                relevant.remove(chunk_id)
        return relevant

    def judged_for_gold(self) -> set[str]:
        """Every chunk judged for the gold: round 1's, plus what a relabel showed."""
        first = set(self.first.judged_ids()) if self.first is not None else set()
        return first | set(self.relabeled.pool if self.relabeled is not None else [])


class Journal(BaseModel):
    """The judgment journal of a question file: question id to its judgments, in file order."""

    model_config = ConfigDict(extra="forbid")

    questions: dict[str, QuestionJournal] = {}

    @model_validator(mode="before")
    @classmethod
    def _wrap(cls, data: Any) -> Any:
        if isinstance(data, dict) and set(data) != {"questions"}:
            return {"questions": data}
        return data

    @model_serializer(mode="plain")
    def _plain(self) -> dict[str, Any]:
        return {q: entry.model_dump() for q, entry in self.questions.items()}

    @classmethod
    def from_json(cls, data: Any) -> Self:
        """The journal a parsed JSON value holds.

        Raises:
            ValueError: `data` is not an object of judgments by question id.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"holds a JSON {type(data).__name__} where an object of rounds by question id is "
                "expected"
            )
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(_first_error(exc)) from exc

    def get(self, question_id: str) -> QuestionJournal:
        """The judgments of a question, empty when the journal has none."""
        return self.questions.get(question_id) or QuestionJournal()

    def entry(self, question_id: str) -> QuestionJournal:
        """The judgments of a question, created empty when the journal has none."""
        return self.questions.setdefault(question_id, QuestionJournal())

    def first(self, question_id: str) -> Judgment | None:
        """Round 1's judgment of a question, None when round 1 did not judge it."""
        return self.get(question_id).first

    def require_first(self, question_id: str) -> Judgment:
        """Round 1's judgment of a question a caller has checked round 1 judged.

        Raises:
            ValueError: Round 1 did not judge the question.
        """
        first = self.first(question_id)
        if first is None:
            raise ValueError(f"{question_id} has no round-1 judgment")
        return first

    def to_text(self) -> str:
        """The journal as its file holds it."""
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False)


def _first_error(exc: ValidationError) -> str:
    """The first validation problem, located by its path in the journal."""
    error = exc.errors()[0]
    where = ".".join(str(p) for p in error["loc"] if p != "questions")
    return f"{where}: {error['msg']}" if where else str(error["msg"])


def rounds_path(questions_path: str | Path) -> Path:
    """Sidecar holding the judgment journal of a question file."""
    path = Path(questions_path)
    return path.parent / "annotations" / (path.name.removesuffix(".jsonl") + ".rounds.json")


def read_journal(path: Path) -> Journal:
    """The journal in the file at `path`.

    Raises:
        UserError: The file cannot be read, is not UTF-8 JSON, or is not a journal; the message
            names the problem.
    """
    try:
        return Journal.from_json(json.loads(path.read_text(encoding="utf-8")))
    # OSError: unreadable file. ValueError: invalid JSON or UTF-8, or not a journal.
    except (OSError, ValueError) as exc:
        raise UserError(
            f"{path}: {exc}", hint="restore the journal from its last good copy"
        ) from exc


def load_rounds(questions_path: str | Path) -> Journal:
    """The judgment journal of a question file, empty when it has none."""
    path = rounds_path(questions_path)
    return read_journal(path) if path.exists() else Journal()


def save_rounds(questions_path: str | Path, journal: Journal) -> None:
    """Writes the judgment journal of a question file to its sidecar."""
    path = rounds_path(questions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(journal.to_text(), encoding="utf-8")


def load_judged_sets(questions_path: str | Path) -> dict[str, set[str]] | None:
    """Question id -> every chunk judged for its gold, from the file's journal.

    That is what round 1 judged, plus what a relabel showed. A frozen dataset's journal is
    the copy taken at freeze time, `evals/datasets/annotations/<name>.rounds.json`. None
    when there is none, so a caller can tell "no history" from "nothing judged".
    """
    if not rounds_path(questions_path).exists():
        return None
    journal = load_rounds(questions_path)
    return {
        qid: entry.judged_for_gold()
        for qid, entry in journal.questions.items()
        if entry.first is not None
    }
