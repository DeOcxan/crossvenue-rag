"""A versioned dataset under `evals/datasets/` as typed records: rows, metadata, versions.

A version is an immutable snapshot of the question file (`v0.1-candidates`, `v0.2-gold`)
with a `.meta.json` recording the corpus it was built against and how.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from .. import formatting, jsonl
from ..corpus.manifest import CorpusSnapshot
from ..corpus.manifest import fingerprint as manifest_fingerprint
from ..errors import UserError
from ..models import (
    ADVERSARIAL,
    COMPARISON,
    MULTI_VENUE,
    Difficulty,
    ExpectedBehavior,
    Question,
    QuestionCategory,
    QuestionTier,
    QuestionType,
    VenueCoverage,
    effective_category,
)

SCHEMA_VERSION = "1.0"

DATASETS_DIR = Path("evals/datasets")
# The shortest question text a dataset accepts: shorter than this is no question.
MIN_QUESTION_CHARS = 8

# `manifest_fingerprint` is re-exported for the package (`dataset/__init__.py`): dataset
# `.meta.json` files and several CHANGELOG entries name it as
# `crossvenue_rag.dataset.manifest_fingerprint`.
__all__ = [
    "DATASETS_DIR",
    "SCHEMA_VERSION",
    "DatasetCounts",
    "DatasetGenerator",
    "DatasetMeta",
    "DatasetSource",
    "QuestionRecord",
    "as_question",
    "load_records",
    "manifest_fingerprint",
]


class QuestionRecord(BaseModel):
    """One dataset row: `models.Question` with the full schema enforced.

    `models.Question` is permissive so that a five-field hand-written question
    loads; a dataset file is a published artifact and gets every rule.
    """

    model_config = {"extra": "forbid"}

    id: str = Field(min_length=1)
    question: str = Field(min_length=MIN_QUESTION_CHARS)
    tier: QuestionTier
    venues: list[str]
    qtype: QuestionType
    category: QuestionCategory
    difficulty: Difficulty
    is_answerable: bool
    expected_behavior: ExpectedBehavior
    adversarial_type: str | None = None

    source_chunk_ids: list[str] = []
    gold_chunk_ids: list[str] = []
    reference_answer: str = ""

    annotator: str = ""
    annotated_at: str = ""
    annotation_round: int = 0
    venue_coverage: dict[str, VenueCoverage] = {}

    created_at: str = ""
    generator: str = ""
    prompt_hash: str = ""
    notes: str = ""

    @field_validator("question")
    @classmethod
    def _looks_like_a_question(cls, v: str) -> str:
        if v != v.strip():
            raise ValueError("question has leading or trailing whitespace")
        return v

    @property
    def is_comparison(self) -> bool:
        """A comparison question by its effective category (`models.effective_category`)."""
        return effective_category(self.category, self.qtype) == COMPARISON

    @model_validator(mode="after")
    def _answerability_is_consistent(self) -> QuestionRecord:
        """The three answerability fields have to agree.

        An unanswerable question with gold chunks, or a `refuse` expectation on
        an answerable one, is the kind of contradiction that silently turns a
        refusal benchmark into noise.
        """
        if self.category == "unanswerable" and self.is_answerable:
            raise ValueError(f"{self.id}: category=unanswerable but is_answerable=true")
        if not self.is_answerable:
            if self.expected_behavior != "refuse":
                raise ValueError(
                    f"{self.id}: is_answerable=false requires expected_behavior=refuse, "
                    f"got {self.expected_behavior!r}"
                )
            if self.gold_chunk_ids:
                raise ValueError(f"{self.id}: unanswerable question carries gold chunks")
        if self.category == "ambiguous" and self.expected_behavior == "answer":
            raise ValueError(
                f"{self.id}: an ambiguous question expects ask_clarification or refuse, "
                "and this one expects answer"
            )
        if self.category == ADVERSARIAL and not self.adversarial_type:
            raise ValueError(f"{self.id}: adversarial questions must name an adversarial_type")
        if self.is_comparison and len(self.venues) < MULTI_VENUE:
            raise ValueError(
                f"{self.id}: a comparison question needs at least "
                f"{formatting.count(MULTI_VENUE)} venues"
            )
        return self


def as_question(record: QuestionRecord) -> Question:
    """A dataset record as the harness's `Question`, with every field `Question` has."""
    return Question.from_dict(record.model_dump())


class DatasetSource(BaseModel):
    """One venue a dataset's questions were written from, as `data/sources.yaml` lists it."""

    model_config = {"extra": "forbid"}

    venue: str
    venue_type: str | None = None
    license: str | None = None
    homepage: str | None = None


class DatasetGenerator(BaseModel):
    """The model and prompt that wrote a dataset's candidates; empty for hand-made questions."""

    model_config = {"extra": "forbid"}

    model: str | None = None
    prompt_hash: str | None = None


class DatasetCounts(BaseModel):
    """The totals and breakdowns a dataset's meta records; a count not recorded is None.

    `validate-evals` recomputes each declared count from the dataset file. The set is
    open: a gold version adds counts of its labeling pass (`judged`, `annotation_round`),
    kept as written and not checked.
    """

    model_config = {"extra": "allow"}

    total: int | None = None
    by_category: dict[str, int] | None = None
    by_qtype: dict[str, int] | None = None
    by_venue: dict[str, int] | None = None
    answerable: int | None = None
    unanswerable: int | None = None
    with_gold: int | None = None
    gold_chunks: int | None = None


class DatasetMeta(BaseModel):
    """The header that makes a dataset file reproducible, in one of two formats.

    The hashed format records the sha256 of the dataset file (`dataset_sha256`) and of its
    journal (`journal_sha256`), and `validate-evals` checks both. The earlier format, of the
    versions frozen before metas recorded file hashes, records neither; the hashes of its
    gold versions are in datasets/CHANGELOG.md.
    """

    model_config = {"extra": "forbid"}

    dataset_version: str
    schema_version: str = SCHEMA_VERSION
    created_at: str
    corpus_snapshot: CorpusSnapshot
    sources: list[DatasetSource]
    generator: DatasetGenerator
    counts: DatasetCounts
    notes: str = ""
    dataset_sha256: str | None = None
    journal_sha256: str | None = None

    @property
    def hashed(self) -> bool:
        """Whether the meta is of the hashed format: it records a file hash."""
        return self.dataset_sha256 is not None or self.journal_sha256 is not None


def load_records(path: str | Path) -> list[QuestionRecord]:
    """The validated rows of a dataset file (`jsonl.read_models`)."""
    return jsonl.read_models(path, QuestionRecord)


def version_key(path: Path) -> tuple[int, ...]:
    """Sorts `v0.1.2` before `v0.1.10`, which a plain string sort gets backward."""
    stem = path.name.removesuffix("-candidates.jsonl").lstrip("v")
    return tuple(int(part) for part in re.findall(r"\d+", stem))


def newest_candidate_pool(datasets_dir: Path) -> Path:
    """The highest-versioned candidate file, so selection follows the supplements.

    Raises:
        UserError: `datasets_dir` holds no candidate file.
    """
    pools = sorted(datasets_dir.glob("*-candidates.jsonl"), key=version_key)
    if not pools:
        raise UserError(
            f"no candidate pool under {datasets_dir}",
            hint="generate one, or pass --dataset explicitly",
        )
    return pools[-1]
