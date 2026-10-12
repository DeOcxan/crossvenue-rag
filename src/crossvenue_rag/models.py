"""Data carried between pipeline stages.

Everything here is a plain dataclass rather than a pydantic model, so an object the program
builds is not validated: these objects cross function boundaries thousands of times per run.
A chunk or question record read from a file, which may be written by hand, is checked against
its dataclass by its loader, through `wrongly_typed_fields`, before it is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, get_args, get_type_hints

from pydantic import TypeAdapter, ValidationError

from .hashes import text_sha256

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

VenueType = Literal["cex", "dex"]
VENUE_TYPES: tuple[VenueType, ...] = get_args(VenueType)


@dataclass(slots=True)
class Document:
    """One normalized documentation page."""

    venue: str
    venue_type: VenueType
    doc_path: str  # path relative to the venue's raw directory
    source_url: str
    text: str


# Bumped whenever the shape of a chunk record changes. Part of `index_version`,
# so labels made against one shape are never scored against another.
CHUNK_SCHEMA_VERSION = "2"
# The fewest venues a multi-venue question names: a comparison needs two.
MULTI_VENUE = 2
# Hex digits of a sha256 an id keeps (a chunk id, an index version, a prompt hash): 64 bits,
# so two different inputs of this corpus never share an id in practice.
ID_HASH_DIGITS = 16

# Structural fields derived from a document's path and headings by the rules in
# data/sources.yaml. `None` means that no rule fired, and says nothing about the document.
MetadataField = Literal["environment", "api_version", "product", "doc_type"]
METADATA_FIELDS: tuple[MetadataField, ...] = get_args(MetadataField)


@dataclass(slots=True)
class ChunkSource:
    """One place a chunk's text occurs in the corpus.

    A canonical chunk has one of these per verbatim occurrence. The structural
    metadata is per occurrence, not per chunk, because the same text can be on a
    mainnet page and on its testnet copy and is equally true of both.
    """

    venue: str
    doc_path: str
    heading_path: list[str]
    source_url: str
    position: int  # index of the chunk within that document
    environment: str | None = None
    api_version: str | None = None
    product: str | None = None
    doc_type: str | None = None


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text plus everything needed to cite it.

    The citation fields (`venue` ... `position`) and the metadata fields describe
    the *primary* occurrence, `sources[0]`; `sources` lists every occurrence.
    `corpus.dedup.primary_key` chooses the primary.
    """

    chunk_id: str
    venue: str
    venue_type: VenueType
    doc_path: str
    heading_path: list[str]
    source_url: str
    text: str
    token_count: int
    position: int  # index of the primary occurrence within its document
    sources: list[ChunkSource] = field(default_factory=list)
    environment: str | None = None
    api_version: str | None = None
    product: str | None = None
    doc_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The record as written to `chunks.jsonl`."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """A chunk from its `chunks.jsonl` record.

        The record, and each of its `sources`, holds every field without a default and
        no other; `corpus.artifacts.read_chunks` refuses any other record before this.
        """
        data = dict(data)
        data["sources"] = [ChunkSource(**s) for s in data.get("sources", [])]
        return cls(**data)

    @property
    def heading_display(self) -> str:
        """The heading path as a breadcrumb (`format_heading_path`)."""
        return format_heading_path(self.heading_path)

    def metadata_values(self, name: str) -> set[str | None]:
        """Every value metadata field `name` takes across this chunk's occurrences."""
        if not self.sources:
            return {getattr(self, name)}
        return {getattr(source, name) for source in self.sources}


def format_heading_path(heading_path: list[str]) -> str:
    """Human-readable breadcrumb, for the retrieval and labeling tables."""
    return " > ".join(heading_path) if heading_path else "(document root)"


def make_chunk_id(venue: str, text: str) -> str:
    """Stable id for a chunk, derived from its content.

    Byte-identical text within a venue gets one id wherever it occurs (the id is
    what deduplication collapses on), so the id cannot depend on position. The
    venue is hashed in because identical text on two *different* venues is never
    collapsed: which venue says something is exactly what this corpus measures.
    Re-running ingestion over an unchanged corpus reproduces the same ids.
    """
    payload = "\x1f".join([venue, text])
    return text_sha256(payload)[:ID_HASH_DIGITS]


@dataclass(slots=True)
class Result:
    """One retrieved chunk with the score that put it there."""

    chunk_id: str
    score: float
    text: str
    venue: str
    venue_type: str
    doc_path: str
    heading_path: list[str]
    source_url: str
    # Rank each contributing method gave it, kept for debugging fusion.
    component_ranks: dict[str, int] = field(default_factory=dict)
    # Every occurrence of this text; more than one means it was deduplicated.
    sources: list[ChunkSource] = field(kw_only=True)

    @property
    def heading_display(self) -> str:
        """The heading path as a breadcrumb (`format_heading_path`)."""
        return format_heading_path(self.heading_path)

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> Result:
        """A result for `chunk` with `score` and no component ranks."""
        return cls(
            chunk_id=chunk.chunk_id,
            score=score,
            text=chunk.text,
            venue=chunk.venue,
            venue_type=chunk.venue_type,
            doc_path=chunk.doc_path,
            heading_path=list(chunk.heading_path),
            source_url=chunk.source_url,
            sources=list(chunk.sources),
        )


# Timings are in milliseconds, and a clock reads seconds.
MS_PER_SECOND = 1000


@dataclass(slots=True)
class Timings:
    """Wall-clock milliseconds per retrieval step.

    Passed into `Retriever.retrieve` by callers that record timings (the eval harness), so
    the retrieval signature stays `-> list[Result]` and the retriever keeps no
    timing state between calls.
    """

    encode_ms: float = 0.0
    search_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        """The sum of the four steps."""
        return self.encode_ms + self.search_ms + self.fuse_ms + self.rerank_ms

    def as_dict(self) -> dict[str, float]:
        """The four steps and `total_ms`, keyed by field name."""
        return {**asdict(self), "total_ms": self.total_ms}


# The retrieval steps `Timings` times, in pipeline order: its field names without `_ms`.
RETRIEVAL_STEPS = tuple(f.name.removesuffix("_ms") for f in fields(Timings))


QuestionCategory = Literal[
    "fact",
    "comparison",
    "multihop",
    "unanswerable",
    "ambiguous",
    "adversarial",
    "version_sensitive",
    "noisy",
]
ExpectedBehavior = Literal["answer", "refuse", "ask_clarification"]
QuestionTier = Literal["gold", "silver", "bronze"]
QuestionType = Literal["fact", "comparison", "multihop"]
Difficulty = Literal["easy", "medium", "hard"]

# How a comparison question's gold set stands for each venue it names.
# `covered`: a marked chunk from that venue. `absent_in_docs`: the annotator
# confirmed the venue's documentation does not answer it. `missing_from_pool`:
# the answer may exist, but no suitable candidate from that venue was shown.
VenueCoverage = Literal["covered", "absent_in_docs", "missing_from_pool"]

# The category, and the qtype, of a comparison question.
COMPARISON = "comparison"
# The category of an adversarial question.
ADVERSARIAL = "adversarial"


def effective_category(category: str | None, qtype: str) -> str:
    """A question's category, or its qtype when it was saved without one.

    The one fallback every breakdown, comparison scope and answer record uses; a
    dataset record always has a category.
    """
    return category or qtype


@dataclass(slots=True)
class Question:
    """One eval question. `gold_chunk_ids` is empty until a human labels it.

    The first five fields are what the retrieval harness needs. The rest is
    dataset bookkeeping (provenance, difficulty, expected behavior, annotation
    round), carried on the same record so a question and the reason it exists
    never drift into separate files. All of it is optional, so a hand-written
    question with only the first five fields loads too.
    """

    id: str
    question: str
    tier: QuestionTier
    venues: list[str]
    qtype: QuestionType
    gold_chunk_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""
    notes: str = ""

    category: QuestionCategory | None = None
    difficulty: Difficulty | None = None
    is_answerable: bool = True
    expected_behavior: ExpectedBehavior = "answer"
    adversarial_type: str | None = None
    # Chunks the question was written from. A provenance hint for the annotator
    # and a check on pool completeness; it is never a gold label.
    source_chunk_ids: list[str] = field(default_factory=list)

    annotator: str = ""
    annotated_at: str = ""
    annotation_round: int = 0
    # Comparison questions only: venue -> VenueCoverage, set when labeling.
    venue_coverage: dict[str, VenueCoverage] = field(default_factory=dict)

    created_at: str = ""
    generator: str = ""
    prompt_hash: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        """A question from its record; fields the class does not define are dropped."""
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        """The record as written to a question file."""
        data = asdict(self)
        # Written only when non-empty, so records that never needed it keep
        # their exact bytes.
        if not data["venue_coverage"]:
            del data["venue_coverage"]
        return data

    @property
    def effective_category(self) -> str:
        """The category, or the qtype for a question saved without one (`effective_category`)."""
        return effective_category(self.category, self.qtype)

    @property
    def is_comparison(self) -> bool:
        """A comparison question by its effective category, as metrics and documents count it."""
        return self.effective_category == COMPARISON

    @property
    def is_labelled(self) -> bool:
        """The question has gold chunks; a question judged to have none is not labeled."""
        return bool(self.gold_chunk_ids)

    @property
    def is_multi_venue(self) -> bool:
        """The question names at least `MULTI_VENUE` distinct venues."""
        return len(set(self.venues)) >= MULTI_VENUE

    @property
    def expects_refusal(self) -> bool:
        """The expected behavior is a refusal."""
        return self.expected_behavior == "refuse"

    @property
    def is_judged(self) -> bool:
        """Round 1 recorded a verdict: gold chunks, or an explicit empty judgment."""
        return bool(self.gold_chunk_ids or self.annotated_at)

    @property
    def needs_venue_coverage(self) -> bool:
        """Answerable comparisons must say, per venue, why it is or is not covered.

        Wider than `is_comparison`: a question whose qtype is comparison needs coverage
        whatever its category (three gold questions have a comparison qtype and another
        category), so the labels of every multi-venue question say which venues they cover.
        """
        return self.is_answerable and (self.qtype == COMPARISON or self.is_comparison)

    def slice_labels(self, dimension: str) -> list[str]:
        """Values this question contributes to a metric breakdown.

        `venue` returns every venue the question names, so a comparison question
        counts under each of its venues and venue rows do not sum to the total.

        Raises:
            ValueError: `dimension` is not `venue`, `qtype` or `category`.
        """
        if dimension == "venue":
            return list(self.venues) or ["(unspecified)"]
        if dimension == "qtype":
            return [self.qtype]
        if dimension == "category":
            return [self.effective_category]
        raise ValueError(f"unknown breakdown dimension: {dimension!r}")


def fields_without_default(record_type: type[DataclassInstance]) -> list[str]:
    """The fields of the record dataclass `record_type` that have no default, in field order.

    A saved record of that type cannot leave one of them out.
    """
    return [
        f.name for f in fields(record_type) if f.default is MISSING and f.default_factory is MISSING
    ]


def missing_fields(record: Mapping[str, object], record_type: type[DataclassInstance]) -> list[str]:
    """The fields of `record_type` without a default that the saved `record` lacks, in order."""
    return [name for name in fields_without_default(record_type) if name not in record]


@cache
def _field_validators(record_type: type[DataclassInstance]) -> dict[str, TypeAdapter[Any]]:
    """One validator per field of the record dataclass `record_type`, by field name."""
    return {name: TypeAdapter(kind) for name, kind in get_type_hints(record_type).items()}


def wrongly_typed_fields(
    record: Mapping[str, object], record_type: type[DataclassInstance]
) -> dict[str, str]:
    """Each field of the saved `record` whose value is not of its type in `record_type`, with why.

    The check is strict, so nothing is converted: a record is written by hand too, and a
    string where a list belongs would be read character by character. Keys `record_type`
    has no field for are left out, and a field whose type is a dataclass takes only an
    instance of it, so a caller checks a nested record against its own type.
    """
    validators = _field_validators(record_type)
    problems: dict[str, str] = {}
    for name, value in record.items():
        if name not in validators:
            continue
        try:
            validators[name].validate_python(value, strict=True)
        except ValidationError as exc:
            problems[name] = exc.errors()[0]["msg"]
    return problems
