"""The reply contract of each model role: the JSON Schema it is held to, checked again on parse.

The JSON Schemas are what the models are constrained to (structured output); the pydantic
models are what the pipeline accepts. Every object of a schema sets
`additionalProperties: false`, and no schema carries a numeric bound (`minimum`,
`maxLength`), which structured output refuses; bounds are enforced on parse.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .. import formatting
from ..errors import ExternalFailure, validation_problems

# The first answer schema version, a full answer or a refusal (prompts/generator.v1.md), and
# the version that adds `partial` and `missing` (prompts/generator.md).
FIRST_SCHEMA = 1
PARTIAL_SCHEMA = 2


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_CLAIMS = {
    "type": "array",
    "items": _object(
        {
            "text": {"type": "string"},
            "venue": {"type": "string"},
            "chunks": {"type": "array", "items": {"type": "integer"}},
        }
    ),
}

# v1 (prompts/generator.v1.md): full answer or refusal.
ANSWER_SCHEMA_V1: dict[str, Any] = _object(
    {
        "answer": {"type": "string"},
        "claims": _CLAIMS,
        "refused": {"type": "boolean"},
        "asked_clarification": {"type": "boolean"},
    }
)

# v2 (prompts/generator.md): adds the partial outcome and what is missing.
ANSWER_SCHEMA_V2: dict[str, Any] = _object(
    {
        "answer": {"type": "string"},
        "claims": _CLAIMS,
        "refused": {"type": "boolean"},
        "asked_clarification": {"type": "boolean"},
        "partial": {"type": "boolean"},
        "missing": {
            "type": "array",
            "items": _object({"venue": {"type": "string"}, "missing": {"type": "string"}}),
        },
    }
)

ANSWER_SCHEMAS = {FIRST_SCHEMA: ANSWER_SCHEMA_V1, PARTIAL_SCHEMA: ANSWER_SCHEMA_V2}
# The answer fields schema v2 added; an answer of schema v1 is saved without them.
V2_ANSWER_FIELDS = frozenset({"partial", "missing"})

# The parsed responses are saved in answer records, which never change once made.
_FROZEN = ConfigDict(frozen=True)

# Rationale before verdict in every object: structured output is generated in
# schema order, and a verdict written before its reasoning can contradict it
# (the trial run showed a rationale concluding "supported" above supported=false).
JUDGEMENT_SCHEMA: dict[str, Any] = _object(
    {
        "claims": {
            "type": "array",
            "items": _object(
                {
                    "index": {"type": "integer"},
                    "rationale": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "venue_correct": {"type": "boolean"},
                }
            ),
        },
        "coverage": {
            "type": "array",
            "items": _object(
                {
                    "venue": {"type": "string"},
                    "rationale": {"type": "string"},
                    "gold_facts": {"type": "integer"},
                    "covered_facts": {"type": "integer"},
                }
            ),
        },
        "behaviour_rationale": {"type": "string"},
        "behaviour_correct": {"type": "boolean"},
    }
)


class Claim(BaseModel):
    """One claim of an answer: its text, the venue it names and the chunk numbers it cites."""

    model_config = _FROZEN

    text: str
    venue: str
    chunks: list[int] = Field(default_factory=list)


class Missing(BaseModel):
    """A venue a partial answer cannot cover, and what the documentation lacks for it."""

    model_config = _FROZEN

    venue: str
    missing: str


class GeneratedAnswer(BaseModel):
    """A generator's parsed response: the answer, its claims and its outcome flags."""

    model_config = _FROZEN

    answer: str
    claims: list[Claim] = Field(default_factory=list)
    refused: bool = False
    asked_clarification: bool = False
    # v2 only; a v1 answer reads as partial=False, missing=[].
    partial: bool = False
    missing: list[Missing] = Field(default_factory=list)

    def numbers_outside(self, chunk_count: int) -> list[int]:
        """The chunk numbers the claims cite outside 1..chunk_count, in claim order."""
        numbers = range(1, chunk_count + 1)
        return [n for claim in self.claims for n in claim.chunks if n not in numbers]

    def stated_partial(self) -> bool | None:
        """`partial` as the answer states it; None for a v1 answer, which has no such field."""
        return self.partial if "partial" in self.model_fields_set else None


Outcome = Literal["full", "partial", "refused", "asked"]
# The outcomes that state an answer.
ANSWERED: tuple[Outcome, ...] = ("full", "partial")


def outcome(answer: GeneratedAnswer) -> Outcome:
    """What an answer counts as in every metric and count (evals/methodology.md, section 9a).

    A partial answer without a single claim states no fact, so it counts as a
    refusal, for every generator alike. The record keeps the flags the model set.
    """
    if answer.refused or (answer.partial and not answer.claims):
        return "refused"
    if answer.asked_clarification:
        return "asked"
    if answer.partial:
        return "partial"
    return "full"


class ClaimVerdict(BaseModel):
    """The judge's verdict on claim `index` (1-based) of an answer."""

    model_config = _FROZEN

    index: int
    supported: bool
    venue_correct: bool
    rationale: str = ""


class VenueCoverage(BaseModel):
    """How many of a venue's gold facts an answer covers; covered never exceeds gold."""

    model_config = _FROZEN

    venue: str
    gold_facts: int = Field(ge=0)
    covered_facts: int = Field(ge=0)
    rationale: str = ""

    @model_validator(mode="after")
    def _covered_within_gold(self) -> VenueCoverage:
        if self.covered_facts > self.gold_facts:
            raise ValueError(
                f"{self.venue}: covered_facts {formatting.count(self.covered_facts)} "
                f"> gold_facts {formatting.count(self.gold_facts)}"
            )
        return self


class Judgement(BaseModel):
    """A judge's parsed response: claim verdicts, coverage per venue and behavior."""

    model_config = _FROZEN

    claims: list[ClaimVerdict] = Field(default_factory=list)
    coverage: list[VenueCoverage] = Field(default_factory=list)
    behaviour_correct: bool
    behaviour_rationale: str = ""


# The form of an absence claim (rule 6 of generator prompt v2): a venue's list lacks a value.
ABSENCE_FORM = "does not list"


def is_absence_claim(text: str) -> bool:
    """Whether a claim's text has the absence form (`ABSENCE_FORM`), in any case."""
    return ABSENCE_FORM in text.lower()


class InvalidResponse(ExternalFailure):
    """A model reply that fails its schema or its checks.

    A failure outside the program: a run records it per answer as it records a
    failed call, and the service answers it with 502, since the model, not the
    request, is at fault.
    """


def _numbers(values: list[int]) -> str:
    return ", ".join(formatting.count(n) for n in values)


def parse_answer(content: dict[str, Any], chunk_count: int, schema_version: int) -> GeneratedAnswer:
    """A generator response, with every cited chunk number inside 1..chunk_count.

    Schema v2 also requires `partial` and `missing`, and their consistency: an
    answer is not both partial and refused, and a partial answer names what is
    missing.

    Raises:
        InvalidResponse: `content` lacks a field of the schema version or fails the schema, is both
            partial and refused, is partial without `missing`, or a claim cites a number outside
            1..chunk_count.
    """
    if schema_version >= PARTIAL_SCHEMA:
        absent = [f for f in ("partial", "missing") if f not in content]
        if absent:
            raise InvalidResponse(
                f"generator response lacks {', '.join(absent)} "
                f"(answer schema v{formatting.own_form(PARTIAL_SCHEMA)})"
            )
    try:
        answer = GeneratedAnswer.model_validate(content)
    except ValidationError as exc:
        raise InvalidResponse(
            f"generator response does not match the schema: {validation_problems(exc)}"
        ) from exc
    if answer.partial and answer.refused:
        raise InvalidResponse("generator response is both partial and refused")
    if answer.partial and not answer.missing:
        raise InvalidResponse("a partial answer must say in `missing` what is missing")
    if bad := answer.numbers_outside(chunk_count):
        raise InvalidResponse(
            f"the claims cite {formatting.plural(len(bad), 'chunk')} outside "
            f"1..{formatting.count(chunk_count)}: {_numbers(bad)}"
        )
    return answer


def parse_judgement(content: dict[str, Any], claim_count: int, gold_venues: set[str]) -> Judgement:
    """A judge response with exactly one verdict per claim, indexed 1..claim_count.

    It also has one coverage entry per venue of `gold_venues`, the venues with gold chunks.

    Raises:
        InvalidResponse: `content` fails the schema, its verdicts are not indexed 1..claim_count, or
            its coverage entries are not one per venue of `gold_venues`.
    """
    try:
        judgement = Judgement.model_validate(content)
    except ValidationError as exc:
        raise InvalidResponse(
            f"judge response does not match the schema: {validation_problems(exc)}"
        ) from exc
    indices = sorted(v.index for v in judgement.claims)
    if indices != list(range(1, claim_count + 1)):
        raise InvalidResponse(
            f"judge returned verdicts for claims {_numbers(indices)}, "
            f"expected 1..{formatting.count(claim_count)}"
        )
    judgement.claims.sort(key=lambda v: v.index)
    covered = [c.venue for c in judgement.coverage]
    if sorted(covered) != sorted(gold_venues):
        raise InvalidResponse(
            f"judge returned coverage for {', '.join(sorted(covered))}, expected "
            f"{', '.join(sorted(gold_venues))}"
        )
    return judgement
