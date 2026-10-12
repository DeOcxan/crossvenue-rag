"""The retrieved chunks as the generator sees them, numbered so a citation resolves to its source.

Each retrieved chunk goes into the prompt with a number, its venue, and the
environment, API version and product its occurrences carry, so the generator
can tell a testnet page from a mainnet one. A citation is the occurrence of a
merged chunk that matches the question (docs/adr/0003-exact-deduplication.md).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..models import ChunkSource, Result

_ENVIRONMENT = [
    ("testnet", re.compile(r"\b(test\s?net|sandbox|paper trading)\b", re.I)),
    ("mainnet", re.compile(r"\b(main\s?net|live|production|prod)\b", re.I)),
]
_PRODUCT = [
    # Not "swap" or "contract": on a DEX they name a token swap and a smart contract.
    ("perp", re.compile(r"\b(perps?|perpetuals?|futures|usdt-m|coin-m)\b", re.I)),
    ("spot", re.compile(r"\bspot\b", re.I)),
]
_VERSION = re.compile(r"\b(v\d+|infinity)\b", re.I)


@dataclass(slots=True)
class QuestionContext:
    """What a question says about environment, product and API version."""

    environment: str | None
    product: str | None
    api_version: str | None

    @classmethod
    def of(cls, question: str) -> QuestionContext:
        """The context `question` names.

        A field stays None when the question names none, and the environment
        when it names two.
        """
        named = [name for name, pattern in _ENVIRONMENT if pattern.search(question)]
        # A question naming both (testnet against live) gets neither.
        env = named[0] if len(named) == 1 else None
        product = next((name for name, pattern in _PRODUCT if pattern.search(question)), None)
        version = _VERSION.search(question)
        return cls(env, product, version.group(1).lower() if version else None)


class Citation(BaseModel):
    """The occurrence of a chunk that one claim cites, as an answer record saves it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_number: int
    chunk_id: str
    venue: str
    doc_path: str
    source_url: str
    environment: str | None
    api_version: str | None
    product: str | None
    is_primary: bool
    matched: list[str]


def _values(sources: Sequence[ChunkSource], name: str) -> str:
    values = sorted({getattr(s, name) for s in sources if getattr(s, name)})
    return ", ".join(values) if values else "-"


def chunk_header(number: int, result: Result) -> str:
    """The header line of chunk `number` in a prompt.

    Gives the venue, the environment, API version and product of its sources,
    and the page with its heading path.
    """
    sources = result.sources or []
    page = result.doc_path + (
        " > " + " > ".join(result.heading_path) if result.heading_path else ""
    )
    return (
        f"[{number}] venue: {result.venue} | environment: {_values(sources, 'environment')} | "
        f"api_version: {_values(sources, 'api_version')} | "
        f"product: {_values(sources, 'product')} | page: {page}"
    )


def render_chunks(results: Sequence[Result]) -> str:
    """The numbered chunk block of the generator prompt, 1-based."""
    blocks = [
        f"{chunk_header(i, result)}\n{result.text.strip()}"
        for i, result in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


def resolve_citation(question: str, number: int, result: Result) -> Citation:
    """The occurrence of chunk `number` that matches the question best.

    The occurrence matching the largest number of the question's environment, product and
    API version wins, and a tie keeps the earlier occurrence in `sources`. The primary
    occurrence is first, so it wins a tie it is part of and a question that says none
    of these. The citation's
    venue is the chunk's, whatever venue the claim names, so a misattributed
    claim shows as a claim and a citation on different venues.
    """
    sources = list(result.sources) or [
        ChunkSource(result.venue, result.doc_path, list(result.heading_path), result.source_url, 0)
    ]
    wanted = QuestionContext.of(question)
    best_index, best_matched = 0, list[str]()
    for index, source in enumerate(sources):
        matched = [
            name
            for name in ("environment", "product", "api_version")
            if getattr(wanted, name) and getattr(source, name) == getattr(wanted, name)
        ]
        # Strictly more matches replaces the chosen occurrence; equal keeps the
        # earlier one, so the primary, first, wins every tie it is part of.
        if len(matched) > len(best_matched):
            best_index, best_matched = index, matched
    chosen = sources[best_index]
    return Citation(
        chunk_number=number,
        chunk_id=result.chunk_id,
        venue=result.venue,
        doc_path=chosen.doc_path,
        source_url=chosen.source_url,
        environment=chosen.environment,
        api_version=chosen.api_version,
        product=chosen.product,
        is_primary=best_index == 0,
        matched=best_matched,
    )
