"""Venue attribution errors by kind (a reporting rule, methodology section 9).

The judge marks a claim `venue_correct: false` when the venue it names is not
the venue whose chunk states it; two different errors reach that verdict, and
they are reported apart from the judge's verdicts alone.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

from ..evaluation.metrics import WRAPPER_VENUES
from ..stats import mean
from .records import AnswerRecord

WRAPPERS = frozenset(WRAPPER_VENUES)
Kind = Literal["cross_exchange", "wrapper", "other"]
KINDS: tuple[Kind, ...] = get_args(Kind)
# A CCXT page that documents one exchange: `wiki/exchanges/<exchange>.md`.
_EXCHANGE_PAGE = re.compile(r"^wiki/exchanges/([^/]+)\.md$")


def documented_exchange(venue: str, doc_path: str) -> str | None:
    """The exchange a wrapper's page documents, or None for any other page."""
    if venue not in WRAPPERS:
        return None
    match = _EXCHANGE_PAGE.match(doc_path)
    return match.group(1).lower() if match else None


def attribution_kind(
    claim_venue: str,
    cited_venues: Collection[str],
    venues: Collection[str],
    documented: Collection[str | None],
) -> Kind:
    """The kind of a misattributed claim, from the venue it names and the chunks it cites.

    `cited_venues` are the venues of the chunks the claim cites; `venues` are
    the corpus's venue names; `documented` holds, for each cited CCXT chunk, the
    exchange its page documents (`documented_exchange`), None for a page about no
    one exchange.

    - `cross_exchange`: the claim names an exchange and cites a chunk of another
      exchange, or a CCXT page about another exchange.
    - `wrapper`: the claim cites only CCXT pages (a client library that documents
      how it calls each exchange) about the exchange it names.
    - `other`: a claim that fits neither: it cites a chunk of the venue it names,
      names CCXT or no known venue, or cites only CCXT pages about no one exchange.
    """
    named = claim_venue.strip().lower()
    cited = {v.lower() for v in cited_venues}
    if named not in venues or named in WRAPPERS or named in cited or not cited:
        return "other"
    if cited - WRAPPERS:
        return "cross_exchange"
    exchanges = set(documented)
    if exchanges == {named}:
        return "wrapper"
    if exchanges - {named, None}:
        return "cross_exchange"
    return "other"


@dataclass
class AttributionSummary:
    """One configuration's claims, misattributed claims and venue_attribution_error by kind.

    The error of a kind is the mean over answers with claims of the answer's
    share of claims of that kind, as the metric itself is averaged.
    """

    answers: int = 0
    claims: int = 0
    misattributed: dict[Kind, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    error: dict[Kind, float | None] = field(default_factory=lambda: dict.fromkeys(KINDS))

    @property
    def total_error(self) -> float | None:
        """venue_attribution_error summed over the kinds; None where no answer has a claim."""
        values = [v for v in self.error.values() if v is not None]
        return sum(values) if values else None


def attribution_summary(
    records: Sequence[AnswerRecord],
    venue_of: Mapping[str, str],
    doc_path_of: Mapping[str, str],
    configs: Sequence[str],
) -> dict[str, AttributionSummary]:
    """Misattributed claims by kind per configuration, from judged answer records.

    `venue_of` and `doc_path_of` give each chunk's venue and document. A kind's error
    is None for a configuration none of whose judged answers has a claim: it does
    not apply there (methodology section 9), and it is never 0. Each claim cites
    numbers of its record's `chunk_ids`, as `read_answers` ensures.

    Raises:
        ValueError: A misattributed claim cites a chunk with no venue in `venue_of`, or a
            wrapper's chunk with no document in `doc_path_of`; the message names the chunk.
    """
    venues = set(venue_of.values())
    out = {config: AttributionSummary() for config in configs}
    shares: dict[str, dict[Kind, list[float]]] = {c: {k: [] for k in KINDS} for c in configs}
    for record in records:
        judged = record.judge
        if record.config not in out or judged is None:
            continue
        generation = record.generated()
        claims = generation.answer.claims
        verdicts = judged.judgement.claims
        if not claims:
            continue
        summary = out[record.config]
        summary.answers += 1
        summary.claims += len(claims)
        counts = dict.fromkeys(KINDS, 0)
        numbered = generation.numbered_chunk_ids()
        for claim, verdict in zip(claims, verdicts, strict=True):
            if verdict.venue_correct:
                continue
            ids = [numbered[n] for n in claim.chunks]
            if lacking := [
                i
                for i in ids
                if i not in venue_of or (venue_of[i] in WRAPPERS and i not in doc_path_of)
            ]:
                raise ValueError(
                    f"{record.answer_id} cites {lacking[0]}, of no known venue or page"
                )
            cited = {venue_of[i] for i in ids}
            documented = {
                documented_exchange(venue_of[i], doc_path_of[i])
                for i in ids
                if venue_of[i] in WRAPPERS
            }
            counts[attribution_kind(claim.venue, cited, venues, documented)] += 1
        for kind in KINDS:
            summary.misattributed[kind] += counts[kind]
            shares[record.config][kind].append(counts[kind] / len(claims))
    for config, summary in out.items():
        for kind in KINDS:
            values = shares[config][kind]
            summary.error[kind] = mean(values) if values else None
    return out
