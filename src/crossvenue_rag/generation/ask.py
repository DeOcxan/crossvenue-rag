"""One ad-hoc question answered: what `crossvenue ask` prints and POST /answer returns.

The question goes through the configured retriever and generator exactly as in
an evaluation run. The venues a question names are read from its text by the
same dictionary venue_split uses.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

from pydantic import BaseModel, ConfigDict

from ..config import Config
from ..config_pipeline import RetrievalMethod
from ..errors import UserError, not_blank
from ..models import Question
from ..retrieval.methods import ad_hoc_method
from ..retrieval.protocols import AskRetriever, DetectingRanks
from ..retrieval.retriever import Retriever
from .answer import generate
from .context import Citation
from .llm import LLM, LLMFactory
from .schemas import Missing, Outcome, outcome

AD_HOC_ID = "ask"

# What an ad-hoc question uses when it names nothing: `crossvenue ask` and POST /answer.
DEFAULT_RETRIEVER = "venue_split"
DEFAULT_GENERATOR = "gemini"


class AskCitation(BaseModel):
    """Where one claim's cited chunk is: its number in the prompt, venue, page and URL."""

    model_config = ConfigDict(frozen=True)

    chunk: int
    venue: str
    doc_path: str
    url: str
    environment: str | None
    api_version: str | None
    product: str | None


class AskClaim(BaseModel):
    """One claim of an ad-hoc answer, with the venue it names and its citations."""

    model_config = ConfigDict(frozen=True)

    text: str
    venue: str
    citations: list[AskCitation]


class AskResult(BaseModel):
    """An ad-hoc question answered, as `crossvenue ask` prints it and POST /answer returns it.

    It holds the answer and each claim with its venue and its citations, resolved
    to a document path and URL (each citation carries its chunk's venue). It also
    holds the outcome under the rule every metric uses (full, partial, refused,
    asked), and for a partial answer the venues the chunks lack.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    venues_detected: list[str]
    retriever: str
    generator: str
    answer: str
    claims: list[AskClaim]
    outcome: Outcome
    refused: bool
    asked_clarification: bool
    partial: bool
    missing: list[Missing]
    model: str
    provider: str
    cost_usd: float | None
    latency_ms: float
    cached: bool


def _ask_citation(c: Citation) -> AskCitation:
    return AskCitation(
        chunk=c.chunk_number,
        venue=c.venue,
        doc_path=c.doc_path,
        url=c.source_url,
        environment=c.environment,
        api_version=c.api_version,
        product=c.product,
    )


def checked_options(
    cfg: Config, question: str, retriever_name: str, generator_name: str
) -> tuple[str, RetrievalMethod]:
    """The stripped question and the retrieval method, after checking all three options.

    Needs no index, database or model, so a command can call it first.

    Raises:
        UserError: The question is blank, the retriever is an oracle or no retrieval method,
            or the generator is not one of `generation.generators`.
    """
    question = not_blank("the question", question).strip()
    gen = cfg.generation
    if generator_name not in gen.generators:
        raise UserError(
            f"--generator {generator_name!r} is not a generator of generation.generators",
            hint=f"pass --generator one of: {', '.join(gen.generators)}",
        )
    return question, ad_hoc_method(retriever_name, "--retriever")


def answer_question(
    cfg: Config,
    retriever: DetectingRanks,
    llm: LLM,
    question: str,
    retriever_name: str,
    generator_name: str,
) -> AskResult:
    """One question answered with the configured retriever and generator.

    Raises:
        UserError: `question` is blank, `retriever_name` is an oracle or no retrieval method,
            or `generator_name` is not a generator of `generation.generators`.
    """
    question, method = checked_options(cfg, question, retriever_name, generator_name)
    gen = cfg.generation
    venues = retriever.venue_detector.detect(question)
    record = Question(AD_HOC_ID, question, "bronze", list(venues), "fact")
    results = retriever.retrieve(question, method, k=gen.top_k)
    generated = generate(llm, gen, gen.generators[generator_name], record, results)
    answer, call = generated.answer, generated.call
    claims = [
        AskClaim(
            text=claim.text,
            venue=claim.venue,
            citations=[_ask_citation(c) for c in citations],
        )
        for claim, citations in zip(answer.claims, generated.citations, strict=True)
    ]
    kind = outcome(answer)
    return AskResult(
        question=question,
        venues_detected=list(venues),
        retriever=retriever_name,
        generator=generator_name,
        answer=answer.answer,
        claims=claims,
        # The outcome rule of methodology section 9a: a partial answer with
        # no claim is a refusal.
        outcome=kind,
        refused=kind == "refused",
        asked_clarification=answer.asked_clarification,
        partial=answer.partial,
        missing=list(answer.missing),
        model=call.actual_model or call.requested_model,
        provider=call.provider,
        cost_usd=call.cost_usd,
        latency_ms=call.latency_ms,
        cached=call.cached,
    )


def ask(
    cfg: Config,
    question: str,
    retriever_name: str,
    generator_name: str,
    *,
    open_retriever: Callable[[Config], AbstractContextManager[AskRetriever]] | None = None,
    make_llm: LLMFactory | None = None,
) -> AskResult:
    """One question answered as `crossvenue ask` answers it, from a retriever opened for it.

    The options are checked before the index or the database is opened; the
    retriever is closed when the answer is made, and the model client is one run
    over `cfg.generation`. `open_retriever` and `make_llm` build the retriever and the
    client; `Retriever` and `LLM` unless given.

    Raises:
        UserError: `question` is blank, `retriever_name` is an oracle or no retrieval method,
            or `generator_name` is not a generator of `generation.generators`.
    """
    checked_options(cfg, question, retriever_name, generator_name)
    with (open_retriever or Retriever)(cfg) as retriever:
        retriever.require_current_index()
        return answer_question(
            cfg,
            retriever,
            (make_llm or LLM)(cfg.generation),
            question,
            retriever_name,
            generator_name,
        )
