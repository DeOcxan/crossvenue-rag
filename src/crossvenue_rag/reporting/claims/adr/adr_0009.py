"""What docs/adr/0009-generators.md claims.

The generators, their token and retry limits, venue_split's coverage, the temperature, the
five-question trial run, and OpenRouter's prices, uptimes and provider listings of
2026-09-25.
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import answers
from ...run_ids import RETRIEVAL_RUN
from ...sources import Sources
from ..comparisons import comparison_questions, coverage_parts, top_k_text
from ..reasons import OPENROUTER, TRIAL


def _generators(src: Sources) -> str:
    return formatting.count(len(answers.generators(src)))


def generators_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0009-generators.md."""
    generators = _generators(src)
    families = formatting.count(answers.model_families(src))
    doc.check(". {} generators from {} model families", generators, families)
    _coverage(doc, src)
    doc.check("OpenRouter serves all {} generator families", families)
    _trial(doc, src)
    doc.check("- {} generators, each from a different family", generators)
    _prices(doc, src)
    _parameters(doc, src)


def _coverage(doc: Doc, src: Sources) -> None:
    """Context: venue_split against hybrid_rerank on the comparison questions."""
    doc.check(
        "On the {} comparison questions of `v0.2.1-gold`, venue_split puts every named venue "
        "in the top {} on {} ({}/{}) of them, against {} ({}/{}) for hybrid_rerank",
        comparison_questions(src),
        top_k_text(src),
        *coverage_parts(src, "venue_split"),
        *coverage_parts(src, "hybrid_rerank"),
    )
    doc.ref("for hybrid_rerank (`results/{}`).", RETRIEVAL_RUN)


def _trial(doc: Doc, src: Sources) -> None:
    """Context: what the answer trial run measured."""
    doc.check(
        "In a {}-question trial",
        formatting.count(len(src.default_config.generation.calibration_exclude)),
    )
    doc.given(
        "Qwen averaged {} s per answer and once took over {} s. At effort minimal it used {} "
        "to {} reasoning tokens and took {} to {} s on a comparison question.",
        "57",
        "300",
        "371",
        "416",
        "23",
        "31",
        reason=TRIAL,
    )
    doc.given("Mercury at minimal used {} reasoning tokens in {} s.", "257", "1.8", reason=TRIAL)


def _prices(doc: Doc, src: Sources) -> None:
    """Decision: the providers and prices OpenRouter listed."""
    doc.given(
        "the cheapest of the {} (${} / ${} per million input / output tokens",
        _generators(src),
        "0.04",
        "0.15",
        reason=OPENROUTER,
    )
    doc.given("OpenRouter listed {} providers for it", "16", reason=OPENROUTER)
    doc.given(
        "(${} / ${} per million tokens), supports structured output, and had {}% uptime over "
        "the {} hours",
        "0.15",
        "1.875",
        "99.1",
        "24",
        reason=OPENROUTER,
    )
    doc.given("Parasail had higher uptime ({}%)", "99.97", reason=OPENROUTER)


def _parameters(doc: Doc, src: Sources) -> None:
    """Decision: the parameters sent to every generator."""
    doc.given(
        "the lowest level all {} accept",
        _generators(src),
        reason="the configured generators, `generation.matrix`",
    )
    doc.given(
        "at most {} output tokens;",
        formatting.count(answers.generator_max_tokens(src)),
        reason="generation.generator_params",
    )
    # An accepted ADR is not edited (docs/style.md rule 41): it keeps the form it was
    # accepted with, where configs/default.yaml and evals/methodology.md write 0.0.
    doc.definition("temperature 0 where", reason="the decision as ADR 0009 was accepted")
    doc.given(
        "asked for again, at most {}.",
        formatting.count(answers.max_invalid_retries(src)),
        reason="generation.max_invalid_retries",
    )
