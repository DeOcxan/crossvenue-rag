"""evals/methodology.md, D16: the published `v0.2.1-gold` results reported as effects below `DELTA`.

The table after the deviations lists them, each with its run, n and the rule it was judged by.
"""

from __future__ import annotations

from .... import formatting
from ....config_eval import (
    COMPARISON_SCOPE,
    LABELLED_COMPARISON_SCOPE,
    LABELLED_SCOPE,
    MULTI_VENUE_SCOPE,
)
from ....evaluation.comparisons import TOP_ONE_OFF_VENUE
from ...check import Doc
from ...compute import (
    answers,
    prompts,
    retrieval,
    sensitivity,
)
from ...compute import retrieval_off_venue as off_venue
from ...compute.answers import PROMPT_V1
from ...compute.estimates import Estimate, is_significant, is_small_effect
from ...outcomes import SKIP
from ...run_ids import (
    JUDGED_V1,
    PROMPTS_RUN,
    PVALUE_RUN,
    RETRIEVAL_RUN,
    SENSITIVITY_BASE_RUN,
)
from ...sources import Sources
from .. import comparisons
from ..reasons import INTERVAL_LEVEL
from .significance import DELTA


def _rule(section: str, p: float) -> str:
    """The decision a registered Holm rule gives a result with adjusted p `p`."""
    verdict = "significant" if is_significant(p) else "inconclusive at this size"
    return f"\u00a7{section}: {verdict}"


def _d16_row(
    effects: dict[str, list[object]],
    key: str,
    source: tuple[str, int],
    difference: Estimate,
    p: float,
    rule: str,
) -> bool:
    """Adds the D16 row `key` from `source` (run, n); whether its result is a small effect."""
    run, n = source
    effects[key] = [
        f"`{run}`",
        formatting.count(n),
        comparisons.estimate(difference),
        formatting.p_value(p),
        SKIP,
        rule,
    ]
    return is_small_effect(difference, p, DELTA)


def small_effects(doc: Doc, src: Sources) -> None:
    """D16: every published result on `v0.2.1-gold` reported as an effect below `DELTA`."""
    rows: dict[str, list[object]] = {}
    holds = []
    recall = retrieval.at_top_k(src, "recall")
    coverage = retrieval.at_top_k(src, "venue_coverage")
    for a, b, metric, scope, label in (
        ("hybrid_rerank", "bm25", recall, LABELLED_COMPARISON_SCOPE, "comparison"),
        ("hybrid_rerank", "bm25", recall, LABELLED_SCOPE, "labeled"),
        ("hybrid_rerank", "dense", coverage, COMPARISON_SCOPE, "comparison"),
        ("venue_split_oracle", "venue_split", coverage, MULTI_VENUE_SCOPE, MULTI_VENUE_SCOPE),
    ):
        p = retrieval.paired_p(src, a, b, metric, scope, "holm_p")
        below = metric == recall and label == "comparison"
        rule = "\u00a77: below \u03b4, inconclusive at this size" if below else _rule("7", p)
        interval = retrieval.paired_interval(src, a, b, metric, scope)
        source = (PVALUE_RUN, retrieval.paired_questions(src, a, b, metric, scope))
        key = f"{a} - {b}, `{metric}`, {label}"
        holds.append(_d16_row(rows, key, source, interval, p, rule))
    for pair, metric, scope in (
        (("gemini@venue_split", "gemini@hybrid_rerank"), "coverage", "comparison"),
        (("gemini@venue_split", "qwen@venue_split"), "coverage", "comparison"),
        (("gemini@venue_split", "qwen@venue_split"), "behaviour_correct", "comparison"),
        (("qwen@venue_split", "mercury@venue_split"), "coverage", "comparison"),
        (("qwen@venue_split", "mercury@venue_split"), "behaviour_correct", "comparison"),
        (("qwen@venue_split", "mercury@venue_split"), "venue_attribution_error", "all"),
        (("mercury@venue_split", "gemini@hybrid_rerank"), "coverage", "comparison"),
    ):
        p = answers.judged_p(src, PROMPT_V1, pair, metric, scope, "holm_p")
        role = answers.judged_role(src, PROMPT_V1, pair, metric, scope)
        interval = answers.judged_interval(src, PROMPT_V1, pair, metric, scope)
        source = (JUDGED_V1, answers.judged_questions(src, PROMPT_V1, pair, metric, scope))
        key = f"{pair[0]} - {pair[1]}, `{metric}`, {scope}"
        holds.append(_d16_row(rows, key, source, interval, p, _rule(f"9 {role}", p)))
    for config in ("gemini@venue_split", "qwen@venue_split", "mercury@venue_split"):
        change = prompts.change(src, config, "wrong_refusal")
        rule = _rule("9a H-v2", change.holm_p)
        key = f"{config} v2 - v1, `wrong_refusal`"
        source = (PROMPTS_RUN, prompts.questions(src, "wrong_refusal", config))
        holds.append(_d16_row(rows, key, source, change.difference, change.holm_p, rule))
    share = retrieval.at_top_k(src, "off_venue")
    for metric in (share, TOP_ONE_OFF_VENUE):
        p = off_venue.off_venue_holm_p(src, "bm25", metric, "comparison")
        interval = off_venue.off_venue_interval(src, "bm25", metric, "comparison")
        key = f"hybrid_rerank - bm25, `{metric}`, comparison"
        source = (RETRIEVAL_RUN, off_venue.off_venue_questions(src, "bm25", metric, "comparison"))
        holds.append(_d16_row(rows, key, source, interval, p, _rule("9b", p)))
    labeled = sensitivity.Comparison("hybrid_rerank", "hybrid", recall, LABELLED_SCOPE)
    row = sensitivity.paired_row(src, SENSITIVITY_BASE_RUN, labeled)
    p = sensitivity.tested_p(row)
    key = f"hybrid_rerank - hybrid, `{recall}`, labeled"
    source = (SENSITIVITY_BASE_RUN, row.n)
    holds.append(_d16_row(rows, key, source, Estimate.of(row), p, _rule("7", p)))
    doc.definition("an interval that excludes 0", reason="the definition of an effect")
    doc.table("| result | run | n | difference | p_Holm | reported in | decision rule |", rows)
    doc.given(
        "each difference has its {}% paired bootstrap interval:",
        comparisons.level(),
        reason=INTERVAL_LEVEL,
    )
    doc.claim(
        "every published result on `v0.2.1-gold` reported as an effect whose size is below \u03b4",
        holds=all(holds),
        what="each D16 row is an effect (interval excluding 0 or p_Holm below the level) "
        f"of size below {formatting.design_parameter(DELTA)}",
    )
