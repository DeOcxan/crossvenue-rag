"""docs/results/findings.md, the judged answers: H-v2 and venue attribution."""

from __future__ import annotations

from .... import formatting
from ....evaluation.labels import fmt
from ....generation.answer_rows import mean_column
from ...check import Doc
from ...compute import answers, answers_judged, prompts
from ...compute.answers import PROMPT_V1, PROMPT_V2
from ...compute.answers_costs import MERCURY
from ...run_ids import JUDGED_V1
from ...sources import Sources
from ..comparisons import (
    change_text,
    comparison_questions,
    estimate,
    level,
    signed_parts,
    significance,
)

# The columns of the generators table, from the judged run's summary.csv.
GENERATOR_COLUMNS = (
    mean_column("coverage", "comparison"),
    mean_column("coverage", "all"),
    mean_column("behaviour_correct", "comparison"),
    mean_column("behaviour_correct", "all"),
    mean_column("venue_attribution_error", "all"),
)
# The pair of configurations whose venue attribution and behavior findings.md singles out.
QWEN_MERCURY = ("qwen@venue_split", MERCURY)
# The cells of the pairs table where the pair is the primary comparison.
PRIMARY_CELLS = ["primary", formatting.MISSING]
H_V2 = ("wrong_refusal", "venue_attribution_error")


def primary(doc: Doc, src: Sources) -> None:
    """The primary comparison of the answers: venue_split against hybrid_rerank."""
    pair = answers.primary_pair(src)
    doc.table(
        "| metric | n | venue_split | hybrid_rerank | difference | p_rand | p_Holm |",
        {
            r.metric: [
                formatting.count(r.questions),
                formatting.metric(r.mean_a),
                formatting.metric(r.mean_b),
                estimate(r.difference),
                formatting.p_value(r.randomisation_p),
                formatting.p_value(r.holm_p),
            ]
            for r in answers_judged.primary_rows(src, answers.primary_metrics(src), "comparison")
        },
    )
    doc.check(
        "both metrics in one Holm family, {}% paired bootstrap intervals (deviations D4, D16 and "
        "D17):",
        level(),
    )
    doc.check(
        "gemini@venue_split - gemini@hybrid_rerank on the {} comparison questions of "
        "`v0.2.1-gold`, prompt v1, run",
        comparison_questions(src),
    )
    doc.ref("prompt v1, run `results/{}`, both metrics", JUDGED_V1)
    doc.claim(
        "venue_attribution_error is {} for both",
        formatting.metric(
            answers.judged_means(src, pair, "venue_attribution_error", "comparison")[0]
        ),
        holds=answers_judged.means_zero(src, pair, "venue_attribution_error", "comparison"),
        what="neither configuration has a misattributed claim on the paired questions",
    )
    doc.claim(
        "Under prompt v1, retrieving once per named venue raises coverage, the share of each "
        "venue's gold facts the answer states averaged over the venues with gold facts, by {} "
        "[{}, {}], p_Holm = {} ([D24](../../evals/methodology.md#deviations-from-protocol)). "
        "Under prompt v2 (exploratory) the same comparison is {} [{}, {}], p_Holm = {}, "
        "inconclusive at this size",
        *signed_parts(answers.judged_interval(src, PROMPT_V1, pair, "coverage", "comparison")),
        formatting.p_value(answers.primary_holm(src, PROMPT_V1, pair, "coverage", "comparison")),
        *signed_parts(answers.judged_interval(src, PROMPT_V2, pair, "coverage", "comparison")),
        formatting.p_value(answers.primary_holm(src, PROMPT_V2, pair, "coverage", "comparison")),
        holds=answers.judged_includes_zero(src, PROMPT_V2, pair, "coverage", "comparison"),
        what="the prompt v2 coverage interval includes 0",
    )
    doc.check(
        "On it the comparison is inconclusive at this size, {} [{}, {}], p_Holm = {}",
        *signed_parts(
            answers.judged_interval(src, PROMPT_V1, pair, "venue_attribution_error", "comparison")
        ),
        formatting.p_value(
            answers.primary_holm(src, PROMPT_V1, pair, "venue_attribution_error", "comparison")
        ),
    )
    doc.check(
        "on the {} questions where both answers carry claims",
        formatting.count(
            answers.judged_questions(src, PROMPT_V1, pair, "venue_attribution_error", "comparison")
        ),
    )


def refusal_coverage(doc: Doc, src: Sources) -> None:
    """The coverage of the refusals and of the answers of the primary pair."""
    pair = answers.primary_pair(src)
    split = answers_judged.coverage_by_outcome(src, pair[0])
    rerank = answers_judged.coverage_by_outcome(src, pair[1])
    doc.check(
        "gemini@venue_split refuses {} comparison questions and its refusals average coverage {}",
        formatting.count(split.refused.count),
        fmt(split.refused.mean),
    )
    doc.check(
        "since {} of them state some of the gold facts before refusing",
        formatting.count(answers_judged.refusals_stating_facts(src, pair[0])),
    )
    doc.check(
        "gemini@hybrid_rerank refuses {}, averaging {}. On the questions each answered, the {} "
        "average {} and {}.",
        formatting.count(rerank.refused.count),
        fmt(rerank.refused.mean),
        formatting.count(len(answers.primary_pair(src))),
        fmt(split.answered.mean),
        fmt(rerank.answered.mean),
    )


def generator_comparison_claims(doc: Doc, src: Sources) -> None:
    """The generators against each other."""
    pair = answers.primary_pair(src)
    doc.table(
        "| configuration | coverage, comparison |",
        {
            c: [formatting.metric(v) for v in values]
            for c, values in answers_judged.judged_summary_rows(src, GENERATOR_COLUMNS)
        },
    )
    doc.table(
        "| a - b | coverage | p_Holm |",
        {
            f"{r.a} - {r.b}": [
                *(
                    PRIMARY_CELLS
                    if r.coverage is None
                    else [
                        estimate(r.coverage.difference),
                        formatting.p_value(r.coverage.holm_p),
                    ]
                ),
                estimate(r.behaviour.difference),
                formatting.p_value(r.behaviour.holm_p),
            ]
            for r in answers_judged.pair_rows(src, "comparison")
        },
    )
    doc.check(
        "Mercury refuses {} of the {} comparison questions, against {} for gemini@venue_split",
        formatting.count(answers.comparison_refusals(src, MERCURY)),
        comparison_questions(src),
        formatting.count(answers.comparison_refusals(src, pair[0])),
    )
    attribution = answers_judged.attribution_pair_rows(src)
    doc.table(
        "| a - b | n, comparison |",
        {
            f"{r.a} - {r.b}": [
                formatting.count(r.comparison_questions),
                estimate(r.comparison.difference),
                formatting.p_value(r.comparison.holm_p),
                formatting.count(r.all_questions),
                estimate(r.all.difference),
                formatting.p_value(r.all.holm_p),
            ]
            for r in attribution
        },
    )
    apart = answers_judged.pair_of(attribution, QWEN_MERCURY)
    doc.claim(
        "On the comparison questions no paired answer carries a claim Opus marks misattributed, "
        "so every pair is inconclusive at this size. On all questions the intervals of {} pairs "
        "include {}, inconclusive at this size; qwen@venue_split - mercury@venue_split is {} "
        "[{}, {}] with p_Holm = {}, inconclusive at this size.",
        formatting.count(answers_judged.attribution_pairs_touching_zero(src, "all")),
        formatting.count(0),
        *signed_parts(apart.all.difference),
        formatting.p_value(apart.all.holm_p),
        holds=answers_judged.attribution_pairs_touching_zero(src, "comparison") == len(attribution)
        and not answers_judged.any_pair_significant(src, "venue_attribution_error"),
        what="every comparison-scope interval is at 0 and no venue_attribution_error comparison "
        f"reaches p_Holm < {significance()}",
    )
    doc.claim(
        "Qwen has the highest coverage.",
        holds=answers_judged.ahead_of_every_pair(src, QWEN_MERCURY[0], "coverage", "comparison"),
        what="Qwen's comparison coverage is above every other configuration's, each pair at "
        f"p_Holm < {significance()}",
    )
    behaviour = answers_judged.pair_of(
        answers_judged.pair_rows(src, "comparison"), QWEN_MERCURY
    ).behaviour
    doc.claim(
        "qwen@venue_split - mercury@venue_split on `behaviour_correct` is {} [{}, {}], "
        "p_Holm = {}, inconclusive at this size",
        *signed_parts(behaviour.difference),
        formatting.p_value(behaviour.holm_p),
        holds=not answers_judged.behaviour_significant(src, *QWEN_MERCURY, "comparison"),
        what="qwen - mercury on behaviour_correct does not reach "
        f"p_Holm < {significance()} on comparison",
    )


def h_v2(doc: Doc, src: Sources) -> None:
    """H-v2: prompt v2 against v1 per configuration."""
    doc.table(
        "| configuration | wrong refusals, v1 to v2 |",
        {
            r.config: [
                change_text(r.first.v1, r.first.v2, formatting.metric),
                estimate(r.first.difference),
                formatting.p_value(r.first.holm_p),
                change_text(r.second.v1, r.second.v2, formatting.metric),
                formatting.count(prompts.questions(src, H_V2[1], r.config)),
                estimate(r.second.difference),
                formatting.p_value(r.second.holm_p),
            ]
            for r in prompts.change_rows(src, H_V2)
        },
    )
    doc.check(
        "Wrong refusals are over the {} questions",
        formatting.count(prompts.questions(src, H_V2[0])),
    )
    doc.claim(
        "and on none of the {} questions does v2 refuse where v1 did not",
        formatting.count(prompts.questions(src, H_V2[0])),
        holds=prompts.v2_never_worse(src, H_V2[0]),
        what="no configuration refuses in v2 a question v1 answered",
    )
    changed = prompts.change_range(src, H_V2[1])
    doc.claim(
        "Per configuration {} or {} questions change, so the smallest p an exact randomization "
        "test can reach is {} or {}. The run's p-values, sampled from random sign flips, equal "
        "those floors up to sampling error, and all {} of those changes go against v2.",
        formatting.count(changed.fewest),
        formatting.count(changed.most),
        formatting.p_value(prompts.smallest_p(changed.fewest)),
        formatting.p_value(prompts.smallest_p(changed.most)),
        formatting.count(changed.total),
        holds=prompts.changes_against_v2_at_floor(src, H_V2[1]),
        what="every change goes against v2 and every p is at its floor",
    )


def attribution_claims(doc: Doc, src: Sources) -> None:
    """Venue attribution in both matrices."""
    doc.table(
        "| configuration, prompt | answers with claims |",
        {
            f"{r.config}, {r.prompt}": [
                formatting.count(r.summary.answers),
                formatting.count(r.summary.claims),
                formatting.count(r.summary.misattributed["cross_exchange"]),
                formatting.count(r.summary.misattributed["wrapper"]),
                fmt(r.summary.total_error),
            ]
            for r in answers_judged.attribution_rows(src)
        },
    )
    doc.claim(
        "Every claim Opus marks misattributed in both matrices is of the wrapper kind",
        holds=answers.misattributions_are_wrappers(src),
        what="no cross-exchange or other misattribution in either matrix",
    )
    mis = answers_judged.misattributed_questions(src)
    doc.check(
        "The {} under v2 come from {} questions, {} of them labeled unanswerable",
        *[formatting.count(v) for v in (mis.claims, mis.questions, mis.unanswerable)],
    )


def prompt_v2_primary(doc: Doc, src: Sources) -> None:
    """The primary comparison under prompt v2, and the partial answers."""
    pair = answers.primary_pair(src)
    doc.check(
        "coverage {} [{}, {}], p_rand {}. On venue_attribution_error it is {} [{}, {}] on the {} "
        "questions where both v2 answers carry claims, p_rand {}. "
        "Both are inconclusive at this size (p_Holm {} and {})",
        *signed_parts(answers.judged_interval(src, PROMPT_V2, pair, "coverage", "comparison")),
        formatting.p_value(
            answers.judged_p(src, PROMPT_V2, pair, "coverage", "comparison", "randomisation_p")
        ),
        *signed_parts(
            answers.judged_interval(src, PROMPT_V2, pair, "venue_attribution_error", "comparison")
        ),
        formatting.count(
            answers.judged_questions(src, PROMPT_V2, pair, "venue_attribution_error", "comparison")
        ),
        formatting.p_value(
            answers.judged_p(
                src, PROMPT_V2, pair, "venue_attribution_error", "comparison", "randomisation_p"
            )
        ),
        formatting.p_value(answers.primary_holm(src, PROMPT_V2, pair, "coverage", "comparison")),
        formatting.p_value(
            answers.primary_holm(src, PROMPT_V2, pair, "venue_attribution_error", "comparison")
        ),
    )
    doc.check(
        "Under v1 the coverage difference is {} [{}, {}], p_Holm = {}.",
        *signed_parts(answers.judged_interval(src, PROMPT_V1, pair, "coverage", "comparison")),
        formatting.p_value(answers.primary_holm(src, PROMPT_V1, pair, "coverage", "comparison")),
    )
    doc.check("on the {} comparison questions (\u00a79a's", comparison_questions(src))
    partial = answers_judged.partial_misattributed(src)
    doc.check(
        "of the {} v2 answers counted as partial, {} carry a claim Opus marks misattributed, "
        "all {} of them the CCXT case above",
        *[
            formatting.count(v)
            for v in (partial.partial, partial.misattributed, partial.ccxt_alone)
        ],
    )
    doc.claim(
        "every claim a partial answer attributes to an exchange cites a chunk of that exchange "
        "or of CCXT",
        holds=answers_judged.partial_exchange_citations_ok(src),
        what="every exchange-attributed claim in a partial answer cites that exchange or CCXT",
    )
