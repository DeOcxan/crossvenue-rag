"""evals/methodology.md section 7: sample size, power and significance testing."""

from __future__ import annotations

from .... import formatting
from ....dataset.questions import MIN_QUESTIONS_PER_CATEGORY
from ....dataset.selection import TARGETS, target_size
from ....evaluation import mcnemar_power, t_test_size
from ....evaluation.power import TARGET_POWER_PERCENT
from ...check import Doc
from ...compute import datasets, power
from ...outcomes import Param
from ...run_ids import POWER_RUN
from ...sources import Sources
from .. import comparisons
from ..reasons import INTERVAL_LEVEL, POWER_MODEL, REGISTERED
from . import protocol
from .protocol import (
    FLOOR_SCORABLE_SIZE,
    FLOOR_SIZE,
    LARGER_COMPARISON_SIZE,
    PSI,
    PSI_HIGH,
    PSI_LOW,
    REDUCED_COMPARISON_SIZE,
    SCORABLE_SIZE,
    SMALLER_COMPARISON_SIZE,
)
from .texts import (
    LITERATURE,
    POWER_STUDIES,
    SAMPLE_SIZE_RULES,
)

# The differences the power model is evaluated at; its discordance is `protocol.PSI`.
DELTA = 0.20
SMALL_DELTA = 0.10
# The example of a significant split: questions.
EXAMPLE_SIZE = 120
# Cohen's conventional effect sizes, and the effect this project sizes its set for.
LARGE, MEDIUM, COHEN_SMALL = 0.8, 0.5, 0.2
TARGET_EFFECT = 0.3


def _delta(size: int) -> str:
    """The smallest difference `size` questions detect at `PSI`."""
    return formatting.design_parameter(mcnemar_power.detectable_delta(size, PSI))


def _delta_row(size: int) -> list[str]:
    """A row of the detectable-difference table: the difference, and the power at `DELTA`."""
    return [
        _delta(size),
        formatting.design_parameter(mcnemar_power.mcnemar_power(size, PSI, DELTA)),
    ]


def _effect(size: int) -> str:
    """The effect size d a paired t-test on `size` questions detects."""
    return formatting.design_parameter(t_test_size.detectable_effect(size))


def _needed(effect: float, shown: str) -> list[str]:
    """A row of the effect-size table: d as `shown`, and the questions it needs.

    Cohen's conventions keep his one decimal (`own_form`); the target is a design value (two).
    """
    return [shown, formatting.count(t_test_size.questions_for_effect(effect))]


def sample_size(doc: Doc) -> None:
    """Section 7: the literature, the effect-size tables and the McNemar model."""
    power = f"{formatting.count(TARGET_POWER_PERCENT)}%"
    doc.definitions(POWER_STUDIES, reason=LITERATURE)
    doc.definition(
        f"at \u03b1 = {comparisons.significance()} two-sided and {power} power is",
        reason=POWER_MODEL,
    )
    doc.given(
        "Cohen's conventional large and medium effects, and this project's target, "
        "{}, between Cohen's small ({}) and medium ({}):",
        formatting.design_parameter(TARGET_EFFECT),
        formatting.own_form(COHEN_SMALL, formatting.EFFECT_SIZE_DECIMALS),
        formatting.own_form(MEDIUM, formatting.EFFECT_SIZE_DECIMALS),
        reason=LITERATURE + "; the target is this project's choice",
    )
    doc.definition("Three cases:", reason="the table that follows")
    doc.table(
        "| Effect | d | n needed |",
        {
            "Large": _needed(LARGE, formatting.own_form(LARGE, formatting.EFFECT_SIZE_DECIMALS)),
            "Medium": _needed(MEDIUM, formatting.own_form(MEDIUM, formatting.EFFECT_SIZE_DECIMALS)),
            "Project target": _needed(TARGET_EFFECT, formatting.design_parameter(TARGET_EFFECT)),
        },
    )
    doc.given(
        "this project's target, between Cohen's small ({}) and medium ({}) effects,",
        formatting.own_form(COHEN_SMALL, formatting.EFFECT_SIZE_DECIMALS),
        formatting.own_form(MEDIUM, formatting.EFFECT_SIZE_DECIMALS),
        reason=LITERATURE,
    )
    doc.table(
        "| Set | scorable n | smallest detectable *d* |",
        {
            f"{formatting.count(target_size())} questions "
            f"({formatting.count(TARGETS['unanswerable'])} unanswerable excluded from recall)": [
                formatting.count(SCORABLE_SIZE),
                _effect(SCORABLE_SIZE),
            ],
            f"{formatting.count(FLOOR_SIZE)} floor "
            f"({formatting.count(FLOOR_SIZE - FLOOR_SCORABLE_SIZE)} excluded)": [
                formatting.count(FLOOR_SCORABLE_SIZE),
                _effect(FLOOR_SCORABLE_SIZE),
            ],
            f"comparison subgroup, {formatting.count(SMALLER_COMPARISON_SIZE)}": [
                formatting.count(SMALLER_COMPARISON_SIZE),
                _effect(SMALLER_COMPARISON_SIZE),
            ],
            f"comparison subgroup, {formatting.count(TARGETS['comparison'])}": [
                formatting.count(TARGETS["comparison"]),
                _effect(TARGETS["comparison"]),
            ],
        },
    )
    _scorable(doc)
    doc.definition(
        "a discordant question favors the better method with probability \u00bd + \u03b4/2\u03c8",
        reason=POWER_MODEL,
    )
    doc.definition(f"reaches \u03b1 = {comparisons.significance()}.", reason=POWER_MODEL)
    doc.definition(
        f"Computed exactly at \u03c8 = {formatting.design_parameter(PSI)} (the two methods "
        f"disagreeing on a quarter of questions), the smallest \u03b4 reaching {power} power.",
        reason=POWER_MODEL,
    )
    _mcnemar(doc)


def _scorable(doc: Doc) -> None:
    """Section 7: the scorable sizes the text states."""
    doc.check(
        "so the {} unanswerable questions are excluded from every recall average",
        Param(formatting.count(TARGETS["unanswerable"]), REGISTERED),
    )
    doc.check(
        "A set of {} therefore has a scorable n of {}",
        Param(formatting.count(target_size()), REGISTERED),
        Param(formatting.count(SCORABLE_SIZE), REGISTERED),
    )
    doc.check(
        "all {} comparison questions must be answerable",
        Param(formatting.count(TARGETS["comparison"]), REGISTERED),
    )
    doc.check(
        "If {} of the fifty turned out unanswerable",
        Param(formatting.count(TARGETS["comparison"] - REDUCED_COMPARISON_SIZE), POWER_MODEL),
    )
    doc.check(
        "whose comparison subgroup has fewer than {} answerable questions",
        formatting.count(TARGETS["comparison"]),
    )


def _mcnemar(doc: Doc) -> None:
    """Section 7: the detectable differences, and the example at n = 120."""
    header = (
        "| n | smallest detectable \u03b4 (recall@10) | "
        f"power at \u03b4 = {formatting.design_parameter(DELTA)} |"
    )
    unanswerable = TARGETS["comparison"] - REDUCED_COMPARISON_SIZE
    reduced = (
        f"{formatting.count(REDUCED_COMPARISON_SIZE)} (if {formatting.count(unanswerable)} of "
        f"the {formatting.count(TARGETS['comparison'])} were unanswerable)"
    )
    subgroup = f"{formatting.count(TARGETS['comparison'])} (the comparison subgroup)"
    scorable = f"{formatting.count(SCORABLE_SIZE)} (the whole scorable set)"
    doc.table(
        header,
        {
            formatting.count(SMALLER_COMPARISON_SIZE): _delta_row(SMALLER_COMPARISON_SIZE),
            reduced: _delta_row(REDUCED_COMPARISON_SIZE),
            subgroup: _delta_row(TARGETS["comparison"]),
            formatting.count(LARGER_COMPARISON_SIZE): _delta_row(LARGER_COMPARISON_SIZE),
            scorable: _delta_row(SCORABLE_SIZE),
        },
    )
    doc.definition(header, reason=POWER_MODEL)
    for key in (reduced, subgroup, scorable):
        doc.definition(f"| {key} |", reason=POWER_MODEL)
    doc.definition(
        f"At n = {formatting.count(EXAMPLE_SIZE)}, \u03c8 = {formatting.design_parameter(PSI)} "
        f"and \u03b4 = {formatting.design_parameter(SMALL_DELTA)}",
        reason=POWER_MODEL,
    )
    wins, losses = mcnemar_power.expected_split(EXAMPLE_SIZE, PSI, SMALL_DELTA)
    doc.check(
        "the expected discordant split is {}/{}, which is significant (p = {}).",
        formatting.count(wins),
        formatting.count(losses),
        formatting.p_value(mcnemar_power.expected_split_p(EXAMPLE_SIZE, PSI, SMALL_DELTA)),
    )
    doc.check(
        "The probability of landing on a significant split is {}, a coin flip.",
        formatting.design_parameter(mcnemar_power.mcnemar_power(EXAMPLE_SIZE, PSI, SMALL_DELTA)),
    )
    doc.check(
        "Detecting \u03b4 = {} at \u03c8 = {} with {}% power needs n \u2265 {}.",
        Param(formatting.design_parameter(SMALL_DELTA), POWER_MODEL),
        Param(formatting.design_parameter(PSI), POWER_MODEL),
        Param(formatting.count(TARGET_POWER_PERCENT), POWER_MODEL),
        formatting.count(mcnemar_power.questions_needed(PSI, SMALL_DELTA)),
    )


def observed_power(doc: Doc, src: Sources) -> None:
    """Section 7: the power analysis of the observed discordance (the power run)."""
    study = src.power
    family = formatting.count(power.family_size(src))
    doc.check("the gold chunk is in the top {} or it is not", comparisons.top_k_text(src))
    doc.check(
        "Per-question recall@10 is fractional for a question with more than one gold chunk ({} of "
        "the {} values of the {} methods on the {} comparison questions)",
        formatting.count(study.fractional_values),
        formatting.count(study.values),
        formatting.count(power.family_methods(src)),
        formatting.count(study.questions),
    )
    doc.check("each comparison is Holm-adjusted within a family of {}", family)
    doc.check(
        "the family's observed discordance ({} of {} question pairs differ, \u03c8 = {}) and "
        "difference sizes (mean {})",
        formatting.count(study.discordant_pairs),
        formatting.count(study.question_pairs),
        formatting.design_parameter(study.psi),
        formatting.design_parameter(study.mean_magnitude),
    )
    doc.ref(
        "varying only which method a discordant question favors (`results/{}`, on `v0.2.1-gold`)",
        POWER_RUN,
    )
    doc.check(
        "It reaches {}% power from \u03b4 = {} when the comparison is Holm's first step "
        "(\u03b1/{}) and from \u03b4 = {} unadjusted.",
        Param(formatting.count(TARGET_POWER_PERCENT), POWER_MODEL),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        family,
        formatting.design_parameter(power.detectable_gap(src, "unadjusted")),
    )
    doc.check(
        "At \u03b4 = {} the power is {} and {}.",
        Param(formatting.design_parameter(DELTA), POWER_MODEL),
        formatting.design_parameter(power.power_at_gap(src, DELTA, "holm_first_step")),
        formatting.design_parameter(power.power_at_gap(src, DELTA, "unadjusted")),
    )
    doc.check(
        "a \u03b4 above {} cannot occur in this model",
        formatting.design_parameter(power.gap_ceiling(src)),
    )
    doc.definitions(SAMPLE_SIZE_RULES, reason=REGISTERED)
    _size_rules(doc, src)


def _size_rules(doc: Doc, src: Sources) -> None:
    """Section 7: the sample-size decision and the reporting rules it sets."""
    total = Param(formatting.count(target_size()), REGISTERED)
    comparison = Param(formatting.count(TARGETS["comparison"]), REGISTERED)
    unanswerable = Param(formatting.count(TARGETS["unanswerable"]), REGISTERED)
    doc.check(
        "{} questions, hard floor {}, sized from the subgroups rather than the total. Comparison "
        ">= {} and all of them answerable, unanswerable >= {}, no category below {}, and the "
        "rest as decision 2 sets out. {} is what those constraints add up to.",
        total,
        Param(formatting.count(FLOOR_SIZE), REGISTERED),
        comparison,
        unanswerable,
        Param(formatting.count(MIN_QUESTIONS_PER_CATEGORY), REGISTERED),
        total,
    )
    doc.check(
        "- {} total gives {} scorable items once the {} unanswerable ones are excluded from "
        "recall averages.",
        total,
        Param(formatting.count(SCORABLE_SIZE), REGISTERED),
        unanswerable,
    )
    doc.check(
        "The floor of {} gives {} scorable",
        Param(formatting.count(FLOOR_SIZE), REGISTERED),
        Param(formatting.count(FLOOR_SCORABLE_SIZE), REGISTERED),
    )
    decisions = formatting.count(datasets.target_decisions(src))
    doc.check(
        "at {} pooled candidates per question, {} questions is already {} relevance decisions "
        "by one person",
        protocol.pool_display_text(src),
        total,
        decisions,
    )
    doc.check(
        "- Comparison at {} is {}% of the {}-question design.",
        comparison,
        formatting.percent_figure(TARGETS["comparison"], target_size()),
        total,
    )
    doc.check(
        "\u03b4 = {} in recall@10 is not detectable",
        Param(formatting.design_parameter(SMALL_DELTA), POWER_MODEL),
    )
    doc.check(
        "against the {} relevance decisions that {} questions already require", decisions, total
    )
    doc.check(
        "A comparison result smaller than {} is reported as inconclusive",
        Param(formatting.design_parameter(DELTA), REGISTERED),
    )
    tested = Param(comparisons.min_tested(), REGISTERED)
    doc.check("fewer than {} questions carries no p-value at all", tested)
    doc.check("over a scope of {} or more questions keeps", tested)
    doc.check("A category or slice below {} supports", tested)
    doc.check(
        '"method A finds a gold chunk in the top {} on k of the n adversarial questions, method '
        'B on m"',
        comparisons.top_k_text(src),
    )


def sample_size_decision(doc: Doc, src: Sources) -> None:
    """Section 7: the decision on the set's size, restated with the model's numbers."""
    doc.check(
        "the subgroup would be n = {} and the detectable \u03b4 would rise from {} to {}.",
        Param(formatting.count(REDUCED_COMPARISON_SIZE), POWER_MODEL),
        _delta(TARGETS["comparison"]),
        _delta(REDUCED_COMPARISON_SIZE),
    )
    doc.check("That detects *d* = {}", _effect(SCORABLE_SIZE))
    doc.check("and detects *d* = {}.", _effect(FLOOR_SCORABLE_SIZE))
    doc.check(
        "At {} the smallest detectable \u03b4 would be {}, against {} at {}.",
        Param(formatting.count(SMALLER_COMPARISON_SIZE), POWER_MODEL),
        _delta(SMALLER_COMPARISON_SIZE),
        _delta(TARGETS["comparison"]),
        Param(formatting.count(TARGETS["comparison"]), POWER_MODEL),
    )
    doc.check(
        "At \u03c8 = {} it needs n \u2265 {} comparison questions "
        "({} at \u03c8 = {}, {} at \u03c8 = {})",
        Param(formatting.design_parameter(PSI), POWER_MODEL),
        formatting.count(mcnemar_power.questions_needed(PSI, SMALL_DELTA)),
        formatting.count(mcnemar_power.questions_needed(PSI_LOW, SMALL_DELTA)),
        Param(formatting.design_parameter(PSI_LOW), POWER_MODEL),
        formatting.count(mcnemar_power.questions_needed(PSI_HIGH, SMALL_DELTA)),
        Param(formatting.design_parameter(PSI_HIGH), POWER_MODEL),
    )
    doc.check(
        "Comparison subgroup n = {}, all answerable. Smallest recall@10 difference detectable "
        "at {}% power, \u03b1 = {}, in the binary McNemar model: \u03b4 = {}.",
        Param(formatting.count(TARGETS["comparison"]), POWER_MODEL),
        Param(formatting.count(TARGET_POWER_PERCENT), POWER_MODEL),
        Param(comparisons.significance(), POWER_MODEL),
        _delta(TARGETS["comparison"]),
    )
    doc.check(
        "Whole scorable set n = {}, \u03b4 = {}. A \u03b4 of {} would require n \u2265 {} "
        "in the subgroup",
        Param(formatting.count(SCORABLE_SIZE), POWER_MODEL),
        _delta(SCORABLE_SIZE),
        Param(formatting.design_parameter(SMALL_DELTA), POWER_MODEL),
        formatting.count(mcnemar_power.questions_needed(PSI, SMALL_DELTA)),
    )
    doc.check(
        "with the observed discordance: \u03b4 = {} at Holm's first step, {} unadjusted.",
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        formatting.design_parameter(power.detectable_gap(src, "unadjusted")),
    )
    _testing(doc, src)


def _testing(doc: Doc, src: Sources) -> None:
    """Section 7: the tests, the primary comparison and the significance level."""
    resamples, seed = protocol.bootstrap_resamples(src), protocol.bootstrap_seed_text(src)
    doc.check(
        "Paired bootstrap over questions with {} resamples (seed {}) gives {}% percentile "
        "intervals",
        resamples,
        seed,
        Param(comparisons.level(), INTERVAL_LEVEL),
    )
    doc.check("A paired randomization test (sign flip, {} permutations, seed {})", resamples, seed)
    k = comparisons.top_k_text(src)
    doc.check("(a hit in the top {}, every named venue in the top {})", k, k)
    comparison = comparisons.comparison_questions(src)
    doc.check(
        "venue_split \u2212 hybrid_rerank on venue_coverage@10 over the {} comparison questions "
        "(`eval.primary_comparison`)",
        comparison,
    )
    doc.check(
        "had measured the coverage failure on the same {} questions (deviation D13)", comparison
    )
    doc.given(
        "Only a secondary result whose adjusted p is below {} is called significant.",
        comparisons.significance(),
        reason=REGISTERED,
    )
