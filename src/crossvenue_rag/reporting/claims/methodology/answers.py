"""evals/methodology.md sections 9 to 9b: answers, prompt v2 and off-venue retrieval."""

from __future__ import annotations

from .... import formatting
from ....dataset.sampling import cited_words
from ....evaluation.paired import MIN_TESTED_QUESTIONS
from ....generation.grader_agreement import BINARY_DIMENSIONS
from ...check import Doc
from ...compute import answers, calibration, datasets, retrieval
from ...compute.answers_costs import MERCURY
from ...compute.estimates import Estimate
from ...compute.lookup import entry
from ...outcomes import SKIP, Param
from ...run_ids import ANSWERS_V1, ANSWERS_V2, RETRIEVAL_RUN
from ...sources import Sources
from .. import comparisons
from ..reasons import INTERVAL_LEVEL, REGISTERED
from . import protocol
from .texts import (
    ANSWER_CONVENTIONS,
    CONVENTION,
    TRIAL_QUESTIONS,
)

# What each dimension of the calibration table counts, as its label names it.
# Style rule 35: a judgment field is written in code font; coverage is also a metric's name.
FIELDS = frozenset(BINARY_DIMENSIONS)
UNITS = {
    "supported": "claims",
    "venue_correct": "claims",
    "behaviour_correct": "answers",
    "coverage": "answers",
}


def generator_protocol_claims(doc: Doc, src: Sources) -> None:
    """Section 9: the generator configurations and their hypotheses."""
    doc.check(
        "The {}-question trial run that checks the infrastructure",
        formatting.count(len(src.default_config.generation.calibration_exclude)),
    )
    doc.check(
        "reasoning effort minimal, at most {} output tokens, temperature {} where the endpoint "
        "accepts it",
        formatting.count(answers.generator_max_tokens(src)),
        formatting.own_form(answers.generator_temperature(src), formatting.TEMPERATURE_DECIMALS),
    )
    doc.check(
        "and at most {} retries of an invalid response",
        formatting.count(answers.max_invalid_retries(src)),
    )
    doc.check(
        "{} configurations, each a generator over a retriever's top {} chunks "
        "(`generation.matrix`):",
        formatting.count(len(answers.configurations(src))),
        formatting.count(answers.answer_top_k(src)),
    )
    doc.definitions(ANSWER_CONVENTIONS, reason=CONVENTION)
    doc.check(
        "{} candidates, `anthropic/claude-sonnet-5` and `anthropic/claude-opus-5.5`, get the "
        "same prompt (`prompts/judge.md` v2)",
        formatting.count(calibration.judge_candidate_count(src)),
    )
    doc.check(
        "on venue_attribution_error and on coverage, over the {} comparison questions. The "
        "generator is held fixed",
        comparisons.comparison_questions(src),
    )
    resamples = protocol.bootstrap_resamples(src)
    doc.check(
        "{}% paired bootstrap interval ({} resamples, seed {}) and a",
        Param(comparisons.level(), INTERVAL_LEVEL),
        resamples,
        protocol.bootstrap_seed_text(src),
    )
    doc.check("paired randomization test ({} permutations, same seed). The", resamples)
    doc.given(
        "a primary result is significant when its adjusted p is below {}.",
        comparisons.significance(),
        reason=REGISTERED,
    )
    doc.given(
        "A secondary result is significant only when its adjusted p is below {};",
        comparisons.significance(),
        reason=REGISTERED,
    )


def calibration_sample_claims(doc: Doc, src: Sources) -> None:
    """Section 9: the calibration sample and its thresholds."""
    size = formatting.count(len(calibration.sample_ids(src)))
    doc.check("The calibration sample is {} answers of the full generation run", size)
    doc.ref("The sample drawn from `results/{}` holds", ANSWERS_V1)
    doc.ref("from `results/{}/calibration.json`", ANSWERS_V1)
    doc.check(
        "Answers to the {} trial questions (`cv-",
        formatting.count(len(src.default_config.generation.calibration_exclude)),
    )
    cited = ", ".join(f'`{q} "{cited_words(src.questions[q])}"`' for q in TRIAL_QUESTIONS)
    doc.claim(
        f"trial questions ({cited})",
        holds=calibration.excludes_exactly(src, TRIAL_QUESTIONS),
        what="the listed questions are `generation.calibration_exclude`",
    )
    configs = calibration.sample_by(src, "config")
    categories = calibration.sample_by(src, "category")
    other = calibration.other_category_size(src)
    doc.claim(
        "holds gemini@venue_split {}, mercury@venue_split {}, qwen@venue_split {} and "
        "gemini@hybrid_rerank {} answers; comparison {}, unanswerable {} and {} of each other "
        "category.",
        formatting.count(configs["gemini@venue_split"]),
        formatting.count(configs[MERCURY]),
        formatting.count(configs["qwen@venue_split"]),
        formatting.count(configs["gemini@hybrid_rerank"]),
        *[formatting.count(categories[c]) for c in calibration.OVERSAMPLED],
        comparisons.optional_count(other),
        holds=other is not None,
        what="every other category holds the same number of calibration answers",
    )
    doc.check("Each judge candidate grades the same {} answers", size)
    _thresholds(doc, src)


def _thresholds(doc: Doc, src: Sources) -> None:
    """Section 9: the calibration thresholds, and how they were recorded."""
    size = formatting.count(len(calibration.sample_ids(src)))
    doc.table(
        "| dimension | measure | threshold |",
        {
            "`supported` (faithfulness)": [SKIP, _at_least(src, "supported")],
            "`venue_correct` (venue_attribution_error)": [SKIP, _at_least(src, "venue_correct")],
            "`behaviour_correct`": [SKIP, _at_least(src, "behaviour_correct")],
            "coverage": [SKIP, _at_least(src, "coverage")],
        },
    )
    doc.check(
        "The coverage threshold, Spearman {}, is a judgment fixed before any rating",
        formatting.own_form(
            calibration.threshold(src, "coverage"), formatting.CONFIG_THRESHOLD_DECIMALS
        ),
    )
    doc.check("on a repeated run of {} answers is reported beside each threshold", size)
    doc.check("Recorded 2026-09-25, after the {} answers were rated", size)
    doc.check(
        "{}% bootstrap intervals over answers ({} resamples, seed {}), from",
        Param(comparisons.level(), INTERVAL_LEVEL),
        formatting.count(calibration.resamples(src)),
        formatting.own_form(calibration.bootstrap_seed(src)),
    )


def _at_least(src: Sources, dimension: str) -> str:
    """The threshold of `dimension` as the table writes it: `>= 0.61`, typeset by the checker."""
    return ">= " + formatting.own_form(
        calibration.threshold(src, dimension), formatting.CONFIG_THRESHOLD_DECIMALS
    )


def _parts(agreement: Estimate) -> list[str]:
    return [formatting.metric(v) for v in (agreement.value, agreement.low, agreement.high)]


def agreement_parts(src: Sources, comparison: str, dimension: str) -> list[str]:
    """The statistic of `comparison` on `dimension`, then its interval's ends.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension`, or records no interval
          of its statistic there.
    """
    return _parts(calibration.agreement_interval(src, comparison, dimension))


def metric_bounds(agreement: Estimate) -> list[str]:
    """The ends of an agreement's interval."""
    return [formatting.metric(agreement.low), formatting.metric(agreement.high)]


def agreement_cell(agreement: Estimate) -> str:
    """A cell of a calibration table: an agreement with its interval, `0.620 [0.412, 0.790]`."""
    value, low, high = _parts(agreement)
    return f"{value} [{low}, {high}]"


def threshold_cell(statistic: str, threshold: float) -> str:
    """The threshold cell of a calibration table: `Spearman >= 0.70`, typeset by the checker.

    The statistic is written as `calibration.STATISTIC_LABELS` names it.

    Raises:
        ValueError: `statistic` is not a statistic of `calibration.STATISTIC_LABELS`.
    """
    shown = formatting.own_form(threshold, formatting.CONFIG_THRESHOLD_DECIMALS)
    label = entry(calibration.STATISTIC_LABELS, statistic, "STATISTIC_LABELS")
    return f"{label} >= {shown}"


def dimension_label(dimension: str, compared: str, between: str | None = None) -> str:
    """The label of a calibration table row: the dimension and what it counts.

    `between`, when given, is what the two candidates compare, where it differs from what a
    candidate and the human compare.

    Raises:
        ValueError: `dimension` is not a dimension of `UNITS`.
    """
    written = f"`{dimension}`" if dimension in FIELDS else dimension
    judges = f"; {between} between the judges" if between is not None else ""
    return f"{written} ({compared} {entry(UNITS, dimension, 'UNITS')}{judges})"


def calibration_table(doc: Doc, src: Sources) -> None:
    """Section 9: each judge candidate's agreement with the human."""
    doc.table(
        "| dimension | threshold | sonnet | opus |",
        {
            dimension_label(
                r.dimension,
                formatting.count(
                    calibration.compared_count(src, calibration.judge_with_human(src), r.dimension)
                ),
            ): [
                threshold_cell(r.statistic, r.threshold),
                *[
                    agreement_cell(interval)
                    for interval in r.intervals[: calibration.TABLE_CANDIDATES]
                ],
            ]
            for r in calibration.table_rows(src)
        },
    )


def judge_protocol_claims(doc: Doc, src: Sources) -> None:
    """Section 9: the judge chosen, its repeat agreement and the refusals it reports."""
    judge = src.judged_v1.config.generation.chosen_judge()
    interval, point_only = calibration.passes_by_reach(src, judge)
    doc.claim(
        "coverage meets its threshold on the point estimate and on the interval; `venue_correct` "
        "on the point estimate only",
        holds="coverage" in interval and "venue_correct" in point_only,
        what="Opus passes coverage on the interval and venue_correct on the point estimate only",
    )
    doc.claim(
        "so its interval is near-degenerate and cannot establish the threshold",
        holds=not calibration.interval_informative(
            src, calibration.judge_with_human(src), "venue_correct"
        ),
        what="Opus's venue_correct kappa is undefined in some resamples",
    )
    doc.claim(
        "`behaviour_correct` passes on the point estimate only",
        holds="behaviour_correct" in point_only,
        what="Opus's behaviour_correct interval reaches below its threshold",
    )
    doc.check(
        "Opus meets {} thresholds and Sonnet {}, so Opus is the judge",
        formatting.count(calibration.passes(src, judge)),
        formatting.count(calibration.passes(src, calibration.other_candidate(src))),
    )
    either, _ = calibration.venue_disagreement(src, judge)
    doc.check(
        "The {} of {} claims either grader marks wrong are both in one answer",
        formatting.count(either),
        formatting.count(calibration.human_negatives(src).claims),
    )
    doc.check(
        "Its \u03ba lies in [{}, {}] or is undefined (undefined in {} of {} resamples)",
        *metric_bounds(
            calibration.agreement_interval(src, calibration.judge_with_human(src), "venue_correct")
        ),
        formatting.count(
            calibration.kappa_undefined(src, calibration.judge_with_human(src), "venue_correct")
        ),
        formatting.count(calibration.resamples(src)),
    )
    doc.check(
        "Opus's agreement with its own repeat on the {} answers",
        formatting.count(calibration.answers_rated(src)),
    )
    doc.claim(
        "\u03ba {} [{}, {}] on `supported`, {} [{}, {}] on `venue_correct` and "
        "`behaviour_correct`, Spearman {} [{}, {}] on coverage.",
        *agreement_parts(src, calibration.judge_with_repeat(src), "supported"),
        *agreement_parts(src, calibration.judge_with_repeat(src), "venue_correct"),
        *agreement_parts(src, calibration.judge_with_repeat(src), "coverage"),
        holds=calibration.same_agreement(
            src, calibration.judge_with_repeat(src), "behaviour_correct", "venue_correct"
        ),
        what="behaviour_correct's repeat agreement equals venue_correct's",
    )
    doc.check(
        "That dimension is measured on {} answers; with it, \u03ba was {} for Sonnet and {} for "
        "Opus",
        formatting.count(
            calibration.compared_count(src, calibration.judge_with_human(src), "behaviour_correct")
        ),
        formatting.metric(calibration.behaviour_kappa(src, calibration.other_candidate(src))),
        formatting.metric(calibration.behaviour_kappa(src, judge)),
    )
    doc.check(
        "It stays in the other {} dimensions", formatting.count(len(calibration.DIMENSIONS) - 1)
    )
    doc.definition("is reported in one of two kinds", reason="the two kinds that follow")
    doc.check(
        "The calibration sample is {} answers of `results/",
        formatting.count(calibration.answers_rated(src)),
    )
    doc.ref("answers of `results/{}`, on the questions of `v0.2.1-gold`, with", ANSWERS_V1)
    doc.check(
        "with {} claims. Every claim of it that either grader",
        formatting.count(calibration.sample_claims(src)),
    )
    doc.definition(
        "`paired.csv` of the two committed judged runs",
        reason="the judged runs of prompt v1 and v2",
    )
    _refusals(doc, src)


def _refused(src: Sources, config: str) -> str:
    """`config`'s refusals of comparison questions under prompt v1."""
    return formatting.count(answers.comparison_refusals(src, config))


def _refusals(doc: Doc, src: Sources) -> None:
    """Section 9: the comparison refusals of the answer matrix."""
    doc.ref("Recorded after the matrix `results/{}`", ANSWERS_V1)
    doc.check(
        "Gemini refused {} of the {} comparison questions with venue_split and {} with "
        "hybrid_rerank; Qwen refused {}, Mercury {}.",
        _refused(src, "gemini@venue_split"),
        comparisons.comparison_questions(src),
        _refused(src, "gemini@hybrid_rerank"),
        _refused(src, "qwen@venue_split"),
        _refused(src, MERCURY),
    )
    refusals = _refused(src, answers.refusing_config(src))
    doc.check(
        "Of Gemini's {} venue_split refusals, {} had a gold chunk of every named venue in the "
        "context, {} of some named venues and {} of none. In {} of the {} the context held "
        "chunks of every named venue.",
        refusals,
        *[formatting.count(n) for _, n in answers.refusal_group_sizes(src)],
        formatting.count(len(answers.refusals_with_every_venue(src))),
        refusals,
    )
    doc.ref("matrix `results/{}` (generator prompt v1", ANSWERS_V1)


def prompt_v2_protocol_claims(doc: Doc, src: Sources) -> None:
    """Section 9a: generator prompt v2."""
    doc.definition("Rule 3 (a fact of one venue", reason=CONVENTION)
    doc.definition("Rule 6 of v2", reason=CONVENTION)
    doc.definition("Rule 5 allows three outcomes", reason="the three outcomes that follow")
    doc.check(
        "the {} configurations, models, providers, parameters, retrievers and top {}",
        formatting.count(len(answers.configurations(src))),
        formatting.count(answers.answer_top_k(src)),
    )
    doc.claim(
        "under v2 it applies to {} answers, all mercury@venue_split.",
        formatting.count(len(answers.claimless_partial_configs(src))),
        holds=answers.claimless_partials_all_from(src, MERCURY),
        what="every claim-less partial answer is Mercury's",
    )
    doc.ref("The v1 and v2 answers are `results/{}` and `results/{}`", ANSWERS_V1, ANSWERS_V2)
    doc.check(
        "answer or ask_clarification, {} questions)",
        formatting.count(datasets.answer_expected_count(src)),
    )
    doc.check(
        "randomization test, seed {})",
        protocol.bootstrap_seed_text(src),
    )
    doc.given(
        "faithfulness by the \u00a79 procedure ({}% paired bootstrap",
        comparisons.level(),
        reason=INTERVAL_LEVEL,
    )
    doc.check(
        "Holm-adjusted within each metric across the {} configurations.",
        formatting.count(len(answers.configurations(src))),
    )
    doc.given(
        "H-v2 holds for a configuration when, at adjusted p < {}",
        comparisons.significance(),
        reason=REGISTERED,
    )
    doc.given(
        "A guard whose adjusted p is {} or above", comparisons.significance(), reason=REGISTERED
    )
    doc.check(
        "over the {} comparison questions, by the \u00a79 procedure, Holm-adjusted within the",
        comparisons.comparison_questions(src),
    )
    doc.check(
        "The {} calibration answers of `results/",
        formatting.count(calibration.answers_rated(src)),
    )
    doc.ref("calibration answers of `results/{}`, on the questions of `v0.2.1-gold`", ANSWERS_V1)
    doc.check(
        "None is a partial answer and none of their {} claims has the absence form",
        formatting.count(calibration.sample_claims(src)),
    )


def off_venue(doc: Doc, src: Sources) -> None:
    """Section 9b: off-venue retrieval."""
    k = comparisons.top_k_text(src)
    doc.check("- Off-venue share@{}: the share of a question's top {}", k, k)
    doc.check("counted separately, as via-ccxt share@{}", k)
    comparison = comparisons.comparison_questions(src)
    adversarial = formatting.count(datasets.category_count(src, "adversarial"))
    doc.check(
        "The comparison ({}) and adversarial ({}) questions of `v0.2.1-gold`, all {} methods, "
        "run `results/",
        comparison,
        adversarial,
        formatting.count(retrieval.method_count(src)),
    )
    doc.ref("methods, run `results/{}`.", RETRIEVAL_RUN)
    doc.ref("the retrieval results of `results/{}` were seen", RETRIEVAL_RUN)
    doc.check(
        "on off-venue share@{} and on top-1 off-venue, each by a {}% paired bootstrap interval "
        "({} resamples, seed {}, as in \u00a77). On the {} comparison questions",
        k,
        Param(comparisons.level(), INTERVAL_LEVEL),
        protocol.bootstrap_resamples(src),
        protocol.bootstrap_seed_text(src),
        comparison,
    )
    doc.check(
        "The {} adversarial questions are a category of fewer than {} questions",
        adversarial,
        formatting.count(MIN_TESTED_QUESTIONS),
    )
    doc.check("competition for a slot in the top {}", k)
