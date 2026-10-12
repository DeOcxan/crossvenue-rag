"""evals/methodology.md: deviations D1 to D26; D16's table of effects is in `small_effects`."""

from __future__ import annotations

from .... import formatting
from ....dataset.questions import MIN_QUESTIONS_PER_CATEGORY
from ....dataset.selection import TARGETS, target_size
from ...check import Doc
from ...compute import (
    agreement,
    answers,
    calibration,
    corpus,
    datasets,
    datasets_rounds,
    retrieval,
    retrieval_earlier,
)
from ...compute.answers import PROMPT_V1
from ...outcomes import Param
from ...run_ids import (
    ANSWERS_V1,
    ANSWERS_V2,
    COVERAGE_FAILURE_RUN,
    FIRST_GOLD_RUN,
    JUDGED_V1,
    JUDGED_V2,
    PRE_DEDUP_RUN,
    RETRIEVAL_RUN,
)
from ...sources import Sources
from .. import comparisons
from ..reasons import LIST_IN_SENTENCE, REGISTERED, TRIAL
from . import protocol
from .texts import CONVENTION

# The total decision 2 says the three constraints do not fit in.
ROUND_TOTAL = 100
# The categories decision 2 names apart from the floor: comparison and unanswerable.
NAMED_CATEGORIES = ("comparison", "unanswerable")


def selection(doc: Doc, src: Sources) -> None:
    """Deviations D1 and D2: how the questions were drafted and selected."""
    questions = comparisons.question_count_text(src)
    doc.check("drafted all {} questions, which were", questions)
    doc.check(
        "accepted with the draft text unchanged: {} of {} identical to `v0.1.6-candidates`.",
        formatting.count(datasets.unchanged_drafts(src)),
        questions,
    )
    doc.check(
        "their median length is {} words",
        formatting.ratio(datasets.median_question_words(src), formatting.COUNT_MEAN_DECIMALS),
    )
    doc.check(
        "| D2 | Decisions {}--{}: {} questions |",
        Param("1", CONVENTION),
        Param("2", CONVENTION),
        Param(formatting.count(target_size()), REGISTERED),
    )
    excess = datasets.category_excess(src)
    doc.claim(
        "selection accepted {} above target: unanswerable {} (+{}), adversarial {} (+{}), "
        "ambiguous, noisy and version_sensitive {} (+{} each). Comparison is exactly {}, all "
        "answerable.",
        formatting.count(datasets.excess_total(src)),
        formatting.count(datasets.category_count(src, "unanswerable")),
        formatting.count(excess["unanswerable"]),
        formatting.count(datasets.category_count(src, "adversarial")),
        formatting.count(excess["adversarial"]),
        formatting.count(datasets.category_count(src, "ambiguous")),
        formatting.count(excess["ambiguous"]),
        comparisons.comparison_questions(src),
        holds=datasets.excess_is_as_stated(src),
        what="ambiguous, noisy and version_sensitive have the same count and excess; "
        "comparison is at target",
    )
    doc.check("| {}. Targets are floors in `select", questions)
    doc.check("records no reason for the extra {}.", formatting.count(datasets.excess_total(src)))
    doc.check(
        "The {} extra unanswerable items enter no recall average; the other {} add one labeled "
        "question each",
        formatting.count(excess["unanswerable"]),
        formatting.count(datasets.excess_total(src, excluding=("unanswerable",))),
    )
    doc.check(
        "the {} extra unanswerable questions enter no recall average.",
        formatting.count(excess["unanswerable"]),
    )


def venues(doc: Doc, src: Sources) -> None:
    """Deviation D3: the venues the questions name."""
    doc.check(
        "| D3 | Decision 3: every venue the answer venue for >= {} questions |",
        Param(protocol.VENUE_FLOOR, REGISTERED),
    )
    ccxt = formatting.count(datasets.questions_naming(src, "ccxt"))
    dex, questions = datasets.questions_naming_dex(src), datasets.question_count(src)
    doc.check(
        "| ccxt names {} questions. Every other venue names {} or more; {} of {} questions ({}%) "
        "name a DEX, meeting the \u2265 {}% DEX floor.",
        ccxt,
        formatting.count(datasets.fewest_naming_other(src, "ccxt")),
        formatting.count(dex),
        formatting.count(questions),
        formatting.percent_figure(dex, questions),
        Param("45", REGISTERED),
    )
    naming, supplement = datasets.supplement_naming(src, "ccxt")
    doc.check(
        "held deliberately small in the comparison supplement ({} of {}).",
        formatting.count(naming),
        formatting.count(supplement),
    )
    doc.check("Results that concern ccxt have {} questions", ccxt)


def augmentation_deviation_claims(doc: Doc, src: Sources) -> None:
    """Deviations on pooling: the chunks judged in augmentation and those left unjudged."""
    others = formatting.count(datasets.non_comparison_count(src))
    doc.check(
        "on the {} comparison questions the never-judged top {} of `venue_split` and "
        "`venue_split_oracle` was judged by augmentation (rule 3). The {} other questions are "
        "not augmented.",
        comparisons.comparison_questions(src),
        formatting.count(retrieval.pool_depth(src)),
        others,
    )
    doc.check(
        "Round 1's pool left {} named venues `missing_from_pool` and an unjudged@10 of {} for "
        "`venue_split` on the {} comparison questions.",
        formatting.count(datasets.missing_from_pool_before_augmentation(src)),
        formatting.metric(retrieval_earlier.first_run_unjudged(src, "venue_split")),
        comparisons.comparison_questions(src),
    )
    doc.ref("the unjudged@10 from the retrievals of `results/{}`", COVERAGE_FAILURE_RUN)
    doc.check(
        "The {} questions that were not augmented keep round-1 gold, and venue_split_oracle's "
        "unjudged@10 on the {} ambiguous questions is {}, so more judging could move its recall "
        "there either way.",
        others,
        formatting.count(datasets.category_count(src, "ambiguous")),
        formatting.metric(retrieval.oracle_unjudged(src, "ambiguous")),
    )
    doc.ref("That share is from `results/{}`", RETRIEVAL_RUN)
    doc.definition(
        "The trial run: Qwen at effort low took 57 s per answer on average and once over 300 s",
        reason=TRIAL,
    )


def later_deviations(doc: Doc, src: Sources) -> None:
    """The deviations of prompt v2, the exposed questions and the first registration."""
    judge = src.judged_v1.config.generation.chosen_judge()
    claimless = formatting.count(len(answers.claimless_partial_configs(src)))
    doc.check("{} Mercury v2 answers flag `partial` and state no fact.", claimless)
    doc.ref("before either matrix was judged: in `results/{}`", ANSWERS_V2)
    doc.check("under v2, {} Mercury answers count as refusals.", claimless)
    doc.check(
        "both answers carry claims: {} for the primary pair.",
        formatting.count(
            answers.judged_questions(
                src, PROMPT_V1, answers.primary_pair(src), "venue_attribution_error", "comparison"
            )
        ),
    )
    sample = comparisons.round2_sample_text(src)
    doc.check(
        "The answer-rating screen showed the gold chunks of {} of the {} round-2 questions on "
        "2026-09-25",
        formatting.count(datasets_rounds.exposed_sample_count(src)),
        sample,
    )
    doc.check(
        "\u03ba over all {} questions stays the registered estimate; \u03ba over the {} "
        "unexposed questions",
        sample,
        formatting.count(datasets_rounds.unexposed_sample_count(src)),
    )
    doc.ref("(`results/{}` already holds the result)", COVERAGE_FAILURE_RUN)
    doc.check(
        "measured the coverage failure on the same {} questions (`results/",
        comparisons.comparison_questions(src),
    )
    doc.definition("Reported in total and in two kinds", reason="the two kinds named")
    either, _ = calibration.venue_disagreement(src, judge)
    doc.check(
        "These are {} of the {} claims of the calibration sample of `results/",
        formatting.count(either),
        formatting.count(calibration.sample_claims(src)),
    )
    doc.ref("calibration sample of `results/{}` that either grader marked", ANSWERS_V1)
    doc.claim(
        "that either grader marked, and all {} claims Opus marks in `results/",
        formatting.count(answers.misattributed_count(src)),
        holds=answers.misattributions_are_wrappers(src),
        what="no cross-exchange or other misattribution in either judged matrix",
    )
    doc.ref("Opus marks in `results/{}` and `results/{}`, on the questions", JUDGED_V1, JUDGED_V2)
    doc.check(
        "each judge candidate grades the {} calibration answers a second time, for "
        "self-consistency",
        formatting.count(calibration.answers_rated(src)),
    )
    doc.check(
        "Both candidates graded the {} calibration answers once and Opus was selected",
        formatting.count(calibration.answers_rated(src)),
    )
    doc.definition("| D1 |", reason=CONVENTION)
    doc.ref(
        "The `report.md` of `results/{}`, `results/{}` and `results/{}` has neither column, and "
        "that of `results/{}` has no `ndcg_prime@k`.",
        PRE_DEDUP_RUN,
        FIRST_GOLD_RUN,
        COVERAGE_FAILURE_RUN,
        RETRIEVAL_RUN,
    )
    rated = formatting.count(calibration.answers_rated(src))
    doc.check("a judge dimension is measured on the {} calibration answers", rated)
    doc.check(
        "`behaviour_correct` against the human is measured on {} of the {}: one answer is left out",
        formatting.count(
            calibration.compared_count(src, calibration.judge_with_human(src), "behaviour_correct")
        ),
        rated,
    )
    doc.check(
        "had been computed (\u03ba {} for Sonnet, {} for Opus)",
        formatting.metric(calibration.behaviour_kappa(src, calibration.other_candidate(src))),
        formatting.metric(calibration.behaviour_kappa(src, judge)),
    )
    doc.ref("on the calibration sample of `results/{}`, drawn", ANSWERS_V1)
    doc.check(
        "Round 1 pooled {} methods (bm25, dense, hybrid, hybrid_rerank)",
        formatting.count(len(retrieval.ROUND1_POOL)),
    )
    doc.check(
        "without temperature; {} judge candidates, both through OpenRouter",
        formatting.count(calibration.judge_candidate_count(src)),
    )
    doc.check(
        "generated over the same {} configurations",
        formatting.count(len(answers.configurations(src))),
    )
    doc.definition(
        "every generator at reasoning effort low and temperature 0",
        reason=REGISTERED + " (\u00a79 as first registered)",
    )
    doc.check(
        '`crossvenue count-term` counts "tick size" in {} Hyperliquid chunks and `szDecimals` '
        "in {}.",
        formatting.count(corpus.term_chunks(src, "tick size", "hyperliquid")),
        formatting.count(corpus.term_chunks(src, "szDecimals", "hyperliquid")),
    )


def _registered(n: int) -> Param:
    """A count the protocol registers, as a given hole."""
    return Param(formatting.count(n), REGISTERED)


def size_deviation(doc: Doc) -> None:
    """D26: the sum decision 2 gives for its three constraints, against what they sum to."""
    three = Param("3", LIST_IN_SENTENCE)
    two = Param("2", CONVENTION)
    total = target_size()
    least = MIN_QUESTIONS_PER_CATEGORY
    named = [TARGETS[c] for c in NAMED_CATEGORIES]
    others = len(TARGETS) - len(NAMED_CATEGORIES)
    constraints = sum(named) + others * least

    doc.check(
        "| D26 | Decision {}: the {} constraints (comparison {}, unanswerable {}, nothing below "
        '{}) "do not fit in {}; {} is what they sum to" |',
        two,
        three,
        *[_registered(n) for n in (*named, least, ROUND_TOTAL, total)],
    )
    doc.check(
        "| Over the {} categories the {} constraints sum to {} ({} + {} + {} \u00d7 {}), so a "
        "total of {} holds them with no category below {}.",
        formatting.count(len(TARGETS)),
        three,
        formatting.count(constraints),
        *[_registered(n) for n in named],
        formatting.count(others),
        _registered(least),
        _registered(ROUND_TOTAL),
        _registered(least),
    )
    doc.claim(
        "holds them with no category below",
        holds=constraints <= ROUND_TOTAL < total,
        what="the three constraints fit in the round total, and the registered total exceeds it",
    )
    doc.check(
        "The {} of decisions {} and {} holds adversarial at its target of {} and fact at {}, the "
        "targets selection used",
        _registered(total),
        Param("1", CONVENTION),
        two,
        _registered(TARGETS["adversarial"]),
        _registered(TARGETS["fact"]),
    )
    doc.check(
        "| The sum was written beside the category split of decision {} and was not recomputed "
        "from the {} constraints alone. |",
        two,
        three,
    )
    doc.check(
        "selection used the targets of decision {} (D2), so no count, test or power figure "
        "changes. The total of {} follows from the adversarial and fact targets as well as from "
        "the {} constraints.",
        two,
        _registered(total),
        three,
    )


def amendment_deviation(doc: Doc, src: Sources) -> None:
    """D25: the adjudication and the D23 relabel under the criterion amended after round 2."""
    study = agreement.study(src)
    outside = agreement.outside_sample(src)
    kept = agreement.round1_criterion_questions(src)
    sample = formatting.count(study.report.questions)
    doc.check(
        "The adjudication of the {} disagreeing pairs of round 2",
        formatting.count(study.report.disagreements),
    )
    doc.check(
        "Of the {} questions outside the round-2 sample, {} keep labels made under the round-1 "
        "criterion until the full relabel; the other {} are D23 questions.",
        formatting.count(len(outside)),
        formatting.count(len(kept)),
        formatting.count(len(outside) - len(kept)),
    )
    doc.definition(
        "the criterion ambiguous in the three cases the amendment decides",
        reason="the cases the amendment lists",
    )
    doc.check(
        "In `v0.2.2-gold` the {} disagreeing pairs of the {} sampled questions and the D23 "
        "questions follow the amended criterion; every other label follows the round-1 criterion, "
        "among them every label of {} questions.",
        formatting.count(study.report.disagreements),
        sample,
        formatting.count(len(kept)),
    )
