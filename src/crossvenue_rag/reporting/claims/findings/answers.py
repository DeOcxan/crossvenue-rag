"""docs/results/findings.md, the answers of both prompts: their outcomes and cost."""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import answers, answers_costs, answers_outcomes, calibration, datasets
from ...compute.answers import PROMPT_V1, PROMPT_V2
from ...compute.answers_costs import MERCURY
from ...outcomes import SKIP
from ...run_ids import ANSWERS_V1, ANSWERS_V2, JUDGED_V1, JUDGED_V2
from ...sources import Sources
from ..comparisons import (
    change_text,
    comparison_questions,
    min_tested,
    optional_count,
    question_count_text,
)


def matrix(doc: Doc, src: Sources) -> None:
    """The answer matrix and its refusals under prompt v1."""
    top_k = formatting.count(answers.answer_top_k(src))
    doc.check(
        "{} configurations answer the {} questions of `v0.2.1-gold` from their top {} chunks",
        formatting.count(len(answers.configurations(src))),
        formatting.count(answers.answered_question_count(src)),
        top_k,
    )
    rows: dict[str, list[object]] = {
        r.category: [
            formatting.count(r.questions),
            r.expected or SKIP,
            *[formatting.count(v1) for v1, _ in r.changes],
        ]
        for r in answers_outcomes.outcome_table(src, "refused")
    }
    rows["all"] = [
        question_count_text(src),
        f"{formatting.count(datasets.refuse_expected_count(src))} refuse",
        *[
            formatting.count(v)
            for v in answers_outcomes.outcome_counts(src, None, "refused", PROMPT_V1)
        ],
    ]
    heading = "### Under prompt v1, gemini@venue_split refuses {} of the {} comparison questions"
    doc.check(
        heading,
        formatting.count(answers.comparison_refusals(src, "gemini@venue_split")),
        comparison_questions(src),
    )
    doc.table("### Under prompt v1, gemini@venue_split refuses", rows)
    doc.ref(
        "the behavior the questions expect; run `results/{}`, `v0.2.1-gold`, n per category:",
        ANSWERS_V1,
    )
    doc.check(
        "Gemini refuses {} (venue_split) and {} (hybrid_rerank) times, Qwen {} and Mercury {}.",
        formatting.count(answers.wrong_refusals(src, "gemini@venue_split")),
        formatting.count(answers.wrong_refusals(src, "gemini@hybrid_rerank")),
        formatting.count(answers.wrong_refusals(src, "qwen@venue_split")),
        formatting.count(answers.wrong_refusals(src, MERCURY)),
    )


def refusals(doc: Doc, src: Sources) -> None:
    """Gemini's comparison refusals under prompt v1, by what its context held."""
    top_k = formatting.count(answers.answer_top_k(src))
    refused = formatting.count(len(answers.refusal_ids(src)))
    even = answers.even_context_chunks(src)
    doc.check(
        "In {} of the {} comparison refusals of gemini@venue_split the context held chunks of "
        "every named venue ({} of each)",
        formatting.count(len(answers.refusals_with_every_venue(src))),
        refused,
        optional_count(even),
    )
    doc.caption(
        "The {} refusals by what the top {} held, run `results/{run}` against the gold of "
        "`v0.2.1-gold` (deviation D4):",
        (ANSWERS_V1,),
        refused,
        top_k,
    )
    doc.table(
        "| group | refusals |",
        {g: [formatting.count(k), SKIP] for g, k in answers.refusal_group_sizes(src)},
    )
    doc.check(
        "for some named venues; in {} of them the other venue has no gold at all",
        formatting.count(answers.gold_less_refusals(src, "b")),
    )
    doc.check(
        "for none; in {} of them one venue has no gold at all",
        formatting.count(answers.gold_less_refusals(src, "c")),
    )
    doc.check(
        "Of the {} refusals, {} cite chunks as [n] in the refusal text itself, and the {} carry "
        "{} claim between them.",
        refused,
        formatting.count(answers_outcomes.citing_refusals(src)),
        refused,
        formatting.count(answers_outcomes.refusal_claims(src)),
    )
    doc.check(
        "On the same {} questions Qwen answers {} and Mercury {}.",
        refused,
        formatting.count(answers_outcomes.answered_refusals(src, "qwen@venue_split")),
        formatting.count(answers_outcomes.answered_refusals(src, MERCURY)),
    )


def prompt_v2(doc: Doc, src: Sources) -> None:
    """Prompt v2: refusals, partial answers and clarifying questions against v1."""
    rows: dict[str, list[object]] = {
        r.category: [
            formatting.count(r.questions),
            r.expected or SKIP,
            *[change_text(a, b, formatting.count) for a, b in r.changes],
        ]
        for r in answers_outcomes.outcome_table(src, "refused")
    }
    rows["all"] = [
        question_count_text(src),
        f"{formatting.count(datasets.refuse_expected_count(src))} refuse",
        *[
            change_text(a, b, formatting.count)
            for a, b in answers_outcomes.outcome_changes(src, None, "refused")
        ],
    ]
    doc.table("Refusals, v1 to v2", rows)
    doc.claim(
        "Partial answers under v2 (v1 has none)",
        holds=answers_outcomes.no_outcome(src, "partial", PROMPT_V1),
        what="v1 has no partial answer",
    )
    doc.check(
        "comparison {}, {}, {} and {} in the column order above; unanswerable {}, {}, {} and {}; "
        "all categories {}, {}, {} and {}.",
        *[
            formatting.count(v)
            for v in answers_outcomes.outcome_counts(src, "comparison", "partial", PROMPT_V2)
        ],
        *[
            formatting.count(v)
            for v in answers_outcomes.outcome_counts(src, "unanswerable", "partial", PROMPT_V2)
        ],
        *[
            formatting.count(v)
            for v in answers_outcomes.outcome_counts(src, None, "partial", PROMPT_V2)
        ],
    )


def clarification_claims(doc: Doc, src: Sources) -> None:
    """Clarifying questions under prompt v1 and v2."""
    doc.check(
        "Clarifying questions: Qwen {} to {} and Mercury {} to {}",
        *[formatting.count(v) for v in answers_outcomes.clarifications(src, "qwen@venue_split")],
        *[formatting.count(v) for v in answers_outcomes.clarifications(src, MERCURY)],
    )
    doc.claim(
        "each on an ambiguous question under v2; Gemini asks none in either run.",
        holds=answers_outcomes.clarifications_only_on(
            src, ("qwen@venue_split", MERCURY), "ambiguous"
        )
        and answers_outcomes.never_clarify(src, ("gemini@venue_split", "gemini@hybrid_rerank")),
        what="v2 clarifications on ambiguous questions only; Gemini asks none",
    )


def wrong_refusal_claims(doc: Doc, src: Sources) -> None:
    """Prompt v2's wrong refusals, and what became of Gemini's refusals."""
    doc.claim(
        "### Prompt v2 lowers wrong refusals in every configuration (exploratory)",
        holds=answers_outcomes.wrong_refusals_fall(src),
        what="every configuration has fewer wrong refusals under v2",
    )
    doc.check(
        "{} questions), v1 to v2: gemini@venue_split {} to {}, qwen {} to {}, mercury {} to {}, "
        "gemini@hybrid_rerank {} to {}.",
        formatting.count(datasets.answer_expected_count(src)),
        *[
            formatting.count(v)
            for change in answers_outcomes.wrong_refusal_changes(src)
            for v in change
        ],
    )
    a = answers_outcomes.group_outcomes(src, "a")
    b = answers_outcomes.group_outcomes(src, "b")
    c = answers_outcomes.group_outcomes(src, "c")
    doc.check(
        "Of gemini@venue_split's {} v1 comparison refusals, v2 gives group a {} partial and "
        "{} refusal, group b {} partial and {} refusals, group c {} full, {} partial and {} "
        "refusals.",
        formatting.count(len(answers.refusal_ids(src))),
        formatting.count(a["partial"]),
        formatting.count(a["refused"]),
        formatting.count(b["partial"]),
        formatting.count(b["refused"]),
        formatting.count(c["full"]),
        formatting.count(c["partial"]),
        formatting.count(c["refused"]),
    )


def mercury(doc: Doc, src: Sources) -> None:
    """Mercury on the unanswerable questions, and the absence form under v2."""
    size = formatting.count(datasets.category_count(src, "unanswerable"))
    before, after = answers_outcomes.judged_behaviour(src, MERCURY, "unanswerable")
    doc.check(
        "Mercury refuses {} of the {} unanswerable questions under v2, against {} under v1. "
        "Opus grades its behavior correct on {} of them under v2, against {} under v1. The {} "
        "unanswerable questions are a category of fewer than {}, so under methodology \u00a77 "
        "the counts carry no p-value. {} of its {} non-refusals are partial answers, and {} of "
        "its answers flagged partial carry no claim",
        formatting.count(answers.outcome_count(src, MERCURY, "unanswerable", "refused", PROMPT_V2)),
        size,
        formatting.count(answers.outcome_count(src, MERCURY, "unanswerable", "refused", PROMPT_V1)),
        formatting.count(after),
        formatting.count(before),
        size,
        min_tested(),
        formatting.count(answers.outcome_count(src, MERCURY, "unanswerable", "partial", PROMPT_V2)),
        formatting.count(answers_outcomes.non_refusals(src, MERCURY, "unanswerable", PROMPT_V2)),
        formatting.count(len(answers.claimless_partial_configs(src))),
    )
    doc.claim(
        "outcome rule counts those as refusals in every table here",
        holds=answers.claimless_partials_all_from(src, MERCURY),
        what="every claim-less partial answer is Mercury's",
    )
    absent, claims = answers_outcomes.absence_claims(src)
    doc.check(
        '{} of the {} v2 claims contain "does not list"',
        formatting.count(absent),
        formatting.count(claims),
    )
    in_table = answers.attributed_claims(src, PROMPT_V2)
    doc.claim(
        "The attribution table above counts {} fewer: those {} claims are in answers whose "
        "judgment failed.",
        formatting.count(claims - in_table),
        formatting.count(answers_outcomes.v2_unjudged_claims(src)),
        holds=claims - in_table == answers_outcomes.v2_unjudged_claims(src),
        what="the v2 claims the attribution table leaves out are those of failed judgments",
    )


def cost_claims(doc: Doc, src: Sources) -> None:
    """Cost and generator call duration per configuration, v1 to v2."""
    doc.table(
        "Cost and generator call duration per configuration, v1 to v2,",
        {
            row.config: [
                *[
                    change_text(v1, v2, formatting.cost_figure)
                    for v1, v2 in (row.generator_usd, row.judge_usd)
                ],
                *[
                    change_text(v1, v2, formatting.call_seconds)
                    for v1, v2 in (row.p50_s, row.p95_s)
                ],
                change_text(*row.output_tokens, formatting.count),
                formatting.count(row.invalid_responses),
            ]
            for row in answers_costs.cost_rows(src)
        },
    )
    doc.caption(
        "v1 to v2, the {} questions of `v0.2.1-gold`. Generator columns are from the answer runs "
        "`results/{run}` and `results/{run}`, the judge column from the judged runs "
        "`results/{run}` and `results/{run}`.",
        (ANSWERS_V1, ANSWERS_V2, JUDGED_V1, JUDGED_V2),
        question_count_text(src),
    )
    doc.check(
        "The judge costs {} to {} times what the generator costs per answer.",
        *[
            formatting.ratio(v, formatting.COST_RATIO_DECIMALS)
            for v in answers_costs.judge_cost_ratios(src)
        ],
    )


def judge_cost(doc: Doc, src: Sources) -> None:
    """What judging and calibrating cost, and the slow calls."""
    judge = src.judged_v1.config.generation.chosen_judge()
    judged1, judged2 = (
        answers_costs.judge_calls(src, PROMPT_V1),
        answers_costs.judge_calls(src, PROMPT_V2),
    )
    doc.check(
        "${} for v1 ({} calls served from the cache, the {} calibration answers among them) and "
        "${} for v2 ({} cached). The p50 per judge call is {} s for v1 and {} s for v2.",
        formatting.cost_figure(judged1.paid_usd),
        formatting.count(judged1.cached),
        formatting.count(calibration.answers_rated(src)),
        formatting.cost_figure(judged2.paid_usd),
        formatting.count(judged2.cached),
        formatting.call_seconds(judged1.p50_s),
        formatting.call_seconds(judged2.p50_s),
    )
    doc.check(
        "The calibration cost ${} (Sonnet), ${} (Opus) and ${} (the Opus repeat). The {} "
        "judgments that stayed invalid",
        formatting.cost_figure(calibration.judgement_cost(src, calibration.other_candidate(src))),
        formatting.cost_figure(calibration.judgement_cost(src, judge)),
        formatting.cost_figure(calibration.judgement_cost(src, judge, 1)),
        formatting.count(answers_costs.invalid_judgements(src)),
    )
    doc.check(
        "The gemini@hybrid_rerank configuration shares {} cached calls with gemini@venue_split, "
        "so v2 spent ${} on uncached calls.",
        formatting.count(answers_costs.cached_calls(src, "gemini@hybrid_rerank", PROMPT_V2)),
        formatting.cost_figure(answers_costs.uncached_cost(src, PROMPT_V2)),
    )
    v2, v1 = answers_costs.slow_calls(src, PROMPT_V2), answers_costs.slow_calls(src, PROMPT_V1)
    doc.check(
        "Mercury's p95 is its calls with an invalid response: {} of its {} such calls took over "
        "{} s under v2 (the longest {} s, with {} invalid responses), and {} of {} under v1.",
        formatting.count(v2.over),
        formatting.count(v2.calls),
        formatting.count(answers_costs.SLOW_CALL_S),
        formatting.call_seconds(v2.longest_s),
        formatting.count(v2.longest_invalid),
        formatting.count(v1.over),
        formatting.count(v1.calls),
    )
