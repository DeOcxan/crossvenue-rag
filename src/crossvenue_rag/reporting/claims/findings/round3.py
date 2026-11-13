"""What docs/results/findings.md claims about `v0.3-gold`, beside the registered results."""

from __future__ import annotations

from .... import formatting
from ....config_eval import LABELLED_SCOPE
from ....dataset.journal import FIRST_ROUND, FULL_RELABEL_ROUND
from ....evaluation.power import TARGET_POWER_PERCENT
from ...check import Doc
from ...compute import power, retrieval, round3, sensitivity
from ...compute.estimates import Estimate, holm_or_one
from ...outcomes import Param
from ...run_ids import (
    JUDGED_V1,
    JUDGED_V2,
    POWER_RUN,
    PROMPTS_RUN,
    ROUND3_JUDGED_V1,
    ROUND3_JUDGED_V2,
    ROUND3_PROMPTS_RUN,
    ROUND3_RUN,
    SENSITIVITY_BASE_RUN,
    SENSITIVITY_RUN,
)
from ...sources import AnswerRun, Sources
from ..comparisons import (
    comparison_questions,
    estimate,
    level,
    signed_parts,
    significance,
)
from ..reasons import INTERVAL_LEVEL
from .rounds import comparison_cells

RUNS = (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN, ROUND3_RUN)


def retrieval_section(doc: Doc, src: Sources) -> None:
    """The six methods on `v0.3-gold` beside the two runs on earlier versions."""
    changed, tested = sensitivity.retrieval_changes(src)
    doc.check(
        "## On v0.3-gold the primary comparison is unchanged and {} of the {} retrieval "
        "comparisons change their reading",
        formatting.count(len(changed)),
        formatting.count(tested),
    )
    doc.ref(
        "Runs `results/{}` on `v0.2.1-gold`, `results/{}` on `v0.2.2-gold` and `results/{}` on "
        "`v0.3-gold`, one config",
        *RUNS,
    )
    sizes = [sensitivity.tested_questions(src, run, LABELLED_SCOPE) for run in RUNS]
    doc.check(
        "on the CPU, {} methods, k = {}, over {} comparison questions, {} multi-venue ones and "
        "the labeled ones: {}, {} and {}.",
        formatting.count(len(src.run(ROUND3_RUN).methods)),
        formatting.count(retrieval.top_k(src, ROUND3_RUN)),
        comparison_questions(src),
        formatting.count(sensitivity.tested_questions(src, ROUND3_RUN, "multi-venue")),
        *(formatting.count(n) for n in sizes),
    )
    doc.ref("venue_coverage@k, off_venue@k and via_ccxt@k equals `results/{}`.", SENSITIVITY_RUN)
    doc.claim(
        "unjudged@k equals it per question on the {} questions with gold in both versions",
        formatting.count(len(round3.shared_gold_questions(src))),
        holds=round3.unjudged_equal(src) and round3.gold_free_equal(src),
        what="the lists, the gold-free metrics and per-question unjudged@k match",
    )
    _direction(doc, src)
    doc.check("Each method on the {} comparison questions, in", comparison_questions(src))
    doc.ref(
        "questions, in `results/{}` on `v0.2.1-gold`, `results/{}` on `v0.2.2-gold` and "
        "`results/{}` on `v0.3-gold`:",
        *RUNS,
    )
    doc.table(
        "| method | recall@10 v0.2.1 | recall@10 v0.2.2 | recall@10 v0.3 |",
        {
            m: [
                formatting.metric(sensitivity.run_metric(src, run, "comparison", m, column))
                for column in ("recall@10", "ndcg@10")
                for run in RUNS
            ]
            for m in retrieval.METHODS
        },
    )
    _comparisons(doc, src, tested, len(changed))


def _direction(doc: Doc, src: Sources) -> None:
    """The change in gold and the per-category counts, as facts."""
    a, change = round3.rounds_agreement(src), round3.gold_change(src)
    doc.check(
        "removes {} gold chunks and adds {}, from {} to {}, on {} of {} questions.",
        formatting.count(change.removed),
        formatting.count(change.added),
        formatting.count(change.before),
        formatting.count(change.after),
        formatting.count(change.questions_changed),
        formatting.count(change.questions),
    )
    doc.check(
        "Of the {} pairs rounds {} and {} both judged, in the journal of `v0.3-gold`, over {} "
        "questions, per category:",
        formatting.count(a.total.pairs),
        formatting.own_form(FIRST_ROUND),
        formatting.own_form(FULL_RELABEL_ROUND),
        formatting.count(a.questions),
    )
    doc.table(
        "| Category | Relevant in round 1 only | Relevant in round 3 only |",
        {
            category: [formatting.count(v.first_only), formatting.count(v.second_only)]
            for category, v in a.by_category
        },
    )


def _comparisons(doc: Doc, src: Sources, tested: int, changed: int) -> None:
    """The primary and `recall@10` comparisons on `v0.2.1-gold` and `v0.3-gold`."""
    runs = (SENSITIVITY_BASE_RUN, ROUND3_RUN)
    rows = sensitivity.tested_comparisons(src)
    recall = [c for c in rows if c.metric == "recall@10"]
    doc.check("The primary and {} recall@10 comparisons of", formatting.count(len(recall)))
    doc.ref("comparisons of `results/{}` (`v0.2.1-gold`) and `results/{}` (`v0.3-gold`) on", *runs)
    doc.check(
        "(`v0.3-gold`) on the {} comparison and the {} and {} questions with gold, {}% paired "
        "bootstrap intervals, p_Holm (unadjusted for the primary):",
        comparison_questions(src),
        *(formatting.count(sensitivity.tested_questions(src, run, LABELLED_SCOPE)) for run in runs),
        Param(level(), INTERVAL_LEVEL),
    )
    doc.table("| comparison | v0.2.1-gold | p | v0.3-gold | p |", comparison_cells(src, runs))
    against_adjudicated = sensitivity.changed_conclusions(src, SENSITIVITY_RUN, ROUND3_RUN)
    doc.claim(
        "Of the {} primary and secondary comparisons, {} change sign, significance at "
        "p_Holm < {} or whether the interval excludes {}, and the same {} change against",
        formatting.count(tested),
        formatting.count(changed),
        significance(),
        "0",
        formatting.count(len(against_adjudicated)),
        holds=against_adjudicated
        == sensitivity.changed_conclusions(src, SENSITIVITY_BASE_RUN, ROUND3_RUN),
        what="the comparisons that change are the same against both earlier versions",
    )
    doc.claim(
        "The interval of hybrid_rerank - hybrid over the labeled questions includes {} on",
        "0",
        holds=Estimate.of(
            sensitivity.paired_row(
                src,
                ROUND3_RUN,
                sensitivity.Comparison("hybrid_rerank", "hybrid", "recall@10", "labelled"),
            )
        ).includes_zero(),
        what="that comparison's interval includes 0 on v0.3-gold",
    )
    doc.check("Neither has p_Holm below {} on any version", significance())
    reranker = sensitivity.Comparison("hybrid_rerank", "hybrid", "recall@10", "labelled")
    oracle = sensitivity.Comparison(
        "venue_split_oracle", "hybrid_rerank", "recall@10", "labelled comparison"
    )
    reranker_rows = [sensitivity.paired_row(src, run, reranker) for run in runs]
    doc.check(
        "On `v0.2.1-gold` and `v0.3-gold`, hybrid_rerank - hybrid is {} [{}, {}] and {} [{}, {}], "
        "p_Holm {} and {}, and venue_split_oracle - hybrid_rerank is {} [{}, {}] and {} [{}, {}]",
        *[part for row in reranker_rows for part in signed_parts(Estimate.of(row))],
        *[formatting.p_value(sensitivity.tested_p(row)) for row in reranker_rows],
        *[
            part
            for run in runs
            for part in signed_parts(Estimate.of(sensitivity.paired_row(src, run, oracle)))
        ],
    )
    doc.claim(
        "; on `v0.2.1-gold` the test of recall@10 on the {} comparison questions reaches {}% power "
        "from a gap of {}",
        formatting.count(src.power.questions),
        formatting.count(TARGET_POWER_PERCENT),
        formatting.design_parameter(power.detectable_gap(src, "holm_first_step")),
        holds=("venue_split_oracle", "hybrid_rerank") in src.power.family,
        what="the power run's recall family holds venue_split_oracle - hybrid_rerank",
    )
    doc.ref("(`results/{}`). On `v0.3-gold`", POWER_RUN)
    bm25 = sensitivity.Comparison("hybrid_rerank", "bm25", "recall@10", "labelled comparison")
    row = sensitivity.paired_row(src, ROUND3_RUN, bm25)
    doc.claim(
        "On `v0.3-gold` hybrid_rerank - bm25 on recall@10 over the comparison questions has an "
        "interval that excludes {} with p_Holm above {}: {} [{}, {}], p_Holm {}, inconclusive at "
        "this size.",
        "0",
        significance(),
        *signed_parts(Estimate.of(row)),
        formatting.p_value(sensitivity.tested_p(row)),
        holds=not Estimate.of(row).includes_zero(),
        what="that comparison's interval excludes 0 on v0.3-gold",
    )


def answers_section(doc: Doc, src: Sources) -> None:
    """The judged runs on `v0.3-gold`: the primary, the rows that change, and H-v2."""
    old = round3.answer_primary_row(src, src.judged_v1)
    new = round3.answer_primary_row(src, src.round3_judged_v1)
    v1 = round3.answer_changes(src.judged_v1, src.round3_judged_v1)
    v2 = round3.answer_changes(src.judged_v2, src.round3_judged_v2)
    tested = formatting.count(round3.tested_answer_rows(src.judged_v1))
    doc.check(
        "## On v0.3-gold the primary answer comparison moves from p_Holm {} to {}, and {} of the "
        "{} tested prompt v1 rows change their reading",
        formatting.p_value(holm_or_one(old)),
        formatting.p_value(holm_or_one(new)),
        formatting.count(len(v1)),
        tested,
    )
    doc.ref(
        "Registered results stay on `v0.2.1-gold`, in `results/{}` and `results/{}`,",
        JUDGED_V1,
        JUDGED_V2,
    )
    doc.ref(
        "over the same answers in `results/{}` (prompt v1) and `results/{}` (prompt v2, "
        "exploratory),",
        ROUND3_JUDGED_V1,
        ROUND3_JUDGED_V2,
    )
    doc.check("{} answers each.", formatting.count(len(src.round3_judged_v1.records)))
    doc.check(
        "The runs made ${} and ${} of judge calls.",
        formatting.cost_figure(round3.new_judge_cost(src.round3_judged_v1)),
        formatting.cost_figure(round3.new_judge_cost(src.round3_judged_v2)),
    )
    first = round3.rejudged(src.judged_v1, src.round3_judged_v1)
    second = round3.rejudged(src.judged_v2, src.round3_judged_v2)
    doc.check(
        "chosen after these runs were read: the {} answers judged anew under prompt v1 and the {} "
        "under prompt v2 have a verdict in both runs. Faithfulness changed on {} and {} of them "
        "and venue attribution on {} and {}. A further {} and {} had no verdict in the registered "
        "runs, whose judge calls failed on them.",
        formatting.count(first.answers),
        formatting.count(second.answers),
        formatting.count(first.faithfulness),
        formatting.count(second.faithfulness),
        formatting.count(first.attribution),
        formatting.count(second.attribution),
        formatting.count(first.first_verdicts),
        formatting.count(second.first_verdicts),
    )
    doc.check(
        "on coverage over the {} comparison questions, is {} [{}, {}], p_Holm = {}, on "
        "`v0.2.1-gold` and {} [{}, {}], p_Holm = {}, on `v0.3-gold`, {}% paired bootstrap "
        "intervals.",
        comparison_questions(src),
        *signed_parts(Estimate.of(old)),
        formatting.p_value(holm_or_one(old)),
        *signed_parts(Estimate.of(new)),
        formatting.p_value(holm_or_one(new)),
        Param(level(), INTERVAL_LEVEL),
    )
    doc.check(
        "Under prompt v1, {} of the {} primary and secondary rows change sign, significance at "
        "p_Holm < {} or whether the interval excludes {}.",
        formatting.count(len(v1)),
        tested,
        significance(),
        "0",
    )
    doc.check(
        "In both tables below, a calibrated row's cell whose interval excludes {} with p_Holm "
        "at or above {} is inconclusive at this size.",
        "0",
        significance(),
    )
    _changes(
        doc, src.judged_v1, src.round3_judged_v1, "The {} prompt v1 rows, in `results/", len(v1)
    )
    doc.ref(
        "prompt v1 rows, in `results/{}` on `v0.2.1-gold` and `results/{}` on `v0.3-gold`,",
        JUDGED_V1,
        ROUND3_JUDGED_V1,
    )
    doc.check(
        "Under prompt v2 (exploratory), {} of the {} rows change.",
        formatting.count(len(v2)),
        tested,
    )
    _changes(
        doc, src.judged_v2, src.round3_judged_v2, "The {} prompt v2 rows, in `results/", len(v2)
    )
    doc.ref(
        "prompt v2 rows, in `results/{}` on `v0.2.1-gold` and `results/{}` on `v0.3-gold`,",
        JUDGED_V2,
        ROUND3_JUDGED_V2,
    )
    _wrong_refusals(doc, src)


def _changes(doc: Doc, old: AnswerRun, new: AnswerRun, template: str, rows: int) -> None:
    doc.check(template, formatting.count(rows))
    doc.check(
        "over the answers each row pairs, counted in its n columns, {}% paired bootstrap "
        "intervals:",
        Param(level(), INTERVAL_LEVEL),
    )
    cells = {}
    for a, b in round3.answer_changes(old, new):
        # An identifier keeps code font (style rule 20); coverage and faithfulness are words.
        metric = f"`{a.metric}`" if "_" in a.metric else a.metric
        cells[f"{a.a} - {a.b}, {metric}, {a.scope}"] = [
            "yes" if b.calibrated else "no",
            formatting.count(a.n),
            estimate(Estimate.of(a)),
            formatting.p_value(holm_or_one(a)),
            formatting.count(b.n),
            estimate(Estimate.of(b)),
            formatting.p_value(holm_or_one(b)),
        ]
    doc.table(f"{template.format(formatting.count(rows))}", cells)


def _wrong_refusals(doc: Doc, src: Sources) -> None:
    """H-v2 of methodology section 9a on both gold versions."""
    old, new = src.prompts.comparison, src.round3_prompts.comparison
    doc.ref(
        "`wrong_refusal` v2 - v1, in `results/{}` on `v0.2.1-gold` and `results/{}` on "
        "`v0.3-gold`,",
        PROMPTS_RUN,
        ROUND3_PROMPTS_RUN,
    )
    doc.check(
        "over the questions that expect an answer or a clarifying question, {}% paired bootstrap",
        Param(level(), INTERVAL_LEVEL),
    )
    rows = [k for k in new if k[1] == "wrong_refusal"]
    doc.check(
        "The count is {} on `v0.3-gold` against {} because {} questions are fact",
        formatting.count(new[rows[0]].n),
        formatting.count(old[rows[0]].n),
        formatting.count(new[rows[0]].n - old[rows[0]].n),
    )
    doc.table(
        "| config | n v0.2.1 | v0.2.1-gold |",
        {
            key[0]: [
                formatting.count(old[key].n),
                estimate(Estimate.of(old[key])),
                formatting.p_value(holm_or_one(old[key])),
                formatting.count(new[key].n),
                estimate(Estimate.of(new[key])),
                formatting.p_value(holm_or_one(new[key])),
            ]
            for key in rows
        },
    )
