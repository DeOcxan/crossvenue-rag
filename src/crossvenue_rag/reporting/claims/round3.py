"""What the documents claim about annotation round 3.

The sections of evals/agreement.md and evals/datasheet.md, the methodology's D27, D28 and section 5b
record, and the README's lines on round 3. docs/results/findings.md has its own
(`findings.round3`).
"""

from __future__ import annotations

from ... import formatting
from ...dataset.journal import FIRST_ROUND, FULL_RELABEL_ROUND, RETEST_ROUND
from ...dataset.sampling import cited_words
from ..check import Doc
from ..compute import agreement, gold_sheet, round3, sensitivity
from ..compute.estimates import Estimate, holm_or_one
from ..run_ids import RETRIEVAL_RUN, ROUND3_JUDGED_V1, ROUND3_RUN, SENSITIVITY_RUN
from ..sources import Sources
from .comparisons import comparison_questions, p_bound, signed_parts

SESSION_ROUNDS = (FIRST_ROUND, RETEST_ROUND, FULL_RELABEL_ROUND)


def require_cited(doc: Doc, src: Sources, questions: list[str]) -> None:
    """Each question's citation, against the question file (docs/style.md rule 38)."""
    gold = round3.gold(src)
    for question in questions:
        doc.definition(
            f'{question} "{cited_words(gold[question])}"', reason="the first words of the record"
        )


def rounds_section(doc: Doc, src: Sources) -> None:
    """evals/agreement.md: rounds 1 and 3 across the criterion revision, and the round times."""
    a = round3.rounds_agreement(src)
    t, change = a.total, round3.gold_change(src)
    doc.check(
        "alone marked {} pairs and round {} alone marked {}, over {} pairs",
        formatting.count(t.first_only),
        formatting.own_form(FULL_RELABEL_ROUND),
        formatting.count(t.second_only),
        formatting.count(t.pairs),
    )
    doc.check("judged all {} questions again, blind, on", formatting.count(a.questions))
    doc.check(
        "\u03ba {} stays the only test-retest figure",
        formatting.metric(agreement.registered(src).report.kappa),
    )
    doc.check(
        "over the journal `evals/datasets/annotations/v0.3-gold.rounds.json`, {} questions:",
        formatting.count(a.questions),
    )
    doc.table(
        "Verdicts over the pairs both rounds judged",
        {
            "pairs": [formatting.count(t.pairs)],
            "both relevant": [formatting.count(t.both_relevant)],
            "both not relevant": [formatting.count(t.both_irrelevant)],
            "relevant in round 1 only": [formatting.count(t.first_only)],
            "relevant in round 3 only": [formatting.count(t.second_only)],
            "raw agreement": [formatting.metric(t.raw_agreement)],
        },
    )
    doc.check(
        "Per category, from the journal of `v0.3-gold`, {} pairs over {} questions, most "
        "disagreement first:",
        formatting.count(t.pairs),
        formatting.count(a.questions),
    )
    doc.table(
        "Per category, from the journal of `v0.3-gold`",
        {
            category: [
                formatting.count(v.pairs),
                formatting.count(v.disagreements),
                formatting.count(v.first_only),
                formatting.count(v.second_only),
                formatting.metric(v.raw_agreement),
            ]
            for category, v in a.by_category
        },
    )
    doc.check(
        "its gold goes from {} to {} chunks: {} removed and {} added, on {} of {} questions",
        formatting.count(change.before),
        formatting.count(change.after),
        formatting.count(change.removed),
        formatting.count(change.added),
        formatting.count(change.questions_changed),
        formatting.count(change.questions),
    )
    third = round3.pace(src, FULL_RELABEL_ROUND)
    doc.check(
        "took {} s at a median of {} s per question",
        formatting.count(int(third.session_seconds)),
        formatting.number(third.per_question, 1),
    )
    doc.definition(
        "A gap is the time between two consecutive entries", reason="the definition of a gap"
    )
    paces = {number: round3.pace(src, number) for number in SESSION_ROUNDS}
    doc.check(
        "Per round, from the journal of `v0.3-gold`, {}, {} and {} questions:",
        *[formatting.count(p.questions) for p in paces.values()],
    )
    rows: dict[str, list[object]] = {}
    for number, p in paces.items():
        rows[str(number)] = [
            formatting.count(p.questions),
            formatting.count(p.decisions),
            formatting.count(int(p.session_seconds)),
            formatting.count(int(p.longest_gap_seconds)),
            formatting.number(p.per_question, 1),
            formatting.number(p.per_decision, 2),
        ]
    doc.table("| Round | Questions | Candidate decisions | Session (s) |", rows)


def version_section(doc: Doc, src: Sources) -> None:
    """evals/datasheet.md: `v0.3-gold` beside `v0.2.2-gold`, and its empty and flagged questions."""
    old, new = agreement.adjudicated_gold(src), round3.gold(src)
    change = round3.gold_change(src)
    doc.check(
        "It holds the same {} questions in the same order, with the", formatting.count(len(new))
    )
    counts = agreement.version_counts(old), agreement.version_counts(new)
    values = [gold_sheet.coverage_values(src, list(q.values())) for q in (old, new)]
    doc.table(
        "| Count | `v0.2.2-gold` | `v0.3-gold` |",
        {
            "questions with gold": [formatting.count(c.with_gold) for c in counts],
            "gold chunks": [formatting.count(c.gold_chunks) for c in counts],
            "venues covered": [formatting.count(v["covered"]) for v in values],
            "venues missing from the pool": [
                formatting.count(v["missing_from_pool"]) for v in values
            ],
            "venues absent from the docs": [formatting.count(v["absent_in_docs"]) for v in values],
        },
    )
    doc.check(
        "The gold of {} questions differs from `v0.2.2-gold`: {} chunks added and {} removed.",
        formatting.count(change.questions_changed),
        formatting.count(change.added),
        formatting.count(change.removed),
    )
    left, kept = round3.left_gold(src), round3.stayed_empty(src)
    doc.check(
        "{} answerable questions have gold in `v0.2.2-gold` and empty gold in `v0.3-gold`.",
        formatting.count(len(left)),
    )
    doc.check(
        "{} other answerable questions have empty gold in both versions, so `v0.3-gold` holds {}:",
        formatting.count(len(kept)),
        formatting.count(len(left) + len(kept)),
    )
    doc.check("These {} have no recall in a run.", formatting.count(len(left) + len(kept)))
    doc.check("flagged {} unanswerable question", formatting.count(len(round3.flagged(src))))
    require_cited(doc, src, left + kept + round3.joined_gold(src) + round3.flagged(src))


def deviation(doc: Doc, src: Sources) -> None:
    """The methodology: D27 on `unjudged@k`, the section 5b record row and D28."""
    doc.claim(
        "`unjudged@k` on the {} questions with gold in both `v0.2.2-gold` and `v0.3-gold`",
        formatting.count(len(round3.shared_gold_questions(src))),
        holds=round3.unjudged_equal(src),
        what="unjudged@k agrees per question on the questions with gold in both versions",
    )
    doc.claim(
        "{} of the {} `unjudged@k` means in `summary.csv` differ.",
        formatting.count(round3.unjudged_means_differ(src)),
        formatting.count(round3.unjudged_mean_cells(src)),
        holds=round3.gold_free_equal(src),
        what="every retrieved list and gold-free metric equals the v0.2.2-gold run",
    )
    moved = round3.left_gold(src) + round3.joined_gold(src)
    doc.check(
        "and {} questions left or joined those:",
        formatting.count(len(moved)),
    )
    require_cited(doc, src, moved)
    doc.check(
        "The `unjudged@k` means of `v0.3-gold` describe its {} questions with gold.",
        formatting.count(agreement.version_counts(round3.gold(src)).with_gold),
    )
    doc.ref("checked against `results/{}` |", SENSITIVITY_RUN)
    doc.check(
        "| Round 3, the full relabel, registered in \u00a75b: all {} questions judged, frozen",
        formatting.count(len(round3.gold(src))),
    )
    doc.ref("the check failed on `results/{}`", ROUND3_RUN)
    doc.check(
        "`make kappa` passes no round and prints \u03ba {} for rounds {} and {}.",
        formatting.metric(agreement.registered(src).report.kappa),
        formatting.own_form(FIRST_ROUND),
        formatting.own_form(RETEST_ROUND),
    )


def status_line(doc: Doc, src: Sources) -> None:
    """README.md: the lines on round 3 in Status and Limitations."""
    a = round3.rounds_agreement(src)
    change = round3.gold_change(src)
    doc.check("judged all {} questions again, blind", formatting.count(a.questions))
    doc.check(
        "{} gold chunks against {} in `v0.2.2-gold`.",
        formatting.count(change.after),
        formatting.count(change.before),
    )
    doc.check(
        "Round 3 moved the gold from {} chunks in `v0.2.2-gold` to {} in `v0.3-gold`",
        formatting.count(change.before),
        formatting.count(change.after),
    )


def results_line(doc: Doc, src: Sources) -> None:
    """README.md: the sentence on `v0.3-gold` below the results table."""
    changed, tested = sensitivity.retrieval_changes(src)
    primary = sensitivity.primary_comparison(src)
    primary_row = sensitivity.paired_row(src, ROUND3_RUN, primary)
    doc.claim(
        "venue_split - hybrid_rerank on venue_coverage@10 over the {} comparison questions is "
        "unchanged, {} [{}, {}], McNemar p < {} (",
        comparison_questions(src),
        *signed_parts(Estimate.of(primary_row)),
        p_bound(),
        holds=primary not in changed
        and sensitivity.identical(src, primary, RETRIEVAL_RUN, ROUND3_RUN)
        and primary_row.value("mcnemar_p") < formatting.SMALLEST_SHOWN_P,
        what=f"the retrieval primary on v0.3-gold keeps its conclusion and the difference and "
        f"interval of {RETRIEVAL_RUN}, McNemar p < {p_bound()}",
    )
    doc.ref("([`results/{}`](results/{}/report.md)). ", ROUND3_RUN, ROUND3_RUN)
    doc.check(
        "{} of the {} retrieval comparisons over the comparison, multi-venue and labeled "
        "questions change their reading",
        formatting.count(len(changed)),
        formatting.count(tested),
    )
    row = round3.answer_primary_row(src, src.round3_judged_v1)
    doc.check(
        "gemini@venue_split - gemini@hybrid_rerank on coverage over the {} comparison questions "
        "is {} [{}, {}], p_Holm = {} (",
        comparison_questions(src),
        *signed_parts(Estimate.of(row)),
        formatting.p_value(holm_or_one(row)),
    )
    doc.ref("([`results/{}`](results/{}/report.md), [findings]", ROUND3_JUDGED_V1, ROUND3_JUDGED_V1)
