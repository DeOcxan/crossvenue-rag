"""What evals/agreement.md claims.

One function per section, in document order.
"""

from __future__ import annotations

from ... import formatting
from ...dataset.agreement import KAPPA_TARGET, MIN_RETEST_DAYS, VerdictTable
from ...stats import Interval
from ..check import Doc
from ..compute import agreement, sensitivity
from ..outcomes import Param
from ..run_ids import SENSITIVITY_BASE_RUN, SENSITIVITY_RUN
from ..sources import Sources
from .comparisons import kappa_bounds, kappa_cell, level, sensitivity_sizes
from .reasons import INTERVAL_LEVEL, REGISTERED

# The readings in the order the readings table lists them, by their first cell.
READINGS = {
    "corrected, all questions (registered)": ("corrected", "all"),
    "corrected, unexposed questions": ("corrected", "unexposed"),
    "first judged, all questions": ("first pass", "all"),
    "first judged, unexposed questions": ("first pass", "unexposed"),
}


def _undefined(interval: Interval | None) -> str:
    if interval is None:
        return formatting.MISSING
    return f"{formatting.count(interval.undefined)} of {formatting.count(interval.resamples)}"


def agreement_doc(doc: Doc, src: Sources) -> None:
    """States the claims of evals/agreement.md."""
    _header(doc, src)
    _registered(doc, src)
    _readings(doc, src)
    _by_category(doc, src)
    _direction(doc, src)
    _sources(doc, src)
    _gap(doc, src)


def _header(doc: Doc, src: Sources) -> None:
    study = agreement.study(src)
    questions = formatting.count(study.report.questions)
    doc.check("round 2 re-judged {} of its questions blind", questions)
    doc.claim(
        "Its round-1 entries for the {} questions are identical to those frozen in",
        questions,
        holds=agreement.round1_is_gold(src),
        what="the working journal's round 1 equals the gold set's journal on the re-judged "
        "questions",
    )
    doc.check(
        "decision: {} (question, chunk) pairs over {} questions.",
        formatting.count(study.report.pairs),
        questions,
    )
    boot = agreement.bootstrap(src)
    doc.check(
        "Intervals are {}% percentile bootstrap intervals over questions: each of {} resamples",
        Param(level(), INTERVAL_LEVEL),
        Param(formatting.count(boot.resamples), REGISTERED),
    )
    doc.check("draws {} questions with replacement", questions)


def _registered(doc: Doc, src: Sources) -> None:
    v = agreement.registered(src)
    r = v.report
    kappa = formatting.metric(r.kappa)
    low, high = kappa_bounds(v)
    target = formatting.own_form(KAPPA_TARGET)
    doc.claim(
        "## Registered estimate: \u03ba {} [{}, {}], below the {} target",
        kappa,
        low,
        high,
        target,
        holds=r.kappa < KAPPA_TARGET,
        what="the registered kappa is below the target",
    )
    doc.check(
        "over all {} sampled questions, {} pairs,",
        formatting.count(r.questions),
        formatting.count(r.pairs),
    )
    doc.given(
        "pairs, with a {}% percentile bootstrap interval over questions:",
        level(),
        reason=INTERVAL_LEVEL,
    )
    doc.table(
        "| Quantity | Value |",
        {
            "both relevant": [formatting.count(r.both_relevant)],
            "both not relevant": [formatting.count(r.both_irrelevant)],
            "disagreements": [formatting.count(r.disagreements)],
            "raw agreement": [formatting.metric(r.observed_agreement)],
            "expected agreement by chance": [formatting.metric(r.expected_agreement)],
            "Cohen's \u03ba": [kappa_cell(v)],
            "resamples with undefined \u03ba": [_undefined(v.interval)],
        },
    )
    doc.claim(
        f'\u03ba {{}} is "{r.interpretation}" on the Landis and Koch bands that [methodology '
        "\u00a75](methodology.md#5-annotator-agreement-with-one-annotator) cites, and the target "
        "registered there is \u03ba >= {}",
        kappa,
        target,
        holds=v.interval is not None
        and v.interval.high is not None
        and v.interval.high >= KAPPA_TARGET > r.kappa,
        what="the kappa is below the target and its interval includes the target",
    )
    doc.check("by their own annotator at \u03ba {} [{}, {}].", kappa, low, high)
    doc.check(
        "ran the {} retrieval methods on `v0.2.2-gold`",
        formatting.count(len(src.run(SENSITIVITY_RUN).methods)),
    )
    doc.ref(
        "on `v0.2.2-gold` (`results/{}`) and on `v0.2.1-gold` (`results/{}`)",
        SENSITIVITY_RUN,
        SENSITIVITY_BASE_RUN,
    )
    doc.check(
        "over the {} comparison questions, the {} multi-venue ones and the {} and {} labeled ones.",
        *sensitivity_sizes(src),
    )
    doc.claim(
        "No comparison changes sign, significance or whether its interval excludes {}",
        "0",
        holds=sensitivity.conclusions_unchanged(src),
        what="every primary and secondary comparison keeps sign, significance and whether its "
        "interval excludes 0",
    )


def _readings(doc: Doc, src: Sources) -> None:
    study = agreement.study(src)
    variants = study.variants
    doc.check(
        "## The {} sensitivity readings lie within {} of the registered \u03ba",
        formatting.count(len(variants) - 1),
        formatting.metric(agreement.sensitivity_spread(src)),
    )
    unexposed = agreement.reading(src, "corrected", "unexposed").report.questions
    doc.check(
        "Methodology \u00a75 registers {} readings.",
        formatting.count(len(variants)),
    )
    doc.check(
        "over all {} questions or over the {} unexposed questions,",
        formatting.count(study.report.questions),
        formatting.count(unexposed),
    )
    doc.check("lists the {} others.", formatting.count(agreement.exposed_count(src)))
    doc.given(
        "pairs and a {}% percentile bootstrap interval over questions:",
        level(),
        reason=INTERVAL_LEVEL,
    )
    rows = {}
    for label, (reference, scope) in READINGS.items():
        v = agreement.reading(src, reference, scope)
        rows[label] = [
            formatting.count(v.report.questions),
            formatting.count(v.report.pairs),
            formatting.metric(v.report.observed_agreement),
            kappa_cell(v),
            _undefined(v.interval),
        ]
    doc.table("| Reading | Questions |", rows)
    doc.claim(
        "Each reading's interval includes the registered point estimate.",
        holds=agreement.intervals_hold_registered(src),
        what="every reading's interval includes the registered kappa",
    )


def _category_kappa(table: VerdictTable, kappa: float) -> str:
    """The category's kappa, or `formatting.MISSING` when no pair is relevant in either round."""
    return (
        formatting.MISSING
        if table.both_relevant + table.disagreements == 0
        else formatting.metric(kappa)
    )


def _by_category(doc: Doc, src: Sources) -> None:
    study = agreement.study(src)
    by = dict(study.by_category)
    directions = dict(study.direction_by_category)
    worst, *_ = study.by_category
    lowest = min(study.by_category, key=lambda item: item[1].kappa)
    doc.claim(
        "## Comparison holds {} of the {} disagreements, and ambiguous has the lowest \u03ba",
        formatting.count(by["comparison"].disagreements),
        formatting.count(study.report.disagreements),
        holds=worst[0] == "comparison" and lowest[0] == "ambiguous",
        what="comparison has the most disagreements and ambiguous the lowest kappa",
    )
    doc.check(
        "{} pairs over {} questions, most disagreement first; a category of one",
        formatting.count(study.report.pairs),
        formatting.count(study.report.questions),
    )
    doc.table(
        "| Category | Questions | Pairs | Disagreements | Raw agreement | \u03ba |",
        {
            category: [
                formatting.count(r.questions),
                formatting.count(r.pairs),
                formatting.count(r.disagreements),
                formatting.metric(r.observed_agreement),
                _category_kappa(directions[category], r.kappa),
            ]
            for category, r in study.by_category
        },
    )
    none = directions["unanswerable"]
    doc.claim(
        "The unanswerable questions' {} pairs are not relevant in both rounds, so \u03ba is "
        "undefined there (`crossvenue kappa` prints {} for this case).",
        formatting.count(by["unanswerable"].pairs),
        formatting.metric(by["unanswerable"].kappa),
        holds=none.both_relevant + none.disagreements == 0,
        what="no unanswerable pair is relevant in either round",
    )


def _direction(doc: Doc, src: Sources) -> None:
    study = agreement.study(src)
    d = study.direction
    first, second = agreement.relevant_counts(src)
    doc.claim(
        "## Round 2 marks more pairs relevant: {} of the {} disagreements",
        formatting.count(d.second_only),
        formatting.count(d.disagreements),
        holds=second > first,
        what="round 2 marks more pairs relevant than round 1",
    )
    doc.check(
        "Round 1 marked {} of the {} pairs relevant and round 2 marked {}.",
        formatting.count(first),
        formatting.count(study.report.pairs),
        formatting.count(second),
    )
    doc.check(
        "Of the {} disagreements, {} are relevant in round 1 only and {} in round 2 only.",
        formatting.count(d.disagreements),
        formatting.count(d.first_only),
        formatting.count(d.second_only),
    )
    doc.check(
        "on `v0.2.1-gold`, {} pairs over {} questions:",
        formatting.count(study.report.pairs),
        formatting.count(study.report.questions),
    )
    rows = {
        category: [
            formatting.count(t.disagreements),
            formatting.count(t.first_only),
            formatting.count(t.second_only),
        ]
        for category, t in [*study.direction_by_category, ("total", d)]
    }
    doc.table("| Category | Disagreements | Relevant in round 1 only |", rows)


def _sources(doc: Doc, src: Sources) -> None:
    study = agreement.study(src)
    s = study.sources.verdicts
    judged = s.both_relevant + s.disagreements + s.both_irrelevant
    doc.check(
        "## Source chunks, marked only on round 1's screen, are {} of the {} round-1-only pairs",
        formatting.count(s.first_only),
        formatting.count(study.direction.first_only),
    )
    doc.check(
        "The {} questions list {} source chunks (`source_chunk_ids`). {} of them were among the "
        "candidates both rounds judged; the other {} were in neither round's candidates. "
        "Verdicts on the {} source chunks, from the journal",
        formatting.count(study.report.questions),
        formatting.count(judged + study.sources.unjudged),
        formatting.count(judged),
        formatting.count(study.sources.unjudged),
        formatting.count(judged),
    )
    doc.table(
        "| Verdicts | Source chunks |",
        {
            "relevant in both rounds": [formatting.count(s.both_relevant)],
            "relevant in round 1 only": [formatting.count(s.first_only)],
            "relevant in round 2 only": [formatting.count(s.second_only)],
            "not relevant in either round": [formatting.count(s.both_irrelevant)],
        },
    )
    doc.claim(
        "Round 1 marked all {} source chunks relevant and round 2 marked {} of them.",
        formatting.count(judged),
        formatting.count(s.both_relevant + s.second_only),
        holds=s.both_relevant + s.first_only == judged,
        what="round 1 marked every judged source chunk relevant",
    )
    doc.check(
        "The {} source chunks relevant in round 1 only are {} of the {} round-1-only "
        "disagreements. The other {} round-1-only pairs and all {} round-2-only pairs are "
        "chunks the marker never flagged.",
        formatting.count(s.first_only),
        formatting.count(s.first_only),
        formatting.count(study.direction.first_only),
        formatting.count(study.direction.first_only - s.first_only),
        formatting.count(study.direction.second_only),
    )
    doc.claim(
        "all {} round-2-only pairs are chunks the marker never flagged",
        formatting.count(study.direction.second_only),
        holds=s.second_only == 0,
        what="no source chunk is relevant in round 2 only",
    )


def _gap(doc: Doc, src: Sources) -> None:
    gaps = agreement.study(src).gaps.values()
    low, high = formatting.number(min(gaps), 1), formatting.number(max(gaps), 1)
    doc.check("## The retest gap is {} to {} days", low, high)
    doc.claim(
        "Each question's round-1 and round-2 judgments are {} to {} days apart",
        low,
        high,
        holds=min(gaps) >= MIN_RETEST_DAYS,
        what="every retest gap meets the registered minimum",
    )
    doc.check(
        "That meets the registered minimum of {} days, so the",
        Param(formatting.count(MIN_RETEST_DAYS), REGISTERED),
    )
