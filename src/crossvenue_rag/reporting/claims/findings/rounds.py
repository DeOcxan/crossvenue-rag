"""What docs/results/findings.md claims about the gold set's second labeling round.

The agreement as `crossvenue kappa` computes it, the adjudication of `v0.2.2-gold`, and the
retrieval check on that version.
"""

from __future__ import annotations

from .... import formatting
from ....config_eval import LABELLED_SCOPE
from ....dataset.agreement import KAPPA_TARGET
from ...check import Doc
from ...compute import agreement, retrieval, sensitivity
from ...compute.estimates import Estimate
from ...outcomes import Param
from ...run_ids import PVALUE_RUN, SENSITIVITY_BASE_RUN, SENSITIVITY_RUN
from ...sources import Sources
from ..agreement import READINGS
from ..comparisons import (
    comparison_questions,
    estimate,
    kappa_bounds,
    kappa_cell,
    level,
    sensitivity_sizes,
    signed_parts,
    significance,
)
from ..reasons import INTERVAL_LEVEL, REGISTERED

# The six methods of the sensitivity table, in its order.
METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank", "venue_split", "venue_split_oracle")
# The comparisons of the sensitivity table, by the method pair its first cell names.
PAIRS = (
    ("venue_split", "hybrid_rerank"),
    ("venue_split_oracle", "hybrid_rerank"),
    ("venue_split_oracle", "venue_split"),
    ("hybrid_rerank", "hybrid"),
    ("hybrid_rerank", "dense"),
    ("hybrid_rerank", "bm25"),
)
# The comparisons the text names as inconclusive at this size, as paired.csv keys them.
INCONCLUSIVE = {
    ("hybrid_rerank", "hybrid", "recall@10", "labelled"),
    ("hybrid_rerank", "bm25", "recall@10", "labelled comparison"),
    ("hybrid_rerank", "dense", "venue_coverage@10", "comparison"),
    ("venue_split_oracle", "venue_split", "venue_coverage@10", "multi-venue"),
}
# How the table names a paired.csv scope.
SCOPES = {"labelled comparison": "comparison", "labelled": "labeled"}


def agreement_finding(doc: Doc, src: Sources) -> None:
    """The section on round 2: kappa, its readings, the direction, the marker, the adjudication."""
    study = agreement.study(src)
    registered = agreement.registered(src)
    report = registered.report
    kappa, (low, high) = formatting.metric(report.kappa), kappa_bounds(registered)
    doc.claim(
        "## Round 2 reproduces the round-1 labels at \u03ba {} [{}, {}], below the {} target",
        kappa,
        low,
        high,
        formatting.own_form(KAPPA_TARGET),
        holds=report.kappa < KAPPA_TARGET,
        what="the registered kappa is below the target",
    )
    doc.check(
        "`v0.2.1-gold`, {} candidate decisions over the {} questions of the round-2 sample.",
        formatting.count(report.pairs),
        formatting.count(report.questions),
    )
    gaps = study.gaps.values()
    doc.check(
        "{} to {} days after round 1, counting its augmentation",
        formatting.number(min(gaps), 1),
        formatting.number(max(gaps), 1),
    )
    bootstrap = agreement.bootstrap(src)
    doc.check(
        "Intervals are {}% percentile bootstrap intervals over questions, {} resamples (seed {});",
        Param(level(), INTERVAL_LEVEL),
        Param(formatting.count(bootstrap.resamples), REGISTERED),
        Param(formatting.own_form(bootstrap.seed), REGISTERED),
    )
    doc.check(
        "Raw agreement is {} against {} expected by chance. The registered target is \u03ba >= {},",
        formatting.metric(report.observed_agreement),
        formatting.metric(report.expected_agreement),
        Param(formatting.own_form(KAPPA_TARGET), REGISTERED),
    )
    doc.check(
        "The {} readings methodology \u00a75 registers", formatting.count(len(study.variants))
    )
    unexposed = agreement.reading(src, "corrected", "unexposed").report
    doc.check(
        "The {} unexposed questions are those whose gold",
        formatting.count(unexposed.questions),
    )
    doc.given(
        "with {}% percentile bootstrap intervals over questions,", level(), reason=INTERVAL_LEVEL
    )
    doc.check(
        "on {} decisions over {} questions or {} over the {} unexposed ones:",
        formatting.count(report.pairs),
        formatting.count(report.questions),
        formatting.count(unexposed.pairs),
        formatting.count(unexposed.questions),
    )
    rows = {}
    for label, (reference, scope) in READINGS.items():
        v = agreement.reading(src, reference, scope)
        rows[label] = [
            formatting.count(v.report.questions),
            kappa_cell(v),
        ]
    doc.table("| reading | questions | \u03ba |", rows)
    _direction(doc, src)


def _direction(doc: Doc, src: Sources) -> None:
    """The direction of the disagreements, the marker's share and the adjudication."""
    study = agreement.study(src)
    d, sources = study.direction, study.sources.verdicts
    doc.claim(
        "of the {} disagreements, {} are relevant in round 1 only and {} in round 2 only.",
        formatting.count(d.disagreements),
        formatting.count(d.first_only),
        formatting.count(d.second_only),
        holds=d.second_only > d.first_only,
        what="round 2 marks more pairs relevant than round 1",
    )
    doc.claim(
        "{} of the {} round-1-only pairs are source chunks, which round 1's screen marked",
        formatting.count(sources.first_only),
        formatting.count(d.first_only),
        holds=sources.second_only == 0,
        what="every disagreeing source chunk is relevant in round 1 only",
    )
    split = agreement.adjudication_split(src)
    doc.check(
        "to the {} pairs: {} decisions agree with round 1 and {} with round 2.",
        formatting.count(split.as_first + split.as_second),
        formatting.count(split.as_first),
        formatting.count(split.as_second),
    )
    doc.check(
        "it changed the gold of {} questions, which `v0.2.2-gold` holds",
        formatting.count(len(agreement.changed_gold(src))),
    )


def _p_text(src: Sources, run: str, comparison: sensitivity.Comparison) -> list[str]:
    row = sensitivity.paired_row(src, run, comparison)
    return [estimate(Estimate.of(row)), formatting.p_value(sensitivity.tested_p(row))]


def sensitivity_finding(doc: Doc, src: Sources) -> None:
    """The section on the retrieval runs on `v0.2.2-gold` and on its baseline."""
    doc.claim(
        "## On v0.2.2-gold no retrieval comparison changes sign, significance or whether its "
        "interval excludes {}",
        "0",
        holds=sensitivity.conclusions_unchanged(src),
        what="every primary and secondary comparison keeps sign, significance and whether its "
        "interval excludes 0",
    )
    doc.ref(
        "Runs `results/{}` on `v0.2.1-gold` and `results/{}` on `v0.2.2-gold`",
        SENSITIVITY_BASE_RUN,
        SENSITIVITY_RUN,
    )
    depths = {retrieval.top_k(src, run) for run in (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN)}
    doc.claim(
        "one config, {} methods, k = {}.",
        formatting.count(len(src.run(SENSITIVITY_RUN).methods)),
        formatting.count(retrieval.top_k(src, SENSITIVITY_RUN)),
        holds=len(depths) == 1,
        what="both runs retrieve to the same depth",
    )
    comparison, multi_venue, labeled_v022, labeled_v021 = sensitivity_sizes(src)
    doc.check(
        "Each run scores {} comparison questions, {} multi-venue ones and the labeled ones, {} on "
        "`v0.2.1-gold` and {} on `v0.2.2-gold`.",
        comparison,
        multi_venue,
        labeled_v021,
        labeled_v022,
    )
    doc.ref("matches `results/{}` in recall@10", PVALUE_RUN)
    doc.check("Each method on the {} comparison questions", comparison_questions(src))
    doc.ref(
        "questions, in `results/{}` on `v0.2.1-gold` and `results/{}` on `v0.2.2-gold`:",
        SENSITIVITY_BASE_RUN,
        SENSITIVITY_RUN,
    )
    columns = ("recall@10", "ndcg@10", "venue_coverage@10")
    runs = (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN)
    doc.table(
        "| method | recall@10 v0.2.1 |",
        {
            m: [
                formatting.metric(sensitivity.run_metric(src, run, "comparison", m, c))
                for c in columns
                for run in runs
            ]
            for m in METHODS
        },
    )
    tested = sensitivity.tested_comparisons(src)
    coverage = [c for c in tested if c.metric == "venue_coverage@10"]
    doc.claim(
        "The {} venue_coverage@10 comparisons are identical in both runs",
        formatting.count(len(coverage)),
        holds=all(sensitivity.identical(src, c) for c in coverage),
        what="the venue_coverage@10 comparisons are the same in both runs",
    )
    recall = [c for c in tested if c.metric == "recall@10"]
    doc.check("The primary and {} recall@10 comparisons of", formatting.count(len(recall)))
    doc.ref(
        "comparisons of `results/{}` (`v0.2.1-gold`) and `results/{}` (`v0.2.2-gold`) on", *runs
    )
    doc.check(
        "on the {} comparison and the {} and {} labeled questions, {}% paired bootstrap intervals, "
        "p_Holm (unadjusted for the primary):",
        comparison_questions(src),
        *(formatting.count(sensitivity.tested_questions(src, run, LABELLED_SCOPE)) for run in runs),
        Param(level(), INTERVAL_LEVEL),
    )
    rows = {}
    primary = next(c for c in tested if sensitivity.paired_row(src, runs[0], c).role == "primary")
    rows[f"{primary.a} - {primary.b}, {primary.metric}, comparison (primary)"] = [
        *_p_text(src, runs[0], primary),
        *_p_text(src, runs[1], primary),
    ]
    for a, b in PAIRS:
        for scope in ("labelled comparison", "labelled"):
            c = sensitivity.Comparison(a, b, "recall@10", scope)
            rows[f"{a} - {b}, recall@10, {SCOPES[scope]}"] = [
                *_p_text(src, runs[0], c),
                *_p_text(src, runs[1], c),
            ]
    doc.table("| comparison | v0.2.1-gold |", rows)
    doc.claim(
        "None of the {} primary and secondary comparisons changes sign, significance at "
        "p_Holm < {} or whether its interval excludes {}.",
        formatting.count(len(tested)),
        significance(),
        "0",
        holds=sensitivity.conclusions_unchanged(src),
        what="every primary and secondary comparison keeps sign, significance and whether its "
        "interval excludes 0",
    )
    doc.claim(
        "{} keep an interval that excludes {} with p_Holm above {}, so they stay inconclusive",
        formatting.count(len(sensitivity.inconclusive(src))),
        "0",
        significance(),
        holds={(c.a, c.b, c.metric, c.scope) for c in sensitivity.inconclusive(src)}
        == INCONCLUSIVE,
        what="the inconclusive comparisons are the four the sentence names",
    )
    coverage_parts = []
    for a, b, metric, scope in sorted(INCONCLUSIVE):
        if metric == "venue_coverage@10":
            row = sensitivity.paired_row(src, runs[0], sensitivity.Comparison(a, b, metric, scope))
            coverage_parts += [
                *signed_parts(Estimate.of(row)),
                formatting.p_value(sensitivity.tested_p(row)),
            ]
    doc.check(
        "On venue_coverage@10 they are hybrid_rerank \u2212 dense over the "
        "comparison questions, {} [{}, {}], p_Holm {}, and venue_split_oracle \u2212 venue_split "
        "over the multi-venue questions, {} [{}, {}], p_Holm {}.",
        *coverage_parts,
    )
