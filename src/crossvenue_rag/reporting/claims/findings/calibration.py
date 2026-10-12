"""docs/results/findings.md: the judge calibration."""

from __future__ import annotations

from .... import formatting, listing
from ....generation.grader_agreement import BINARY_DIMENSIONS
from ...check import Doc
from ...compute import answers, calibration
from ...sources import Sources
from ..comparisons import level
from ..methodology.answers import (
    agreement_cell,
    agreement_parts,
    dimension_label,
    metric_bounds,
    threshold_cell,
)
from ..reasons import INTERVAL_LEVEL

# What three answers of the calibration sample claim, as the text quotes them.
LIMIT_DEFAULT = ("20",)
SIZE_DEFAULT = ("20",)
LIMIT_RANGE = ("1", "1000", "500")


def table(doc: Doc, src: Sources) -> None:
    """The calibration table: each candidate's agreement with the human and the other."""
    doc.table(
        "| dimension | threshold | sonnet | opus | sonnet vs opus |",
        {
            dimension_label(
                r.dimension,
                formatting.count(r.with_human),
                formatting.count(r.between) if r.between != r.with_human else None,
            ): [
                threshold_cell(r.statistic, r.threshold),
                *[agreement_cell(interval) for interval in r.intervals],
            ]
            for r in calibration.table_rows(src)
        },
    )
    doc.check(
        "{} judge candidates graded the {} answers",
        formatting.count(calibration.judge_candidate_count(src)),
        formatting.count(calibration.answers_rated(src)),
    )
    doc.given(
        "human, {}% percentile bootstrap intervals resampling", level(), reason=INTERVAL_LEVEL
    )
    doc.check(
        "percentile bootstrap intervals resampling answers ({} resamples, seed {}), "
        "`behaviour_correct` under deviation D20:",
        formatting.count(calibration.resamples(src)),
        formatting.own_form(calibration.bootstrap_seed(src)),
    )


def judge_choice(doc: Doc, src: Sources) -> None:
    """The judge chosen, and where the judges and the human disagree."""
    judge = src.judged_v1.config.generation.chosen_judge()
    doc.claim(
        "its interval above is near-degenerate and cannot establish the threshold",
        holds=not calibration.interval_informative(
            src, calibration.judge_with_human(src), "venue_correct"
        ),
        what="Opus's venue_correct kappa is undefined in some resamples",
    )
    interval, point_only = calibration.passes_by_reach(src, judge)
    doc.claim(
        "coverage, on the point estimate and the interval; `behaviour_correct`, on the point "
        "estimate only; and venue_attribution_error, whose `venue_correct` dimension meets it "
        "on the point estimate only, with an interval that is not informative",
        holds=interval == {"coverage"} and point_only == {"behaviour_correct", "venue_correct"},
        what="Opus passes coverage on the interval, behaviour_correct and venue_correct on the "
        "point estimate only",
    )
    doc.check(
        "Opus meets {} thresholds and Sonnet {}, so Opus judges both matrices",
        formatting.count(calibration.passes(src, judge)),
        formatting.count(calibration.passes(src, calibration.other_candidate(src))),
    )
    doc.claim(
        "so Opus judges both matrices",
        holds=calibration.selected_passes_most(src, judge),
        what="Opus passes the most thresholds and is selected",
    )
    doc.check(
        "every binary dimension ({} to {} for Opus)",
        *[
            formatting.metric(v)
            for v in calibration.agreement_range(
                src, calibration.judge_with_human(src), BINARY_DIMENSIONS
            )
        ],
    )
    human = calibration.human_negatives(src)
    doc.check(
        "the human marked {} of {} claims unsupported and {} misattributed",
        formatting.count(human.unsupported),
        formatting.count(human.claims),
        formatting.count(human.misattributed),
    )
    either, both = calibration.venue_disagreement(src, judge)
    doc.check(
        "passes on {} of {} claims that either grader marked misattributed, {} that both marked",
        formatting.count(either),
        formatting.count(human.claims),
        formatting.count(both),
    )
    doc.check(
        "\u03ba lies in [{}, {}] or is undefined (undefined in {} of the {} resamples)",
        *metric_bounds(
            calibration.agreement_interval(src, calibration.judge_with_human(src), "venue_correct")
        ),
        formatting.count(
            calibration.kappa_undefined(src, calibration.judge_with_human(src), "venue_correct")
        ),
        formatting.count(calibration.resamples(src)),
    )
    doc.claim(
        "The {} calibration answers also come from prompt v1: none is a partial answer and none "
        "of their {} claims has the absence form",
        formatting.count(len(calibration.sample_ids(src))),
        formatting.count(calibration.sample_claims(src)),
        holds=calibration.sample_has_no_v2_forms(src),
        what="no calibration answer is partial and no calibration claim has the absence form",
    )


def rejection_claims(doc: Doc, src: Sources) -> None:
    """The claims the human rejects, and what each judge made of them."""
    judge = src.judged_v1.config.generation.chosen_judge()
    unsupported = calibration.human_negatives(src).unsupported
    doc.check(
        "### Opus, as judge, agrees with the human on {} of the {} claims the human rejects",
        formatting.count(calibration.verdict_pairs(src, judge, "supported")[(False, False)]),
        formatting.count(unsupported),
    )
    doc.check(
        "Of the {} claims the human marks unsupported, Opus agrees on {} and Sonnet on {}; of "
        "the {} the human accepts, Opus rejects {} and Sonnet",
        formatting.count(unsupported),
        formatting.count(calibration.verdict_pairs(src, judge, "supported")[(False, False)]),
        formatting.count(
            calibration.verdict_pairs(src, calibration.other_candidate(src), "supported")[
                (False, False)
            ]
        ),
        formatting.count(calibration.human_accepted(src)),
        formatting.count(calibration.verdict_pairs(src, judge, "supported")[(False, True)]),
    )
    doc.claim(
        "and Sonnet none",
        holds=calibration.rejects_none_accepted(src, calibration.other_candidate(src)),
        what="Sonnet rejected none of the claims the human accepted",
    )
    _answer_claim(doc, src, "limit defaults to {} for", "gemini@venue_split:cv-0127", LIMIT_DEFAULT)
    _answer_claim(doc, src, "size defaults to {} for", "mercury@venue_split:cv-0127", SIZE_DEFAULT)
    _answer_claim(
        doc,
        src,
        "limit accepts {} to {}, default {}, for",
        "mercury@venue_split:cv-0067",
        LIMIT_RANGE,
    )


def _answer_claim(
    doc: Doc, src: Sources, template: str, answer_id: str, values: tuple[str, ...]
) -> None:
    """What one calibration answer claims, as the text quotes it."""
    doc.claim(
        template,
        *values,
        holds=answers.claims_mention(src, answer_id, values),
        what=f"{answer_id} claims {listing.prose(values)}",
    )


def repeat(doc: Doc, src: Sources) -> None:
    """Opus's second grading of the calibration answers: their count, its self-agreement."""
    doc.check(
        "graded the {} calibration answers a second time",
        formatting.count(calibration.answers_rated(src)),
    )
    repeat_comparison = calibration.judge_with_repeat(src)
    doc.claim(
        "agrees with itself on `supported` at \u03ba {} [{}, {}], against {} with the human, and "
        "on the other dimensions at \u03ba {} (`venue_correct`, `behaviour_correct`) and "
        "Spearman {} [{}, {}]",
        *agreement_parts(src, repeat_comparison, "supported"),
        formatting.metric(
            calibration.comparison_kappa(src, calibration.judge_with_human(src), "supported")
        ),
        formatting.metric(calibration.comparison_kappa(src, repeat_comparison, "venue_correct")),
        *agreement_parts(src, repeat_comparison, "coverage"),
        holds=calibration.same_agreement(
            src, repeat_comparison, "behaviour_correct", "venue_correct"
        ),
        what="behaviour_correct's repeat agreement equals venue_correct's",
    )
