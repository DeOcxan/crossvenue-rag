"""What evals/methodology.md claims, by one function per part of the document, `naming` last."""

from __future__ import annotations

from ...check import Doc
from ...sources import Sources
from . import answers, decisions, deviations, labeling, naming, significance, small_effects


def methodology(doc: Doc, src: Sources) -> None:
    """States the claims of evals/methodology.md.

    Raises:
        ValueError: `doc`'s text has no `references.REFERENCES_HEADING` line.
    """
    decisions.opening(doc, src)
    decisions.decision_table(doc, src)
    decisions.pooling(doc, src)
    decisions.generation(doc)
    decisions.unanswerable(doc)
    labeling.adversarial(doc, src)
    labeling.agreement(doc, src)
    labeling.instructions(doc)
    labeling.round3(doc, src)
    labeling.deduplication(doc, src)
    significance.sample_size(doc)
    significance.observed_power(doc, src)
    significance.sample_size_decision(doc, src)
    answers.generator_protocol_claims(doc, src)
    answers.calibration_sample_claims(doc, src)
    answers.calibration_table(doc, src)
    answers.judge_protocol_claims(doc, src)
    answers.prompt_v2_protocol_claims(doc, src)
    answers.off_venue(doc, src)
    deviations.selection(doc, src)
    deviations.venues(doc, src)
    deviations.augmentation_deviation_claims(doc, src)
    deviations.later_deviations(doc, src)
    deviations.amendment_deviation(doc, src)
    deviations.size_deviation(doc)
    small_effects.small_effects(doc, src)
    naming.wording(doc)
    naming.metric_names(doc)
    naming.reference_claims(doc, src)
