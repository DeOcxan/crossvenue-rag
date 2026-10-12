"""The registered protocol's parameters that more than one section or document states.

evals/methodology.md, findings.md, the datasheet and the ADRs state them.
"""

from __future__ import annotations

from .... import formatting
from ....dataset.selection import TARGETS, target_size
from ....generation.grader_agreement import BINARY_DIMENSIONS
from ...compute import config_reference, datasets, retrieval
from ...sources import Sources

# The per-venue floor of decision 3: questions each venue is the answer venue for.
VENUE_FLOOR = "8"
# The discordance the power model assumes (section 7), and the discordances the
# questions it needs are also stated at.
PSI = 0.25
PSI_LOW, PSI_HIGH = 0.20, 0.30
# The comparison subgroup sizes section 7 weighs beside `TARGETS["comparison"]`: a smaller
# and a larger one, and the subgroup if some of its questions were unanswerable.
SMALLER_COMPARISON_SIZE = 35
LARGER_COMPARISON_SIZE = 60
REDUCED_COMPARISON_SIZE = 40
# The hard floor of decision 1, unanswerable questions included; `selection.target_size` is
# the set size.
FLOOR_SIZE = 85
# The scorable questions of the set and of the floor: the unanswerable ones enter no recall.
SCORABLE_SIZE = target_size() - TARGETS["unanswerable"]
FLOOR_SCORABLE_SIZE = FLOOR_SIZE - TARGETS["unanswerable"]


def bootstrap_resamples(src: Sources) -> str:
    """`eval.bootstrap_resamples`: bootstrap resamples, and randomization test permutations."""
    return formatting.count(src.retrieval.config.eval.bootstrap_resamples)


def bootstrap_seed_text(src: Sources) -> str:
    """`eval.bootstrap_seed`, written without a thousands separator."""
    return formatting.own_form(src.retrieval.config.eval.bootstrap_seed)


def pool_depth_text(src: Sources) -> str:
    """`eval.pool_depth_per_method`: each pool method's depth in the labeling pool."""
    return formatting.count(retrieval.pool_depth(src))


def pool_display_text(src: Sources) -> str:
    """The fused candidates shown to the annotator, `datasets.pool_display`, written as a count."""
    return formatting.count(datasets.pool_display(src))


def duplicate_threshold(src: Sources) -> str:
    """`eval.duplicate_threshold`: the cosine that flags two questions as near-duplicates."""
    return formatting.own_form(
        src.default_config.eval.duplicate_threshold, formatting.CONFIG_THRESHOLD_DECIMALS
    )


def judge_thresholds(src: Sources) -> tuple[str, str]:
    """The judge calibration thresholds of configs/default.yaml: the kappa, then coverage's.

    The kappa is the one the binary dimensions share; `formatting.MISSING` if they differ.
    """
    kappa = config_reference.shared_threshold(src, BINARY_DIMENSIONS)
    coverage = src.default_config.generation.calibration_thresholds["coverage"]
    return (
        formatting.optional(
            kappa, lambda v: formatting.own_form(v, formatting.CONFIG_THRESHOLD_DECIMALS)
        ),
        formatting.own_form(coverage, formatting.CONFIG_THRESHOLD_DECIMALS),
    )
