"""The config section of the retrieval evaluation, with the names of its comparison scopes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from . import formatting
from .config_pipeline import RetrievalMethod
from .config_section import Section, distinct

# What eval.primary_comparison names, in order.
PRIMARY_COMPARISON_FIELDS = ("a", "b", "metric", "scope")
# The scopes of the paired retrieval comparisons (`evaluation.comparisons`): venue_coverage
# is compared on the first two, recall on the last two.
MULTI_VENUE_SCOPE = "multi-venue"
COMPARISON_SCOPE = "comparison"
LABELLED_SCOPE = "labelled"
LABELLED_COMPARISON_SCOPE = "labelled comparison"
# The cut-off of the top-1 off-venue comparison (`evaluation.comparisons.TOP_ONE_OFF_VENUE`).
TOP_ONE_K = 1


# A list or dict default is given through `Field`: ruff (RUF012) treats a class as a pydantic
# model only when it has a `BaseModel` base in its own file, and `Section` is in config_section.
class EvalConfig(Section):
    """The retrieval evaluation: question file, methods, cutoffs, labeling pool and comparisons."""

    questions: Path = Path("evals/questions.jsonl")
    methods: list[RetrievalMethod] = Field(["bm25", "dense", "hybrid", "hybrid_rerank"])
    k_values: list[int] = Field([1, 5, 10])
    # Dimensions the ablation breaks metrics down by, on top of the aggregate.
    breakdowns: list[Literal["venue", "qtype", "category"]] = Field(["venue", "qtype"])
    # Cosine similarity at or above which two questions count as near-duplicates.
    duplicate_threshold: float = Field(0.92, ge=0.0, le=1.0)
    # Labeling pool: how deep each of the five pool methods contributes, and how many
    # fused candidates the annotator is shown.
    pool_depth_per_method: int = Field(20, gt=0)
    pool_display: int = Field(25, gt=0)
    # Paired method comparisons written into report.md: [a, b] means a - b.
    paired_comparisons: list[tuple[RetrievalMethod, RetrievalMethod]] = Field([])
    # The one comparison the study is designed around: [a, b, metric, scope].
    # Every other paired comparison is secondary and Holm-adjusted.
    primary_comparison: list[str] = Field([])
    bootstrap_resamples: int = Field(10_000, gt=0)
    # numpy's generators take a non-negative seed.
    bootstrap_seed: int = Field(20260924, ge=0)

    @field_validator("methods")
    @classmethod
    def _methods_named_once(cls, v: list[RetrievalMethod]) -> list[RetrievalMethod]:
        if not v:
            raise ValueError("methods must name at least one method")
        return distinct(v, "methods")

    @field_validator("k_values")
    @classmethod
    def _cutoffs(cls, v: list[int]) -> list[int]:
        if TOP_ONE_K not in v:  # an empty list included
            top = formatting.count(TOP_ONE_K)
            raise ValueError(
                f"k_values needs {top}: the top-{top} off-venue comparison reads off_venue@{top}"
            )
        if any(k <= 0 for k in v):
            given = ", ".join(formatting.own_form(k) for k in v)
            raise ValueError(f"k_values must all be positive, got {given}")
        return sorted(distinct(v, "k_values"))

    @field_validator("paired_comparisons")
    @classmethod
    def _pairs_of_two_methods(
        cls, v: list[tuple[RetrievalMethod, RetrievalMethod]]
    ) -> list[tuple[RetrievalMethod, RetrievalMethod]]:
        """A method against itself compares nothing; a repeated pair is counted twice by Holm.

        A pair whose methods `methods` does not run is kept and skipped by the run, so an
        ablation (`--set eval.methods=[bm25, dense]`) needs no edit of the pairs.
        """
        for a, b in v:
            if a == b:
                raise ValueError(f"eval.paired_comparisons pairs {a} with itself")
        distinct([f"[{a}, {b}]" for a, b in v], "eval.paired_comparisons")
        return v

    @model_validator(mode="after")
    def _primary_is_a_configured_pair(self) -> EvalConfig:
        """A mistyped primary would silently make every comparison secondary."""
        if not self.primary_comparison:
            return self
        if len(self.primary_comparison) != len(PRIMARY_COMPARISON_FIELDS):
            raise ValueError("eval.primary_comparison is [a, b, metric, scope]")
        a, b, metric, scope = self.primary_comparison
        if (a, b) not in [tuple(p) for p in self.paired_comparisons]:
            raise ValueError(
                f"eval.primary_comparison [{a}, {b}] is not in eval.paired_comparisons"
            )
        # The rows `evaluation.comparisons` writes: two metrics at max(k_values),
        # each on two scopes.
        k = max(self.k_values)
        scopes = {
            f"venue_coverage@{k}": (MULTI_VENUE_SCOPE, COMPARISON_SCOPE),
            f"recall@{k}": (LABELLED_SCOPE, LABELLED_COMPARISON_SCOPE),
        }
        if metric not in scopes:
            raise ValueError(
                f"eval.primary_comparison metric {metric!r} is not compared; "
                f"expected one of {', '.join(scopes)}"
            )
        if scope not in scopes[metric]:
            raise ValueError(
                f"eval.primary_comparison scope {scope!r} is not a scope of {metric}; "
                f"expected one of {', '.join(scopes[metric])}"
            )
        return self

    @model_validator(mode="after")
    def _pool_exceeds_evaluation_depth(self) -> EvalConfig:
        """The judged set has to extend past the ranks being scored.

        If the pool is only as deep as the largest k, the judgments end exactly at the
        deepest rank the metric scores. Recall@k is then measured against a gold
        set that was never allowed to contain anything below rank k. Pooling bias
        is unavoidable (evals/methodology.md section 1); making it coincide with the
        evaluation depth is not.
        """
        deepest = max(self.k_values)
        pool_depth = formatting.own_form(self.pool_depth_per_method)
        if self.pool_depth_per_method <= deepest:
            raise ValueError(
                f"eval.pool_depth_per_method ({pool_depth}) must exceed max(k_values) "
                f"({formatting.own_form(deepest)}); a pool no deeper than the evaluation "
                "depth adds no judgment below the ranks the metric scores"
            )
        if self.pool_display < self.pool_depth_per_method:
            raise ValueError(
                f"eval.pool_display ({formatting.own_form(self.pool_display)}) is below "
                f"pool_depth_per_method ({pool_depth}); candidates would be fetched and then "
                "hidden from the annotator"
            )
        return self
