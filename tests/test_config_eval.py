"""Tests for `crossvenue_rag.config_eval`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crossvenue_rag.config_eval import EvalConfig
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError

DEFAULT = Path("configs/default.yaml")


def test_pool_must_be_deeper_than_the_evaluation_depth(cfg):
    """A pool only as deep as max(k) adds no judgment below the ranks recall@k scores,
    so the judgments end exactly at the deepest rank it scores."""
    with pytest.raises(ValidationError, match="must exceed"):
        EvalConfig(k_values=[1, 5, 10], pool_depth_per_method=10, pool_display=25)
    with pytest.raises(ValidationError, match="below"):
        EvalConfig(k_values=[1, 5, 10], pool_depth_per_method=20, pool_display=15)
    # The configured geometry is the one evals/methodology.md decision 5 specifies.
    assert cfg.eval.pool_depth_per_method == 20
    assert cfg.eval.pool_display == 25


def test_the_refused_pool_depths_are_written_through_the_formatting_module(monkeypatch):
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(ValidationError, match=r"\(<10>\) must exceed max\(k_values\) \(<10>\)"):
        EvalConfig(k_values=[1, 5, 10], pool_depth_per_method=10, pool_display=25)
    with pytest.raises(ValidationError, match=r"\(<15>\) is below pool_depth_per_method \(<20>\)"):
        EvalConfig(k_values=[1, 5, 10], pool_depth_per_method=20, pool_display=15)


def test_primary_comparison_must_be_a_configured_pair():
    ok = EvalConfig(
        paired_comparisons=[("venue_split", "hybrid_rerank")],
        primary_comparison=["venue_split", "hybrid_rerank", "venue_coverage@10", "comparison"],
    )
    assert ok.primary_comparison[0] == "venue_split"
    with pytest.raises(ValidationError):
        EvalConfig(
            paired_comparisons=[("venue_split", "hybrid_rerank")],
            primary_comparison=["dense", "bm25", "recall@10", "labelled"],
        )


@pytest.mark.parametrize(
    ("metric", "scope", "message"),
    [
        ("venue_coverage@5", "comparison", "metric 'venue_coverage@5' is not compared"),
        ("venue_coverage@10", "Comparison", "scope 'Comparison' is not a scope"),
        ("recall@10", "comparison", "scope 'comparison' is not a scope of recall@10"),
    ],
)
def test_a_mistyped_primary_metric_or_scope_is_refused(metric, scope, message):
    with pytest.raises(ValidationError, match=message):
        EvalConfig(
            paired_comparisons=[("venue_split", "hybrid_rerank")],
            primary_comparison=["venue_split", "hybrid_rerank", metric, scope],
        )


def test_k_values_without_1_is_refused_naming_the_key_and_the_reason():
    with pytest.raises(UserError) as refused:
        load_config(DEFAULT, ["eval.k_values=[5, 10]"])
    message = str(refused.value)
    assert "eval.k_values" in message
    assert "needs 1: the top-1 off-venue comparison reads off_venue@1" in message


def test_a_k_value_that_is_not_positive_is_refused_with_the_values_as_given(monkeypatch):
    with pytest.raises(ValidationError, match="k_values must all be positive, got 1, -5, 10000"):
        EvalConfig(k_values=[1, -5, 10000])
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(ValidationError, match="got <1>, <-5>, <10000>"):
        EvalConfig(k_values=[1, -5, 10000])


def test_a_paired_comparison_pairs_two_methods_once():
    with pytest.raises(UserError, match="pairs bm25 with itself"):
        load_config(
            DEFAULT, ["eval.paired_comparisons=[[bm25, bm25]]", "eval.primary_comparison=[]"]
        )
    with pytest.raises(UserError, match=r"repeats \[bm25, dense\]"):
        load_config(
            DEFAULT,
            [
                "eval.paired_comparisons=[[bm25, dense], [bm25, dense]]",
                "eval.primary_comparison=[]",
            ],
        )


def test_a_pair_of_methods_the_run_skips_is_kept():
    cfg = load_config(DEFAULT, ["eval.methods=[bm25, dense]"])
    assert ("venue_split", "hybrid_rerank") in cfg.eval.paired_comparisons


# Each key given a value outside its domain through `--set`, and the key the refusal names.
KEY_OUT_OF_DOMAIN = [
    ("eval.methods=[]", "eval.methods"),
    ("eval.methods=[bm25, bm25]", "eval.methods"),
    ("eval.methods=[nosuch]", "eval.methods"),
    ("eval.k_values=[]", "eval.k_values"),
    ("eval.k_values=[5, 5, 10]", "eval.k_values"),
    ("eval.duplicate_threshold=1.5", "eval.duplicate_threshold"),
    ("eval.bootstrap_seed=-1", "eval.bootstrap_seed"),
    ("eval.bootstrap_resamples=0", "eval.bootstrap_resamples"),
]


@pytest.mark.parametrize(("override", "key"), KEY_OUT_OF_DOMAIN)
def test_a_key_outside_its_domain_is_refused_naming_it(override, key):
    with pytest.raises(UserError) as refused:
        load_config(DEFAULT, [override])
    assert str(refused.value).startswith(f"invalid --set {override}: ")
    assert key in str(refused.value)
