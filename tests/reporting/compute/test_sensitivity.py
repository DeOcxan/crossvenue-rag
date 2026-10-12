"""Tests for `reporting.compute.sensitivity`, on the two committed runs it compares."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import sensitivity
from crossvenue_rag.reporting.run_ids import SENSITIVITY_BASE_RUN, SENSITIVITY_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.published import ROOT

PRIMARY = sensitivity.Comparison("venue_split", "hybrid_rerank", "venue_coverage@10", "comparison")


@pytest.fixture(scope="module")
def src() -> Sources:
    return Sources(ROOT)


def test_the_tested_comparisons_are_the_primary_and_secondary_rows(src):
    tested = sensitivity.tested_comparisons(src)
    assert len(tested) == 24 and PRIMARY in tested
    roles = {sensitivity.paired_row(src, SENSITIVITY_BASE_RUN, c).role for c in tested}
    assert roles == {"primary", "secondary"}


def test_the_primary_uses_its_unadjusted_p_and_a_secondary_its_holm_p(src):
    primary = sensitivity.paired_row(src, SENSITIVITY_RUN, PRIMARY)
    assert primary.holm_p is None and sensitivity.tested_p(primary) == primary.randomisation_p
    secondary = sensitivity.Comparison("hybrid_rerank", "bm25", "recall@10", "labelled")
    row = sensitivity.paired_row(src, SENSITIVITY_RUN, secondary)
    assert sensitivity.tested_p(row) == row.holm_p


def test_no_conclusion_changes_and_venue_coverage_needs_no_gold(src):
    assert sensitivity.conclusions_unchanged(src)
    coverage = [c for c in sensitivity.tested_comparisons(src) if c.metric == "venue_coverage@10"]
    assert all(sensitivity.identical(src, c) for c in coverage)
    recall = sensitivity.Comparison("venue_split", "hybrid_rerank", "recall@10", "labelled")
    assert not sensitivity.identical(src, recall)


def test_inconclusive_comparisons_exclude_0_with_a_p_at_or_above_the_level(src):
    keys = {(c.a, c.b, c.metric, c.scope) for c in sensitivity.inconclusive(src)}
    assert ("hybrid_rerank", "hybrid", "recall@10", "labelled") in keys
    assert PRIMARY not in sensitivity.inconclusive(src)
    assert len(keys) == 4


def test_each_scope_pairs_the_questions_of_its_gold_version(src):
    sizes = {
        run: [
            sensitivity.tested_questions(src, run, s)
            for s in ("comparison", "multi-venue", "labelled")
        ]
        for run in (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN)
    }
    assert sizes == {SENSITIVITY_BASE_RUN: [50, 59, 90], SENSITIVITY_RUN: [50, 59, 92]}
    with pytest.raises(ValueError, match="pair no single number of questions"):
        sensitivity.tested_questions(src, SENSITIVITY_RUN, "nowhere")


def test_a_metric_or_a_comparison_a_run_lacks_is_an_error(src):
    assert sensitivity.run_metric(src, SENSITIVITY_RUN, "comparison", "bm25", "recall@10") > 0
    with pytest.raises(ValueError, match="has no"):
        sensitivity.run_metric(src, SENSITIVITY_RUN, "comparison", "nope", "recall@10")
    with pytest.raises(ValueError, match="has no"):
        sensitivity.paired_row(src, SENSITIVITY_RUN, sensitivity.Comparison("a", "b", "m", "s"))
