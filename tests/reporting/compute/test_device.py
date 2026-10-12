"""Tests for `reporting.compute.device`, on the committed CPU and MPS runs it compares."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import device
from crossvenue_rag.reporting.sources import Sources
from tests.published import ROOT


@pytest.fixture(scope="module")
def src() -> Sources:
    return Sources(ROOT)


def test_the_device_changes_dense_dependent_lists_and_no_bm25_list(src):
    lists = device.list_change(src)
    assert 0 < lists.differing < lists.shared
    assert "bm25" not in lists.methods and "dense" in lists.methods


def test_changed_cells_are_largest_first_and_membership_cells_a_subset(src):
    cells = device.changed_cells(src)
    sizes = [abs(c.cpu - c.mps) for c in cells]
    assert sizes == sorted(sizes, reverse=True)
    membership = device.membership_changes(src)
    assert membership and all(c in cells for c in membership)
    assert all(c.metric.split("@")[0] in device.MEMBERSHIP_METRICS for c in membership)


def test_on_comparison_questions_only_the_ranking_moves(src):
    assert device.ranking_only_changes(src, "comparison")
    assert not device.ranking_only_changes(src, "version_sensitive")
