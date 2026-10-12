"""Tests for `reporting.claims.comparisons`: the kappa cell and bounds the documents state."""

from __future__ import annotations

import pytest

from crossvenue_rag.dataset.agreement import AgreementReport, AgreementVariant
from crossvenue_rag.reporting.claims import comparisons
from crossvenue_rag.stats import Interval


def test_a_kappa_cell_holds_its_interval_or_the_kappa_alone():
    report = AgreementReport(10, 2, 7, 1, 0.9, 0.8, 0.47, 3)
    defined = AgreementVariant("corrected", "all", report, Interval(0.317, 0.645, 0, 10))
    assert comparisons.kappa_cell(defined) == "0.470 [0.317, 0.645]"
    assert comparisons.kappa_bounds(defined) == ("0.317", "0.645")
    undefined = AgreementVariant("corrected", "all", report, Interval(None, None, 10, 10))
    assert comparisons.kappa_cell(undefined) == "0.470"
    with pytest.raises(ValueError, match="has no kappa interval"):
        comparisons.kappa_bounds(undefined)
