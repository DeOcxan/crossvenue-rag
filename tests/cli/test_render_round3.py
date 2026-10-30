"""Tests for `cli.render_round3`: the tables of `crossvenue round-agreement`."""

from __future__ import annotations

from rich.console import Console

from crossvenue_rag.cli.render_round3 import (
    round_agreement_categories_table,
    round_agreement_table,
)
from crossvenue_rag.dataset.agreement import RoundAgreement, VerdictTable


def _report() -> RoundAgreement:
    table = VerdictTable(both_relevant=2, first_only=1, second_only=3, both_irrelevant=14)
    return RoundAgreement(questions=2, total=table, by_category=[("comparison", table)])


def _printed(table: object) -> str:
    console = Console(record=True, width=160)
    console.print(table)
    return console.export_text()


def test_the_totals_show_every_count_and_the_raw_agreement_and_no_kappa():
    out = _printed(round_agreement_table(_report()))
    assert "20" in out and "0.800" in out and "relevant in round 3 only" in out
    assert "descriptive" in out and "kappa" not in out.lower()


def test_the_categories_show_the_direction_of_disagreement():
    out = _printed(round_agreement_categories_table(_report()))
    assert "comparison" in out and "relevant in round 1 only" in out and "0.800" in out
