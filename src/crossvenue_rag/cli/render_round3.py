"""Terminal tables of `crossvenue round-agreement`: counts and raw agreement, with no kappa."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from .. import formatting
from ..dataset.agreement import RoundAgreement
from ..rich_text import add_text_row
from .render import BORDERLESS_PADDING, TITLE_STYLE

_FIRST_ONLY = "relevant in round 1 only"
_LATER_ONLY = "relevant in round 3 only"


def round_agreement_table(report: RoundAgreement) -> Table:
    """The verdict counts and raw agreement of rounds 1 and 3."""
    table = Table(
        title=Text(
            "rounds 1 and 3, descriptive: the criterion changed between them", style=TITLE_STYLE
        ),
        show_header=False,
        box=None,
        padding=BORDERLESS_PADDING,
    )
    t = report.total
    add_text_row(table, "questions compared", formatting.count(report.questions))
    add_text_row(table, "(question, chunk) pairs", formatting.count(t.pairs))
    add_text_row(table, "both relevant", formatting.count(t.both_relevant))
    add_text_row(table, "both irrelevant", formatting.count(t.both_irrelevant))
    add_text_row(table, _FIRST_ONLY, formatting.count(t.first_only))
    add_text_row(table, _LATER_ONLY, formatting.count(t.second_only))
    add_text_row(table, "raw agreement", formatting.metric(t.raw_agreement))
    return table


def round_agreement_categories_table(report: RoundAgreement) -> Table:
    """Raw agreement and the direction of disagreement per category, most disagreement first."""
    table = Table(title="by category")
    for column in ("category", "pairs", "disagreements", _FIRST_ONLY, _LATER_ONLY, "raw agreement"):
        table.add_column(column, justify="left" if column == "category" else "right")
    for category, t in report.by_category:
        add_text_row(
            table,
            category,
            formatting.count(t.pairs),
            formatting.count(t.disagreements),
            formatting.count(t.first_only),
            formatting.count(t.second_only),
            formatting.metric(t.raw_agreement),
        )
    return table
