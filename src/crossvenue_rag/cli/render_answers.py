"""Terminal rendering of the answer commands' results: an answer, a cost estimate, an agreement.

Every value is literal text, as `render` explains.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import formatting
from ..evaluation.labels import fmt
from ..generation.answer_report import latency_seconds
from ..generation.estimate import estimate_total
from ..rich_text import add_text_row
from .app import say
from .render import TITLE_STYLE

if TYPE_CHECKING:
    from ..generation.ask import AskResult
    from ..generation.estimate import EstimateRow
    from ..generation.grader_agreement import DimensionAgreement


def print_answer(console: Console, result: AskResult) -> None:
    """`crossvenue ask`: the answer, its outcome and its claims per venue with citations."""
    missing = "; ".join(f"{m.venue}: {m.missing}" for m in result.missing)
    outcome = {
        "refused": "refused: the documentation does not answer this",
        "asked": "asks for clarification",
        "partial": f"partial, not in the documentation: {missing or formatting.MISSING}",
        "full": "full",
    }[result.outcome]
    say(
        console,
        Panel(
            Text(result.answer),
            title=Text(f"{result.generator} over {result.retriever}"),
            border_style="green",
        ),
    )
    say(console, outcome, style="green" if result.outcome == "full" else "yellow")
    for i, claim in enumerate(result.claims, start=1):
        say(console, f"({formatting.count(i)}) [{claim.venue}] {claim.text}")
        for c in claim.citations:
            where = "/".join(v for v in (c.environment, c.api_version, c.product) if v)
            say(
                console,
                f"      [{formatting.count(c.chunk)}] {c.venue} {c.doc_path}"
                + (f" ({where})" if where else "")
                + (f"\n          {c.url}" if c.url else ""),
                style="dim",
            )
    cost = "cached" if result.cached else formatting.optional(result.cost_usd, formatting.cost)
    say(
        console,
        f"venues detected: {', '.join(result.venues_detected) or 'none'}; "
        f"model {result.model} via {result.provider or formatting.MISSING}; "
        f"{latency_seconds(result.latency_ms)} s; {cost}",
        style="dim",
    )


def estimate_table(rows: Sequence[EstimateRow]) -> Table:
    """Cost estimate per `EstimateRow`, with a total that notes any unknown cost."""
    table = Table(title="cost estimate (no call made)")
    for column in ("item", "model", "calls", "cached", "mean USD/call", "estimate USD"):
        table.add_column(column, justify="left" if column in ("item", "model") else "right")
    for row in rows:
        add_text_row(
            table,
            row.item,
            row.model,
            formatting.count(row.calls),
            formatting.MISSING if row.cached is None else formatting.count(row.cached),
            formatting.optional(row.mean_cost_usd, formatting.cost),
            "unknown" if row.estimate_usd is None else formatting.cost(row.estimate_usd),
        )
    total, unknown = estimate_total(rows)
    add_text_row(
        table,
        "total",
        "",
        "",
        "",
        "",
        formatting.cost(total) + (" + unknown" if unknown else ""),
        style="bold",
    )
    return table


def agreement_table(title: str, rows: Sequence[DimensionAgreement]) -> Table:
    """Agreement, kappa and correlations per `DimensionAgreement` row."""
    table = Table(title=Text(title, style=TITLE_STYLE))
    for column in ("dimension", "n", "agreement", "kappa", "pearson", "spearman"):
        table.add_column(column, justify="left" if column == "dimension" else "right")

    for row in rows:
        add_text_row(
            table,
            row.dimension,
            formatting.count(row.n),
            fmt(row.agreement),
            fmt(row.kappa),
            fmt(row.pearson),
            fmt(row.spearman),
        )
    return table
