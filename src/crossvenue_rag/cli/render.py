"""Terminal rendering of the CLI's output tables, every value literal text.

A cell or title that holds a value is a `Text`, never a markup string. Rich reads
square brackets as style markup, so `[/x]` raises and `[bold]` vanishes, and API
docs, paths and model answers are full of them (`[text](url)`, `[/x]`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from .. import formatting
from ..corpus.artifacts import CorpusStats, DedupSummary, MetadataCoverage, TermCount
from ..corpus.ingest import LENGTH_BINS
from ..corpus.manifest import ManifestRow
from ..dataset.agreement import (
    AgreementReport,
    AgreementVariant,
    SourceChunks,
    VerdictTable,
)
from ..dataset.selection import TARGETS, CompositionReport, target_size
from ..dataset.validate import ValidationReport, ValidationStats
from ..evaluation.power import TARGET_POWER_PERCENT
from ..models import Result, Timings
from ..rich_text import add_text_row
from ..stats import INTERVAL_LEVEL_PERCENT

if TYPE_CHECKING:
    from ..evaluation.power import PowerStudy

PREVIEW_CHARS = 400

# The widest bar of the chunk-length histogram, in characters.
HISTOGRAM_WIDTH = 40
# The column widths of the retrieval results table, in characters: rank, score, venue and
# heading path; the text takes the rest.
RANK_WIDTH, SCORE_WIDTH, VENUE_WIDTH, HEADING_WIDTH = 3, 8, 12, 30
# The status column of the accepted-composition table, in characters.
STATUS_WIDTH = 10
# The space between the columns of a borderless table: no rows above or below, two
# characters beside.
BORDERLESS_PADDING = (0, 2)
# The decimals of a retrieval score in the results table.
SCORE_DECIMALS = 3
# The power lines list the simulated deltas that lie on this grid.
POWER_LINE_STEP = 0.05
# A title holding a value is a `Text`, which rich prints without its default title style.
TITLE_STYLE = "table.title"


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:PREVIEW_CHARS] + (" ..." if len(flat) > PREVIEW_CHARS else "")


def fetch_summary(rows: Sequence[ManifestRow]) -> Table:
    """Documents and pinned commit per venue, with the total, after `crossvenue fetch`."""
    table = Table(title="fetched", show_footer=True)
    table.add_column("venue", footer="total")
    table.add_column("type")
    table.add_column("documents", justify="right", footer=Text(formatting.count(len(rows))))
    table.add_column("commit")

    by_venue: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_venue.setdefault(row.venue, []).append(row)
    for name, venue_rows in sorted(by_venue.items()):
        add_text_row(
            table,
            name,
            venue_rows[0].venue_type,
            formatting.count(len(venue_rows)),
            # GitBook sources have no commit to pin to; the manifest holds each full sha.
            formatting.short_hash(venue_rows[0].commit_sha)
            if venue_rows[0].commit_sha
            else formatting.MISSING,
        )
    return table


def corpus_stats_table(stats: CorpusStats) -> Table:
    """Documents, chunks and token lengths per venue, from the corpus statistics."""
    table = Table(title="corpus by venue")
    for column in ("venue", "type", "docs", "chunks", "tokens", "mean", "p50", "p95", "max"):
        table.add_column(column, justify="left" if column in ("venue", "type") else "right")
    for row in stats.by_venue:
        add_text_row(
            table,
            row.venue,
            row.venue_type,
            formatting.count(row.documents),
            formatting.count(row.chunks),
            formatting.count(row.tokens_total),
            formatting.ratio(row.tokens_mean, formatting.COUNT_MEAN_DECIMALS),
            formatting.count(row.tokens_p50),
            formatting.count(row.tokens_p95),
            formatting.count(row.tokens_max),
        )
    totals = stats.totals
    table.add_section()
    add_text_row(
        table,
        Text("total", style="bold"),
        "",
        formatting.count(totals.documents),
        formatting.count(totals.chunks),
        formatting.count(totals.tokens),
        "",
        formatting.count(totals.tokens_p50),
        formatting.count(totals.tokens_p95),
        "",
    )
    return table


def chunk_length_histogram(stats: CorpusStats) -> Table:
    """The chunk-length histogram from the corpus statistics, as bars, one per `LENGTH_BINS` bin.

    Each bucket is labeled by its token range, both bounds included, with an en dash
    (`750\u2013999`), and the last as `1,000 and over`.
    """
    table = Table(title="chunk length distribution (tokens)")
    table.add_column("bucket")
    table.add_column("chunks", justify="right")
    table.add_column("", ratio=1)
    counts = stats.chunk_length_histogram.values()
    # An empty corpus has every bucket at 0; 1 keeps the bar width defined.
    peak = max(counts, default=0) or 1
    for (low, high), count in zip(LENGTH_BINS, counts, strict=True):
        label = (
            f"{formatting.count(low)}\u2013{formatting.count(high - 1)}"
            if high is not None
            else f"{formatting.count(low)} and over"
        )
        add_text_row(
            table, label, formatting.count(count), "█" * int(HISTOGRAM_WIDTH * count / peak)
        )
    return table


def dedup_summary(dedup: DedupSummary) -> Table:
    """What exact deduplication merged, and what it deliberately left alone."""
    table = Table(
        title="exact deduplication", show_header=False, box=None, padding=BORDERLESS_PADDING
    )
    add_text_row(table, "chunks before", formatting.count(dedup.chunks_before))
    add_text_row(table, "chunks after", formatting.count(dedup.chunks_after))
    add_text_row(table, "removed copies", formatting.count(dedup.removed))
    add_text_row(
        table,
        "groups merged",
        f"{formatting.count(dedup.groups)} "
        f"({formatting.plural(dedup.occurrences_in_groups, 'occurrence')})",
    )
    add_text_row(
        table,
        "  by venue",
        "  ".join(f"{venue}={formatting.count(n)}" for venue, n in dedup.groups_by_venue.items()),
    )
    add_text_row(
        table, "  spanning testnet + non-testnet", formatting.count(dedup.testnet_mirror_groups)
    )
    add_text_row(
        table, "cross-venue groups (kept apart)", formatting.count(dedup.cross_venue_groups)
    )
    return table


def metadata_coverage(coverage: MetadataCoverage) -> Table:
    """Share of chunks each structural field was assigned on, and where not."""
    table = Table(
        title=Text(
            f"structural metadata coverage ({formatting.plural(coverage.chunks, 'chunk')})",
            style=TITLE_STYLE,
        )
    )
    table.add_column("field")
    table.add_column("assigned", justify="right")
    table.add_column("share", justify="right")
    table.add_column("values")
    table.add_column("unassigned by venue")
    for name, row in coverage.fields.items():
        add_text_row(
            table,
            name,
            formatting.count(row.assigned),
            formatting.percent(row.share),
            "  ".join(f"{k}={formatting.count(v)}" for k, v in row.values.items()),
            "  ".join(f"{k}={formatting.count(v)}" for k, v in row.unassigned_by_venue.items()),
        )
    return table


def retrieval_results(method: str, k: int, results: Sequence[Result], timings: Timings) -> Table:
    """Ranked results with a text preview, titled with each stage's latency."""
    table = Table(
        title=Text(
            f"{method} | k={formatting.own_form(k)} | "
            f"encode {formatting.ms(timings.encode_ms)} ms | "
            f"search {formatting.ms(timings.search_ms)} ms | "
            f"fuse {formatting.ms(timings.fuse_ms)} ms | "
            f"rerank {formatting.ms(timings.rerank_ms)} ms | "
            f"total {formatting.ms(timings.total_ms)} ms",
            style=TITLE_STYLE,
        ),
        show_lines=True,
        expand=True,
    )
    table.add_column("#", width=RANK_WIDTH, justify="right")
    table.add_column("score", width=SCORE_WIDTH, justify="right")
    table.add_column("venue", width=VENUE_WIDTH)
    table.add_column("heading path", width=HEADING_WIDTH, overflow="fold")
    table.add_column("text", overflow="fold")
    for position, result in enumerate(results, start=1):
        add_text_row(
            table,
            formatting.count(position),
            formatting.number(result.score, SCORE_DECIMALS),
            result.venue,
            result.heading_display,
            _preview(result.text),
        )
    return table


def validation_stats(stats: ValidationStats) -> Table:
    """Question counts of a validated set: totals and per category, qtype and venue."""
    table = Table(show_header=False, box=None, padding=BORDERLESS_PADDING)
    add_text_row(table, "total", formatting.count(stats.total))
    add_text_row(table, "with gold chunks", formatting.count(stats.with_gold))
    add_text_row(
        table,
        "answerable / unanswerable",
        f"{formatting.count(stats.answerable)} / {formatting.count(stats.unanswerable)}",
    )
    for name, counts in (
        ("category", stats.by_category),
        ("qtype", stats.by_qtype),
        ("venue", stats.by_venue),
    ):
        add_text_row(
            table, name, "  ".join(f"{k}={formatting.count(n)}" for k, n in sorted(counts.items()))
        )
    return table


def validation_messages(report: ValidationReport) -> list[Text]:
    """Warnings then errors, as literal text: messages quote question text and ids."""
    return [Text.assemble("  ", ("warn", "yellow"), "  ", m) for m in report.warnings] + [
        Text.assemble("  ", ("error", "red"), " ", m) for m in report.errors
    ]


def _rounds(first: int, second: int) -> str:
    """Two compared rounds as a title names them: `rounds 1 vs 2`."""
    return f"rounds {formatting.own_form(first)} vs {formatting.own_form(second)}"


def _only(round_number: int) -> str:
    """The verdicts one round alone gave: `relevant in round 1 only`."""
    return f"relevant in round {formatting.own_form(round_number)} only"


def rounds_agreement_table(report: AgreementReport, first: int, second: int) -> Table:
    """Pair counts, observed and chance agreement, and Cohen's kappa for two rounds."""
    table = Table(
        title=Text(_rounds(first, second), style=TITLE_STYLE),
        show_header=False,
        box=None,
        padding=BORDERLESS_PADDING,
    )
    add_text_row(table, "questions compared", formatting.count(report.questions))
    add_text_row(table, "(question, chunk) pairs", formatting.count(report.pairs))
    add_text_row(table, "both relevant", formatting.count(report.both_relevant))
    add_text_row(table, "both irrelevant", formatting.count(report.both_irrelevant))
    add_text_row(table, "disagreements", formatting.count(report.disagreements))
    add_text_row(table, "observed agreement", formatting.metric(report.observed_agreement))
    add_text_row(table, "expected by chance", formatting.metric(report.expected_agreement))
    add_text_row(table, "Cohen's kappa", formatting.metric(report.kappa), style="bold")
    add_text_row(table, "interpretation", report.interpretation)
    return table


def kappa_by_category_table(rows: list[tuple[str, AgreementReport]]) -> Table:
    """Agreement per question category, most disagreement first."""
    table = Table(title="agreement by category")
    for column in ("category", "questions", "pairs", "disagreements", "observed", "kappa"):
        table.add_column(column, justify="left" if column == "category" else "right")
    for category, r in rows:
        add_text_row(
            table,
            category,
            formatting.count(r.questions),
            formatting.count(r.pairs),
            formatting.count(r.disagreements),
            formatting.metric(r.observed_agreement),
            formatting.metric(r.kappa),
        )
    return table


def agreement_variants_table(variants: list[AgreementVariant], first: int, second: int) -> Table:
    """The registered reading first, then the sensitivity checks (methodology section 5)."""
    table = Table(title=Text(f"{_rounds(first, second)}: kappa by reading", style=TITLE_STYLE))
    for column in ("round 1 as", "questions", "pairs", "questions compared", "observed", "kappa"):
        table.add_column(column, justify="left" if column == "round 1 as" else "right")
    for v in variants:
        registered = v.reference == "corrected" and v.scope == "all"
        label = f"{v.reference}{'' if registered else ' (sensitivity)'}"
        add_text_row(
            table,
            label,
            v.scope,
            formatting.count(v.report.pairs),
            formatting.count(v.report.questions),
            formatting.metric(v.report.observed_agreement),
            formatting.metric(v.report.kappa),
        )
    return table


def kappa_intervals_table(variants: list[AgreementVariant], first: int, second: int) -> Table:
    """Each reading's kappa with its bootstrap interval over questions."""
    table = Table(title=Text(f"{_rounds(first, second)}: kappa interval", style=TITLE_STYLE))
    level = f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% interval"
    for column in ("round 1 as", "questions", "kappa", level, "undefined resamples"):
        table.add_column(column, justify="left" if column == "round 1 as" else "right")
    for v in variants:
        ci = v.interval
        if ci is None:
            continue
        bounds = (
            formatting.MISSING
            if ci.low is None or ci.high is None
            else formatting.bounds(ci.low, ci.high)
        )
        add_text_row(
            table,
            v.reference,
            v.scope,
            formatting.metric(v.report.kappa),
            bounds,
            f"{formatting.count(ci.undefined)} of {formatting.count(ci.resamples)}",
        )
    return table


def disagreement_direction_table(
    total: VerdictTable, rows: list[tuple[str, VerdictTable]], first: int, second: int
) -> Table:
    """Disagreements split by the round that judged the pair relevant, in total and per category."""
    table = Table(title=Text("direction of disagreement", style=TITLE_STYLE))
    for column in ("category", "disagreements", _only(first), _only(second)):
        table.add_column(column, justify="left" if column == "category" else "right")
    for category, t in [*rows, ("total", total)]:
        add_text_row(
            table,
            category,
            formatting.count(t.disagreements),
            formatting.count(t.first_only),
            formatting.count(t.second_only),
        )
    return table


def source_chunks_table(sources: SourceChunks, first: int, second: int) -> Table:
    """How each round judged the compared questions' source chunks.

    Round 1 marked them on screen and the later round hid the marker.
    """
    table = Table(
        title=Text("source chunks", style=TITLE_STYLE),
        show_header=False,
        box=None,
        padding=BORDERLESS_PADDING,
    )
    v = sources.verdicts
    add_text_row(table, "both relevant", formatting.count(v.both_relevant))
    add_text_row(table, _only(first), formatting.count(v.first_only))
    add_text_row(table, _only(second), formatting.count(v.second_only))
    add_text_row(table, "both irrelevant", formatting.count(v.both_irrelevant))
    add_text_row(table, "not judged in both rounds", formatting.count(sources.unjudged))
    return table


def composition(report: CompositionReport) -> Table:
    """Accepted against target per category, for `select --finalize`."""
    table = Table(title="accepted composition")
    table.add_column("category")
    table.add_column("accepted", justify="right")
    table.add_column("target", justify="right")
    table.add_column("", width=STATUS_WIDTH)
    for category, target in TARGETS.items():
        accepted = report.accepted[category]
        add_text_row(
            table,
            category,
            formatting.count(accepted),
            formatting.count(target),
            _mark(accepted, target),
        )
    table.add_section()
    add_text_row(
        table,
        Text("total", style="bold"),
        formatting.count(sum(report.accepted.values())),
        formatting.count(target_size()),
        "",
    )
    add_text_row(
        table,
        "answerable comparison",
        formatting.count(report.answerable_comparison),
        formatting.count(TARGETS["comparison"]),
        _mark(report.answerable_comparison, TARGETS["comparison"]),
    )
    return table


def _mark(accepted: int, target: int) -> Text:
    """`ok` in green when `accepted` meets `target`, `short` in red when not."""
    return Text("ok", style="green") if accepted >= target else Text("short", style="red")


def rejection_reasons(reasons: Counter[str]) -> Table:
    """Why candidates were dropped: the record of how the set was chosen."""
    table = Table(title="rejections")
    table.add_column("reason")
    table.add_column("count", justify="right")
    for reason, count in reasons.most_common():
        add_text_row(table, reason, formatting.count(count))
    table.add_section()
    add_text_row(table, Text("total", style="bold"), formatting.count(sum(reasons.values())))
    return table


def term_counts(term: str, counts: TermCount, venue: str | None) -> Table:
    """Chunks containing a term, per venue, by the project's one counting convention."""
    scope = f" in {venue}" if venue else ""
    table = Table(
        title=Text(f'chunks containing "{term}"{scope} (case-insensitive)', style=TITLE_STYLE)
    )
    table.add_column("venue")
    table.add_column("chunks", justify="right")
    for name, count in sorted(counts.by_venue.items(), key=lambda kv: -kv[1]):
        add_text_row(table, name, formatting.count(count))
    table.add_section()
    add_text_row(table, "total", formatting.count(counts.total), style="bold")
    return table


def power_lines(study: PowerStudy) -> list[str]:
    """The summary `crossvenue power` prints.

    The family, the reachable deltas in steps of 0.05, and the smallest delta
    with 80% power.
    """
    family = formatting.count(len(study.family))
    lines = [
        f"{study.metric} on {formatting.count(study.questions)} {study.category} "
        f"questions, family of {family}: {formatting.count(study.fractional_values)} of "
        f"{formatting.count(study.values)} per-question values fractional; "
        f"{formatting.count(study.discordant_pairs)} of "
        f"{formatting.count(study.question_pairs)} question pairs differ "
        f"(psi {formatting.design_parameter(study.psi)}), mean |difference| "
        f"{formatting.metric(study.mean_magnitude)}, largest reachable delta "
        f"{formatting.design_parameter(study.ceiling)}"
    ]
    for p in study.points:
        # Only reachable points are listed, and a reachable point carries both powers.
        holm, unadjusted = p.power_holm_first_step, p.power_unadjusted
        if (
            p.reachable
            and holm is not None
            and unadjusted is not None
            and math.isclose(p.delta / POWER_LINE_STEP, round(p.delta / POWER_LINE_STEP))
        ):
            lines.append(
                f"  delta {formatting.design_parameter(p.delta)}: power "
                f"{formatting.design_parameter(holm)} at alpha/{family}, "
                f"{formatting.design_parameter(unadjusted)} at alpha"
            )
    lines.append(
        f"{formatting.count(TARGET_POWER_PERCENT)}% power from delta "
        f"{_design_or_none(study.delta_80_holm_first_step)} "
        f"(alpha/{family}), {_design_or_none(study.delta_80_unadjusted)} (alpha)"
    )
    return lines


def _design_or_none(value: float | None) -> str:
    """A delta, or `not reached` when no simulated delta reached the power."""
    return "not reached" if value is None else formatting.design_parameter(value)
