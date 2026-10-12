"""report.md of a retrieval run: its settings, metric tables, comparisons and step latency.

Numbers go through `formatting`, the seed and the run's settings (chunk sizes, the RRF k) in
their own form (`formatting.own_form`, docs/style.md rule 11), a cutoff in a metric name as
part of the name. A metric a slice cannot have (no labels, no judgment history) shows `-`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import formatting
from ..config import Config
from ..models import MULTI_VENUE, RETRIEVAL_STEPS, Question
from ..stats import INTERVAL_LEVEL_PERCENT
from .comparisons import PairedRow, paired_table
from .evaluator import (
    MethodSummary,
    QueryRun,
    SliceSummary,
    metric_names_for,
    venue_metric_names_for,
)
from .labels import POOL_NOTE, fmt, markdown_table, method_label
from .paired import MIN_TESTED_QUESTIONS

# The heading of the per-step latency table, which reporting reads back.
LATENCY_HEADING = "## Latency (p50, ms per step)"


@dataclass(frozen=True)
class RunTables:
    """What the report shows.

    The per-query runs, the method summaries, the breakdowns by dimension, the
    paired and off-venue comparisons, and the chunks of each venue in the corpus
    the run retrieved from. `venue_filter` is set when `eval --venue-filter`
    restricted each query to the question's labeled venues.
    """

    runs: Sequence[QueryRun]
    summaries: Sequence[MethodSummary]
    breakdowns: Mapping[str, list[SliceSummary]]
    paired: Sequence[PairedRow]
    off_venue: Sequence[PairedRow]
    venue_chunks: Mapping[str, int]
    venue_filter: bool = False


def report_markdown(
    cfg: Config, run_name: str, index_version: str, questions: Sequence[Question], tables: RunTables
) -> str:
    """The whole report.md."""
    lines = [
        *_header(cfg, run_name, index_version, questions, venue_filter=tables.venue_filter),
        *_metrics(cfg, questions, tables),
    ]
    if tables.paired:
        lines += _paired(cfg, tables.paired)
    if tables.off_venue:
        lines += _off_venue(tables.off_venue)
    lines += ["", LATENCY_HEADING, "", _latency_table(tables.summaries), ""]
    return "\n".join(lines)


def _header(
    cfg: Config,
    run_name: str,
    index_version: str,
    questions: Sequence[Question],
    *,
    venue_filter: bool,
) -> list[str]:
    labelled = sum(1 for q in questions if q.is_labelled)
    filtered = (
        ["- venue filter: each query restricted to the question's labeled venues"]
        if venue_filter
        else []
    )
    return [
        "# Retrieval ablation",
        "",
        f"- run: `{run_name}`",
        f"- `index_version`: `{index_version}`",
        f"- questions file: `{cfg.eval.questions}`",
        f"- questions: {formatting.count(len(questions))} "
        f"({formatting.count(labelled)} with gold labels)",
        f"- embedding: `{cfg.embedding.model_name}`",
        f"- reranker: `{cfg.reranker.model_name}`",
        f"- chunking: max {formatting.own_form(cfg.chunking.max_tokens)} tokens, "
        f"overlap {formatting.own_form(cfg.chunking.overlap_tokens)}",
        f"- RRF k: {formatting.own_form(cfg.retrieval.rrf_k)}",
        *filtered,
        "",
        "## Metrics",
        "",
    ]


def _metrics(cfg: Config, questions: Sequence[Question], tables: RunTables) -> list[str]:
    if not (any(q.is_labelled for q in questions) or any(r.multi_venue for r in tables.runs)):
        return [
            "The question file holds no gold labels, so this report has no retrieval "
            "metrics. `crossvenue label` marks relevant chunks, and an evaluation after it "
            "reports the metrics."
        ]
    k_values = cfg.eval.k_values
    lines = [
        _markdown_table(tables.summaries, k_values),
        f"\n`venue_coverage@k`: share of questions naming {formatting.count(MULTI_VENUE)} or "
        "more venues (`multi-venue`) whose top k hold a chunk from every one of them. It "
        "needs no gold labels.",
        "\n" + POOL_NOTE,
    ]
    for dimension, slices in tables.breakdowns.items():
        if not slices:
            continue
        lines += ["", f"### By {dimension}", "", _slice_table(slices, k_values)]
        if dimension == "venue":
            lines.append(
                "\nA question naming two venues counts in both rows, so venue rows "
                "do not sum to the total. " + _balance_note(tables.venue_chunks)
            )
    return lines


def _paired(cfg: Config, rows: Sequence[PairedRow]) -> list[str]:
    ev = cfg.eval
    return [
        "",
        "## Paired comparisons",
        "",
        "Same questions, both methods. `a \u2212 b` with a "
        f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% paired bootstrap interval "
        f"({formatting.plural(ev.bootstrap_resamples, 'resample')}, seed "
        f"{formatting.own_form(ev.bootstrap_seed)}); exact two-sided McNemar on the binary "
        "venue_coverage; two-sided paired randomization test (sign flip, "
        f"{formatting.plural(ev.bootstrap_resamples, 'permutation')}, seed "
        f"{formatting.own_form(ev.bootstrap_seed)}) on every metric. The primary comparison is "
        f"{' '.join(ev.primary_comparison) or 'not declared'}; every other row is "
        "secondary, with its randomization p-value Holm-adjusted within its metric "
        "and scope." + _descriptive_note(rows),
        "",
        paired_table(rows),
    ]


def _descriptive_note(rows: Sequence[PairedRow]) -> str:
    """The sentence that names the descriptive rows, when the table has any."""
    if not any(r.role == "descriptive" for r in rows):
        return ""
    return (
        f" A row of a scope under {formatting.plural(MIN_TESTED_QUESTIONS, 'question')} is "
        "descriptive instead: an interval and no p-value."
    )


def _off_venue(rows: Sequence[PairedRow]) -> list[str]:
    return [
        "",
        "## Off-venue retrieval (exploratory, methodology section 9b)",
        "",
        "`off_venue@k`: share of the top k whose venue the question does not name "
        "(ccxt apart, as `via_ccxt@k`); `off_venue@1` is the top-1 off-venue rate. "
        "Tests as in the paired comparisons, Holm-adjusted within each metric and scope. "
        "Lower is better for both metrics: `better` counts the questions where a has "
        "the smaller share." + _descriptive_note(rows),
        "",
        paired_table(rows),
    ]


def _shown_venue_metrics(k_values: Sequence[int]) -> list[str]:
    """venue_coverage@k for the markdown tables.

    k = 1 is 0 by construction (one chunk covers one venue), so it stays in the
    CSV files and out of the tables.
    """
    return venue_metric_names_for([k for k in k_values if k >= MULTI_VENUE])


def _metric_header(k_values: Sequence[int]) -> list[str]:
    """The metric columns both metric tables show, after their leading columns."""
    return [
        *metric_names_for(k_values),
        f"ndcg_prime@{max(k_values)}",
        "multi-venue",
        *_shown_venue_metrics(k_values),
        f"unjudged@{max(k_values)}",
    ]


def _metric_cells(summary: MethodSummary | SliceSummary, k_values: Sequence[int]) -> list[str]:
    """One summary's cells under `_metric_header`; a metric it lacks shows `-`."""
    return [
        *[fmt(summary.metrics.get(name)) for name in metric_names_for(k_values)],
        fmt(summary.judged_metrics.get(f"ndcg_prime@{max(k_values)}")),
        formatting.count(summary.multi_venue_questions),
        *[fmt(summary.venue_metrics.get(name)) for name in _shown_venue_metrics(k_values)],
        fmt(summary.judged_metrics.get(f"unjudged@{max(k_values)}")),
    ]


def _slice_table(summaries: Sequence[SliceSummary], k_values: Sequence[int]) -> str:
    """One row per (slice, method).

    On a slice with no labeled question, the gold metrics, `ndcg_prime@k` and `unjudged@k`
    show `-`. The question counts, the multi-venue count and p50 ms need no labels, and
    venue_coverage@k needs only a multi-venue question in the slice (`-` without one).
    """
    header = [summaries[0].dimension, "method", "questions", "labeled", *_metric_header(k_values)]
    rows = [
        [
            s.slice,
            s.method,
            formatting.count(s.questions),
            formatting.count(s.labelled_questions),
            *_metric_cells(s, k_values),
            formatting.ms(s.latency_p50_ms),
        ]
        for s in summaries
    ]
    return markdown_table([*header, "p50 ms"], rows)


def _markdown_table(summaries: Sequence[MethodSummary], k_values: Sequence[int]) -> str:
    """One row per method, with the p50 and p95 of its steps sum."""
    rows = [
        [
            method_label(s.method),
            *_metric_cells(s, k_values),
            formatting.ms(s.latency_p50_ms),
            formatting.ms(s.latency_p95_ms),
        ]
        for s in summaries
    ]
    return markdown_table(["method", *_metric_header(k_values), "p50 ms", "p95 ms"], rows)


def _balance_note(venue_chunks: Mapping[str, int]) -> str:
    """The venue imbalance, from the chunks of each venue the run retrieved from."""
    counts = Counter(venue_chunks)
    if len(counts) < MULTI_VENUE:
        return ""
    (large, most), (small, least) = counts.most_common()[0], counts.most_common()[-1]
    return (
        f"The corpus is unbalanced ({large} {formatting.count(most)} chunks vs {small} "
        f"{formatting.count(least)}), so the aggregate above is weighted toward whichever "
        "venues the question set names most."
    )


def _latency_table(summaries: Sequence[MethodSummary]) -> str:
    rows = [
        [
            summary.method,
            *(formatting.ms(summary.step_latency_p50_ms[f"{s}_ms"]) for s in RETRIEVAL_STEPS),
            formatting.ms(summary.latency_p50_ms),
        ]
        for summary in summaries
    ]
    return markdown_table(["method", *RETRIEVAL_STEPS, "steps sum p50"], rows)
