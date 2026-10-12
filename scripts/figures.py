"""Draws each findings figure from saved runs into docs/figures/, as SVG plus PNG: `make figures`.

It reads only `results/`, `evals/datasets/` and `data/processed/chunks.jsonl` through
`crossvenue_rag.reporting.sources`, and the latency, calibration and wrong-refusals figures
draw the run's own tables, the last two with the bootstrap intervals their runs report. A
recomputed value must equal the number its run reports before it is drawn (`agree`), except
the per-method means' bootstrap intervals, which no run reports, and the outcomes figure's
per-category bars, whose sums are checked (`answer_figures`).
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from answer_figures import draw_calibration, draw_outcomes, draw_wrong_refusals
from figure_style import (
    DARK,
    INK,
    LIGHT,
    MUTED,
    ORACLE,
    STEP_COLORS,
    add_caption,
    agree,
    save_figure,
    value_grid,
)
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from crossvenue_rag import formatting
from crossvenue_rag.evaluation.comparisons import OFF_VENUE_SCOPES
from crossvenue_rag.evaluation.evaluator import METRIC_DECIMALS
from crossvenue_rag.evaluation.metrics import ndcg_at_k, recall_at_k
from crossvenue_rag.evaluation.results import PER_QUERY_FILE, breakdown_file
from crossvenue_rag.models import ADVERSARIAL, COMPARISON, RETRIEVAL_STEPS
from crossvenue_rag.reporting.compute.figures import (
    bootstrap_mean_interval,
    category_per_query,
    labelled_values,
    off_venue_shares,
)
from crossvenue_rag.reporting.compute.lookup import entry
from crossvenue_rag.reporting.compute.retrieval import (
    METHODS,
    VENUE_FILTERED,
    at_top_k,
    category_metric,
    pool_depth,
    recall_against,
    step_p50,
    summary_metric,
    top_k,
)
from crossvenue_rag.reporting.compute.retrieval_off_venue import off_venue_means
from crossvenue_rag.reporting.run_ids import COVERAGE_FAILURE_RUN, RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources
from crossvenue_rag.retrieval.methods import ORACLE_METHODS
from crossvenue_rag.run_files import REPORT_FILE
from crossvenue_rag.stats import INTERVAL_LEVEL_PERCENT

# A step faster than this is not drawn as a bar; the caption gives how many there are.
SMALLEST_BAR_MS = 0.05
# The fixed axis limits, each above the largest value its figure draws: a share axis ends
# at its top, a range is (left, right) or (bottom, top), a latency axis ends in ms.
COVERAGE_TOP = 1.12
RECALL_TOP = 0.85
POOLING_RANGE = (0.3, 0.75)
LATENCY_RIGHT_MS = 10_000
OFF_VENUE_RIGHT = 0.6
# Where a value is written beside what it labels, in axis units: above a bar, and right of a
# dot and, since the pooling figure's y axis is inverted, below it.
VALUE_GAP = 0.02
POOLING_LABEL_OFFSET = (0.008, 0.18)
# Half the height of the latency figure's steps-sum mark, and the spread of its step bars
# about each method's row.
TOTAL_HALF_HEIGHT = 0.38
STEP_SPREAD = 0.27


def interval_note(src: Sources) -> str:
    """The caption sentence naming the interval level, method, resamples and seed."""
    cfg = src.retrieval.config.eval
    return (
        f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% percentile bootstrap interval over "
        "questions, "
        f"{formatting.count(cfg.bootstrap_resamples)} resamples, "
        f"seed {formatting.own_form(cfg.bootstrap_seed)}"
    )


def method_axis(ax: Axes) -> None:
    """Labels the x axis with the method names, each on one line.

    A slight rotation and a smaller font keep venue_split_oracle unbroken.
    """
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=15, ha="right", rotation_mode="anchor", fontsize=8)


def draw_venue_coverage(src: Sources) -> None:
    """Draws venue_coverage@k per method on the comparison questions, k the run's top k."""
    k = top_k(src)
    coverage = at_top_k(src, "venue_coverage")
    rows = category_per_query(src, COMPARISON)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, m in enumerate(METHODS):
        interval = bootstrap_mean_interval([r.value(coverage) for r in rows[m]], src)
        mean, low, high = interval.value, interval.low, interval.high
        agree(mean, category_metric(src, COMPARISON, m, coverage), m)
        color = ORACLE if m in ORACLE_METHODS else DARK
        ax.bar(i, mean, width=0.6, color=color)
        ax.errorbar(i, mean, yerr=[[mean - low], [high - mean]], color=INK, capsize=3, lw=0.8)
        ax.text(i, high + VALUE_GAP, formatting.metric(mean), ha="center", va="bottom", fontsize=8)
    method_axis(ax)
    ax.set_ylim(0, COVERAGE_TOP)
    ax.set_ylabel(f"{coverage} (share of questions)")
    value_grid(ax)
    n = len(rows["bm25"])
    add_caption(
        fig,
        f"Share of the {formatting.count(n)} comparison questions whose top "
        f"{formatting.count(k)} hold a chunk of every named venue. "
        f"{interval_note(src)}. venue_split_oracle (gray) uses the labeled venues: it bounds "
        "what better venue detection can add to coverage. "
        f"Run results/{RETRIEVAL_RUN}.",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "venue_coverage_comparison")


def draw_recall_ndcg(src: Sources) -> None:
    """Draws recall@k and nDCG@k per method on the labeled comparison questions."""
    k = top_k(src)
    recall = at_top_k(src, "recall")
    metrics = (
        (recall, recall, recall_at_k, DARK),
        (at_top_k(src, "ndcg"), f"nDCG@{formatting.own_form(k)}", ndcg_at_k, LIGHT),
    )
    rows = category_per_query(src, COMPARISON)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.38
    for i, m in enumerate(METHODS):
        for j, (metric, label, fn, color) in enumerate(metrics):
            interval = bootstrap_mean_interval(labelled_values(src, rows[m], fn), src)
            mean, low, high = interval.value, interval.low, interval.high
            agree(
                round(mean, METRIC_DECIMALS),
                category_metric(src, COMPARISON, m, metric),
                f"{m} {metric}",
            )
            x = i + (j - 0.5) * width
            ax.bar(x, mean, width=width, color=color, label=label if i == 0 else None)
            ax.errorbar(x, mean, yerr=[[mean - low], [high - mean]], color=INK, capsize=2, lw=0.7)
    method_axis(ax)
    ax.set_ylim(0, RECALL_TOP)
    ax.set_ylabel("mean over comparison questions")
    ax.legend(loc="upper left", ncols=2)
    value_grid(ax)
    add_caption(
        fig,
        f"{metrics[0][1]} and {metrics[1][1]} on the "
        f"{formatting.count(len(rows['bm25']))} comparison questions, "
        "against "
        f"v0.2.1-gold. {interval_note(src)}. Run results/{RETRIEVAL_RUN}.",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, "recall_ndcg_comparison")


def draw_pooling(src: Sources) -> None:
    """Draws each method's comparison recall@k against both gold versions."""
    recall = at_top_k(src, "recall")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, m in enumerate(METHODS):
        old = recall_against(src, m, src.questions_before_augmentation)
        new = recall_against(src, m, src.questions)
        # The run on v0.2-gold holds the same retrievals scored against that gold.
        before = entry(
            src.run(COVERAGE_FAILURE_RUN).by_category,
            (COMPARISON, m),
            f"{breakdown_file('category')} of results/{COVERAGE_FAILURE_RUN}",
        )
        agree(round(old, METRIC_DECIMALS), before.value(recall), f"{m} against v0.2-gold")
        agree(round(new, METRIC_DECIMALS), category_metric(src, COMPARISON, m, recall), m)
        ax.plot([old, new], [i, i], color=MUTED, lw=1.2, zorder=1)
        ax.scatter([old], [i], color=LIGHT, s=36, zorder=2, label="v0.2-gold" if i == 0 else None)
        ax.scatter([new], [i], color=DARK, s=36, zorder=3, label="v0.2.1-gold" if i == 0 else None)
        ax.text(
            old + POOLING_LABEL_OFFSET[0],
            i + POOLING_LABEL_OFFSET[1],
            formatting.signed(new - old),
            fontsize=7.5,
            color=MUTED,
        )
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    ax.invert_yaxis()
    ax.set_xlim(*POOLING_RANGE)
    ax.set_xlabel(f"{recall} on comparison questions (same retrievals)")
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1.0, 1.0))
    value_grid(ax, "x")
    add_caption(
        fig,
        "v0.2.1-gold adds the gold chunks found by judging venue_split's and the oracle's "
        f"never-judged top {formatting.count(pool_depth(src))}. Run results/{RETRIEVAL_RUN}, "
        "scored against both "
        "gold versions.",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "recall_gold_versions")


def draw_latency(src: Sources) -> None:
    """Draws each method's p50 latency per retrieval step, with the steps' sum.

    Every method of `METHODS` is drawn, so the run must hold each one.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    offsets = dict(
        zip(
            RETRIEVAL_STEPS,
            np.linspace(-STEP_SPREAD, STEP_SPREAD, len(RETRIEVAL_STEPS)),
            strict=True,
        )
    )
    zeros = []
    for i, m in enumerate(METHODS):
        for step in RETRIEVAL_STEPS:
            value = step_p50(src, m, step)
            if value < SMALLEST_BAR_MS:
                zeros.append(f"{m} {step}")
                continue
            ax.barh(i + offsets[step], value, height=0.17, color=STEP_COLORS[step])
        total = summary_metric(src, m, "p50_ms")
        ax.plot([total, total], [i - TOTAL_HALF_HEIGHT, i + TOTAL_HALF_HEIGHT], color=INK, lw=1.2)
    handles: list[Artist] = [
        Rectangle((0, 0), 1, 1, color=STEP_COLORS[s_]) for s_ in RETRIEVAL_STEPS
    ]
    handles.append(Line2D([], [], color=INK, lw=1.2))
    ax.legend(
        handles,
        [*RETRIEVAL_STEPS, "steps sum p50"],
        loc="lower right",
        ncols=5,
        fontsize=7.5,
        bbox_to_anchor=(1.0, 1.0),
    )
    ax.set_xscale("log")
    ax.set_xlim(SMALLEST_BAR_MS, LATENCY_RIGHT_MS)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    ax.invert_yaxis()
    ax.set_xlabel("p50 per query, ms (log scale)")
    value_grid(ax, "x")
    add_caption(
        fig,
        f"Nearest-rank p50 of each step over {formatting.count(len(src.questions))} questions; "
        "steps under "
        f"{formatting.own_form(SMALLEST_BAR_MS, formatting.CODE_BOUND_DECIMALS)} ms are not drawn "
        f"({formatting.count(len(zeros))} of "
        f"{formatting.count(len(METHODS) * len(RETRIEVAL_STEPS))}). "
        f"Run results/{RETRIEVAL_RUN}, {REPORT_FILE}.",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(fig, "latency_steps")


def draw_off_venue(src: Sources) -> None:
    """Draws off-venue and via-ccxt share@k per method, on comparison and adversarial questions.

    The two shares are stacked, and the two categories drawn side by side. Each
    mean is recomputed from the run's saved `retrieved_venues` and must equal
    the value findings uses (methodology 9b).
    """
    k = top_k(src)
    off_metric, via_metric = at_top_k(src, "off_venue"), at_top_k(src, "via_ccxt")
    panels = OFF_VENUE_SCOPES
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 3.4), sharey=True)
    sizes = {}
    for ax, category in zip(axes, panels, strict=True):
        rows = category_per_query(src, category)
        for i, m in enumerate(METHODS):
            off, via = off_venue_shares(src, rows[m])
            size, means = off_venue_means(src, m, category)
            agree(off, means[off_metric], f"{m} {category} {off_metric}")
            agree(via, means[via_metric], f"{m} {category} {via_metric}")
            sizes[category] = size
            color = ORACLE if m in VENUE_FILTERED else DARK
            ax.barh(i, off, height=0.6, color=color, label="off-venue" if i == 0 else None)
            ax.barh(i, via, left=off, height=0.6, color=LIGHT, label="via ccxt" if i == 0 else None)
        ax.set_title(
            f"{category} ({formatting.count(sizes[category])} questions)", fontsize=8.5, color=INK
        )
        ax.set_xlim(0, OFF_VENUE_RIGHT)
        ax.set_xlabel(f"share of the top {formatting.count(k)}")
        value_grid(ax, "x")
    axes[0].set_yticks(range(len(METHODS)))
    axes[0].set_yticklabels(METHODS)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.0))
    add_caption(
        fig,
        f"Exploratory (methodology \u00a79b). Mean share of each question's top "
        f"{formatting.count(k)} from a venue "
        "the question does not name (off-venue), with ccxt chunks on questions that do not "
        "name ccxt stacked apart (via ccxt); "
        f"n = {formatting.count(sizes[COMPARISON])} comparison "
        f"and {formatting.count(sizes[ADVERSARIAL])} adversarial questions. "
        "venue_split and venue_split_oracle "
        "(gray) filter by venue by construction. Run "
        f"results/{RETRIEVAL_RUN}, retrieved_venues of {PER_QUERY_FILE}.",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    save_figure(fig, "off_venue")


def main() -> None:
    """Draws every figure from the runs under the working directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src = Sources(".")
    draw_venue_coverage(src)
    draw_recall_ndcg(src)
    draw_pooling(src)
    draw_latency(src)
    draw_outcomes(src)
    draw_calibration(src)
    draw_wrong_refusals(src)
    draw_off_venue(src)


if __name__ == "__main__":
    main()
