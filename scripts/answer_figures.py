"""The findings figures of the answers: refusals and partial answers, the judge calibration, H-v2.

Each configuration's refused and partial totals and wrong-refusal shares are checked against
its run's summary.csv and comparison.csv (`figure_style.agree`) before they are drawn; the
per-category bars of those totals are drawn unchecked. The calibration figure draws
calibration.json as the run saved it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_style import (
    DARK,
    INK,
    LIGHT,
    MUTED,
    ORACLE,
    add_caption,
    agree,
    save_figure,
    value_grid,
)

from crossvenue_rag import formatting, listing
from crossvenue_rag.generation.calibration_file import CALIBRATION_FILE, human_comparison
from crossvenue_rag.reporting.compute import calibration
from crossvenue_rag.reporting.compute.answers import (
    PROMPT_V1,
    PROMPT_V2,
    configurations,
    outcome_count,
)
from crossvenue_rag.reporting.compute.answers import wrong_refusals as wrong_refusal_count
from crossvenue_rag.reporting.compute.answers_outcomes import CATEGORIES
from crossvenue_rag.reporting.compute.datasets import category_count
from crossvenue_rag.reporting.compute.lookup import entry
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, ANSWERS_V2
from crossvenue_rag.reporting.sources import Sources
from crossvenue_rag.run_files import SUMMARY_FILE
from crossvenue_rag.stats import INTERVAL_LEVEL_PERCENT

# The fixed axis limits, each past the largest value its figure draws: the answers per
# category, the agreement (bottom, top), the wrong-refusal share.
OUTCOMES_RIGHT = 50
AGREEMENT_RANGE = (-0.1, 1.1)
WRONG_REFUSALS_RIGHT = 0.68
# Where a mark or a label is written, in axis units: above an interval, (right, up) of a dot.
MARK_GAP = 0.03
REFUSAL_LABEL_OFFSET = (0.012, -0.12)
# Half the width of a calibration threshold line.
THRESHOLD_HALF_WIDTH = 0.4
# The judge candidates the calibration figure draws, side by side per dimension.
CANDIDATES_DRAWN = 2


def draw_outcomes(src: Sources) -> None:
    """Draws refused and partial answers per category and configuration, prompt v1 and v2."""
    configs = configurations(src)
    fig, axes = plt.subplots(1, len(configs), figsize=(7.2, 3.8), sharey=True)
    y = np.arange(len(CATEGORIES))
    h = 0.36
    for ax, config in zip(axes, configs, strict=True):
        v1 = [outcome_count(src, config, c, "refused", PROMPT_V1) for c in CATEGORIES]
        v2r = [outcome_count(src, config, c, "refused", PROMPT_V2) for c in CATEGORIES]
        v2p = [outcome_count(src, config, c, "partial", PROMPT_V2) for c in CATEGORIES]
        v1_row = entry(src.v1.summary, config, f"{SUMMARY_FILE} of results/{ANSWERS_V1}")
        v2_row = entry(src.v2.summary, config, f"{SUMMARY_FILE} of results/{ANSWERS_V2}")
        agree(sum(v1), v1_row.refused, f"{config} v1 refused")
        agree(sum(v2r), v2_row.refused, f"{config} v2 refused")
        agree(sum(v2p), v2_row.partial, f"{config} v2 partial")
        ax.barh(y - h / 2, v1, height=h, color=ORACLE, label="v1 refused")
        ax.barh(y + h / 2, v2r, height=h, color=DARK, label="v2 refused")
        ax.barh(y + h / 2, v2p, left=v2r, height=h, color=LIGHT, label="v2 partial")
        ax.set_title(config, fontsize=8.5, color=INK)
        ax.set_xlim(0, OUTCOMES_RIGHT)
        ax.set_xlabel("answers")
        value_grid(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [f"{c} ({formatting.count(category_count(src, c))})" for c in CATEGORIES]
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.0))
    add_caption(
        fig,
        "Answers per category (questions in parentheses). Each v2 bar is stacked: refused, "
        "then partial.\nA partial answer without a claim "
        f"counts as a refusal (methodology section 9a). Runs results/{ANSWERS_V1} (v1) "
        f"and results/{ANSWERS_V2} (v2).",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    save_figure(fig, "refusals_partial")


def candidate_colors(judges: list[str]) -> dict[str, str]:
    """The bar color of each judge candidate; the figure draws `CANDIDATES_DRAWN`, side by side.

    Raises:
        ValueError: The calibration has another number of candidates.
    """
    if len(judges) != CANDIDATES_DRAWN:
        raise ValueError(
            f"the calibration figure draws {formatting.count(CANDIDATES_DRAWN)} judge candidates "
            "side by side; "
            f"the run has {formatting.count(len(judges))}: {listing.prose(judges)}"
        )
    return {judges[0]: LIGHT, judges[1]: DARK}


def draw_calibration(src: Sources) -> None:
    """Draws each judge candidate's agreement with the human ratings per dimension."""
    report = src.calibration
    dims = list(report.thresholds)
    judges = calibration.candidates(src)
    colors = candidate_colors(judges)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.34
    ticks = []
    for i, dim in enumerate(dims):
        statistic = calibration.statistic(src, dim)
        label = entry(calibration.STATISTIC_LABELS, statistic, "STATISTIC_LABELS")
        ticks.append(f"{dim}\n({label})")
        for j, judge in enumerate(judges):
            comparison = human_comparison(judge)
            agreement = calibration.agreement_interval(src, comparison, dim)
            value, low, high = agreement.value, agreement.low, agreement.high
            x = i + (j - 0.5) * width
            ax.bar(x, value, width=width, color=colors[judge], label=judge if i == 0 else None)
            ax.errorbar(
                x, value, yerr=[[value - low], [high - value]], color=INK, capsize=2, lw=0.7
            )
            if not calibration.interval_informative(src, comparison, dim):
                ax.text(x, high + MARK_GAP, "*", ha="center", fontsize=9, color=INK)
        threshold = calibration.threshold(src, dim)
        ax.plot(
            [i - THRESHOLD_HALF_WIDTH, i + THRESHOLD_HALF_WIDTH],
            [threshold, threshold],
            color=INK,
            lw=1.0,
            ls="--",
        )
    ax.plot([], [], color=INK, lw=1.0, ls="--", label="threshold")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(ticks)
    ax.set_ylim(*AGREEMENT_RANGE)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel("agreement with the human")
    ax.legend(loc="upper left", ncols=3)
    value_grid(ax)
    left_out = len(report.excluded_from_behaviour_correct)
    add_caption(
        fig,
        f"Cohen's \u03ba (binary dimensions) or Spearman (coverage) against "
        f"{formatting.count(calibration.answers_rated(src))} human-rated answers "
        f"({formatting.count(left_out)} "
        f"left out of behaviour_correct). {formatting.count(INTERVAL_LEVEL_PERCENT)}% percentile "
        "bootstrap interval "
        f"resampling answers, {formatting.count(calibration.resamples(src))} resamples, seed "
        f"{formatting.own_form(calibration.bootstrap_seed(src))}; * \u03ba undefined in part of "
        f"the resamples. Run {report.run}, {CALIBRATION_FILE}.",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "judge_calibration")


def draw_wrong_refusals(src: Sources) -> None:
    """Draws each configuration's wrong-refusal share under prompt v1 and v2."""
    rows = {
        c: row for (c, metric), row in src.prompts.comparison.items() if metric == "wrong_refusal"
    }
    configs = configurations(src)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, config in enumerate(configs):
        row = entry(rows, config, f"the wrong_refusal rows of results/{src.prompts.dir.name}")
        v1, v2 = row.v1, row.v2
        agree(v1, wrong_refusal_count(src, config, PROMPT_V1) / row.n, f"{config} v1")
        agree(v2, wrong_refusal_count(src, config, PROMPT_V2) / row.n, f"{config} v2")
        ax.plot([v1, v2], [i, i], color=MUTED, lw=1.2, zorder=1)
        ax.scatter([v1], [i], color=ORACLE, s=36, zorder=2, label="v1" if i == 0 else None)
        ax.scatter([v2], [i], color=DARK, s=36, zorder=3, label="v2" if i == 0 else None)
        low, high = row.low, row.high
        diff = row.diff
        label = f"{formatting.signed(diff)} {formatting.interval(low, high)}"
        ax.text(
            v1 + REFUSAL_LABEL_OFFSET[0],
            i + REFUSAL_LABEL_OFFSET[1],
            label,
            fontsize=7.5,
            color=MUTED,
            va="bottom",
        )
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs)
    ax.invert_yaxis()
    ax.set_xlim(0, WRONG_REFUSALS_RIGHT)
    ax.set_xlabel(
        f"share refused of the {formatting.count(rows[configs[0]].n)} questions that expect\n"
        "an answer or a clarifying question"
    )
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1.0, 1.0))
    value_grid(ax, "x")
    add_caption(
        fig,
        "Wrong refusals under prompt v1 and v2, the same questions and configurations; "
        f"v2 \u2212 v1 with its {formatting.count(INTERVAL_LEVEL_PERCENT)}% paired bootstrap "
        "interval. Run "
        f"results/{src.prompts.dir.name}.",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, "wrong_refusals")
