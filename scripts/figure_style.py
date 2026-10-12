"""What every figure shares: palette, fonts, grid lines, caption, file output, agreement check."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from crossvenue_rag import formatting
from crossvenue_rag.evaluation.evaluator import METRIC_DECIMALS
from crossvenue_rag.models import RETRIEVAL_STEPS

log = logging.getLogger(__name__)

OUT = Path("docs/figures")
# Where a caption starts, in figure coordinates: the bottom left corner, inset.
CAPTION_POSITION = (0.01, 0.01)

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
DARK = "#2b5c8a"
LIGHT = "#9dbad6"
ORACLE = "#b8b8b8"
# A color for each retrieval step, in pipeline order: the first is the main color.
STEP_COLORS = dict(zip(RETRIEVAL_STEPS, (DARK, "#6f9fc9", "#b0b0b0", "#c0504d"), strict=True))

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "svg.hashsalt": "crossvenue",
        "svg.fonttype": "none",
    }
)


def add_caption(fig: Figure, text: str) -> None:
    """Writes `text` as the caption along the bottom left of `fig`."""
    fig.text(*CAPTION_POSITION, text, fontsize=7, color=MUTED, ha="left", va="bottom", wrap=True)


def save_figure(fig: Figure, name: str) -> None:
    """Writes `fig` to docs/figures/<name>.svg and .png, then closes it."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None, "Creator": None})
    fig.savefig(OUT / f"{name}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    log.info("%s.svg, .png", OUT / name)


def value_grid(ax: Axes, axis: Literal["x", "y"] = "y") -> None:
    """Draws light grid lines along the value `axis`, behind the data."""
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def agree(drawn: float, reported: float, what: str) -> None:
    """Checks that `drawn` equals `reported` to `METRIC_DECIMALS`.

    The values checked are venue coverage, recall and nDCG, the two ends of each pooling gap,
    the wrong refusals, each configuration's refusal and partial totals and the off-venue
    shares. The drawn difference between a gap's two ends is not checked on its own.

    Raises:
        ValueError: The two differ at `METRIC_DECIMALS`; the message names `what`.
    """
    drawn_text, reported_text = (formatting.fixed(v, METRIC_DECIMALS) for v in (drawn, reported))
    if drawn_text != reported_text:
        raise ValueError(f"{what}: drawn {drawn_text}, the run reports {reported_text}")
