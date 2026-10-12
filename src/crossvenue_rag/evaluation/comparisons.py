"""Paired method comparisons for a finished run: which pairs, which scopes, which test.

Each pair of `eval.paired_comparisons` ([a, b] means a minus b) is compared on
venue_coverage@K over multi-venue questions and the comparison category, and on
recall@K over labeled questions and labeled comparisons; the tests are in `evaluation.paired`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .. import formatting
from ..config import Config
from ..config_eval import (
    COMPARISON_SCOPE,
    LABELLED_COMPARISON_SCOPE,
    LABELLED_SCOPE,
    MULTI_VENUE_SCOPE,
    TOP_ONE_K,
)
from ..metric_direction import higher_is_better
from ..models import ADVERSARIAL, COMPARISON, Question
from ..run_files import CsvTable, OptionalFloat, read_table
from ..stats import INTERVAL_LEVEL_PERCENT
from .evaluator import QueryRun
from .labels import markdown_table, method_label, scope_label
from .paired import (
    PairedEstimate,
    Role,
    compare_pair,
    holm_by_family,
    pair_tests,
    tested,
)

# The scope of each category whose labeled questions the comparisons scope on.
LABELLED_CATEGORY_SCOPES = {COMPARISON: LABELLED_COMPARISON_SCOPE}


class PairedRow(PairedEstimate):
    """One paired comparison as reported: a row of a retrieval run's paired.csv.

    `read_paired` reads every format of the file:

    - the runs of 2026-09-25 before 13:00 UTC: up to `mcnemar_p`, without
      `randomisation_p`, `role`, `holm_p` and `higher_is_better`;
    - 20260925T132613Z and 20260925T144223Z: without `higher_is_better`;
    - the later runs: every column.
    """

    better: int
    worse: int
    ties: int
    mcnemar_p: OptionalFloat  # binary metrics only
    # sign-flip, any metric; None below MIN_TESTED_QUESTIONS (role descriptive)
    randomisation_p: OptionalFloat = None
    # "primary" for eval.primary_comparison; "descriptive" below 50 questions
    role: Role = "secondary"
    holm_p: OptionalFloat = None  # secondary rows: Holm within (metric, scope)
    # The metric's direction; `better` and `worse` count questions in it.
    higher_is_better: bool = True


def read_paired(path: Path) -> CsvTable[PairedRow]:
    """A retrieval run's paired.csv or off_venue.csv, in any format `PairedRow` lists."""
    return read_table(path, PairedRow)


@dataclass(frozen=True)
class Cell:
    """One comparison: method a against b on a metric over a scope."""

    a: str
    b: str
    metric: str
    scope: str


def _paired_row(
    cfg: Config, cell: Cell, xs: list[float], ys: list[float], *, binary: bool, with_p: bool
) -> PairedRow:
    """The estimate and tests of one cell.

    McNemar runs on a binary metric; p-values are computed only when the scope is
    large enough (`with_p`), else the row is descriptive.
    """
    up = higher_is_better(cell.metric)
    test = compare_pair(
        xs,
        ys,
        resamples=cfg.eval.bootstrap_resamples,
        seed=cfg.eval.bootstrap_seed,
        higher_is_better=up,
        tests=pair_tests(randomisation=with_p, mcnemar=binary and with_p),
    )
    boot = test.boot
    return PairedRow(
        a=cell.a,
        b=cell.b,
        metric=cell.metric,
        scope=cell.scope,
        n=test.n,
        mean_a=test.mean_a,
        mean_b=test.mean_b,
        diff=boot.mean_diff,
        low=boot.low,
        high=boot.high,
        better=boot.better,
        worse=boot.worse,
        ties=boot.ties,
        mcnemar_p=test.mcnemar_p,
        randomisation_p=test.randomisation_p,
        role="secondary" if with_p else "descriptive",
        higher_is_better=up,
    )


@dataclass(frozen=True)
class _Scope:
    metric: str
    scope: str
    keep: Callable[[QueryRun], bool]  # selects the question runs the scope covers
    # Marks venue_coverage, the metric McNemar applies to.
    binary: bool = field(kw_only=True)


class _Scopes:
    """The scopes of `paired_comparisons` over one question set."""

    def __init__(self, questions: Sequence[Question], k: int) -> None:
        self.by_id = {q.id: q for q in questions}
        multi = [q for q in questions if q.is_multi_venue]
        labelled = [q for q in questions if q.is_labelled]
        # Questions each scope is defined over, whatever the metric's n.
        self.sizes = {
            MULTI_VENUE_SCOPE: len(multi),
            COMPARISON_SCOPE: sum(self.is_comparison(q.id) for q in multi),
            LABELLED_SCOPE: len(labelled),
            LABELLED_COMPARISON_SCOPE: sum(self.is_comparison(q.id) for q in labelled),
        }
        self.all = [
            _Scope(f"venue_coverage@{k}", MULTI_VENUE_SCOPE, lambda r: r.multi_venue, binary=True),
            _Scope(
                f"venue_coverage@{k}",
                COMPARISON_SCOPE,
                lambda r: r.multi_venue and self.is_comparison(r.question_id),
                binary=True,
            ),
            _Scope(f"recall@{k}", LABELLED_SCOPE, lambda r: r.labelled, binary=False),
            _Scope(
                f"recall@{k}",
                LABELLED_COMPARISON_SCOPE,
                lambda r: r.labelled and self.is_comparison(r.question_id),
                binary=False,
            ),
        ]

    def is_comparison(self, question_id: str) -> bool:
        """Whether the question `question_id` is a comparison question."""
        return self.by_id[question_id].is_comparison


def paired_comparisons(
    cfg: Config, runs: Sequence[QueryRun], questions: Sequence[Question]
) -> list[PairedRow]:
    """Every configured pair on venue_coverage@K and recall@K, K = `max(k_values)`.

    Scopes: all multi-venue questions and the comparison category for
    venue_coverage; all labeled questions and labeled comparisons for recall.
    """
    scopes = _Scopes(questions, max(cfg.eval.k_values))
    table = _by_question_and_method(runs)
    ran = {r.method for r in runs}
    rows: list[PairedRow] = []
    for a, b in cfg.eval.paired_comparisons:
        if a not in ran or b not in ran:
            continue
        for each in scopes.all:
            metric, scope, binary = each.metric, each.scope, each.binary
            ids = sorted(_kept_ids(runs, a, each.keep) & _kept_ids(runs, b, each.keep))
            if not ids:
                continue
            source = "venue_metrics" if binary else "metrics"
            xs, ys = _values(table, ids, a, source, metric), _values(table, ids, b, source, metric)
            with_p = tested(scopes.sizes[scope], whole_set=scope == LABELLED_SCOPE)
            cell = Cell(a, b, metric, scope)
            rows.append(_paired_row(cfg, cell, xs, ys, binary=binary, with_p=with_p))
    return mark_primary_and_adjust(rows, cfg.eval.primary_comparison)


def _by_question_and_method(runs: Sequence[QueryRun]) -> dict[tuple[str, str], QueryRun]:
    return {(r.question_id, r.method): r for r in runs}


def _values(
    table: Mapping[tuple[str, str], QueryRun],
    ids: Sequence[str],
    method: str,
    source: str,
    metric: str,
) -> list[float]:
    """One method's value of `metric` on each question, from the run's `source` metrics."""
    return [table[(q, method)].metric_value(source, metric) for q in ids]


def _kept_ids(runs: Sequence[QueryRun], method: str, keep: Callable[[QueryRun], bool]) -> set[str]:
    return {r.question_id for r in runs if r.method == method and keep(r)}


# Methodology section 9b: the reranked hybrid against each first-stage method.
OFF_VENUE_PAIRS = (
    ("hybrid_rerank", "bm25"),
    ("hybrid_rerank", "dense"),
    ("hybrid_rerank", "hybrid"),
)
OFF_VENUE_SCOPES = (COMPARISON, ADVERSARIAL)
# The top-1 off-venue rate, the binary metric of the off-venue comparisons.
TOP_ONE_OFF_VENUE = f"off_venue@{TOP_ONE_K}"


def off_venue_comparisons(
    cfg: Config, runs: Sequence[QueryRun], questions: Sequence[Question]
) -> list[PairedRow]:
    """The exploratory off-venue comparisons of methodology section 9b.

    Each pair of `OFF_VENUE_PAIRS` is compared on off_venue@K (bootstrap and
    randomization test) and off_venue@1, the top-1 off-venue rate (also exact
    McNemar), per category of `OFF_VENUE_SCOPES`. A row is secondary, Holm-adjusted
    within its metric and scope, when its scope has at least `MIN_TESTED_QUESTIONS`
    questions, and descriptive, with no p-value, otherwise.
    """
    k = max(cfg.eval.k_values)
    by_id = {q.id: q for q in questions}
    table = _by_question_and_method(runs)
    ran = {r.method for r in runs}
    rows: list[PairedRow] = []
    for metric, binary in ((f"off_venue@{k}", False), (TOP_ONE_OFF_VENUE, True)):
        for scope in OFF_VENUE_SCOPES:
            with_p = tested(sum(q.effective_category == scope for q in by_id.values()))
            for a, b in OFF_VENUE_PAIRS:
                if a not in ran or b not in ran:
                    continue
                cell = Cell(a, b, metric, scope)
                ids = _off_venue_ids(by_id.values(), table, cell)
                if not ids:
                    continue
                xs = _values(table, ids, a, "off_venue_metrics", metric)
                ys = _values(table, ids, b, "off_venue_metrics", metric)
                rows.append(_paired_row(cfg, cell, xs, ys, binary=binary, with_p=with_p))
    return mark_primary_and_adjust(rows, [])


def _off_venue_ids(
    questions: Iterable[Question], table: Mapping[tuple[str, str], QueryRun], cell: Cell
) -> list[str]:
    """The scope's questions both methods ran on and that have off-venue metrics."""
    return sorted(
        q.id
        for q in questions
        if q.effective_category == cell.scope
        and (q.id, cell.a) in table
        and (q.id, cell.b) in table
        and table[(q.id, cell.a)].off_venue_metrics
    )


def mark_primary_and_adjust(rows: list[PairedRow], primary: Sequence[str]) -> list[PairedRow]:
    """The rows with the primary one tagged and the secondary ones Holm-adjusted per family.

    A family is every secondary row on one metric and one scope: the tests that
    answer one question about one set of questions. Descriptive rows (below 50
    questions) carry no p-value and join no family.
    """
    rows = [
        row.model_copy(update={"role": "primary"})
        if list(primary) == [row.a, row.b, row.metric, row.scope]
        else row
        for row in rows
    ]
    adjusted = holm_by_family(
        rows,
        lambda r: None if r.role == "primary" else (r.metric, r.scope),
        lambda r: r.randomisation_p,
    )
    return [
        row.model_copy(update={"holm_p": holm_p})
        for row, holm_p in zip(rows, adjusted, strict=True)
    ]


def paired_table(rows: Sequence[PairedRow]) -> str:
    """The paired comparisons as a Markdown table, one line per row."""
    header = [
        "a vs b",
        "metric",
        "scope",
        "n",
        "a",
        "b",
        "a \u2212 b",
        f"{formatting.count(INTERVAL_LEVEL_PERCENT)}% CI",
        "a better / worse / tie",
        "better is",
        "McNemar p",
        "randomization p",
        "role",
        "Holm p",
    ]
    cells = [
        [
            f"{method_label(r.a)} vs {method_label(r.b)}",
            r.metric,
            scope_label(r.scope),
            formatting.count(r.n),
            formatting.metric(r.mean_a),
            formatting.metric(r.mean_b),
            formatting.signed(r.diff),
            formatting.interval(r.low, r.high),
            " / ".join(formatting.count(n) for n in (r.better, r.worse, r.ties)),
            "higher" if r.higher_is_better else "lower",
            formatting.optional(r.mcnemar_p, formatting.p_value),
            formatting.optional(r.randomisation_p, formatting.p_value),
            r.role,
            formatting.optional(r.holm_p, formatting.p_value),
        ]
        for r in rows
    ]
    return markdown_table(header, cells)
