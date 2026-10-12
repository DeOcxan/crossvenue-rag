"""Paired comparisons of configurations, question by question.

The pre-registered comparison (`generation.primary_pair` on `primary_metrics` over
`primary_scope`) is its own Holm family, and every other pair, metric and scope is
Holm-adjusted within its metric and scope, as for retrieval. A row is primary for that
comparison under the registered prompt (evals/methodology.md sections 9 and 9a), else
secondary, or descriptive with no p-value when its scope is too small to test
(`evaluation.paired.tested`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from ..config import Config
from ..config_generation import GenerationConfig
from ..evaluation.paired import (
    PairedEstimate,
    Role,
    compare_pair,
    holm_by_family,
    pair_tests,
    tested,
)
from ..metric_direction import higher_is_better
from ..run_files import PAIRED_FILE, CsvTable, OptionalFloat, read_table
from .answer_rows import SCOPES, category_in_scope
from .metrics import METRICS
from .records import AnswerRecord, named_judges
from .runs import uncalibrated_metrics


class ConfigComparison(PairedEstimate):
    """One row of an answer run's paired.csv: configuration a against b on a metric and scope.

    `read_config_comparisons` reads every format of the file:

    - the committed answer and judged runs of 2026-09-25 and 2026-09-26: up to
      `holm_p`, without `calibrated`, `better`, `worse`, `ties` and
      `higher_is_better`, which read as their defaults;
    - every other run: every column.
    """

    # None when the scope is too small to test (`evaluation.paired.tested`). The role is then
    # descriptive, or primary for the pre-registered pair under the registered prompt.
    randomisation_p: OptionalFloat
    mcnemar_p: OptionalFloat
    role: Role
    # Set by `_adjust_families` once every row of a family exists.
    holm_p: OptionalFloat = None
    # False when the metric's judge dimension failed calibration.
    calibrated: bool = True
    # Questions where a is better than b, worse, tied, in the metric's direction.
    better: int = 0
    worse: int = 0
    ties: int = 0
    higher_is_better: bool = True


@dataclass(frozen=True)
class _Comparison:
    """One cell of the paired table: configuration a against b on a metric and scope."""

    a: str
    b: str
    metric: str
    scope: str


def paired_rows(records: Sequence[AnswerRecord], cfg: Config) -> list[ConfigComparison]:
    """Every pair of configurations on every metric and scope, Holm-adjusted per family."""
    gen = cfg.generation
    configs = [m.name for m in gen.matrix]
    by_key = _judged_by_key(records)
    categories = {r.question_id: r.category for r in records}
    registered = _registered_prompt(records, gen)
    uncalibrated = set(uncalibrated_metrics(gen, _judge_of(records)))
    rows: list[ConfigComparison] = []
    families: list[tuple[str, ...]] = []
    for a, b in combinations(configs, 2):
        if {a, b} == set(gen.primary_pair):
            a, b = gen.primary_pair  # the pre-registered direction
        for metric in METRICS:
            for scope in SCOPES:
                comparison = _Comparison(a, b, metric, scope)
                xs, ys = _paired_values(by_key, categories, comparison)
                if not xs:
                    continue
                with_p = _scope_tested(categories, scope)
                pre_registered = _is_pre_registered(comparison, gen)
                role: Role = (
                    "primary"
                    if pre_registered and registered
                    else ("secondary" if with_p else "descriptive")
                )
                rows.append(
                    _answer_pair(
                        cfg,
                        comparison,
                        (xs, ys),
                        role=role,
                        with_p=with_p,
                        calibrated=metric not in uncalibrated,
                    )
                )
                families.append(("pre-registered",) if pre_registered else (metric, scope))
    return _adjust_families(rows, families)


def _judged_by_key(records: Sequence[AnswerRecord]) -> dict[tuple[str, str], AnswerRecord]:
    return {(r.config, r.question_id): r for r in records if r.judge is not None}


def _scope_tested(categories: Mapping[str, str], scope: str) -> bool:
    """Whether a scope is large enough for p-values.

    The scope's size is its questions, not the answers with a value: the
    section 9 families over the comparison questions keep their p-values when
    only some answers carry claims.
    """
    size = sum(category_in_scope(c, scope) for c in categories.values())
    return tested(size, whole_set=scope == "all")


def _registered_prompt(records: Sequence[AnswerRecord], gen: GenerationConfig) -> bool:
    """Whether the run used the prompt the primary comparison was registered for.

    Under another prompt the same comparison is the exploratory secondary comparison
    of methodology 9a, which stays Holm-adjusted within its own metrics.
    """
    prompts = {r.generation.prompt_sha256 for r in records if r.generation is not None}
    return gen.primary_prompt_sha256 is None or prompts == {gen.primary_prompt_sha256}


def _is_pre_registered(comparison: _Comparison, gen: GenerationConfig) -> bool:
    return (
        {comparison.a, comparison.b} == set(gen.primary_pair)
        and comparison.metric in gen.primary_metrics
        and comparison.scope == gen.primary_scope
    )


def _paired_values(
    by_key: Mapping[tuple[str, str], AnswerRecord],
    categories: Mapping[str, str],
    comparison: _Comparison,
) -> tuple[list[float], list[float]]:
    """The metric of a and of b on each question of the scope both have a value for."""
    xs, ys = [], []
    for qid, category in sorted(categories.items()):
        if not category_in_scope(category, comparison.scope):
            continue
        ra, rb = by_key.get((comparison.a, qid)), by_key.get((comparison.b, qid))
        if ra is None or rb is None:
            continue
        va = ra.judged().metrics.value(comparison.metric)
        vb = rb.judged().metrics.value(comparison.metric)
        if va is not None and vb is not None:
            xs.append(va)
            ys.append(vb)
    return xs, ys


def _answer_pair(
    cfg: Config,
    comparison: _Comparison,
    values: tuple[list[float], list[float]],
    *,
    role: Role,
    with_p: bool,
    calibrated: bool,
) -> ConfigComparison:
    """The estimate and tests of one comparison; p-values only when `with_p`."""
    xs, ys = values
    up = higher_is_better(comparison.metric)
    test = compare_pair(
        xs,
        ys,
        resamples=cfg.eval.bootstrap_resamples,
        seed=cfg.eval.bootstrap_seed,
        higher_is_better=up,
        tests=pair_tests(
            randomisation=with_p, mcnemar=comparison.metric == "behaviour_correct" and with_p
        ),
    )
    boot = test.boot
    return ConfigComparison(
        a=comparison.a,
        b=comparison.b,
        metric=comparison.metric,
        scope=comparison.scope,
        n=test.n,
        mean_a=test.mean_a,
        mean_b=test.mean_b,
        diff=boot.mean_diff,
        low=boot.low,
        high=boot.high,
        randomisation_p=test.randomisation_p,
        mcnemar_p=test.mcnemar_p,
        role=role,
        calibrated=calibrated,
        better=boot.better,
        worse=boot.worse,
        ties=boot.ties,
        higher_is_better=up,
    )


def _adjust_families(
    rows: Sequence[ConfigComparison], families: Sequence[tuple[str, ...]]
) -> list[ConfigComparison]:
    """The rows with Holm within each row's family, `families[i]` naming row i's.

    An adjusted p needs every p of its family, which exist only once every row
    is built. Descriptive rows, which have no p, join no family.
    """
    adjusted = holm_by_family(
        range(len(rows)), lambda i: families[i], lambda i: rows[i].randomisation_p
    )
    return [
        row.model_copy(update={"holm_p": holm_p})
        for row, holm_p in zip(rows, adjusted, strict=True)
    ]


def read_config_comparisons(run_dir: Path) -> CsvTable[ConfigComparison]:
    """paired.csv of an answer or judged run, in any format `ConfigComparison` lists.

    An empty file, as a run with one configuration writes, is an empty table.
    """
    return read_table(run_dir / PAIRED_FILE, ConfigComparison)


def _judge_of(records: Sequence[AnswerRecord]) -> str | None:
    judges = named_judges(records)
    return judges[0] if len(judges) == 1 else None
