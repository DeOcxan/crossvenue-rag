"""Tests for `generation.answer_pairs`: paired comparisons of configurations."""

from __future__ import annotations

import csv

from crossvenue_rag.generation.records import AnswerRecord
from crossvenue_rag.run_files import PAIRED_FILE
from tests.generation.factories import (
    judge_json,
    metrics_json,
    record,
)
from tests.published import ROOT


def _judged(
    config: str, qid: str, category: str, vae: float | None, coverage: float
) -> AnswerRecord:
    metrics = metrics_json(
        faithfulness=1.0 if vae is not None else None,
        venue_attribution_error=vae,
        coverage=coverage,
    )
    return record(config, qid, category=category, judge=judge_json(metrics=metrics))


def test_answer_pairs_count_better_in_each_metrics_direction(cfg):
    from crossvenue_rag.generation.answer_pairs import paired_rows

    a, b = (m.name for m in cfg.generation.matrix[:2])

    def rec(config: str, qid: str, vae: float, coverage: float) -> AnswerRecord:
        metrics = metrics_json(venue_attribution_error=vae, coverage=coverage)
        return record(config, qid, judge=judge_json(metrics=metrics))

    records = [
        rec(a, "q1", 0.0, 1.0), rec(b, "q1", 0.5, 0.5),
        rec(a, "q2", 0.0, 1.0), rec(b, "q2", 0.5, 0.5),
    ]  # fmt: skip
    rows = {(r.metric, r.scope): r for r in paired_rows(records, cfg) if {r.a, r.b} == {a, b}}
    vae, cov = rows[("venue_attribution_error", "comparison")], rows[("coverage", "comparison")]
    first = vae.a
    # The config with the lower error and the higher coverage is better on both.
    assert not vae.higher_is_better and cov.higher_is_better
    expected = (2, 0) if first == a else (0, 2)
    assert (vae.better, vae.worse) == expected and (cov.better, cov.worse) == expected


def test_primary_family_keeps_p_values_when_few_answers_carry_claims(cfg):
    """evals/methodology.md section 9: the primary family is over the 50 comparison
    questions. That both configurations' answers carry claims on only 17 of them (34
    answers) does not make venue_attribution_error descriptive, and coverage's p_Holm is
    adjusted over the two tests of the family.
    """
    from crossvenue_rag.evaluation.paired import holm
    from crossvenue_rag.generation.answer_pairs import paired_rows

    cfg.generation.primary_prompt_sha256 = None
    a, b = cfg.generation.primary_pair
    records = []
    for i in range(50):
        vae = (0.0, 0.5) if i < 17 else (None, None)
        records += [
            _judged(a, f"q{i}", "comparison", vae[0], 1.0 if i % 3 else 0.5),
            _judged(b, f"q{i}", "comparison", vae[1], 0.5),
        ]
    rows = {(r.metric, r.scope): r for r in paired_rows(records, cfg) if (r.a, r.b) == (a, b)}
    cov, vae_row = rows[("coverage", "comparison")], rows[("venue_attribution_error", "comparison")]
    assert cov.role == vae_row.role == "primary" and vae_row.n == 17
    assert cov.randomisation_p is not None and vae_row.randomisation_p is not None
    assert cov.holm_p == holm([cov.randomisation_p, vae_row.randomisation_p])[0]


def test_a_category_subset_below_50_is_descriptive_and_the_whole_set_is_not(cfg):
    from crossvenue_rag.generation.answer_pairs import paired_rows

    cfg.generation.primary_prompt_sha256 = None
    a, b = (
        m.name for m in cfg.generation.matrix if set([m.name]) - set(cfg.generation.primary_pair)
    )
    records = []
    for i in range(20):
        records += [
            _judged(a, f"q{i}", "comparison", 0.0, 1.0 if i % 2 else 0.5),
            _judged(b, f"q{i}", "comparison", 0.5, 0.5),
        ]
    rows = {(r.metric, r.scope): r for r in paired_rows(records, cfg) if {r.a, r.b} == {a, b}}
    sub, whole = rows[("coverage", "comparison")], rows[("coverage", "all")]
    assert sub.role == "descriptive" and sub.randomisation_p is None and sub.holm_p is None
    assert sub.low <= sub.diff <= sub.high
    assert whole.role == "secondary" and whole.randomisation_p is not None


# The committed v2 judged run (prompt v2): methodology 9a's secondary comparison is the
# pre-registered pair under a prompt other than the registered one.
V2_JUDGED = ROOT / "results" / "20260926T004836Z_judged"


def _recomputed_and_committed(overrides: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """The run's paired.csv rows recomputed from its answers with its config, and as committed,
    each as the cells under the committed columns."""
    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.generation.answer_pairs import paired_rows
    from crossvenue_rag.generation.records import read_answers

    with (V2_JUDGED / PAIRED_FILE).open(encoding="utf-8", newline="") as fh:
        committed = list(csv.reader(fh))
    columns = committed[0]
    cfg = load_config(V2_JUDGED / "config.yaml", overrides)
    rows = paired_rows(read_answers(V2_JUDGED / "answers.jsonl"), cfg)
    cells = [row.cells() for row in rows]
    recomputed = [["" if c[name] is None else str(c[name]) for name in columns] for c in cells]
    return [columns, *recomputed], committed


def test_the_v2_judged_run_recomputes_to_its_committed_paired_table():
    recomputed, committed = _recomputed_and_committed([])
    assert recomputed == committed


def test_the_pair_keeps_its_own_family_as_secondary_under_another_prompt():
    """With the v1 pin of configs/default.yaml, the v2 run's pair is methodology 9a's
    secondary comparison. Every cell is as committed (coverage's p_Holm 0.771, Holm within
    the two metrics), except the role of the pair's two rows, which is secondary."""
    from crossvenue_rag.config_loading import load_config

    pin = load_config(ROOT / "configs" / "default.yaml").generation.primary_prompt_sha256
    recomputed, committed = _recomputed_and_committed([f"generation.primary_prompt_sha256={pin}"])
    role = committed[0].index("role")
    expected = [
        [*row[:role], "secondary", *row[role + 1 :]] if row[role] == "primary" else row
        for row in committed
    ]
    assert sum(row[role] == "primary" for row in committed) == 2
    assert recomputed == expected
