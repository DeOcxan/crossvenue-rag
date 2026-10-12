"""Tests for `evaluation.power`: the simulated power of the randomization test."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.evaluation.power import (
    DELTAS,
    POWER_FILE,
    PowerDesign,
    PowerPoint,
    PowerRequest,
    power_json,
    power_run,
    power_study,
    read_power_study,
    run_questions_path,
    simulate_power,
    smallest_simulated_delta,
    write_power_study,
)
from crossvenue_rag.run_files import RUN_FILE_HINT
from tests.conftest import committed_of_kind, power_study_of
from tests.evaluation.factories import fake_run


def test_no_effect_rejects_at_about_alpha_and_power_grows_with_delta():
    points = simulate_power(
        PowerDesign(
            magnitudes=[1.0, 0.5],
            psi=0.5,
            deltas=[0.0, 0.1, 0.3],
            n=50,
            alpha=0.05,
            family=6,
            studies=400,
            permutations=400,
            seed=7,
        )
    )
    null, small, large = (_value(p.power_unadjusted) for p in points)
    assert null < 0.10  # the test's size, near alpha
    assert small < large
    # Holm's first step is stricter than the unadjusted level.
    assert _value(points[2].power_holm_first_step) <= large


def _value(power: float | None) -> float:
    assert power is not None
    return power


def test_a_delta_beyond_the_models_ceiling_is_unreachable():
    # psi 0.2 and magnitudes of 0.5 cap the expected difference at 0.1.
    [point] = simulate_power(PowerDesign([0.5], 0.2, [0.15], 50, 0.05, 6, 10, 10, seed=1))
    assert not point.reachable and point.power_unadjusted is None
    assert smallest_simulated_delta([point], 0.8, holm=False) is None


def test_the_study_pools_the_familys_differences_on_the_category(tmp_path):
    run, questions = fake_run(tmp_path)
    assert run_questions_path(run) == tmp_path / "questions.jsonl"
    study = power_study(PowerRequest(run, "comparison", 10, 5, 5), questions, seed=3)
    # Only secondary recall@10 rows on the category make the family.
    assert study.family == [("hybrid", "bm25"), ("hybrid", "dense")]
    assert study.questions == 6  # the fact question is outside the category
    assert study.values == 18  # three methods of the family, six questions
    assert study.fractional_values == 2  # cv-0 has two gold chunks, one retrieved
    # hybrid against bm25 differs on five questions, hybrid against dense on none.
    assert study.discordant_pairs == 5 and study.question_pairs == 12
    assert study.psi == pytest.approx(5 / 12)
    assert study.mean_magnitude == pytest.approx(0.9)
    assert study.ceiling == pytest.approx(0.375)
    assert [p.delta for p in study.points] == list(DELTAS)
    assert all(p.reachable for p in study.points)
    assert study.metric == "recall@10" and study.seed == 3


def test_a_run_without_the_family_is_a_user_error(tmp_path):
    run, questions = fake_run(tmp_path, family=False)
    with pytest.raises(UserError, match="no recall@10 comparisons on comparison") as refused:
        power_study(PowerRequest(run, "comparison", 10, 2000, 2000), questions, seed=1)
    assert refused.value.hint.startswith("pass the options of a comparison the run makes: ")


def test_the_study_is_written_with_its_source_runs_index(tmp_path):
    source = tmp_path / "20260101T000000Z"
    source.mkdir()
    index = {"index_version": "v1", "questions": "q.jsonl", "questions_sha256": "abc"}
    (source / "index.json").write_text(json.dumps(index), encoding="utf-8")
    study = power_study_of()
    run_dir = write_power_study(tmp_path / "results", study, source)
    assert run_dir.name.endswith("_power")
    assert read_power_study(run_dir) == study
    assert (run_dir / "index.json").read_text() == json.dumps(index, indent=2) + "\n"


def test_a_power_run_writes_the_study_it_returns_under_the_configured_results(tmp_path):
    from tests.conftest import default_config

    run, questions = fake_run(tmp_path)
    cfg = default_config(tmp_path)
    cfg.eval.bootstrap_seed = 3
    power = power_run(cfg, PowerRequest(run, "comparison", 10, 5, 5), questions)
    assert power.run_dir.parent == cfg.paths.results and power.study.seed == 3
    assert read_power_study(power.run_dir) == power.study


def test_a_run_without_index_json_is_refused_as_a_source(tmp_path):
    with pytest.raises(UserError, match=r"has no index\.json"):
        run_questions_path(tmp_path)


@pytest.mark.parametrize("run", committed_of_kind(POWER_FILE, "_power"), ids=lambda p: p.name)
def test_a_committed_power_json_is_written_back_with_the_same_bytes(run):
    original = (run / POWER_FILE).read_bytes()
    assert power_json(read_power_study(run)).encode("utf-8") == original


def test_a_family_without_a_discordant_pair_is_a_user_error(tmp_path):
    run, questions = fake_run(tmp_path)
    # Every method retrieves the same chunk, so no question separates two methods.
    rows = (run / "per_query.csv").read_text(encoding="utf-8").splitlines()
    same = [rows[0]] + [",".join([*row.split(",")[:2], "g"]) for row in rows[1:]]
    (run / "per_query.csv").write_text("\n".join(same) + "\n", encoding="utf-8")
    with pytest.raises(UserError, match="no discordant pair") as refused:
        power_study(PowerRequest(run, "comparison", 10, 5, 5), questions, seed=1)
    assert (refused.value.hint or "").startswith("pass --k or --category")


def test_a_category_without_a_labeled_question_is_a_user_error(tmp_path):
    run, questions = fake_run(tmp_path)
    for q in questions:
        q.gold_chunk_ids = []
    with pytest.raises(UserError, match="no labeled comparison question"):
        power_study(PowerRequest(run, "comparison", 10, 5, 5), questions, seed=1)


def test_an_unknown_adjustment_is_refused_rather_than_read_as_unadjusted():
    point = PowerPoint(delta=0.1, reachable=True, power_holm_first_step=0.2, power_unadjusted=0.5)
    assert point.power_under("holm_first_step") == 0.2
    assert point.power_under("unadjusted") == 0.5
    with pytest.raises(ValueError, match="unknown adjustment 'holm'"):
        point.power_under("holm")


def test_a_question_file_changed_or_gone_since_the_run_is_refused(tmp_path):
    import hashlib

    run, _ = fake_run(tmp_path)
    questions = tmp_path / "questions.jsonl"
    index = {
        "index_version": "v1",
        "questions": str(questions),
        "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
    }
    (run / "index.json").write_text(json.dumps(index), encoding="utf-8")
    assert run_questions_path(run) == questions
    questions.write_text(questions.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    shortened = r"\(sha256 [0-9a-f]{8}\u2026 in its index\.json\)"
    with pytest.raises(UserError, match=f"differs from the question file .* {shortened}"):
        run_questions_path(run)
    questions.unlink()
    with pytest.raises(UserError, match="is missing"):
        run_questions_path(run)


def test_a_cutoff_or_category_the_run_does_not_compare_names_what_it_does(tmp_path):
    run, questions = fake_run(tmp_path)
    for k, category in ((1, "comparison"), (10, "fact")):
        request = PowerRequest(run, category, k, 2, 2)
        with pytest.raises(UserError, match="no recall@") as refused:
            power_study(request, questions, seed=1)
        assert "--k 10 --category comparison" in refused.value.hint


def test_a_question_saved_without_a_category_counts_by_its_qtype_as_the_family_did(tmp_path):
    run, questions = fake_run(tmp_path)
    # The paired family scopes on the effective category, the qtype when no category is saved.
    for q in questions[:6]:
        q.category = None
    study = power_study(PowerRequest(run, "comparison", 10, 5, 5), questions, seed=3)
    assert study.questions == 6


def test_a_question_the_questions_lack_is_a_value_error_naming_it(tmp_path):
    run, questions = fake_run(tmp_path)
    request = PowerRequest(run, "comparison", 10, 5, 5)
    with pytest.raises(ValueError, match="has cv-0, which `questions` lacks"):
        power_study(request, questions[1:], seed=1)


def test_a_family_row_per_query_csv_lacks_is_a_user_error_naming_the_file(tmp_path):
    run, questions = fake_run(tmp_path)
    per_query = run / "per_query.csv"
    lines = per_query.read_text(encoding="utf-8").splitlines(keepends=True)
    per_query.write_text("".join(x for x in lines if not x.startswith("cv-2,dense,")), "utf-8")
    with pytest.raises(UserError, match=r"per_query\.csv: dense has no row on cv-2") as refused:
        power_study(PowerRequest(run, "comparison", 10, 5, 5), questions, seed=1)
    assert refused.value.hint == RUN_FILE_HINT
