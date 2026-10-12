"""Tests for `generation.calibration_sample`: the stratified sample and its file."""

from __future__ import annotations

from collections import Counter

from crossvenue_rag.generation.calibration_sample import calibration_sample, stratum
from tests.generation.factories import matrix_records


def test_calibration_sample_is_stratified_and_reproducible():
    sample = calibration_sample(matrix_records(), 40, seed=20260924, exclude_questions=())
    assert len(sample) == 40
    counts = Counter(stratum(r) for r in sample)
    assert len(counts) == 12  # every configuration x category stratum represented
    assert all(v >= 1 for v in counts.values())
    by_config = Counter(r.config for r in sample)
    assert set(by_config.values()) == {10}
    again = calibration_sample(
        list(reversed(matrix_records())), 40, seed=20260924, exclude_questions=()
    )
    assert {r.answer_id for r in again} == {r.answer_id for r in sample}
    assert {
        r.answer_id for r in calibration_sample(matrix_records(), 40, seed=1, exclude_questions=())
    } != {r.answer_id for r in sample}


def test_calibration_sample_never_draws_excluded_questions():
    excluded = [f"cv-{i:04d}" for i in (0, 1, 2, 15, 25)]
    sample = calibration_sample(matrix_records(), 40, seed=20260924, exclude_questions=excluded)
    assert len(sample) == 40
    assert not {r.question_id for r in sample} & set(excluded)


def test_an_answer_shared_by_two_configurations_is_sampled_at_most_once():
    def one_gemini_answer(config, i):
        # Both gemini configurations make the same call: one cached answer.
        return f"gemini:{i}" if config.startswith("gemini@") else f"{config}:{i}"

    sample = calibration_sample(
        matrix_records(one_gemini_answer), 40, seed=20260924, exclude_questions=()
    )
    keys = [r.generated().call.cache_key for r in sample]
    assert len(keys) == len(set(keys))
    assert all(r.config != "gemini@hybrid_rerank" for r in sample)


def test_the_sample_file_names_its_run_and_its_answers(tmp_path):
    from crossvenue_rag.generation.calibration_sample import read_sample, write_calibration_sample

    sample = calibration_sample(matrix_records(), 12, seed=1, exclude_questions=())
    path = tmp_path / "sample.txt"
    write_calibration_sample(path, "results/r_answers", sample, 1, ["cv-0001"])
    run_dir, ids = read_sample(path)
    assert run_dir == "results/r_answers"
    assert ids == [r.answer_id for r in sample]
    assert "cv-0001" in path.read_text(encoding="utf-8")


def test_a_one_answer_sample_says_answer_and_still_names_its_run(tmp_path):
    from crossvenue_rag.generation.calibration_sample import read_sample, write_calibration_sample

    sample = calibration_sample(matrix_records(), 1, seed=1, exclude_questions=())
    path = tmp_path / "sample.txt"
    write_calibration_sample(path, "results/r_answers", sample, 1, excluded=())
    assert "sample: 1 answer," in path.read_text(encoding="utf-8")
    assert read_sample(path) == ("results/r_answers", [sample[0].answer_id])


def test_a_sample_larger_than_the_run_names_answers_not_questions():
    import pytest

    from crossvenue_rag.errors import UserError

    eligible = len(matrix_records())
    with pytest.raises(UserError, match=f"exceeds the {eligible} eligible answers"):
        calibration_sample(matrix_records(), eligible + 1, seed=1, exclude_questions=())


def test_an_excluded_question_the_run_did_not_answer_is_refused():
    import pytest

    from crossvenue_rag.errors import UserError

    with pytest.raises(UserError, match="calibration_exclude names questions"):
        calibration_sample(matrix_records(), 10, seed=1, exclude_questions=["cv-9999"])


def test_a_run_path_holding_a_comma_reads_back_whole(tmp_path):
    from crossvenue_rag.generation.calibration_sample import read_sample, write_calibration_sample

    sample = calibration_sample(matrix_records(), 1, seed=1, exclude_questions=())
    path = tmp_path / "sample.txt"
    write_calibration_sample(path, "results/a,b_answers", sample, 1, excluded=())
    assert read_sample(path) == ("results/a,b_answers", [sample[0].answer_id])


def test_the_one_line_header_of_committed_samples_still_names_its_run(tmp_path):
    from crossvenue_rag.generation.calibration_sample import read_sample

    path = tmp_path / "sample.txt"
    path.write_text(
        "# Judge calibration sample: 2 answers from results/a,b_answers, stratified by "
        "configuration and category, seed 1.\n"
        "# Excluded questions (judge prompt revised on them): none.\n"
        "g:cv-0001  g|fact\ng:cv-0002  g|fact\n",
        encoding="utf-8",
    )
    assert read_sample(path) == ("results/a,b_answers", ["g:cv-0001", "g:cv-0002"])


def test_a_sample_file_that_is_not_utf8_is_refused_by_name(tmp_path):
    import pytest

    from crossvenue_rag.errors import UserError
    from crossvenue_rag.generation.calibration_sample import read_sample

    path = tmp_path / "sample.txt"
    path.write_bytes(b"# Run: results/r_answers\n\xff\n")
    with pytest.raises(UserError, match=r"sample\.txt is not UTF-8 text") as caught:
        read_sample(path)
    assert "rate-sample" in caught.value.hint


def test_with_more_strata_than_answers_only_the_largest_strata_get_one_each():
    """12 strata and n = 3: the largest win, and a tie in size goes to the name sorting first."""
    records = matrix_records()
    sizes = Counter(stratum(r) for r in records)
    largest = sorted(sizes, key=lambda s: (-sizes[s], s))[:3]
    sample = calibration_sample(records, 3, seed=1, exclude_questions=())
    assert sorted(stratum(r) for r in sample) == sorted(largest)
