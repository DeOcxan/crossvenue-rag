"""Tests for `reporting.compute.datasets_rounds`, on journals written under tmp_path."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.dataset.journal import rounds_path
from crossvenue_rag.reporting.compute import datasets_rounds
from crossvenue_rag.reporting.run_ids import (
    GOLD,
    GOLD_BEFORE_AUGMENTATION,
    ROUND2_EXPOSED,
    ROUND2_SAMPLE,
)
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import augmentation, correction
from tests.reporting.conftest import write_file, write_gold


def test_the_shuffle_seed_is_read_from_the_file_that_sets_it(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/labeling/pool.py", "RETEST_SHUFFLE_SEED = 456\n")
    assert datasets_rounds.retest_shuffle_seed(Sources(tmp_path)) == 456


def test_the_pool_depth_multiplier_is_read_from_the_file_that_sets_it(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/labeling/pool.py", "_MAX_DEPTH_MULTIPLIER = 6\n")
    assert datasets_rounds.pool_depth_multiplier(Sources(tmp_path)) == 6


def test_a_missing_seed_is_an_error(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/labeling/pool.py", "RETEST_SIZE = 25\n")
    with pytest.raises(ValueError, match="matches nothing"):
        datasets_rounds.retest_shuffle_seed(Sources(tmp_path))


def test_round_1_marks_are_split_by_the_chunks_questions_were_drafted_from(tmp_path):
    write_gold(
        tmp_path,
        [{"id": "cv-0001", "source_chunk_ids": ["s1"]}, {"id": "cv-0002"}],
        path=GOLD_BEFORE_AUGMENTATION,
    )
    journal = {
        "cv-0001": {"1": {"pool": ["s1", "c2", "c3"], "relevant": ["s1", "c3"]}},
        "cv-0002": {"1": {"pool": ["s1", "c4"], "relevant": []}},
    }
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    src = Sources(tmp_path)
    assert datasets_rounds.question_count_before_augmentation(src) == 2
    assert datasets_rounds.first_pass_marks(src) == (2, 5)
    # s1 is a source chunk of cv-0001 only; on cv-0002 it is another pool chunk.
    assert datasets_rounds.source_chunk_marks(src) == datasets_rounds.SourceChunkMarks(
        source_shown=1, source_relevant=1, other_shown=4, other_relevant=1
    )


def sample_fixture(root):
    """Three sampled questions; cv-0003 is not sampled, cv-0001 and cv-0003 were exposed.

    cv-0001 carries a review correction and cv-0004 a derived one.
    """
    write_gold(
        root,
        [
            {"id": "cv-0001", "category": "comparison"},
            {"id": "cv-0002", "category": "fact"},
            {"id": "cv-0003", "category": "fact"},
            {"id": "cv-0004", "category": "noisy"},
        ],
    )
    write_file(root, ROUND2_SAMPLE, "cv-0001\ncv-0002\ncv-0004\n")
    write_file(root, ROUND2_EXPOSED, "# exposed\ncv-0001\ncv-0003\n")
    corrected = {"pool": ["a"], "relevant": ["a"], "corrections": [correction([], ["a"])]}
    derived = {**correction(["a"]), "annotator": None, "derived": True}
    augmented = {
        "pool": ["a", "b"],
        "relevant": ["x"],
        "augmentations": [augmentation(["b", "x"], ["x"])],
    }
    rounds = {
        "cv-0001": {"1": corrected},
        "cv-0002": {"1": augmented},
        "cv-0003": {"1": {"pool": ["a"] * 9}},
        "cv-0004": {
            "1": {"pool": ["a", "b", "c", "d"], "relevant": ["a"], "corrections": [derived]}
        },
    }
    write_file(root, str(rounds_path(GOLD)), json.dumps(rounds))
    return Sources(root)


def test_the_sample_is_counted_by_exposure_and_category(tmp_path):
    src = sample_fixture(tmp_path)
    assert datasets_rounds.exposed_sample_count(src) == 1
    assert datasets_rounds.unexposed_sample_count(src) == 2
    assert datasets_rounds.sample_categories(src) == {"comparison": 1, "fact": 1, "noisy": 1}
    assert datasets_rounds.sample_categories(src)["adversarial"] == 0
    assert datasets_rounds.single_sampled_categories(src) == ["comparison", "fact", "noisy"]
    assert datasets_rounds.only_single_sampled(src, ("noisy", "fact", "comparison"))
    assert not datasets_rounds.only_single_sampled(src, ("noisy", "fact"))


def test_the_sample_counts_its_round_one_decisions_augmentations_and_corrections(tmp_path):
    src = sample_fixture(tmp_path)
    # cv-0001 judged 1 chunk, cv-0002 the pool and one augmentation chunk (3), cv-0004 4.
    assert datasets_rounds.sample_decisions(src) == datasets_rounds.SampleDecisions(8, 1, 4)
    assert datasets_rounds.sample_augmented(src) == 1
    assert datasets_rounds.sample_corrected(src) == datasets_rounds.SampleCorrections(2, 1, 1)
    # cv-0001's first pass marked nothing; review added a chunk. cv-0004's correction is
    # derived and changes nothing.
    assert datasets_rounds.sample_correction_changes(src) == 1


def test_the_pace_leaves_out_the_session_break_and_measures_the_longer_session(tmp_path):
    write_gold(
        tmp_path, [{"id": q} for q in ("q1", "q2", "q3", "q4")], path=GOLD_BEFORE_AUGMENTATION
    )
    stamps = ["2026-09-24T10:00:00Z", "2026-09-24T12:00:00Z", "2026-09-24T12:00:30Z",
              "2026-09-24T12:01:10Z"]  # fmt: skip
    journal = {f"q{i}": {"1": {"annotated_at": s}} for i, s in enumerate(stamps, start=1)}
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    times = datasets_rounds.save_times(Sources(tmp_path))
    assert times.median_gap == 30.0  # the gaps of 30 and 40 s; the two-hour break is left out
    assert (times.longer_saves, times.longer_span, times.longer_mean_gap) == (3, 70.0, 35.0)


def test_the_pace_needs_three_stamped_judgments(tmp_path):
    write_gold(tmp_path, [{"id": "q1"}], path=GOLD_BEFORE_AUGMENTATION)
    journal = {"q1": {"1": {"annotated_at": "2026-09-24T10:00:00Z"}}}
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    with pytest.raises(ValueError, match="three stamped judgments"):
        datasets_rounds.save_times(Sources(tmp_path))


def test_the_review_counts_the_annotators_corrections_and_leaves_out_derived_ones(tmp_path):
    write_gold(tmp_path, [{"id": q} for q in ("q1", "q2", "q3")], path=GOLD_BEFORE_AUGMENTATION)
    derived = {**correction(["a"]), "derived": True, "rule": "r"}
    del derived["annotator"]
    journal = {
        "q1": {"1": {"relevant": ["b"], "corrections": [correction(["a"], ["b", "c"])]}},
        "q2": {"1": {"relevant": ["a"], "corrections": [correction(["a"])]}},  # coverage only
        "q3": {"1": {"relevant": ["a"], "corrections": [derived]}},
    }
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    counts = datasets_rounds.review_counts(Sources(tmp_path))
    assert (counts.reopened, counts.changed, counts.added, counts.removed) == (2, 1, 2, 1)
