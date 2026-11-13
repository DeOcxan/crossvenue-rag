"""Tests for `reporting.compute.datasets_rounds`, on files written under `tmp_path`."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.dataset.journal import (
    FIRST_ROUND,
    FULL_RELABEL_ROUND,
    RETEST_ROUND,
    rounds_path,
)
from crossvenue_rag.reporting.compute import datasets_rounds
from crossvenue_rag.reporting.run_ids import (
    DEDUP_IDMAP,
    GOLD,
    GOLD_BEFORE_AUGMENTATION,
    GOLD_ROUND3,
    ROUND2_EXPOSED,
    ROUND2_SAMPLE,
)
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import augmentation, correction
from tests.reporting.conftest import write_file, write_gold


def seeds_fixture(root, round3_seeds):
    """`v0.3-gold` whose journal records seed 7 in round 2 and `round3_seeds` in round 3."""
    entry = {"pool": ["a"], "relevant": []}
    ids = [f"cv-{i:04d}" for i in range(1, len(round3_seeds) + 1)]
    write_gold(root, [{"id": q} for q in ids], path=GOLD_ROUND3)
    rounds = {
        q: {"1": entry, "3": dict(entry, shuffle_seed=s)}
        for q, s in zip(ids, round3_seeds, strict=True)
    }
    rounds[ids[0]]["2"] = dict(entry, shuffle_seed=7)
    write_file(root, str(rounds_path(GOLD_ROUND3)), json.dumps(rounds))
    return Sources(root)


def test_a_rounds_shuffle_seed_is_the_one_its_journal_entries_record(tmp_path):
    src = seeds_fixture(tmp_path, [9, 9])
    assert datasets_rounds.shuffle_seed(src, RETEST_ROUND) == 7
    assert datasets_rounds.shuffle_seed(src, FULL_RELABEL_ROUND) == 9


def test_a_round_whose_entries_record_no_one_seed_is_an_error(tmp_path):
    src = seeds_fixture(tmp_path, [9, 8])
    for number in (FIRST_ROUND, FULL_RELABEL_ROUND):
        with pytest.raises(ValueError, match="do not all record one shuffle_seed"):
            datasets_rounds.shuffle_seed(src, number)


def test_the_pool_depth_multiplier_is_read_from_the_file_that_sets_it(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/labeling/pool.py", "_MAX_DEPTH_MULTIPLIER = 6\n")
    assert datasets_rounds.pool_depth_multiplier(Sources(tmp_path)) == 6


def test_a_missing_depth_multiplier_is_an_error(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/labeling/pool.py", "RETEST_SIZE = 25\n")
    with pytest.raises(ValueError, match="no literal _MAX_DEPTH_MULTIPLIER"):
        datasets_rounds.pool_depth_multiplier(Sources(tmp_path))


def write_idmap(root, mapping):
    """The deduplication id map `mapping` under `root`."""
    idmap = {
        "old_ids": len(mapping),
        "map": mapping,
        "old_chunks_sha256": "",
        "to_index_version": "",
    }
    write_file(root, DEDUP_IDMAP, json.dumps(idmap))


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
    write_idmap(tmp_path, {})
    src = Sources(tmp_path)
    assert datasets_rounds.question_count_before_augmentation(src) == 2
    assert datasets_rounds.round1_marks_after_review(src) == (2, 5)
    # s1 is a source chunk of cv-0001 only; on cv-0002 it is another pool chunk.
    assert datasets_rounds.source_chunk_marks(src) == datasets_rounds.SourceChunkMarks(
        source_shown=1, source_relevant=1, other_shown=4, other_relevant=1
    )


def test_round_1_marks_count_a_migrated_pool_as_it_was_shown(tmp_path):
    write_gold(
        tmp_path, [{"id": "cv-0001", "source_chunk_ids": ["n1"]}], path=GOLD_BEFORE_AUGMENTATION
    )
    shown = {"pool": ["o1", "o2", "o3"], "relevant": ["o1", "o2"], "index_version": None}
    journal = {"cv-0001": {"1": {"pool": ["n1", "n3"], "relevant": ["n1"], "migrated_from": shown}}}
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    # o1 and o2 hold one text, the source chunk n1.
    write_idmap(tmp_path, {"o1": "n1", "o2": "n1", "o3": "n3"})
    src = Sources(tmp_path)
    assert datasets_rounds.round1_marks_after_review(src) == (2, 3)
    assert datasets_rounds.source_chunk_marks(src) == datasets_rounds.SourceChunkMarks(
        source_shown=2, source_relevant=2, other_shown=1, other_relevant=0
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


def test_a_sampled_question_saved_without_a_category_is_counted_by_its_qtype(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001", "category": None, "qtype": "fact"}])
    write_file(tmp_path, ROUND2_SAMPLE, "cv-0001\n")
    assert datasets_rounds.sample_categories(Sources(tmp_path)) == {"fact": 1}


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


def test_a_sampled_question_round_1_never_judged_is_an_error_naming_it(tmp_path):
    src = sample_fixture(tmp_path)
    write_file(tmp_path, ROUND2_SAMPLE, "cv-0001\ncv-0005\n")
    with pytest.raises(ValueError, match="cv-0005 has no round-1 judgment"):
        datasets_rounds.sample_decisions(src)


def _save_times(root, stamps):
    write_gold(root, [{"id": f"q{i}"} for i in range(1, len(stamps) + 1)], GOLD_BEFORE_AUGMENTATION)
    journal = {f"q{i}": {"1": {"annotated_at": s}} for i, s in enumerate(stamps, start=1)}
    write_file(root, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    times = datasets_rounds.save_times(Sources(root))
    return times.longer_saves, times.longer_span, times.longer_mean_gap


def test_the_pace_leaves_out_the_session_break_and_measures_the_longer_session(tmp_path):
    # Three saves, the break, two saves: the longer session is the one before the break.
    before = ["2026-09-24T10:00:00Z", "2026-09-24T10:00:20Z", "2026-09-24T10:01:00Z",
              "2026-09-24T12:00:00Z", "2026-09-24T12:00:30Z"]  # fmt: skip
    assert _save_times(tmp_path / "before", before) == (3, 60.0, 30.0)
    # One save, the break, three saves: the longer session is the one after it.
    after = ["2026-09-24T10:00:00Z", "2026-09-24T12:00:00Z", "2026-09-24T12:00:30Z",
             "2026-09-24T12:01:10Z"]  # fmt: skip
    assert _save_times(tmp_path / "after", after) == (3, 70.0, 35.0)


def test_the_pace_needs_three_stamped_judgments(tmp_path):
    write_gold(tmp_path, [{"id": "q1"}, {"id": "q2"}], path=GOLD_BEFORE_AUGMENTATION)
    stamps = {"q1": "2026-09-24T10:00:00Z", "q2": "2026-09-24T10:00:30Z"}
    journal = {q: {"1": {"annotated_at": s}} for q, s in stamps.items()}
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    with pytest.raises(ValueError, match="three stamped judgments"):
        datasets_rounds.save_times(Sources(tmp_path))


def test_the_review_counts_the_annotators_corrections_and_leaves_out_derived_ones(tmp_path):
    write_gold(tmp_path, [{"id": q} for q in ("q1", "q2", "q3")], path=GOLD_BEFORE_AUGMENTATION)
    derived = {**correction(["a"]), "derived": True, "rule": "r"}
    del derived["annotator"]
    journal = {
        "q1": {"1": {"relevant": ["b"], "corrections": [correction(["a"], ["b", "c"])]}},
        "q2": {"1": {"relevant": ["a"], "corrections": [correction(["a"])]}},  # no change
        "q3": {"1": {"relevant": ["a"], "corrections": [derived]}},
    }
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    counts = datasets_rounds.review_counts(Sources(tmp_path))
    assert (counts.reopened, counts.changed, counts.added, counts.removed) == (2, 1, 2, 1)
