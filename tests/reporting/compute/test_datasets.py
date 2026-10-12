"""Tests for `reporting.compute.datasets`, on a gold set written under tmp_path."""

from __future__ import annotations

import json

from crossvenue_rag.dataset.journal import rounds_path
from crossvenue_rag.dataset.selection import TARGETS
from crossvenue_rag.reporting.compute import datasets
from crossvenue_rag.reporting.run_ids import (
    CANDIDATES_FINAL,
    COMPARISON_SUPPLEMENT,
    DEDUP_IDMAP,
    GOLD,
    GOLD_BEFORE_AUGMENTATION,
    POOL_RUN,
    RETRIEVAL_RUN,
    ROUND2_SAMPLE,
)
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import augmentation
from tests.reporting.conftest import (
    write_file,
    write_fixture,
    write_gold,
    write_retrieval_run,
    write_sources,
)


def test_comparison_questions_are_selected_by_category(tmp_path):
    write_fixture(tmp_path)
    src = Sources(tmp_path)
    assert datasets.comparison_ids(src) == ["cv-0001"]


def test_one_sided_comparisons_have_a_venue_missing_or_absent(tmp_path):
    coverage = [
        {"binance": "covered"},
        {"binance": "missing_from_pool"},
        {"bybit": "absent_in_docs"},
    ]
    questions = [
        {"id": f"cv-000{i}", "category": "comparison", "venue_coverage": c}
        for i, c in enumerate(coverage)
    ]
    questions.append({"id": "cv-0009", "category": "fact", "venue_coverage": coverage[1]})
    write_gold(tmp_path, questions)
    assert datasets.one_sided_comparisons(Sources(tmp_path)) == 2


def test_questions_are_counted_and_labelled_ones_have_gold(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001", "gold_chunk_ids": ["c1"]}, {"id": "cv-0002"}])
    src = Sources(tmp_path)
    assert datasets.question_count(src) == 2
    assert datasets.labelled_count(src) == 1


def test_split_questions_name_two_venues_the_detector_finds(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "cv-0002", "question": "bybit vs binance"},
            {"id": "cv-0001", "question": "binance only"},
            {"id": "cv-0003", "question": "Binance and BYBIT"},
        ],
    )
    write_sources(tmp_path, ["binance", "bybit"])
    src = Sources(tmp_path)
    assert datasets.split_ids(src) == ["cv-0002", "cv-0003"]


def test_augmentation_counts_shown_and_relevant_chunks_by_pool_rank(tmp_path):
    ranked = " ".join(f"c{i}" for i in range(1, 13))  # c11 and c12 below rank 10
    write_retrieval_run(
        tmp_path, POOL_RUN, per_query=[{"question_id": "cv-0001", "retrieved_ids": ranked}]
    )
    rounds = {
        "cv-0001": {"1": {"augmentations": [augmentation(["c2", "c11", "c12"], ["c2", "c12"])]}},
        "cv-0002": {"1": {}},
    }
    write_file(tmp_path, str(rounds_path(GOLD)), json.dumps(rounds))
    assert datasets.augmentation(Sources(tmp_path)) == datasets.AugmentationCounts(
        shown=3, relevant=2, shown_top10=1, relevant_top10=1
    )
    assert datasets.augmentation(Sources(tmp_path)).relevant_lower == 1


def test_the_gold_set_before_augmentation_is_counted(tmp_path):
    coverage = {"binance": "missing_from_pool", "bybit": "missing_from_pool"}
    questions = [
        {"id": "cv-0001", "venue_coverage": coverage},
        {"id": "cv-0002", "venue_coverage": {"binance": "covered"}},
        {"id": "cv-0003", "category": "fact", "venue_coverage": coverage},
    ]
    write_gold(tmp_path, questions, path=GOLD_BEFORE_AUGMENTATION)
    src = Sources(tmp_path)
    assert datasets.comparison_count_before_augmentation(src) == 2
    assert datasets.missing_from_pool_before_augmentation(src) == 2


def test_a_migrated_pool_counts_candidates_and_distinct_texts(tmp_path):
    migrated = {"pool": ["o1", "o2", "o3", "o9"], "relevant": [], "index_version": None}
    journal = {"cv-0001": {"1": {"migrated_from": migrated}}}
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    idmap = {
        "old_ids": 3,
        "map": {"o1": "n1", "o2": "n1", "o3": "n3"},
        "old_chunks_sha256": "",
        "to_index_version": "",
    }
    write_file(tmp_path, DEDUP_IDMAP, json.dumps(idmap))
    # o1 and o2 hold one text; o9 is not in the map and keeps its own id.
    assert datasets.migrated_pool(Sources(tmp_path), "cv-0001") == (4, 3)


def test_the_round2_sample_skips_blank_and_comment_lines(tmp_path):
    write_file(tmp_path, ROUND2_SAMPLE, "# sample\n\ncv-0001\ncv-0002\n  \n")
    assert len(datasets.round2_sample(Sources(tmp_path))) == 2


def test_relabel_decisions_are_questions_times_the_candidates_shown(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}, {"id": "cv-0002"}])
    write_file(tmp_path, f"results/{RETRIEVAL_RUN}/config.yaml", "eval:\n  pool_display: 25\n")
    assert datasets.relabel_decisions(Sources(tmp_path)) == 50


def test_the_target_decisions_are_the_category_targets_summed_times_the_candidates_shown(tmp_path):
    write_file(tmp_path, f"results/{RETRIEVAL_RUN}/config.yaml", "eval:\n  pool_display: 25\n")
    assert datasets.target_decisions(Sources(tmp_path)) == 25 * sum(TARGETS.values())


def test_the_questions_of_a_category_are_its_ids_in_set_order(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "cv-0003", "category": "noisy"},
            {"id": "cv-0001"},
            {"id": "cv-0002", "category": "noisy"},
        ],
    )
    assert datasets.category_ids(Sources(tmp_path), "noisy") == ["cv-0003", "cv-0002"]


def test_a_question_saved_without_a_category_counts_under_its_qtype(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001", "category": None}, {"id": "cv-0002"}])
    assert datasets.category_ids(Sources(tmp_path), "comparison") == ["cv-0001", "cv-0002"]


def test_detection_counts_a_question_that_names_no_venue_and_has_none_detected(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "cv-0001", "question": "binance or bybit"},
            {"id": "cv-0002", "question": "no venue", "venues": [], "category": "fact"},
        ],
    )
    write_sources(tmp_path, ["binance", "bybit"])
    counts = datasets.detection_counts(Sources(tmp_path), "all")
    assert counts == {"exact": 1, "partial": 0, "missed": 0, "wrong": 0, "none": 1}


def test_the_multi_venue_group_counts_distinct_venues(tmp_path):
    write_gold(
        tmp_path,
        [{"id": "cv-0001"}, {"id": "cv-0002", "venues": ["binance", "binance"]}],
    )
    assert datasets.detection_group(Sources(tmp_path), "multi-venue") == ["cv-0001"]


def category_fixture(root):
    """Each category at its target, plus 2 unanswerable, 1 adversarial and 1 each of the
    three even categories; one question expects a refusal."""
    counts = dict(TARGETS)
    counts["unanswerable"] += 2
    counts["adversarial"] += 1
    for c in datasets.EVEN_EXCESS:
        counts[c] += 1
    questions = [
        {"id": f"{c}-{i}", "category": c, "expected_behavior": "answer"}
        for c, k in counts.items()
        for i in range(k)
    ]
    questions[-1]["expected_behavior"] = "refuse"
    write_gold(root, questions)
    return Sources(root)


def test_categories_are_counted_against_their_targets(tmp_path):
    src = category_fixture(tmp_path)
    assert datasets.category_count(src, "unanswerable") == TARGETS["unanswerable"] + 2
    assert datasets.answer_expected_count(src) == datasets.question_count(src) - 1
    assert datasets.category_excess(src)["unanswerable"] == 2
    assert datasets.category_excess(src)["comparison"] == 0
    assert datasets.excess_total(src) == 6
    assert datasets.excess_total(src, excluding=("unanswerable",)) == 4
    assert (
        datasets.non_comparison_count(src) == datasets.question_count(src) - TARGETS["comparison"]
    )
    assert datasets.excess_is_as_stated(src)


def test_an_uneven_or_short_category_is_not_as_stated(tmp_path):
    src = category_fixture(tmp_path)
    questions = [q.to_dict() for q in src.questions.values()]
    write_gold(tmp_path, [q for q in questions if q["id"] != "noisy-0"])
    assert not datasets.excess_is_as_stated(Sources(tmp_path))
    write_gold(tmp_path, [q for q in questions if q["id"] != "fact-0"])
    assert not datasets.excess_is_as_stated(Sources(tmp_path))


def test_a_question_equal_to_its_draft_is_unchanged_and_question_lengths_have_a_median(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "cv-0001", "question": "one two three"},
            {"id": "cv-0002", "question": "one two three four five"},
            {"id": "cv-0003", "question": "one"},
        ],
    )
    drafts = [
        {"id": "cv-0001", "question": "one two three"},
        {"id": "cv-0002", "question": "a different draft"},
    ]
    write_gold(tmp_path, drafts, path=CANDIDATES_FINAL)
    src = Sources(tmp_path)
    assert datasets.unchanged_drafts(src) == 1
    assert datasets.median_question_words(src) == 3.0


def test_a_question_counts_once_per_venue_it_names_and_a_dex_by_its_venue_type(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "cv-0001", "venues": ["binance", "ccxt"]},
            {"id": "cv-0002", "venues": ["binance", "uniswap"]},
            {"id": "cv-0003", "venues": ["bybit", "binance", "bybit"]},
        ],
    )
    write_file(
        tmp_path,
        "data/sources.yaml",
        "sources:\n  - {venue: binance, kind: git, venue_type: cex, license: x, repo: r}\n"
        "  - {venue: uniswap, kind: git, venue_type: dex, license: x, repo: r}\n",
    )
    src = Sources(tmp_path)
    assert datasets.questions_naming(src, "binance") == 3
    assert datasets.questions_naming(src, "bybit") == 1
    assert datasets.fewest_naming_other(src, "bybit") == 1
    assert datasets.fewest_naming_other(src, "ccxt") == 1
    assert datasets.questions_naming_dex(src) == 1


def test_the_supplement_is_the_candidates_of_its_prompt(tmp_path):
    rows = [
        {
            "id": "a",
            "prompt_hash": datasets.COMPARISON_SUPPLEMENT_HASH,
            "venues": ["ccxt", "binance"],
        },
        {
            "id": "b",
            "prompt_hash": datasets.COMPARISON_SUPPLEMENT_HASH,
            "venues": ["bybit", "binance"],
        },
        {"id": "c", "prompt_hash": "other", "venues": ["ccxt"]},
    ]
    write_gold(tmp_path, rows, path=COMPARISON_SUPPLEMENT)
    assert datasets.supplement_naming(Sources(tmp_path), "ccxt") == (1, 2)
