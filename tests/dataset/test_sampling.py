"""The round-2 subsample: stratified by category, reproducible from its seed."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from crossvenue_rag.dataset.sampling import (
    read_sample_file,
    stratified_quotas,
    stratified_sample,
    write_sample_file,
)
from crossvenue_rag.errors import UserError
from tests.dataset.factories import SIZES, gold_like, question


def test_quotas_are_proportional_with_one_per_category():
    quotas = stratified_quotas(SIZES, 25)
    assert sum(quotas.values()) == 25
    assert quotas == {
        "comparison": 11,
        "unanswerable": 5,
        "adversarial": 3,
        "fact": 2,
        "ambiguous": 1,
        "noisy": 1,
        "version_sensitive": 1,
        "multihop": 1,
    }


def test_small_categories_still_get_one_even_against_a_dominant_one():
    assert stratified_quotas({"a": 100, "b": 1, "c": 1}, 3) == {"a": 1, "b": 1, "c": 1}


def test_quota_never_exceeds_the_category_size():
    quotas = stratified_quotas({"a": 3, "b": 3}, 5)
    assert sum(quotas.values()) == 5 and all(q <= 3 for q in quotas.values())


def test_too_small_or_too_large_a_sample_is_refused():
    with pytest.raises(UserError, match="cannot hold one of each of 8 categories"):
        stratified_quotas(SIZES, 7)
    with pytest.raises(UserError, match="exceeds the 2 eligible questions"):
        stratified_quotas({"a": 2}, 3)


def test_every_number_in_a_refused_sample_has_a_thousands_separator():
    with pytest.raises(UserError) as refused:
        stratified_quotas({"a": 1000, "b": 500}, 2000)
    assert str(refused.value) == "a sample of 2,000 exceeds the 1,500 eligible questions"
    assert refused.value.hint == "ask for at most 1,500"


def test_sample_matches_the_quotas_and_contains_only_judged_questions():
    questions = gold_like() + [question(900 + i, "comparison", judged=False) for i in range(10)]
    sample = stratified_sample(questions, 25, seed=20260924)
    assert len(sample) == 25
    assert all(q.annotated_at for q in sample)
    assert Counter(q.effective_category for q in sample) == Counter(stratified_quotas(SIZES, 25))


def test_same_seed_same_sample_and_a_different_seed_a_different_one():
    questions = gold_like()
    first = [q.id for q in stratified_sample(questions, 25, seed=20260924)]
    again = [q.id for q in stratified_sample(list(reversed(questions)), 25, seed=20260924)]
    other = [q.id for q in stratified_sample(questions, 25, seed=1)]
    assert sorted(first) == sorted(again)
    assert first != other


def test_sample_file_round_trips(tmp_path: Path):
    questions = gold_like()
    sample = stratified_sample(questions, 25, seed=20260924)
    path = tmp_path / "round2_sample.txt"
    write_sample_file(path, sample, seed=20260924)
    assert read_sample_file(path) == [q.id for q in sample]


def test_a_sample_file_takes_the_first_token_of_each_line(tmp_path):
    listing = tmp_path / "review.txt"
    listing.write_text('# header\ncv-0043 "binance lets me raise"\n\ncv-0046 x\n')
    assert read_sample_file(listing) == ["cv-0043", "cv-0046"]


def test_a_question_is_cited_by_the_first_six_words_of_its_text():
    from crossvenue_rag.dataset.sampling import cited_words
    from crossvenue_rag.models import Question

    question = Question("cv-0001", "one  two three four five six seven", "gold", [], "fact")
    assert cited_words(question) == "one two three four five six"


def test_a_missing_or_undecodable_sample_file_is_refused(tmp_path):
    with pytest.raises(UserError, match="not found"):
        read_sample_file(tmp_path / "missing.txt")
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(UserError, match="not UTF-8"):
        read_sample_file(binary)
