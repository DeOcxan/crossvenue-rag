"""Tests for `reporting.compute.corpus_pancakeswap`, on a corpus written under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import corpus_pancakeswap as pancakeswap
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.reporting.conftest import write_chunks

UNISWAP = (
    "swap router exact input output amount slippage deadline",
    "concentrated liquidity position range ticks price",
    "governance token voting proposal delegate quorum",
)
# The same text as Uniswap's first chunk; three terms of its second; no shared term.
COPY = UNISWAP[0]
HALF = "concentrated liquidity position farm syrup lottery prediction"
APART = "farm syrup lottery prediction"


def corpus(root, pancakeswap_texts, uniswap_texts=UNISWAP):
    chunks = [make_chunk(f"u{i}", "uniswap", t, "dex") for i, t in enumerate(uniswap_texts)]
    chunks += [
        make_chunk(f"p{i}", "pancakeswap", t, "dex") for i, t in enumerate(pancakeswap_texts)
    ]
    # A chunk of another venue is left out of the match.
    chunks.append(make_chunk("b0", "binance", COPY))
    write_chunks(root, chunks)
    return Sources(root)


def test_each_pancakeswap_chunk_gets_its_best_uniswap_cosine(tmp_path):
    src = corpus(tmp_path, [COPY, HALF, APART])
    best = pancakeswap.best_match(src)
    assert best.tolist() == pytest.approx([1.0, best[1], 0.0])
    assert 0.0 < best[1] < 1.0
    assert pancakeswap.pancakeswap_median_cosine(src) == pytest.approx(best[1])


def test_matches_at_least_count_the_chunks_reaching_each_threshold(tmp_path):
    src = corpus(tmp_path, [COPY, HALF, APART])
    half = float(pancakeswap.best_match(src)[1])
    assert pancakeswap.matches_at_least(src, [0.0, half, 0.99]) == [
        pancakeswap.ThresholdMatches(0.0, 3, 3),
        pancakeswap.ThresholdMatches(half, 2, 3),
        pancakeswap.ThresholdMatches(0.99, 1, 3),
    ]


def test_a_copied_chunk_breaks_the_cosine_bound(tmp_path):
    assert not pancakeswap.pancakeswap_below_bound(corpus(tmp_path / "copy", [COPY, APART]))
    assert pancakeswap.pancakeswap_below_bound(corpus(tmp_path / "apart", [HALF, APART]))


def test_chunks_beyond_one_batch_are_all_matched(tmp_path):
    texts = [APART] * pancakeswap.BATCH + [COPY]
    src = corpus(tmp_path, texts)
    assert pancakeswap.matches_at_least(src, [0.99]) == [
        pancakeswap.ThresholdMatches(0.99, 1, pancakeswap.BATCH + 1)
    ]


def test_a_fee_tier_is_written_at_two_decimals_and_searched_in_its_shortest_form_too():
    assert pancakeswap.fee_tier_text(0.3) == "0.30"
    assert pancakeswap.fee_tier_terms(0.3) == {"0.30%", "0.3%"}


@pytest.mark.parametrize(
    ("uniswap", "pancakeswap_text", "stated"),
    [
        ("The fee tier is 0.3% per swap.", "Fees: 0.25% per trade.", True),
        ("The fee tier is 0.30% per swap.", "Fees: 0.25% per trade.", True),
        ("The fee tier is 0.3% per swap.", "Fees: 0.30% per trade.", False),
        ("The fee tier is 0.05% per swap.", "Fees: 0.25% per trade.", False),
    ],
)
def test_each_venue_must_state_its_own_fee_tier(tmp_path, uniswap, pancakeswap_text, stated):
    src = corpus(tmp_path, [pancakeswap_text], [uniswap])
    assert pancakeswap.fee_tiers_stated(src) is stated
