"""Tests for `crossvenue_rag.retrieval.venue_detect`."""

from __future__ import annotations

from crossvenue_rag.retrieval.venue_detect import VenueDetector, detection_outcome


def test_detector_reads_names_and_aliases_whole_word_in_order():
    detector = VenueDetector.from_file("data/sources.yaml")
    assert detector.detect("Is pancake v3 like UNI V3?") == ["pancakeswap", "uniswap"]
    assert detector.detect("on HL the nonce vs dydx") == ["hyperliquid", "dydx"]
    assert detector.detect("unified margin on bybit") == ["bybit"]  # "uni" is whole-word
    assert detector.detect("the hloc candles") == []
    assert detector.detect("BNB chain gas") == []  # bnb is deliberately not binance


def test_detection_is_exact_partial_missed_wrong_or_none():
    assert detection_outcome(["a", "b"], ["b", "a"]) == "exact"
    assert detection_outcome(["a"], ["a", "b"]) == "partial"
    assert detection_outcome([], ["a"]) == "missed"
    assert detection_outcome(["c"], ["a"]) == "wrong"
    assert detection_outcome([], []) == "none"
