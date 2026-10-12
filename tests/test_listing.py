"""Tests for `crossvenue_rag.listing`."""

from __future__ import annotations

from crossvenue_rag import listing


def test_prose_joins_its_last_two_items_with_and():
    assert listing.prose(["a"]) == "a"
    assert listing.prose(["a", "b"]) == "a and b"
    assert listing.prose(["a", "b", "c"]) == "a, b and c"


def test_first_items_lists_up_to_the_limit_and_counts_the_rest():
    assert listing.first_items(["a", "b"]) == "a, b"
    six = [str(i) for i in range(listing.LISTED_ITEMS + 3)]
    assert listing.first_items(six, "; ") == "0; 1; 2; 3; 4; ... 3 more"
