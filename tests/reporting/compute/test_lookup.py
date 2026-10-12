import pytest

from crossvenue_rag.reporting.compute.lookup import entry


def test_an_entry_is_the_value_of_its_key_and_a_missing_key_is_a_value_error_naming_it():
    assert entry({"bm25": 1}, "bm25", "summary.csv") == 1
    with pytest.raises(ValueError, match=r"summary\.csv has no 'dense'"):
        entry({"bm25": 1}, "dense", "summary.csv")
