"""Tests for `crossvenue_rag.reporting.compute.documents`: list items and table cells."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import documents

LIMITS = """# Limits

Intro line.

- first limit
- second limit
  a continuation line of the second limit
- third limit

## Next section

- an item after the end marker
"""

TABLE = """Some text.

| venue | kind | fee |
|---|---|---|
| binance | cex | 0.10 |
| bybit | cex | 0.10 |
| uniswap | dex | 0.30 |

| venue | other |
|---|---|
| ignored | table |
"""


def test_list_items_are_the_dash_lines_between_start_and_end():
    items = documents.list_items(LIMITS, "# Limits", "## Next section")
    assert items == ["- first limit", "- second limit", "- third limit"]


def test_list_items_end_at_the_first_end_marker_after_the_start():
    text = "## End\n- before\n# Start\n- inside\n## End\n- after\n"
    assert documents.list_items(text, "# Start", "## End") == ["- inside"]


def test_a_missing_start_marker_raises():
    with pytest.raises(ValueError, match="substring not found"):
        documents.list_items(LIMITS, "# Absent", "## Next section")


def test_table_column_reads_the_named_column_until_the_table_ends():
    assert documents.table_column(TABLE, "| venue | kind", "kind") == ["cex", "cex", "dex"]
    assert documents.table_column(TABLE, "| venue | kind", "venue") == [
        "binance",
        "bybit",
        "uniswap",
    ]


def test_table_column_counts_the_rows_per_value():
    counts = documents.table_column_counts(TABLE, "| venue | kind", "kind")
    assert counts == {"cex": 2, "dex": 1}
