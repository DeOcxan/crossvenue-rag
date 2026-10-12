from __future__ import annotations

import pytest
from figure_style import agree


def test_a_drawn_value_equal_to_the_reported_one_at_the_metric_decimals_passes():
    agree(0.12341, 0.12339, "bm25 recall@10")


def test_a_drawn_value_that_differs_from_the_reported_one_is_a_value_error_naming_both():
    with pytest.raises(ValueError, match=r"bm25 recall@10: drawn 0\.5000, the run reports 0\.6000"):
        agree(0.5, 0.6, "bm25 recall@10")
