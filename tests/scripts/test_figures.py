"""Tests for scripts/figures.py: what a figure refuses before it draws."""

from __future__ import annotations

import pytest
from figures import draw_latency, draw_venue_coverage

from crossvenue_rag.models import RETRIEVAL_STEPS
from crossvenue_rag.reporting.compute.retrieval import METHODS
from crossvenue_rag.reporting.run_ids import RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import write_gold, write_retrieval_run


def test_the_latency_figure_refuses_a_run_without_a_method_it_draws(tmp_path):
    steps = {f"{step}_p50_ms": 1.0 for step in RETRIEVAL_STEPS}
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, summary=[{"method": "other", **steps}])
    with pytest.raises(ValueError, match=f"the step table has no '{METHODS[0]}'"):
        draw_latency(Sources(tmp_path))


def test_the_coverage_figure_refuses_a_run_whose_by_category_table_lacks_a_method(tmp_path):
    write_gold(tmp_path, [{"id": "q1"}])
    per_query = [{"method": m, "question_id": "q1", "venue_coverage@10": 1.0} for m in METHODS]
    other = [{"category": "single", "method": "bm25", "venue_coverage@10": 1.0}]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, per_query=per_query, by_category=other)
    with pytest.raises(ValueError, match=r"by_category\.csv has no \('comparison', 'bm25'\)"):
        draw_venue_coverage(Sources(tmp_path))
